import sys

sys.path.append("./dsls")
sys.path.append("../")
sys.path.append("../../")

from tqdm import tqdm
import torch
from ShapeMOD import DSL, Function, ProgNode, OrderedProg
import pickle
import re
import sa_utils as utils
import importlib

def clamp(v, a, b):
    return min(max(v, a), b)

def make_function(name, args):
    args = [str(arg) for arg in args]
    return '{}({})'.format(name, ", ".join(args))

def getCuboidDims(F, f_params, j_param_map, bbox_dims):
    params = [1.0] + bbox_dims.tolist()
    f_params = f_params.tolist()
    for var in j_param_map:
        if var[0] == 'd':
            params.append('dummy')
        else:
            params.append(f_params.pop(0))
        
    cube_dims = []
    for fn, logic in zip(F.structure, F.getLogic()):
        if fn != "Cuboid":
            continue

        cube_line = []
        for var in logic[:3]:
            assert var[0] == 'f'
            sum = 0.
            for ind, scale in var[1]:
                sum += float(scale) * params[ind]
            cube_line.append(sum)
        cube_dims.append(cube_line)

    return cube_dims

def tensorize(prog, func_map, max_cparams, max_dparams):
    if prog is None:
        return None
        
    nmap = {
        'bbox': 0,
        'X': 0,
        'Y': 1,
        'Z': 2,
        'left': 0,
        'right': 1,
        'bot': 2,
        'top': 3,
        'back': 4,
        'front': 5
    }

    cn_map = {0:0}
    t_lines = [('START',)]
    count = 1    
    child_map = [[]]

    for ln, (ret, line) in enumerate(prog):
        child_inds = []
        for n in ret:
            if 'cube' in n:
                nmap[f'temp{int(n[4:])}'] = count
                cn_map[int(n[4:])+1] = count
                child_inds.append(count)
                count += 1
            elif 'bbox' in n:
                child_inds.append(0)
                
        t_lines.append([l.replace('cube','temp') if isinstance(l, str) else l for l in line]) 
        child_map.append(child_inds)

        
    t_lines.append(('STOP',))
    child_map.append([])
    
    prog_funcs = []
    prog_cvars = []
    prog_dvars = []
    
    for t_line in t_lines:        
        prog_funcs.append(func_map[t_line[0]])
        line_cvars = torch.zeros(max_cparams).float()
        line_dvars = torch.zeros(max_dparams).long()

        dv_count = 0
        cv_count = 0

        for par in t_line[1:]:
            if par in nmap:
                line_dvars[dv_count] = nmap[par]
                dv_count += 1
            else:
                line_cvars[cv_count] = float(par)
                cv_count += 1
                
        prog_cvars.append(line_cvars)
        prog_dvars.append(line_dvars)

    prog_funcs = torch.tensor(prog_funcs)
    prog_cvars = torch.stack(prog_cvars)
    prog_dvars = torch.stack(prog_dvars)
    
    return (prog_funcs, prog_cvars, prog_dvars, cn_map, child_map)


def getCatTypeMap(dsl):
    cat_type_map = {}    

    for n, f in dsl.library.items():
        _cat_type_map = {}
        c_count = 0
        seen_inds = set()
        for st, log in zip(f.getStructure(), f.getLogic()):
            for l in log:
                if l[0] == 'c' and l[1] == 'var' and l[2] not in seen_inds:
                    seen_inds.add(l[2])
                    name = f'c_var_{c_count}'
                    c_count += 1
                    if st == 'squeeze':                    
                        _cat_type_map[name] = 'sq'
                    elif st == 'translate' or st == 'reflect':
                        _cat_type_map[name] = 'sym'
        cat_type_map[n] = _cat_type_map
        
    return cat_type_map

def getBestProgram(dsl, node, order_thresh = 1):

    def scoreOrder(ord):
        sum = 0.
        place = 1.
        for o in ord:
            sum += o[1] * place
            place *= .1
        return sum
    
    best_score = 1e8
    best_program = None
            
    res = []

    if len(node.orders) == 0:
        return None
    
    for o in node.orders[:dsl.abs_order_num]:
        canon_sig, line_attrs, ret_attrs = o.canon_info
        score, program, _  = dsl.getApproxBestOps(line_attrs, ret_attrs)
        res.append((score, canon_sig, program))

    res.sort()    
    best_score = res[0][0]
        
    order_res = [
        (scoreOrder(r[1]), r[2]) for r in res
        if r[0] < (best_score + order_thresh)
    ]

    order_res.sort()
    best_program = order_res[0][1]
    
    return best_program

def getBestPrograms(dsl, nodes):
    best_progs = []
    for node in tqdm(nodes):
        bp = getBestProgram(dsl, node)
        best_progs.append(bp)
    return best_progs

def form_training_data(dsl, nodes):
    max_cparams = 0 # start with attach max
    max_dparams = 0 # start with squeeze max
    func_map = {'START': 0, 'STOP': 1}    
    
    cparam_map = {0: [], 1: []}
    dparam_map = {0: [], 1: []}
    jparam_map = {0: [], 1: []}
    
    num_cube_map = {0: 0, 1: 0}
    cat_type_map = getCatTypeMap(dsl)
    
    for n, f in dsl.library.items():

        func_map[n] = len(func_map)
                        
        interf = f.getInterface()
        structure = f.getStructure()
        
        cparam_map[func_map[n]] = []
        dparam_map[func_map[n]] = []
        jparam_map[func_map[n]] = []
        for i in interf:
            _i = i.split('_')[0]
            if _i == 'i':
                jparam_map[func_map[n]].append(f'd_{len(dparam_map[func_map[n]])}')                
                dparam_map[func_map[n]].append(_i)
            elif _i == 'c':
                jparam_map[func_map[n]].append(f'd_{len(dparam_map[func_map[n]])}')
                dparam_map[func_map[n]].append(cat_type_map[n][i])
            else:
                jparam_map[func_map[n]].append(f'c_{len(cparam_map[func_map[n]])}')
                cparam_map[func_map[n]].append(_i)                    

        num_cube_map[func_map[n]] = structure.count("Cuboid")
        max_cparams = max(len(cparam_map[func_map[n]]), max_cparams)
        max_dparams = max(len(dparam_map[func_map[n]]), max_dparams)

    best_progs = getBestPrograms(dsl, nodes)        
    best_progs = [tensorize(bp, func_map, max_cparams, max_dparams) for bp in best_progs]
    
    return best_progs, func_map, cparam_map, dparam_map, jparam_map, max_cparams, max_dparams, num_cube_map
    
def writeData(dsl, out_name, input_data, category):
    full_nodes = pickle.load(open(input_data, 'rb'))    
        
    simp_inds = list(set([n.ind.split('_')[0] for n in full_nodes]))
    
    inds = []
    nodes = []
    for node in full_nodes:
        ind = node.ind
        if ind.split('_')[0] in simp_inds:            
            inds.append(ind)
            nodes.append(node)

    node_tensors, func_map, cparam_map, dparam_map, jparam_map, \
        max_cparams, max_dparams, num_cube_map  = form_training_data(dsl, nodes)
        
    training_data = []

    metadata = {
        'func_map': func_map,
        'cparam_map': cparam_map,
        'dparam_map': dparam_map,
        'jparam_map': jparam_map,
        'max_cparams': max_cparams,
        'max_dparams': max_dparams,
        'num_cube_map': num_cube_map,
        'dsl': dsl,
    }

    metadata['rev_func_map'] = {v:k for k, v in metadata['func_map'].items()}

    for key in ('i', 'sq', 'sym'):
        metadata[f'max_d_{key}_params'] = max([
            len([ __l for __l in _l if __l == key])
            for _i, _l in metadata['dparam_map'].items()            
        ])
        
    for ind in simp_inds:        
        good_ind = True
        if f'{ind}_{category}' not in inds:
            print("Failed a lookup on root")
            continue
        root = {'lookup': f'{ind}_{category}'}
        q = [root]
        while(len(q) > 0):
            node = q.pop(0)            
            index = inds.index(node['lookup'])
            node['children_names'] = nodes[index].children_names

            if node_tensors[index] is None:                
                good_ind = False
                q = []
                break

            node['func_gt'] = node_tensors[index][0].numpy()
            node['cparam_gt'] = node_tensors[index][1].numpy()
            node['dparam_gt'] = node_tensors[index][2].numpy()
            node['child_gt'] = node_tensors[index][4]
                        
            node['children'] = []
            child_order = {v:k for k,v in node_tensors[index][3].items()}
            children_names = node.pop('children_names')
            
            for ci in range(len(children_names)):                
                cn = children_names[child_order[ci]]
                c_lookup = f'{ind}_{cn}'
                if c_lookup in inds:
                    c_node = {'lookup': c_lookup}
                    q.append(c_node)
                    node['children'].append(c_node)
                else:
                    node['children'].append({})

        if good_ind:
            training_data.append((ind, root))
        else:
            print("Failed")

    pickle.dump(training_data, open(f'{out_name}_train.data', 'wb'))
    pickle.dump(metadata, open(f'{out_name}_train.meta', 'wb'))

def indToFace(i):
    m = ['left', 'right', 'bot', 'top', 'back', 'front']
    return m[i]

def indToAxis(i):
    m = ['X', 'Y', 'Z']
    return m[i]

def indToCube(i):
    if i == 0:
        return 'bbox'
    else:
        return f'cube{i-1}'

def getSALine(logic, d_line, d_ret):
    sa_line = []
    
    for var in logic:
        if var[0] == 'f':
            sum = 0.
            for ind, scale in var[1]:
                sum += float(scale) * d_line[ind]
            sa_line.append(sum)
                
        elif var[0] == 'i':
            if var[1] == 'const':
                sa_line.append(var[2])
            elif var[1] == 'ret':
                sa_line.append(d_ret[var[2]])
            else:
                sa_line.append(d_line[var[2]])

        elif var[0] == 'b':
            if var[1] == 'const':
                sa_line.append(var[2])
            else:
                sa_line.append(d_line[var[2]])
                    
        elif var[0] == 'c':
            if var[1] == 'const':
                sa_line.append(var[2])
            else:
                sa_line.append(d_line[var[2]])
        else:
            assert False, 'bad sa line arg'
                    
    return sa_line
    
def makeSALines(F, d_line, d_ret, last_cube):
    ret_num = 0
    sa_lines = []

    for fn, logic in zip(F.structure, F.getLogic()):            
        params = getSALine(logic, d_line, d_ret)

        if fn == 'Cuboid':
            params[0] = clamp(params[0], 0.01, d_line[1])
            params[1] = clamp(params[1], 0.01, d_line[2])
            params[2] = clamp(params[2], 0.01, d_line[3])
        else:
            params.insert(0, last_cube)
                
        if fn == 'attach':                
            for i in range(2, 8):
                params[i] = clamp(params[i], 0.0, 1.0)

            # FLIP BBOX ATTACHES
            if params[1] == 'bbox':
                params[6] = 1 - params[6]

                # Sem Validity that attaches to BBox need to be top or bot
                if params[6] <= .5 and params[6] > .1:
                    params[6] = 0.1

                elif params[6] >= .5 and params[6] <.9:
                    params[6] = 0.9
                    
        elif fn == 'squeeze':
            for i in range(4, 6):
                params[i] = clamp(params[i], 0.0, 1.0)

        elif fn == 'translate':
            params[2] = max(round(params[2] * utils.TRANS_NORM), 1)
            params[3] = clamp(params[3], 0.0, 1.0)
                    
        mf = make_function(fn, params)
        ret = ""
        if fn == "Cuboid":
            ret = f"{d_ret[ret_num]} = "
            last_cube = d_ret[ret_num]
            ret_num += 1
                                
        sa_lines.append(ret + mf)

    return sa_lines, last_cube
    
def makeSAProg(dsl, dsl_prog):
    bbox_dims = dsl_prog[0][1][1:]
    sa_prog = [f"{dsl_prog[0][0][0]} = {utils.make_function(dsl_prog[0][1][0], bbox_dims)}"]
    last_cube = 'bbox'
    for d_ret, d_line in dsl_prog[1:]:
        new_lines, last_cube = makeSALines(
            dsl.library[d_line[0]],
            [1.0] + list(bbox_dims[:3]) + list(d_line[1:]),
            d_ret,
            last_cube
        )
        sa_prog += new_lines
            
    return sa_prog
    
def fillProgram(dsl, hier, meta, func_field, cparam_field, dparam_field):
    prog = []
    cube_num = -1
    
    f_gt = hier[func_field].tolist()
    cp_gt = hier[cparam_field].tolist()
    dp_gt = hier[dparam_field].tolist()
    
    inv_func_map = {v:k for k,v in meta['func_map'].items()}
    
    for f_num, cl_prm, dl_prm in zip(f_gt, cp_gt, dp_gt):
        if f_num <= 1:
            continue
        
        fn = inv_func_map[f_num]
        
        line = [fn]
        j_prm = meta['jparam_map'][f_num]

        for j in j_prm:

            info = j.split('_')
            index = int(info[1])
            if info[0] == 'c':                
                lookup = cl_prm
                ptype = meta['cparam_map'][f_num][index]
            else:
                lookup = dl_prm
                ptype = meta['dparam_map'][f_num][index]
                
            if ptype == 'f':
                line.append(round(float(lookup[index]), 2))
            elif ptype == 'i':
                if lookup[index] == 0:
                    line.append('bbox')
                else:                    
                    line.append(f'cube{int(lookup[index])-1}')
            elif ptype == 'sq':
                line.append(indToFace(lookup[index]))

            elif ptype == 'sym':
                line.append(indToAxis(lookup[index]))
                
            elif ptype == 'b':
                line.append(True if lookup[index] > 0 else False)
            else:
                assert False, f'bad input {j_prm}'

        ret = []
        struct = dsl.library[fn].getStructure() 
        for s in struct:
            if s == "Cuboid":
                if cube_num < 0:
                    ret.append(f'bbox')
                else:
                    ret.append(f'cube{cube_num}')
                cube_num += 1

        prog.append((tuple(ret), tuple(line)))
        
    hier['dsl_prog'] = prog
    hier['prog'] = makeSAProg(dsl, prog)
    
    for c in hier['children']:
        if len(c) > 0:
            fillProgram(dsl, c, meta, func_field, cparam_field, dparam_field)
               
def testGetProgram(dsl, out_name):
    from ShapeAssembly import hier_execute

    data = pickle.load(open(f'{out_name}_train.data', "rb"))
    metadata = pickle.load(open(f'{out_name}_train.meta', "rb"))
    for d in data:
        print(d[0])
        fillProgram(dsl, d[1], metadata, 'func_gt', 'cparam_gt', 'dparam_gt')
        prog_lines = utils.getHierProgLines(d[1], 'dsl_prog')
        for l in prog_lines:
            print(l)
        verts, faces = hier_execute(d[1])
        utils.writeObj(verts, faces, f'{d[0]}_dsl_gt.obj')                   


def main(disc, out_name, in_dir, cat, config_path):        
    
    config_mod = importlib.import_module(config_path)
    config = config_mod.loadConfig()

    dsl = DSL(config)
    
    for name, func in disc.ADD_FUNCS:
        dsl.library[name] = Function(func, dsl)
        dsl.full_library[name] = Function(func, dsl)
        
    for name in disc.RM_FUNCS:
        dsl.library.pop(name)
        
    with torch.no_grad():
        writeData(dsl, out_name, in_dir, cat)
        
        
if __name__ == "__main__":
    
    out_name = sys.argv[1]
    input_data = sys.argv[2]
    cat = sys.argv[3]
    
    disc = importlib.import_module(f'dsls.{sys.argv[4]}')
    config_path = sys.argv[5]
    
    main(disc, out_name, input_data, cat, config_path)
