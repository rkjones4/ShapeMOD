import torch
import sys
import ast
import numpy as np
import random
import os
import pickle
import old_intersect as inter
import random
from copy import deepcopy
import networkx as nx
import symmetry as sym


VERBOSE = False

DO_SHORTEN = True
DO_SIMP_SYMMETRIES = True
DO_SQUEEZE = True
DO_VALID_CHECK = True
DO_NORM_AA_CUBES = True
DO_SEM_FLATTEN = True
DO_SEM_REHIER = True

SQUARE_THRESH = 0.1
SD_THRESH = 0.01
AA_THRESH = 0.995 #??

SCOL_MAP = {
    'chair': set(['caster', 'mechanical_control']),
    'table': set(['caster', 'cabinet_door', 'drawer', 'keyboard_tray']),
    'storage': set(['drawer', 'cabinet_door', 'mirror', 'caster'])
}

SFLAT_MAP = {
    'chair': set(['chair_back', 'chair_arm', 'chair_base', 'chair_seat', 'footrest', 'chair_head']),
    'table': set(['tabletop', 'table_base']),
    'storage': set(['cabinet_frame', 'cabinet_base'])
}

SRH_MAP = {
    'storage':('cabinet_frame', set(['countertop', 'shelf', 'drawer', 'cabinet_door', 'mirror']))
}

def isSquare(a, b, c):
    v = (a - b).abs()/max(a.item(), b.item(), c.item()) < SQUARE_THRESH
    return v
    
def isAxisAligned(props, bbox, thresh=AA_THRESH):    

    xdir = props['xdir']
    ydir = props['ydir']
    zdir = props['zdir']
    
    xdir /= xdir.norm()
    ydir /= ydir.norm()
    zdir /= zdir.norm()
    
    if xdir.dot(bbox['xdir']) < thresh:
        return False
                
    if ydir.dot(bbox['ydir']) < thresh:
        return False
    
    if zdir.dot(bbox['zdir']) < thresh:
        return False

    return True

def shouldChange(props, bbox):    

    if isAxisAligned(props, bbox):
        return True
    
    xsquare = isSquare(props['zd'], props['yd'], props['xd'])
    ysquare = isSquare(props['zd'], props['xd'], props['yd'])
    zsquare = isSquare(props['xd'], props['yd'], props['zd'])
    
    if xsquare and ysquare and zsquare:
        return True

    if xsquare:
        return isSpecAxisAligned(props['xdir'], bbox['xdir'])

    if ysquare:
        return isSpecAxisAligned(props['ydir'], bbox['ydir'])

    if zsquare:
        return isSpecAxisAligned(props['zdir'], bbox['zdir'])
      
    return False

def isSpecAxisAligned(cdir, axis):
    cdir /= cdir.norm()

    if cdir.dot(axis) >= AA_THRESH:
        return True
    else:
        return False

def getDataPath(category):
    if 'storage' == category:
        category += 'furniture'
    return f"/home/{os.getenv('USER')}/data/{category}_hier/"

def getSemOrder(category):
    if category == "chair":
        sem_order_path = "stats/part_semantics/PGP-Chair.txt"
    elif category == "storage":
        sem_order_path = "stats/part_semantics/PGP-Storage.txt"
    elif category == "table":
        sem_order_path = "stats/part_semantics/PGP-Table.txt"
    else:
        assert False, f'Invalid Category {category}'
        
    sem_order = {"bbox": "-1","other":"100"}

    with open(sem_order_path) as f:
        for line in f:
            sem_order[line.split()[1].split('/')[-1]] = line.split()[0]
    return sem_order


def cubeOrder(cubes, names, sem_order):
    d = []
    
    min_c = np.array([1e8,1e8,1e8])
    max_c = np.array([-1e8,-1e8,-1e8])
    
    for rw in cubes:
        min_c = np.min((min_c, rw['center'].numpy()), axis = 0)
        max_c = np.max((max_c, rw['center'].numpy()), axis = 0)

    mac = np.max(max_c)
    mic = np.min(min_c)

    for c_ind, (rw, name) in enumerate(zip(cubes, names)):        
        sc = (rw['center'].numpy() - mic) / (mac - mic)
        
        x_r = round(sc[0]*2)/2
        y_r = round(sc[1]*2)/2
        z_r = round(sc[2]*2)/2
        
        d.append((
            int(sem_order[name]),
            x_r + y_r + z_r,
            x_r,
            y_r,
            z_r,
            sc[0],
            sc[1],
            sc[2],
            c_ind
        ))

    d.sort()
    return [c_ind for _,_,_,_,_,_,_,_,c_ind in d]

Sbbox = {
    'xdir': torch.tensor([1.0, 0.0, 0.0]),
    'ydir': torch.tensor([0.0, 1.0, 0.0]),
    'zdir': torch.tensor([0.0, 0.0, 1.0]),    
}

def vector_cos(norm1, norm2):
    norm1 = np.asarray(norm1)
    norm2 = np.asarray(norm2)
    dot = np.dot(norm1, norm2)
    magnitude = np.linalg.norm(norm1) * np.linalg.norm(norm2)
    if magnitude == 0.:
        return 0.
    return dot / float(magnitude)


def orientProps(center, xd, yd, zd, xdir, ydir, zdir):

    rt = np.asarray([1., 0., 0.])
    up = np.asarray([0., 1., 0.])
    fwd = np.asarray([0., 0., 1.])

    l = [
        (xdir, xd, 0),
        (ydir, yd, 1),
        (zdir, zd, 2),
        (-1 * xdir, xd, 3),
        (-1 * ydir, yd, 4),
        (-1 * zdir, zd, 5)
    ]

    rtdir, rtd, rind = sorted(
        deepcopy(l), key=lambda x: vector_cos(rt, x[0]))[-1]

    if rind >= 3:
        l.pop(rind)
        l.pop((rind+3) % 6)
    else:
        l.pop((rind+3) % 6)
        l.pop(rind)

    for i in range(0, 4):
        p_ind = l[i][2]
        if p_ind > max(rind, (rind+3) % 6):
            l[i] = (l[i][0], l[i][1], l[i][2] - 2)
        elif p_ind > min(rind, (rind+3) % 6):
            l[i] = (l[i][0], l[i][1], l[i][2] - 1)

    updir, upd, upind = sorted(
        deepcopy(l), key=lambda x: vector_cos(up, x[0]))[-1]

    if upind >= 2:
        l.pop(upind)
        l.pop((upind+2) % 4)
    else:
        l.pop((upind+2) % 4)
        l.pop(upind)

    fwdir, fwd, _ = sorted(l, key=lambda x: vector_cos(fwd, x[0]))[-1]

    return {
        'center': torch.tensor(center).float(),
        'xd': torch.tensor(rtd).float(),
        'yd': torch.tensor(upd).float(),
        'zd': torch.tensor(fwd).float(),
        'xdir': torch.tensor(rtdir).float(),
        'ydir': torch.tensor(updir).float(),
        'zdir': torch.tensor(fwdir).float()
    }

def jsonToProps(json):
    json = np.array(json)
    center = np.array(json[:3])
    
    xd = json[3]
    yd = json[4]
    zd = json[5]
    xdir = json[6:9]
    xdir /= np.linalg.norm(xdir)
    ydir = json[9:]
    ydir /= np.linalg.norm(ydir)
    zdir = np.cross(xdir, ydir)
    zdir /= np.linalg.norm(zdir)

    if xd < SD_THRESH or yd < SD_THRESH or zd < SD_THRESH:
        return None
    
    props = orientProps(center, xd, yd, zd, xdir, ydir, zdir)

    if DO_NORM_AA_CUBES:
        if shouldChange(props, Sbbox):
            props['xdir'] = Sbbox['xdir'].clone().detach()
            props['ydir'] = Sbbox['ydir'].clone().detach()
            props['zdir'] = Sbbox['zdir'].clone().detach()
    
    return props


def addAttachments(node, sem_order):
    co = cubeOrder(node['cubes'], node['children_names'], sem_order)
    
    for key in ['cubes', 'children', 'children_names']:    
        node[key] = [node[key][i] for i in co]
        
    ind_to_pc, scene_geom = inter.samplePC(node['cubes'], split_bbox=True)
    inters = inter.findInters(ind_to_pc, scene_geom)
    node['attachments'] = inter.calcAttachments(inters, scene_geom, ind_to_pc)
    for child in node['children']:
        if len(child) > 0:
            addAttachments(child, sem_order)

            
# From raw json, get graph structure and all leaf cuboids
def getShapeHier(ind, category):
    data_path = getDataPath(category)
    with open(data_path + ind + ".json") as f:
        json = None
        for line in f:
            json = ast.literal_eval(line)

    hier = {}
            
    queue = [(json, hier)]
    
    while len(queue) > 0:
        json, node = queue.pop(0)
        
        if "children" not in json:
            continue

        name = json["label"]

        # Don't add sub-programs when we collapse
        collapse_names = SCOL_MAP[category]
        if name in collapse_names:
            continue
        
        while("children" in json and len(json["children"])) == 1:
            json = json["children"][0]

        if "children" not in json:
            continue
        
        node.update({
            "children": [],
            "children_names": [],
            "cubes": [],
            "name": name
        })
        
        for c in json["children"]:
            cprops = jsonToProps(c["box"])
            if cprops is not None:
                new_c = {}
                queue.append((c, new_c))
                node["children"].append(new_c)
                node["cubes"].append(cprops)
                node["children_names"].append(c["label"])

    return hier


def getOBB(parts):
    part_corners = inter.sampleCorners(parts)
    bbox = inter.points_obb(part_corners.cpu(), 1)
    return bbox


def cleanHier(hier):
    for i in range(len(hier['children'])):
        if len(hier['children'][i]) > 0:
            cleanHier(hier['children'][i])
            
    if len(hier['children']) == 0:
        for key in list(hier.keys()):
            hier.pop(key)


def trimHier(hier):
    for i in range(len(hier['children'])):
        if len(hier['children'][i]) > 0:
            if len(hier['children'][i]['children']) == 1:
                hier['cubes'][i] = hier['children'][i]['cubes'][0]
                hier['children_names'][i] = hier['children'][i]['children_names'][0]
                hier['children'][i] = hier['children'][i]['children'][0]
                
        if len(hier['children'][i]) > 0:
            trimHier(hier['children'][i])                

            
def fillHier(hier):    
    for i in range(len(hier['children'])):
        if len(hier['children'][i]) > 0:
            hier['cubes'][i] = fillHier(hier['children'][i])            

    hier['bbox'] = getOBB(hier['cubes'])
    return deepcopy(hier['bbox'])


# centers and orients root bounding box
# propogates transformation to all cuboids
# also instanties bounding box into the cube + children spots
def normalizeHier(hier):
    hier.pop('bbox')
    
    rbbox = {}
    
    samps = inter.sampleCorners(hier['cubes']).cpu()

    dims = samps.max(dim=0).values - samps.min(dim=0).values

    rbbox['xd'] = dims[0]
    rbbox['yd'] = dims[1]
    rbbox['zd'] = dims[2]

    rbbox['center'] = (samps.max(dim=0).values + samps.min(dim=0).values) / 2

    rbbox['xdir'] = torch.tensor([1.,0.,0.])
    rbbox['ydir'] = torch.tensor([0.,1.,0.])
    rbbox['zdir'] = torch.tensor([0.,0.,1.])

    hier['bbox'] = rbbox
    
    offset = rbbox['center']

    q = [hier]

    while len(q) > 0:
        
        n = q.pop(0)
        bbox = n.pop('bbox')
        n['children'] = [{}] + n['children']
        n['children_names'] = ["bbox"] + n['children_names']
        n['cubes'] = [bbox] + n['cubes']
        
        for i in range(len(n['cubes'])):            
            n['cubes'][i]['center'] = n['cubes'][i]['center'] - offset            
            
        for c in n['children']:
            if len(c) > 0:
                q.append(c)                


def markLeafCubes(hier):
    parts = []
    q = [hier]
    while(len(q) > 0):
        n = q.pop(0)
        n['leaf_inds'] = []
        assert(len(n['cubes']) == len(n['children']))
        for cu, ch in zip(n['cubes'], n['children']):
            if len(ch) > 0:
                q.append(ch)
                n['leaf_inds'].append(-1)
            else:
                n['leaf_inds'].append(len(parts))
                parts.append(cu)

    return parts


def replace_parts(hier, parts, key):
    q = [hier]
    while(len(q) > 0):
        n = q.pop(0)

        binds = []
        
        for i in range(len(n[key])):
            if n[key][i] != -1:
                lpart = parts[n[key][i]]
                
                if lpart is None:
                    binds.append(i)
                else:
                    n['cubes'][i] = lpart

        binds.sort(reverse=True)
        for bi in binds:
            n['children'].pop(bi)
            n['cubes'].pop(bi)
            n['children_names'].pop(bi)
            
        for c in n['children']:
            if c is not None and len(c) > 0:
                q.append(c)

        n.pop(key)

# Takes in a hierarchy of just leaf cuboids,
# finds new parameters for leaf cuboids so that
# part-to-part connections are as valid as possible

def shortenLeaves(hier):
    if VERBOSE:
        print("Doing Shortening")

    parts = markLeafCubes(hier)
    bad_inds = inter.findHiddenCubes(parts)

    ind_to_pc, scene_geom = inter.samplePC(parts)
    inters = inter.findInters(ind_to_pc, scene_geom)

    dim_parts = [
        (p['xd'] * p['yd'] * p['zd'], i) for i,p in enumerate(parts)
    ]
    
    dim_parts.sort()

    for _, ind in dim_parts:
        if ind in bad_inds:
            continue
        
        if VERBOSE:
            print(f"Shortening Leaf ind: {ind}")
            
        sres = inter.shorten_cube(inters, parts, ind, scene_geom)
        if sres is not None:
            t_ind_to_pc, t_scene_geom = inter.samplePC([parts[ind]])
            ind_to_pc[ind] = t_ind_to_pc[0]
            scene_geom[ind] = t_scene_geom[0]
            sres = [(int(s.split('_')[0]), int(s.split('_')[1])) for s in sres]
            new_inters = inter.findInters(ind_to_pc, scene_geom, sres)
            inters.update(new_inters)
            if parts[ind]['xd'] < SD_THRESH or \
               parts[ind]['yd'] < SD_THRESH or \
               parts[ind]['zd'] < SD_THRESH:
                bad_inds.append(ind)

    for bi in bad_inds:
        parts[bi] = None
            
    replace_parts(hier, parts, 'leaf_inds')
                
def make_conn_graph(num_nodes, attachments):

    edges = []
    for (ind1, ind2, _, _) in attachments:
        edges.append((ind1, ind2))

    G = nx.Graph()
    G.add_nodes_from(list(range(num_nodes)))
    G.add_edges_from(edges)
    
    return G


def assertConnected(num_nodes, attachments):
    G = make_conn_graph(num_nodes, attachments)
    assert nx.number_connected_components(G) == 1, 'disconnected graph'

    
def checkConnected(node):
    assertConnected(len(node['cubes']), node['attachments'])
    for c in node['children']:
        if len(c) > 0:
            checkConnected(c)
    
def memoize(f):
    def helper(ind, category):
        cdir = "parse_cache"
        cached_res = os.listdir(cdir)
        if ind in cached_res:
            return pickle.load(open(cdir+"/"+ind, "rb"))
        else:
            hier = f(ind, category)
            pickle.dump(hier, open(cdir+"/"+ind, "wb"))
            return hier
    return helper


def checkCubeNum(node):
    assert len(node['cubes']) <= 11, f"Saw program with {len(node['cubes'])} cubes"
    for c in node['children']:
        if len(c) > 0:
            checkCubeNum(c)


def flattenNode(node):
    for c in node['children']:
        if len(c) > 0:
            flattenNode(c)

    # Now everything has one sub-program
    fcubes = []
    fchildren = []
    fchildren_names = []

    for i in range(len(node['children'])):
        if len(node['children'][i]) == 0:
            fcubes.append(node['cubes'][i])
            fchildren.append(node['children'][i])
            fchildren_names.append(node['children_names'][i])
        else:
            fcubes += node['children'][i]['cubes']
            fchildren += node['children'][i]['children']
            fchildren_names += node['children'][i]['children_names']

    node['cubes'] = fcubes
    node['children'] = fchildren
    node['children_names'] = fchildren_names
    
            
def semFlattenHier(hier, category):
    flat_names = SFLAT_MAP[category]
    q = [hier]
    while (len(q) > 0):
        node = q.pop(0)
        if node['name'] in flat_names:
            flattenNode(node)
        else:
            for c in node['children']:
                if len(c) > 0:
                    q.append(c)

def semReHier(hier, category):

    if category not in SRH_MAP:
        return

    rh_tar, rh_names = SRH_MAP[category]

    if rh_tar not in hier['children_names']:
        return

    rhinds = []
    for i,name in enumerate(hier['children_names']):
        if name in rh_names:
            rhinds.append(i)

    rhinds.sort(reverse=True)

    ti = hier['children_names'].index(rh_tar)

    for i in rhinds:        
        for key in ['children_names', 'cubes', 'children']:        
            hier['children'][ti][key].append(hier[key][i])

    for i in rhinds:        
        for key in ['children_names', 'cubes', 'children']:        
            hier[key].pop(i)
    
    if len(hier['children']) == 1:
        hier['children_names'] = hier['children'][0]['children_names']
        hier['cubes'] = hier['children'][0]['cubes']
        hier['children'] = hier['children'][0]['children']
                    

right = torch.tensor([1.0,0.5,0.5])
left = torch.tensor([0.0,0.5,0.5])
top = torch.tensor([0.5,1.0,0.5])
bot = torch.tensor([0.5,0.0,0.5])
front = torch.tensor([0.5,0.5,1.0])
back = torch.tensor([0.5,0.5,0.0])

def isFaceAtt(face, oface):
    if not DO_SQUEEZE:
        return None
    
    face = torch.tensor(face)
    if (face-right).norm() < .1 and abs(oface[0] - 0.0) < .1:        
        return 'right'
    if (face-left).norm() < .1  and abs(oface[0] - 1.0) < .1:
        return 'left'
    if (face-top).norm() < .1  and abs(oface[1] - 0.0) < .1:
        return 'top'
    if (face-bot).norm() < .1  and abs(oface[1] - 1.0) < .1:
        return 'bot'
    if (face-front).norm() < .1  and abs(oface[2] - 0.0) < .1:
        return 'front'
    if (face-back).norm() < .1  and abs(oface[2] - 1.0) < .1:
        return 'back'
    return None

        
def preProc(node):
    node['aligned'] = [isAxisAligned(cube, node['cubes'][0]) for cube in node['cubes']]
    node['cubes'] = [{f : cube[f].tolist() for f in cube } for cube in node['cubes']]

    new_attachments = []
    for i0, i1, a0, a1 in node['attachments']:        
        if i0 == 0:
            if a0[1] < .5:
                i0 = -2
            else:
                i0 = -1

            # FLIP BBOX TOP AND BOT FOR CONSISTENCY
            a0[1] = 1-a0[1]

        f0 = isFaceAtt(a0, a1)
        f1 = isFaceAtt(a1, a0)
            
        assert i1 != 0, 'uh oh'
                
        new_attachments.append((
            i0,
            i1,
            a0,
            a1,
            f0,
            f1
        ))
    node['attachments'] = new_attachments
    
def parseJsonToHier(ind, category, get_gt=False):
    sem_order = getSemOrder(category)
        
    hier = getShapeHier(ind, category)

    assert len(hier) > 0, 'saw empty hier'
    
    if DO_SEM_FLATTEN:
        semFlattenHier(hier, category)

    if DO_SEM_REHIER:
        semReHier(hier, category)

    if DO_SHORTEN:
        shortenLeaves(hier)

    cleanHier(hier)        
    trimHier(hier)
    fillHier(hier)
    
    normalizeHier(hier)

    addAttachments(hier, sem_order)
    if get_gt:
        return hier

    if DO_VALID_CHECK:
        checkConnected(hier)
    
    if DO_SIMP_SYMMETRIES:
        sym.addSymSubPrograms(hier)
        sym.addSimpSymmetries(hier)

    if DO_VALID_CHECK:
        checkCubeNum(hier)
        
    return hier

def memoize(f):
    def helper(ind, category, part):
        cdir = "parse_part_cache"
        cached_res = os.listdir(cdir)
        if f"{ind}_{part}" in cached_res:
            return pickle.load(open(f"{cdir}/{ind}_{part}", "rb"))
        else:
            try:
                hier = f(ind, category, part)
            except Exception as e:
                print(f"Failed {ind} with {e}")
                hier = None
            pickle.dump(hier, open(f"{cdir}/{ind}_{part}", "wb"))
            return hier
    return helper

def genAllData(outdir, ind, category):
    hier = parseJsonToHier(ind, category)
    q = [hier]
    seen = set()
    while(len(q)>0):
        node = q.pop(0)
        for i, c in enumerate(node['children']):            
            while node['children_names'][i] in seen and i > 0:
                node['children_names'][i] += '+'
                
            if len(c) > 0:
                c['name'] = node['children_names'][i]
                q.append(c)
                
        node.pop('children')
        preProc(node)
        part = node['name']
        pickle.dump(node, open(f"{outdir}/{ind}_{part}", "wb"))


def loadAllData(in_dir, max_files=int(1e8)):
    ninds = []
    nodes = []

    files = list(os.listdir(in_dir))
    files.sort()
    for f in files[:max_files]:
        node = pickle.load(open(f"{in_dir}/{f}", "rb"))        
        if node is not None:
            ninds.append(f)
            nodes.append(node)

    return ninds, nodes

    
if __name__ == '__main__':
    category = sys.argv[1]
    outdir = sys.argv[2]
    
    PATH_TO_SA_DATA = sys.argv[3]
    inds = os.listdir(PATH_TO_SA_DATA)
    inds = [i.split('.')[0] for i in inds]

    #inds = ['173', '2307', '44366', '1282']
    #inds += ['42231', '40507', '36402', '41162', '41830']
    os.system(f'mkdir {outdir}')
    from tqdm import tqdm
    for ind in tqdm(inds):
        try:
            hier = genAllData(outdir, str(ind), category)
        except Exception as e:
            print(e)
