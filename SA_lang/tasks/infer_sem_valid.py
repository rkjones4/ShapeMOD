import torch, os, sys, random
import sa_utils as utils
import pickle
import model_prog as mp
from copy import deepcopy
from torch.distributions import Categorical
from make_abs_data import fillProgram, makeSALines, getCuboidDims
from ShapeAssembly import hier_execute, Program

import numpy as np

MAX_TRIES = 50

BE = 1.5 
MAX_SEQ = 30

sq_map = {
    0:'left',
    1:'right',
    2:'bot',
    3:'top',
    4:'back',
    5:'front'
}
sym_map = {
    0: 'X',
    1: 'Y',
    2: 'Z'
}

def sem_eval_forward(net, inp_seq, code, code_start, bb_dims, hier_ind, P, samp_ind):

    bb_dims = bb_dims.unsqueeze(0).unsqueeze(0).repeat(1,inp_seq.shape[1],1)

    hier_oh = torch.zeros(1, inp_seq.shape[1], mp.MAX_DEPTH).to(mp.device)

    hier_oh[0, :, min(hier_ind, 2)] = 1.0
                
    inp = net.inp_net(
        torch.cat(
        (inp_seq, bb_dims, hier_oh), dim=2)
    )
        
    gru_out, h = net.gru(inp, code.view(1,1,-1))
    
    out = torch.zeros(inp_seq.shape, device=mp.device).float()

    commands = None
            
    for _net in net.net_list:                
            
        if _net.func is not None:
            assert commands is not None
            if _net.func != commands:
                continue
                
        if _net.line_cond is not None:
            line_cond = out[:,:,_net.line_cond[0]:_net.line_cond[1]]
        else:
            line_cond = torch.zeros(inp_seq.shape[0], inp_seq.shape[1], 0, device=mp.device)

            
        if _net.bb_cond is True:
            bb_cond = bb_dims
        else:
            bb_cond = torch.zeros(inp_seq.shape[0], inp_seq.shape[1], 0, device=mp.device)
                                    
        raw_out = _net(torch.cat((
            gru_out, line_cond, bb_cond
        ), dim=2))
                
        if _net._type == 'func':

            if samp_ind == 0:
                dist = torch.softmax(raw_out.squeeze(), dim=0)
            else:
                dist = torch.softmax(
                    raw_out.squeeze() \
                    / (samp_ind * 1.0),
                    dim=0
                )
                
            mask = torch.zeros(dist.shape).float()

            if len(P.cuboids) == 0:
                mask[2] = 1.0
            
            else:
                if len(P.cuboids) >= 2:
                    lcmd = P.last_command
                else:
                    lcmd = 'start'
                    
                mask[torch.tensor(P.cmd_to_func_masks[lcmd])] = 1.0


            dist = dist * mask.to(mp.device)                            
            dist[0] = 0.

            if samp_ind > 0:
                cmd = Categorical(
                    dist
                ).sample()
            else:
                cmd = dist.argmax().item()
                
            out[0,0, _net.start+cmd] = 1.0
            assert commands == None
            commands = cmd

        elif _net._type == 'disc':
            dist = torch.softmax(raw_out.squeeze(), dim=0)
            if samp_ind > 0:
                m = Categorical(
                    dist
                ).sample()
            else:
                m = dist.argmax().item()
                
            out[0,0, _net.start+m] = 1.0

        elif _net._type == 'b':
            if samp_ind > 0:
                r = torch.distributions.Bernoulli(
                    torch.sigmoid(raw_out.squeeze()/(1.0 * samp_ind))
                ).sample().float()
            else:
                r = (raw_out.squeeze() >= 0.).float()
                
            out[0,0,_net.start:_net.end] = r
            
        elif _net._type == 'f':
            r = raw_out.squeeze()
            out[0,0,_net.start:_net.end] = r
                    
    double_enc = torch.cat((
        gru_out, code_start.repeat(1, gru_out.shape[1], 1)
    ), dim = 2)
            
    child_pred = net.child_net(
        double_enc
    )
                
    next_codes = net.next_code_net(
        double_enc                        
    ).view(inp_seq.shape[0], inp_seq.shape[1], net.max_children, -1)                
        
    return out, next_codes, child_pred, h

def decode_sa_line(net, P, out, bbox_dims, meta):
    f_num, cl_prm, dl_prm = net.decode_line(out)

    assert f_num != 0, 'this should never happen'
    
    if f_num == 1:        
        return ['<END>']

    inv_func_map = {v:k for k,v in meta['func_map'].items()}
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
            line.append(sq_map[lookup[index].item()])

        elif ptype == 'sym':
            line.append(sym_map[lookup[index].item()])
                
        elif ptype == 'b':
            line.append(bool(lookup[index]))
        else:
            assert False, f'bad input {j_prm}'

    ret = []
    struct = meta['dsl'].library[fn].getStructure()
    cube_num = len([c for c in P.cuboids.values() if c.parent is None])
    for s in struct:
        if s == "Cuboid":
            if cube_num == 0:
                ret.append(f'bbox')
            else:
                ret.append(f'cube{cube_num-1}')
            cube_num += 1

    d_ret = tuple(ret)
    d_line = tuple(line)
    sa_lines, _ = makeSALines(
        meta['dsl'].library[d_line[0]],
        [1.0] + bbox_dims + list(d_line[1:])
        , d_ret,
        P.last_cuboid
    )

    return sa_lines

def insideBBox(P):

    bbox_corners = P.cuboids[f"bbox"].getCorners()
    
    maxb = bbox_corners.max(dim=0).values
    minb = bbox_corners.min(dim=0).values

    assert (maxb > 0).all()
    assert (minb < 0).all()

    maxb = (maxb + 0.1) * BE
    minb = (minb - 0.1) * BE
    
    for ci in P.cuboids:
        
        if ci == 'bbox':
            continue
        corners = P.cuboids[ci].getCorners()
    
        maxc = corners.max(dim=0).values
        minc = corners.min(dim=0).values
    
        if (maxc > maxb).any():            
            return False

        if (minc < minb).any():            
            return False

    return True

def checkValidLine(net, P, out, bbox_dims, meta):
    
    sa_lines = decode_sa_line(net, P, out, bbox_dims, meta)
    
    attaches_to_add = []
    cube_syms = []
    
    for line in sa_lines:
        if '<END>' in line:
            # Make sure nothing is left unmoved
            for v in P.cube_attaches.values():
                if len(v) == 0:
                    return None, False
            
        if 'Cuboid(' in line:
            parse = P.parseCuboid(line)
            P.last_cuboid = parse[0]
            if 'bbox' not in line:
                P.last_command = 'Cuboid'
                P.cube_attaches[parse[0]] = []
                
        if 'attach' in line:
            parse = P.parseAttach(line)            
            if parse[0] != P.last_cuboid:
                assert False, 'how did this fail'
                return None, False
            P.last_command = 'attach'
                                                
            attaches_to_add.append((parse[0], parse[1]))
            
        if 'squeeze' in line:
            parse = P.parseSqueeze(line)
            if parse[0] != P.last_cuboid:
                assert False, 'how did this fail'
                return None, False
            P.last_command = 'squeeze'
            attaches_to_add.append((parse[0], parse[1]))
            attaches_to_add.append((parse[0], parse[2]))

        if 'translate' in line:
            parse = P.parseTranslate(line)
            if parse[0] != P.last_cuboid:
                assert False, 'how did this fail'
                return None, False
            P.last_command = 'translate'
            cube_syms.append(parse[0])
            
        if 'reflect' in line:
            parse = P.parseReflect(line)
            if parse[0] != P.last_cuboid:
                assert False, 'how did this fail'
                return None, False
            P.last_command = 'reflect'
            cube_syms.append(parse[0])
            
        try:
            P.execute(line)
            P.last_command_cuboid = 'Cuboid(' in line    
        except Exception as e:
            if mp.VERBOSE:
                print(f"failed line {line} with {e}")
                
            # Return none
            return None, False


    for a, o in attaches_to_add:
        past_attaches = P.cube_attaches[a]

        # Attached to non bounding box more than once
        if o != 'bbox' and o in past_attaches:
            return None, False

        # Already has max attachments
        if len(past_attaches) == 2:
            return None, False

        P.cube_attaches[a].append(o)

    for c in cube_syms:
        if c in P.cube_syms:
            return None, False

        P.cube_syms.add(c)
        
    return P, insideBBox(P)
    
def getSpecFuncs(meta, types):
    l = []
    if 'end' in types:
        l.append(1)
    dsl = meta['dsl']
    for k, v in meta['func_map'].items():
        if k in dsl.library:
            if dsl.library[k].structure[0] in types:
                l.append(v)
    return l

        
def sem_eval_prog(net, code, node=None):
    is_root = False
    
    if node is None:
        is_root = True
        bb_dims = net.bb_net(code)
        node = {
            'depth': 0,
            'bb_dims': bb_dims
        }
        
    if node['depth'] > mp.MAX_DEPTH:        
        node.pop('depth')
        node.pop('bb_dims')
        return 
                    
    h = code.view(1,1, -1)
    h_start = h.clone()
    
    inp = net.getStartLine()                
    
    out_lines = []        
    children = []

    P = Program()
    
    P.cube_attaches = {}    
    P.cube_syms = set()
    
    meta = net.metadata
    
    P.last_cuboid = None
    P.last_command = 'start'

    P.cmd_to_func_masks = {
        'start': getSpecFuncs(meta, ('Cuboid')),
        'Cuboid': getSpecFuncs(meta, ('attach', 'squeeze')),
        'attach': getSpecFuncs(meta, ('attach', 'reflect', 'translate', 'Cuboid', 'end')),
        'squeeze': getSpecFuncs(meta, ('reflect', 'translate', 'Cuboid', 'end')),
        'reflect': getSpecFuncs(meta, ('Cuboid', 'end')),
        'translate': getSpecFuncs(meta, ('Cuboid', 'end'))
    }
    
    P.cuboids.pop('bbox')
    
    for i in range(MAX_SEQ):        
        should_break = False
        for j in range(MAX_TRIES):

            # Failed to make valid prog
            if j == (MAX_TRIES-1):
                keys = list(node.keys())
                for k in keys:
                    if k in node:
                        node.pop(k)
                should_break = True
                break
            
            new_inp, pnext, pchild, new_h = sem_eval_forward(
                net, inp, h, h_start, node['bb_dims'], node['depth'], P, j
            )
            
            clean_out = new_inp.squeeze()
            
            new_P, valid = checkValidLine(
                net,
                deepcopy(P),
                clean_out,
                node['bb_dims'].tolist(),
                net.metadata
            )
            
            if not valid:                
                continue

            inp = new_inp
            h = new_h
            P = new_P
            break

        if should_break:
            break
        
        fstart, fend = net.metadata['tl_map']['func']
        func_ind = torch.argmax(clean_out[fstart:fend]).item()
        
        if func_ind == 1:
            break
            
        out_lines.append(clean_out)
    
        child_pred = pchild[0][0]
        next_codes = pnext[0][0]

        _, _c, _ = net.decode_line(clean_out)
                                    
        cube_dims = getCuboidDims(
            net.metadata['dsl'].library[
                net.metadata['rev_func_map'][func_ind]
            ],
            _c,
            net.metadata['jparam_map'][func_ind],
            node['bb_dims']
        )                    
        cube_dims = torch.tensor(cube_dims,device=child_pred.device)
        
        for i in range(net.metadata['num_cube_map'][func_ind]):                                
            if child_pred[i].item() >= 0.0:                    
                child = {
                    'depth': node['depth']+1,
                    'bb_dims': cube_dims[i]
                }
                children.append(child)
                sem_eval_prog(net, next_codes[i], child)
            else:
                children.append({})

    if len(node) == 0:
        if not is_root:
            return node
        else:
            return dummy_prog(net)
        
    node['children'] = children
    out_funcs, out_cprms, out_dprms = net.split_lines(out_lines)
    
    node[mp.FUNC_PRED_FIELD] = torch.tensor(out_funcs)
    node[mp.CPARAM_PRED_FIELD] = torch.stack(out_cprms) if len(out_cprms) > 0 else torch.tensor([])
    node[mp.DPARAM_PRED_FIELD] = torch.stack(out_dprms) if len(out_dprms) > 0 else torch.tensor([])
        
    return node

def dummy_prog(net):
    
    node = {}
    node['children'] = [{}, {}]
    node[mp.FUNC_PRED_FIELD] = torch.tensor([2,2])
    node[mp.CPARAM_PRED_FIELD] = torch.ones(2, net.metadata['max_cparams']).float()
    node[mp.DPARAM_PRED_FIELD] = torch.zeros(2, net.metadata['max_dparams']).long()
    return node

        

        
