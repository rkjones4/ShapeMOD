import numpy as np
import itertools
import os
import networkx as nx
from tqdm import tqdm
from random import sample
import sa_utils
import time
from sa_utils import PREC

# Controlling allowable orders / programs from PartNet data

MAX_ATTS = 64
MAX_ORDERS = 10000
MAX_DIRS = 1000
FLEX_AA_THRESH = .9
MIN_SEP_DIST = .25
MAX_TIME = 2

#Assumptions
#  Directioning Should be balanced
#  Attaches of a cuboid should be made sequentially
#  "Grounded ordering" -> where we take "max" dependancy to bbox as grounding level
#   prefer FROM CENTER attachments
#  If face attachment, have an ordering of how they should be ordered

right = np.array([1.0,0.5,0.5])
left = np.array([0.0,0.5,0.5])
top = np.array([0.5,1.0,0.5])
bot = np.array([0.5,0.0,0.5])
front = np.array([0.5,0.5,1.0])
back = np.array([0.5,0.5,0.0])

def isFaceAtt(face):
    face = np.array(face)
    if np.linalg.norm(face-bot) < .1:
        return 'bot'
    if np.linalg.norm(face-top) < .1:
        return 'top'
    if np.linalg.norm(face-left) < .1:
        return 'left'
    if np.linalg.norm(face-right) < .1:
        return 'right'   
    if np.linalg.norm(face-back) < .1:
        return 'back'
    if np.linalg.norm(face-front) < .1:
        return 'front'
    return None

def compare(x):
    a,b,_,_ = x
    
    if a < b:
        return -1
    if a > b:
        return 1
    return 0
    

def get_valid_directions(attachments, aligned, do_face_atts, lower_min=False):
    NA = set()
    BB = set()
    R = []

    mv_parts = set()
    
    for att1, att2, pos1, pos2,_,_ in attachments:
        # DO THIS BECAUSE SQUEEZE ISN"T RELIANT ON THESE FACES
        f1 = isFaceAtt(pos1)
        f2 = isFaceAtt(pos2)
        
        if att1 < 0:
            mv_parts.add(att2)
            BB.add((att2, att1, f2, f1))
        elif att2 < 0:
            mv_parts.add(att1)
            BB.add((att1, att2, f1,f2))
        elif aligned[att1] == 0 and aligned[att2] == 1:
            mv_parts.add(att1)
            NA.add((att1, att2, f1,f2))
        elif aligned[att1] == 1 and aligned[att2] == 0:
            mv_parts.add(att2)
            NA.add((att2, att1,f2,f1))
        else:
            mv_parts.add(att1)
            mv_parts.add(att2)
            R.append((att1, att2,f1,f2))

    if len(mv_parts) == (len(aligned) - 1):
        BB = BB.union(NA)
    else:
        R += list(NA)
        
    if do_face_atts:
        TR = []
        FAs = []

        mv_parts = set()
        
        for att1, att2, f1, f2 in R:
            if f1 is not None and f2 is None:            
                mv_parts.add(att1)
                FAs.append((att1, att2, f1, f2))
            elif f2 is not None and f1 is None:
                mv_parts.add(att2)
                FAs.append((att2, att1, f2, f1))
            else:
                TR.append((att1, att2, f1, f2))
                mv_parts.add(att1)
                mv_parts.add(att2)
        
        for att1, _,_,_ in BB:
            mv_parts.add(att1)
            
        if len(mv_parts) == (len(aligned) - 1):
            R = TR
            BB = BB.union(set(FAs))

    counts = [0 for _ in range(len(aligned))]
    for att1, _, _, _ in BB:
        counts[att1] += 1

    if lower_min:
        minimum = [1 for _ in aligned]
    else:
        minimum = [1 if a == 1 else 2 for a in aligned]
        
    minimum[0] = 0
    minimum = np.array(minimum)

    counts = np.array(counts)

    PR = [((a,b, c, d), (b,a,d,c)) for a, b, c, d in R]

    valid = []
    
    for dir in itertools.product(*PR):
        new = counts.copy()
        for a, _, _, _ in dir:
            new[a] += 1.

        good_dir = (new >= minimum).all()
        if good_dir:
            valid.append(dir)        

    dirs =  [list(BB) + list(v) for v in valid]
        
    for d in dirs:
        d.sort(key=compare)
        
    dirs.sort()
        
    return dirs

def get_simple_valid_directions(attachments, aligned):
    NA = set()
    BB = set()
    R = []

    mv_parts = set()

    bb_atts = set()
    
    for att1, att2, pos1, pos2,_,_ in attachments:

        f1 = isFaceAtt(pos1)
        f2 = isFaceAtt(pos2)
        
        if att1 < 0:
            mv_parts.add(att2)
            BB.add((att2, att1, f2, f1))
            bb_atts.add(att2)
        elif att2 < 0:
            mv_parts.add(att1)
            BB.add((att1, att2, f1,f2))
            bb_atts.add(att1)
        elif aligned[att1] == 0 and aligned[att2] == 1:
            mv_parts.add(att1)
            NA.add((att1, att2, f1,f2))
        elif aligned[att1] == 1 and aligned[att2] == 0:
            mv_parts.add(att2)
            NA.add((att2, att1,f2,f1))            
        else:
            mv_parts.add(att1)
            mv_parts.add(att2)
            R.append((att1, att2,f1,f2))

    if len(mv_parts) == (len(aligned) - 1):
        BB = BB.union(NA)
    else:
        R += list(NA)

    NEW_R = set()

    for (a1, a2, f1, f2) in R:
        if a1 in bb_atts and a2 in bb_atts:
            continue
        elif a1 in bb_atts:
            BB.add((a2, a1, f2, f1))
        elif a2 in bb_atts:
            BB.add((a1, a2, f1, f2))
        else:
            NEW_R.add((a1, a2, f1, f2))

    R = NEW_R
            
    if True:
        TR = []
        FAs = []

        mv_parts = set()
        
        for att1, att2, f1, f2 in R:
            if f1 is not None and f2 is None:            
                mv_parts.add(att1)
                FAs.append((att1, att2, f1, f2))
            elif f2 is not None and f1 is None:
                mv_parts.add(att2)
                FAs.append((att2, att1, f2, f1))
            else:
                TR.append((att1, att2, f1, f2))
                mv_parts.add(att1)
                mv_parts.add(att2)
        
        for att1, _,_,_ in BB:
            mv_parts.add(att1)
            
        if len(mv_parts) == (len(aligned) - 1):
            R = TR
            BB = BB.union(set(FAs))

    counts = [0 for _ in range(len(aligned))]
    for att1, _, _, _ in BB:
        counts[att1] += 1

        
    minimum = [1 for _ in aligned]
        
    minimum[0] = 0
    minimum = np.array(minimum)

    counts = np.array(counts)

    PR = [((a,b, c, d), (b,a,d,c)) for a, b, c, d in R]

    valid = []
    
    for dir in itertools.product(*PR):
        new = counts.copy()
        for a, _, _, _ in dir:
            new[a] += 1.

        good_dir = (new >= minimum).all()
        if good_dir:
            valid.append(dir)        

    dirs =  [list(BB) + list(v) for v in valid]
        
    for d in dirs:
        d.sort(key=compare)
        
    dirs.sort()
    
    return dirs
            
def get_ground_levels(valid_dir):
    g = nx.DiGraph()

    for att1, att2, _, _ in valid_dir:
        att2 = 0 if att2 < 0 else att2
        g.add_node(att1)
        g.add_node(att2)
        g.add_edge(att1, att2)

    assert nx.is_directed_acyclic_graph(g)
    
    levels = []
    for i in range(1, len(g.nodes())):    
        levels.append(max([len(l) - 1 for l in list(nx.all_simple_paths(g, i, 0))]))

    return levels
    
face_lookup = {
    'bot': 1,
    'top':2,
    'left':3,
    'right':4,
    'back':5,
    'front':6,
    None: 7
}

# Given a single valid directioning, returns the valid orderings
# steps:
    #   form graph of cuboids, with directed edges when one attaches to another (including bbox)
    #   "grounded level" is FURTHEST distance to bounding box of any node
    #      -> make sure this is acyclic (TEST)
    #   group cuboids by grounded level
    #   for each cuboid, find all permutations of its attaches
    #   for each group, find all permutations of cuboids -> fill these in with all permutations of attaches of the cuboid
def get_valid_orders(valid_dir):
    levels = get_ground_levels(valid_dir)

    cube_atts = []
    for i in range(1, len(levels)+1):
        cube_atts.append([(a, b, f) for a, b, f,_ in valid_dir if a == i])

    cube_atts = [list(itertools.permutations(c)) for c in cube_atts]
    
    temp = []
    for cube in cube_atts:
        _temp = []            
        for seq in cube:
            prev = 0
            skip = False
            for ind, (a, b, f) in enumerate(seq):
                fv = face_lookup[f]                    
                if fv < prev:
                    skip = True
                    break
                else:
                    prev = fv

            if skip:
                continue
            _temp.append(seq)
        temp.append(_temp)
    cube_atts = temp
    
    level_groups = [[] for _ in range(max(levels))]
    for i, l in enumerate(levels):
        level_groups[l-1].append(i)
        
    level_groups = [list(itertools.permutations(l)) for l in level_groups]
    group_orders = list(itertools.product(*level_groups))
    
    all_orders = []

    group_orders.sort()
    
    for go in group_orders[:MAX_ORDERS]:
        new_ords = []
        for g in go:
            for ci in g:
                new_ords.append(cube_atts[ci])

        ord = []
        for g in list(itertools.product(*new_ords))[0]:
            for att in g:
                ord.append(att)

        all_orders.append(ord)

    return all_orders

def clean_valid(c_info, sq_info, sym_info, att_info, aligned):
    # add squeezes (find all valid combos beforehand)
    #   no attaches for blocks with squeeze
    # add symmetry operations  (one per cube)
    # remove unnecessary attachments
    #   no more than 2
    #   none that are too close

    if c_info[0][0] in sym_info:
        sym_type = sym_info[c_info[0][0]][0]
        if 'ref' in sym_type:
            sym_val = [('ref', c_info[0][0])]
        else:
            sym_val = [('trans', c_info[0][0])]
    else:
        sym_val = []
        
    prev_atts = []
    val_c_info = []
    for c in c_info:
        att_pt = att_info[(c[0], c[1])]
        add = True
        for patt in prev_atts:
            if np.linalg.norm(att_pt-patt) < MIN_SEP_DIST:
                add = False
        if add:
            val_c_info.append(c)
            prev_atts.append(att_pt)        
    
    if len(val_c_info) == 1:
        att_val = [('att', val_c_info[0][0], val_c_info[0][1])]
    else:        
        sc1, sc2, sf1, sf2 = val_c_info[0][1], val_c_info[1][1], val_c_info[0][2], val_c_info[1][2]

        add = False
        if val_c_info[0][0] in sq_info:            
            for tc1, tc2, tf1, tf2, _, _ in sq_info[val_c_info[0][0]]:
                if sc1 == tc1 and sc2 == tc2 and tf1 == sf1 and sf2 == tf2:
                    att_val = [('squeeze', val_c_info[0][0], val_c_info[0][1], val_c_info[1][1])]
                    add = True
                    break
            
        if not add:            
            att_val = [
                ('att', val_c_info[0][0], val_c_info[0][1]),
                ('att', val_c_info[1][0], val_c_info[1][1])
            ] 
            
    return att_val + sym_val


def getAttInfo(node):
    ai = {}
    for i0, i1, a0, a1, _, _ in node['attachments']:
        ai[(i0, i1)] = np.array(a0)
        ai[(i1, i0)] = np.array(a1)
    return ai


def valid_filter(node, orders, sq_info):
    
    # make sure no duplicates
    valid_orders = set()    
    aligned = node['aligned']
    
    if 'att_info' not in node:
        node['att_info'] = getAttInfo(node)
    att_info = node['att_info']

    if 'sym_info' not in node:
        node['sym_info'] = {n[0]:n[1:] for n in node['syms']}
    sym_info = node['sym_info']    
    
    for order in orders:
        seen = set()
        valid = []
        cur = []
        add = True
        for c1, c2, f in order:
            if c1 not in seen:
                if len(cur) > 0:
                    _valid = clean_valid(cur, sq_info, sym_info, att_info, aligned[c1])
                    if _valid is None:
                        add = False                        
                        break
                    else:
                        valid += _valid
                        
                cur = [(c1, c2, f)]
                seen.add(c1)
            else:
                cur.append((c1, c2, f))

        if len(cur) > 0:
            _valid = clean_valid(cur, sq_info, sym_info, att_info, aligned[c1])
            if _valid is None:
                add = False
            else:
                valid += _valid
                
        if add:
            valid_orders.add(tuple(valid))            

    return list(valid_orders)


def make_squeeze(attachments, mode, seen_faces, ind):
    if mode == 'lr':
        f0, f1, ii = 'left', 'right', 0
    elif mode == 'bt':
        f0, f1, ii = 'bot', 'top', 1
    elif mode == 'bf':
        f0, f1, ii = 'back', 'front', 2

    a0s = []
    a1s = []

    for i in seen_faces[f0]:
        a = attachments[i]

        if a[0] == ind:
            a0s.append((a[3], i, a[1]))
        else:
            a0s.append((a[2], i, a[0]))

    for i in seen_faces[f1]:
        a = attachments[i]

        if a[0] == ind:
            a1s.append((a[3], i, a[1]))
        else:
            a1s.append((a[2], i, a[0]))

    sqs = []
    for a0, i0, o0 in a0s:
        for a1, i1, o1 in a1s:
            add = True
            u = None
            v = None
            for j in range(3):
                if j == ii:
                    continue

                if abs(a0[j] - a1[j]) > .1:
                    add = False

                elif u is None:
                    u = (a0[j] + a1[j]) / 2 

                elif v is None:
                    v = (a0[j] + a1[j]) /2 
                    
            if add:
                sqs.append([
                    ind,
                    o0,
                    o1,
                    f0,
                    f1,
                    u,
                    v
                ])
                sqs.append([
                    ind,
                    o1,
                    o0,
                    f1,
                    f0,
                    u,
                    v
                ])
    return sqs


def getSqueezeInfo(node):
    
    face_atts = {}
    for i, (i0, i1, a0, a1, face0, face1) in enumerate(node['attachments']):
        if i0 > 0 and face0 is not None:
            if i0 in face_atts:
                face_atts[i0].append((face0, i))
            else:
                face_atts[i0] = [(face0, i)]
                
        if i1 > 0 and face1 is not None:
            if i1 in face_atts:
                face_atts[i1].append((face1, i))
            else:
                face_atts[i1] = [(face1, i)]

    sqs = []
    for ind, atts in face_atts.items():
        if len(atts) < 2:
            continue
        seen_faces = {}
        for f, ai in atts:            
            if f in seen_faces:
                seen_faces[f].append(ai)
            else:
                seen_faces[f] = [ai]

        sq = None
        if 'bot' in seen_faces and 'top' in seen_faces:
            sq = make_squeeze(node['attachments'], 'bt', seen_faces, ind) 
            sqs += sq
            
        if sq is None and 'left' in seen_faces and 'right' in seen_faces:
            sq = make_squeeze(node['attachments'], 'lr', seen_faces, ind)
            sqs += sq
            
        if sq is None and 'back' in seen_faces and 'front' in seen_faces:
            sq = make_squeeze(node['attachments'], 'bf', seen_faces, ind)
            sqs += sq

    sq_info = {}
    for ind, i0, i1, f0, f1, u, v in sqs:
        if ind in sq_info:            
            sq_info[ind].append((i0, i1, f0, f1, u, v))
        else:
            sq_info[ind] = [(i0, i1, f0, f1, u, v)]
            
    return sq_info
                
def get_all_valid_orders(node):
    start_time = time.time()
    
    if len(node['attachments']) > MAX_ATTS:
        return []

    if 'syms' not in node:
        node['syms'] = []
    
    aligned = sa_utils.get_aligned_flags([node])    
    flex_aligned = sa_utils.get_flex_aligned_flags([node], FLEX_AA_THRESH)
    attachments = node['attachments']
        
    dir_att_set = get_valid_directions(attachments, aligned, True)
        
    if len(dir_att_set) == 0:        
        dir_att_set = get_valid_directions(
            attachments, flex_aligned, True
        )

        if len(dir_att_set) == 0:
            dir_att_set = get_valid_directions(
                attachments, flex_aligned, True, True
            )
            
    orders = []

    if len(dir_att_set) > MAX_DIRS:
        dir_att_set = get_simple_valid_directions(
            attachments, flex_aligned
        )
        
        if len(dir_att_set) > MAX_DIRS:            
            return []
        
    for dir in dir_att_set:
        try:
            orders += get_valid_orders(dir)            
            if (time.time() - start_time) > MAX_TIME or \
               len(orders) >= MAX_ORDERS:
                break
            
        except AssertionError as e:                        
            continue

    if len(orders) == 0:
        dir_att_set = get_simple_valid_directions(
            attachments, aligned
        )
        for dir in dir_att_set:
            try:                
                orders += get_valid_orders(dir)            
                if (time.time() - start_time) > MAX_TIME or \
                   len(orders) >= MAX_ORDERS:
                    break
                
            except AssertionError as e:                        
                continue
                
    if len(orders) == 0:
        dir_att_set = get_valid_directions(
            attachments, aligned, False
        )

        if len(dir_att_set) == 0:        
            dir_att_set = get_valid_directions(
                attachments, flex_aligned, False
            )
        if len(dir_att_set) > MAX_DIRS:
            return []
            
        for dir in dir_att_set:
            try:
                orders += get_valid_orders(dir)
                if (time.time() - start_time) > MAX_TIME:
                    return []
            except AssertionError:
                continue
            
    if 'sq_info' not in node:
        node['sq_info'] = getSqueezeInfo(node)
        
    sq_info = node['sq_info']

    orders.sort()
    
    valid_orders = valid_filter(node, orders, sq_info)
    
    return valid_orders


def getSingleAttrs(node):
    cube_attrs = []

    for i in range(len(node['cubes'])):
        cube_attrs.append((
            round(node['cubes'][i]['xd'], PREC),
            round(node['cubes'][i]['yd'], PREC),
            round(node['cubes'][i]['zd'], PREC),
            node['aligned'][i]
        ))

    att_attrs = {}

    if True:
        for _ind1, _ind2, _at1, _at2, _, _ in node['attachments']:
            for ind1, ind2, at1, at2 in [
                (_ind1, _ind2, _at1, _at2), (_ind2, _ind1, _at2, _at1)
            ]:
                if (ind1, ind2) in att_attrs:
                    att_attrs[(ind1, ind2)].append((
                        max(ind1, 0),
                        max(ind2, 0),
                        round(at1[0], PREC),
                        round(at1[1], PREC),
                        round(at1[2], PREC),
                        round(at2[0], PREC),
                        round(at2[1], PREC),
                        round(at2[2], PREC)
                    ))
                else:                
                    att_attrs[(ind1, ind2)] = [(
                        max(ind1, 0),
                        max(ind2, 0),
                        round(at1[0], PREC),
                        round(at1[1], PREC),
                        round(at1[2], PREC),
                        round(at2[0], PREC),
                        round(at2[1], PREC),
                        round(at2[2], PREC)
                    )]

    sq_attrs = {}
    if True:
        for ind, d in node['sq_info'].items():
            if ind not in sq_attrs:
                sq_attrs[ind] = {}
            for (i1,i2,f1,_,u,v) in d:
                u = round(u, PREC)
                v = round(v, PREC)
                if (i1, i2) in sq_attrs[ind]:
                    sq_attrs[ind][(i1, i2)].append((f1, u, v))
                else:
                    sq_attrs[ind][(i1, i2)] = [(f1, u, v)]

    sym_info = {}
    
    for sym in node['syms']:
        ind = sym[0]
        typ = sym[1].split('_')[0]
        axis = sym[1].split('_')[1]
        if len(sym) == 2:
            d = (typ, axis)
        else:
            d = (typ, axis, sym[2] / sa_utils.TRANS_NORM, round(sym[3], PREC))
        if ind in sym_info:
            sym_info[ind].append(d)
        else:
            sym_info[ind] = [d]
                
    return cube_attrs, att_attrs, sq_attrs, sym_info


def get_canon_map(order):
    cmap = {0:0}
    seen = set()
    count = 1
    for o in order:
        mc = o[1]
        if mc not in seen:
            cmap[mc] = count
            count += 1
            seen.add(mc)
    return cmap


def canonicalize(order, node, use_given_order=False):

    if use_given_order:
        cmap = {i:i for i in range(11)}
        canon_ret_list = None
        canon_attr_list = None
    else:
        _, (_, canon_attr_list, canon_ret_list), (_,_) = canonicalize(
            order, node, True
        )
        cmap = get_canon_map(order)        
        
    co = []
    for o in order:
        no = [o[0]]
        for a in o[1:]:
            if a > 0:
                no.append(cmap[a])
            else:
                no.append(a)
        co.append(tuple(no))

    canon_order = tuple(co)

    cube_attrs, att_attrs, sq_attrs, sym_attrs = getSingleAttrs(node)

    attr_list = []
    return_list = []
    
    seen = set()    
    cube_lookup = {}
    
    for oj, v in enumerate(cube_attrs):
        cj = cmap[oj]
        name = f"cube{cj-1}" if cj > 0 else "bbox"        
        ca = ("Cuboid", v[0], v[1], v[2], v[3])
        if cj == 0:
            attr_list.append(ca)
            return_list.append(name)
        else:
            cube_lookup[cj] = (ca, name)

    fprms = []
    
    for ov, cv in zip(order, canon_order):
        t = cv[0]
        cc = cv[1]
        oc = ov[1]
                
        if cc not in seen:
            att, name = cube_lookup[cc]
            seen.add(cc)
            fprms += att[1:4]
            attr_list.append(att)
            return_list.append(name)

        if t == 'att':
            oo = ov[2]
            co = cv[2]
            d = att_attrs[(oc, oo)]
            
            n2 = f"cube{co-1}" if co > 0 else "bbox"

            fprms += d[0][2:]
            
            attr_list.append(
                ("attach", n2, d[0][2], d[0][3], d[0][4], d[0][5], d[0][6], d[0][7])
            )

        elif t == 'squeeze':
            oo1 = ov[2]
            oo2 = ov[3]
            co1 = cv[2]
            co2 = cv[3]
            
            d = sq_attrs[oc][(oo1, oo2)]

            n2 = f"cube{co1-1}" if co1 > 0 else "bbox"
            n3 = f"cube{co2-1}" if co2 > 0 else "bbox"

            fprms += d[0][1:]
            
            attr_list.append(
                ("squeeze", n2, n3, d[0][0], d[0][1], d[0][2])
            )

        elif t == 'ref':
            d = sym_attrs[oc]

            attr_list.append(
                ("reflect", d[0][1])
            )

        elif t == 'trans':
            d = sym_attrs[oc]
            attr_list.append(
                ("translate", d[0][1], d[0][2], d[0][3])
            )

            fprms += d[0][2:]
            
        return_list.append(None)
            
    return canon_order, (np.array(fprms), attr_list, return_list), (canon_attr_list, canon_ret_list)
