import sys
import os
from copy import copy, deepcopy
import time
from multiprocessing import Pool
import math
import random
from random import sample, shuffle
import numpy as np
from tqdm import tqdm
import pickle
import importlib
from sklearn.cluster import DBSCAN
import networkx as nx

VERBOSE = True
NPROG_COUNT = 0

def make_function(name, args):
    args = [str(arg) for arg in args]
    return '{}({})'.format(name, ", ".join(args))

def assign(var_name, value):
    return '{} = {}'.format(var_name, value)

# Generalize the i-j position in t:
# make a free variable
def replace(t, r, a, count, max_changes):
    if count > max_changes:
        return []
    
    counts = {'f':0, 'b':0, 'i':0, 'c':0}
    seen = set()
    vtype = None
    vcount = None
    vexpr = None
    new = []

    gens = []
    
    for i in range(0, len(t)):
        line = [t[i][0]]
        for j in range(1, len(t[i])):
            _type = t[i][j].split('_')[0]
            if (i, j) == r[0]:
                vtype = _type
                vcount = counts[vtype]
                vexpr = t[i][j]
                line.append(f'{vtype}_var_{vcount}')
                
            elif (i, j) in r:
                assert vexpr
                extra = ' * 1.0' if vtype == 'f' else ''
                line.append(f'{vtype}_var_{vcount}' + extra)                
                
            elif vexpr is None:
                if 'var' in t[i][j]:
                    var = t[i][j].split()[0]
                    if var not in seen:
                        seen.add(var)
                        counts[_type] += 1
                line.append(t[i][j])
            else:
                if (t[i][j] == vexpr and \
                    (i, j) not in r and \
                    i >= a[0] and j >= a[1]
                ):
                    _ngs = replace(t, r + [(i,j)], (i,j), count+1, max_changes )
                    for _ng, _c in _ngs:
                        gens.append((
                            _ng, _c
                        ))
                
                if _type != vtype:
                    line.append(t[i][j])
                else:
                    expr = []
                    for c in t[i][j].split('+'):
                        if 'var' not in c:
                            expr.append(c)
                        else:
                            cs = c.split()
                            vc = cs[0].split('_')
                            if int(vc[2]) >= vcount:
                                vc[2] = str(int(vc[2]) + 1)

                            cs[0] = '_'.join(vc)
                            expr.append(' '.join(cs))
                    line.append(' + '.join(expr))
        new.append(tuple(line))

    gens.append((tuple(new), count + 1))
    return [(g, c) for g, c in gens if c <= max_changes]


# Find all generalization of the abstraction t within max_changes edits
def findGens(t, max_changes):
    gens = set([t])
    q = [(t, 0)]
    while len(q) > 0:        
        cur, count = q.pop()
        assert count <= max_changes
        seen = set()
        for i in range(0, len(cur)):
            for j in range(1, len(cur[i])):
                name = cur[i][j]
                if 'var' in name and '*' not in name and name not in seen:
                    seen.add(name)
                else:
                    new_gens = replace(cur, [(i, j)], (0, 0), count, max_changes)
                    for ng, c in new_gens:
                        if ng not in gens:
                            gens.add(ng)
                            q.append((ng, c))

    return gens


def getType(s, vmap, replace=False, ret_num=0):
    _type = s[0]
    _cat = None
    _expr = copy(s)
    ind = None
    
    if _type == 'b':
        if s[1] == 'const':
            _cat = 'expr'

        elif s[1] == 'var':
            if s[2] in vmap:
                _cat = 'expr'
                _expr = vmap[s[2]] if replace else s
            else:
                _cat = 'var'
                ind = s[2]

    if _type == 'c':
        if s[1] == 'const':
            _cat = 'expr'
        elif s[1] == 'var':
            if s[2] in vmap:
                _cat = 'expr'
                _expr = vmap[s[2]] if replace else s
            else:
                _cat = 'var'
                ind = s[2]

    if _type == 'i':
        
        if s[1] == 'const':
            _cat = 'expr'

        elif s[1] == 'ret':
            _cat = 'expr'
            _expr = (s[0], s[1], s[2] + ret_num)

        elif s[1] == 'var':
            if s[2] in vmap:
                _cat = 'expr'
                _expr = vmap[s[2]] if replace else s
            else:
                _cat = 'var'
                ind = s[2]

    if _type == 'f':
        if len(s[1]) == 1 and s[1][0][0] not in vmap:
            _cat = 'var'
            ind = s[1][0][0]
        else:            
            _cat = 'expr'
            terms = []
            for _ind, _scale in s[1]:
                if replace and _ind in vmap:
                    for o_ind, o_scale in vmap[_ind][1]:
                        terms.append((o_ind, o_scale * _scale))
                else:                    
                    terms.append((_ind, _scale))
            terms.sort()
            _expr = ('f', tuple(terms))            
                
    return _cat, _expr, ind
                
# Checks if g is general case of specific logic s (can g always represent s)
def isGenLogic(G, S, ret_num, dsl):
    vmap = {
        i: ('f', ((i, 1.),)) for i in range(len(dsl.inp_params))                
    }

    for lg, ls in zip(G, S):
        for g, s in zip(lg, ls):
            _, sexpr, _ = getType(s, vmap)
            gcat, gexpr, gind = getType(g, vmap, True, ret_num)

            if gcat == 'var' and gind not in vmap:
                vmap[gind] = sexpr
            else:                
                if gexpr != sexpr:
                    return False
                
    return True

def calcGain(func, dsl):
    gain = func.constrData()
    gain['line'] = -1
    
    logic = func.getLogic()
    structure = func.getStructure()    

    ret_num = 0
    
    while(len(logic) > 0):
        library = dsl.getOrdLibrary()
        for n, f in library:
        
            na = len(f.getStructure())            
            if (f.getStructure() == structure[:na] and
                isGenLogic(f.getLogic(), logic[:na], ret_num, dsl)
            ):                
                logic = logic[na:]
                for rf in dsl.ret_fns:
                    ret_num += structure[:na].count(rf)
                structure = structure[na:]

                cd = f.constrData()
                for k, v in cd.items():
                    gain[k] -= v
                    
                gain['line'] += 1.
                break

    return gain

# lines sep by |
# params sep by :
# free vars gets {}

def getFuncHash(a):
    h = ''
    values = []
    for i in range(len(a)):
        if i > 0:
            h += '|'
        for j in range(len(a[i])):
            if j > 0:
                h += ':'
            exprs = a[i][j].split('+')
            for _i, e in enumerate(exprs):
                if _i > 0:
                    h +='+'
                prms = e.split('*')
                var = prms[0].strip()
                
                if len(prms) > 1:                    
                    var += f'*{len(values)}'
                    values.append(float(prms[1].strip()))
                h += var

    return h, values

def fillHash(h, f, dsl):
    abst = []
    for l in h.split('|'):
        line = []
        for e in l.split(':'):
            expr = []
            for t in e.split('+'):
                if '*' in t:
                    var = t.split('*')[0]
                    scale = f[int(t.split('*')[1])]
                    round_scale = round(int(scale * dsl.scale_prec) / dsl.scale_prec, dsl.prec)
                    expr.append(f'{var} * {round_scale}')
                else:
                    expr.append(t)
            line.append(' + '.join(expr))
        abst.append(tuple(line))

    return tuple(abst)


def groupHashedAbs(_hash, group, dsl):
    if len(group) == 1:
        return {fillHash(_hash, group[0][0], dsl) : group[0][2]}

    db = DBSCAN(eps = dsl.cluster_eps, min_samples = 1, metric = 'manhattan')

    counts = np.array([len(g[2]) for g in group])
    params = np.array([np.array(g[0]) for g in group])        
    res = db.fit(params).labels_

    group_abs = {}
    
    for c in np.unique(res):
        c_inds = (res == c).nonzero()[0]        
        fit = np.sum(params[c_inds] * counts[c_inds].reshape(-1, 1), axis=0) / counts[c_inds].sum()
        g_abs = fillHash(_hash, fit, dsl)

        for ci in c_inds:        
            if g_abs in group_abs:
                group_abs[g_abs] = group_abs[g_abs].union(group[ci][2])
            else:
                group_abs[g_abs] = group[ci][2]

    return group_abs

                
# Post Process all found candidate abstraciton
def genCandAbs(gen_cand_abs, count, max_changes, dsl):
    
    t = time.time()
        
    # store {hash : [(params, full_abs, count)]}
    hashed_abs = {}

    for _abs, _cov in gen_cand_abs.items():
        _hash, params = getFuncHash(_abs)
        if _hash in hashed_abs:
            hashed_abs[_hash].append((params, _abs, _cov))
        else:
            hashed_abs[_hash] = [(params, _abs, _cov)]

    group_cand_abs = {}    
    for _hash, group in hashed_abs.items():
        group_abs = groupHashedAbs(_hash, group, dsl)        
        for _abs, _cov in group_abs.items():

            group_cand_abs[_abs] = _cov
                                            
    # Graph
    # Map from inds (i.e. node index) to abstraction
    node_to_abs = {}
    abs_to_node = {}
    final_cand_abs = {}
    
    G = nx.Graph()

    for k, v in group_cand_abs.items():
        if k not in final_cand_abs:
            G.add_node(len(node_to_abs))
            abs_to_node[k] = len(node_to_abs)
            node_to_abs[len(node_to_abs)] = k            
            final_cand_abs[k] = v            
        else:
            final_cand_abs[k] = final_cand_abs[k].union(v)

        node_i = abs_to_node[k]
        
        gens = findGens(k, max_changes)        
        for g in gens:
            if g not in final_cand_abs:
                G.add_node(len(node_to_abs))
                G.add_edge(node_i, len(node_to_abs))
                abs_to_node[g] = len(node_to_abs)
                node_to_abs[len(node_to_abs)] = g                
                final_cand_abs[g] = v
            else: 
                G.add_edge(node_i, abs_to_node[g])
                final_cand_abs[g] = final_cand_abs[g].union(v)


    ll_abs = []
    count = 0

    # For each candidate abstraction in the same group, find all of the lines that it covers
    # then for speed purposes, remove the candidate abstractions with worse gains that cover
    # the same amount of lines, because they would never be considered during sleep
    for group in nx.connected_components(G):
        num_to_score = {}
        l_abs = []
        for c in group:
            _abs = node_to_abs[c]
            gain = calcGain(Function(_abs, dsl), dsl)
            score = dsl.scoreGain(gain)            
            num_abs = len(final_cand_abs[_abs])

            if num_abs not in num_to_score:
                num_to_score[num_abs] = score
            else:
                num_to_score[num_abs] = max(score, num_to_score[num_abs])

        for c in group:
            _abs = node_to_abs[c]
            gain = calcGain(Function(_abs, dsl), dsl)
            score = dsl.scoreGain(gain)
            num_abs = len(final_cand_abs[_abs])

            if score < num_to_score[num_abs]:
                continue
            
            l_abs.append([
                None,
                _abs,
                final_cand_abs[_abs]
            ])
            count += 1
                    
        ll_abs.append(l_abs)
        
    
    return ll_abs, count

# removes frequency of freq from all generalization of cab in cand_abs
def updateFreq(group_cand_abs, cab_cov):
    for i in range(len(group_cand_abs)):
        for j in range(len(group_cand_abs[i][1])):
            rem_cov =  list(group_cand_abs[i][1][j][2] - cab_cov)

            if len(rem_cov) == 0:
                group_cand_abs[i][1][j][2] = set()
                continue
            
            rem_cov = [(int(r.split('_')[0]), int(r.split('_')[1])) for r in rem_cov]
            rem_cov.sort()
            struct_len = len(group_cand_abs[i][1][j][1])

            new_cov = set()
            cur_group = set([f'{rem_cov[0][0]}_{rem_cov[0][1]}'])            
            prog_index = rem_cov[0][0]
            line_index = rem_cov[0][1]
            for rc in rem_cov[1:]:
                # New group
                if rc[0] != prog_index or rc[1] != line_index+1:
                    if len(cur_group) >= struct_len:
                        new_cov = new_cov.union(cur_group)

                    cur_group = set([f'{rc[0]}_{rc[1]}'])            
                    prog_index = rc[0]
                    line_index = rc[1]

                # continutation
                else:
                    line_index = rc[1]
                    cur_group.add(f'{rc[0]}_{rc[1]}')
                    
            if len(cur_group) >= struct_len:
                new_cov = new_cov.union(cur_group)
            
            group_cand_abs[i][1][j][2] = new_cov


def hasGoodMEFit(fit, targets, scalars, max_error, dsl):

    lines = scalars.reshape(-1, 1) @ fit.reshape(1, -1)
    error = np.abs(lines - targets)
    
    g_inds = (error.max(axis=0) < max_error).nonzero()[0]

    if g_inds.shape[0] < (error.shape[1] * dsl.err_shape_perc):
        return None

    return g_inds

def hasGoodMAEFit(fit, targets, scalars, max_avg_error):
    lines = scalars.reshape(-1, 1) @ fit.reshape(1, -1)
    error = np.abs(lines - targets)
    
    if error.mean() < max_avg_error:
        return True
    else:
        return False

def checkFixedId(cvars, cscalars, targets, scalars, max_error, max_avg_error, dsl):
    def f():
        for var in cvars:
            for s in cscalars:
                expr = f'{var[0]} * {s}'
                fit = var[1] * s                
                g_inds = hasGoodMEFit(
                    fit, targets, scalars, max_error, dsl
                )

                if g_inds is None:
                    continue
                
                res = hasGoodMAEFit(
                    fit[g_inds], targets[:,g_inds], scalars, max_avg_error
                )

                if res:
                    return expr, fit
                
        return None, None
     
    return f
    
def checkFixedComb(cvars, dvars, cscalars, targets, scalars, max_error, max_avg_error, dsl):
    def f():
        for cvar in cvars:
            for dvar in dvars:
                for c,d in cscalars:
                    expr = f'{cvar[0]} * {c} + {dvar[0]} * {d}' 
                    fit = (cvar[1] * c) + (dvar[1] * d)
                    g_inds = hasGoodMEFit(
                        fit, targets, scalars, max_error, dsl
                    )

                    if g_inds is None:
                        continue
                
                    res = hasGoodMAEFit(
                        fit[g_inds], targets[:,g_inds], scalars, max_avg_error
                    )

                    if res:
                        return expr, fit
        return None, None
    return f
    
def checkFitLin(cvars, targets, scalars, max_error, max_avg_error, dsl):
    def f():
        for var in cvars:            
            sol = np.linalg.lstsq(
                np.tile(var[1], targets.shape[0]).reshape(-1, 1),
                (targets / scalars.reshape(-1, 1)).reshape(-1, 1),
                rcond=None
            )[0][0]
            fit = sol * var[1]
            g_inds = hasGoodMEFit(
                fit, targets, scalars, max_error, dsl
            )

            if g_inds is None:
                continue

            sol = np.linalg.lstsq(
                np.tile(var[1][g_inds], targets.shape[0]).reshape(-1, 1),
                (targets[:,g_inds] / scalars.reshape(-1, 1)).reshape(-1, 1),
                rcond=None
            )[0][0]
            fit = sol * var[1][g_inds]

            res = hasGoodMAEFit(
                fit, targets[:, g_inds], scalars, max_avg_error
            )

            if res:                
                expr = f'{var[0]} * {round(sol[0], dsl.prec)}'
                return expr, fit
        return None, None
    return f
    
def checkFitComb(cvars, dvars, targets, scalars, max_error, max_avg_error, dsl):
    def f():
        for cvar in cvars:
            for dvar in dvars:
                
                sol = np.linalg.lstsq(
                    np.stack((np.tile(cvar[1], targets.shape[0]),
                              np.tile(dvar[1], targets.shape[0]))
                    ).T,
                    (targets / scalars.reshape(-1, 1)).reshape(-1, 1),
                    rcond=None
                )[0].reshape(-1)
                
                fit = (sol[0] * cvar[1]) + (sol[1] * dvar[1])
                g_inds = hasGoodMEFit(
                    fit, targets, scalars, max_error, dsl
                )

                if g_inds is None:
                    continue

                sol = np.linalg.lstsq(
                    np.stack((np.tile(cvar[1][g_inds], targets.shape[0]),
                              np.tile(dvar[1][g_inds], targets.shape[0]))
                    ).T,
                    (targets[:,g_inds] / scalars.reshape(-1, 1)).reshape(-1, 1),
                    rcond=None
                )[0].reshape(-1)
                fit = (sol[0] * cvar[1][g_inds]) + (sol[1] * dvar[1][g_inds])

                res = hasGoodMAEFit(
                    fit, targets[:, g_inds], scalars, max_avg_error
                )

                if res:                
                    expr = f'{cvar[0]} * {round(sol[0], dsl.prec)} + {dvar[0]} * {round(sol[1], dsl.prec)}'
                    return expr, fit
        return None, None
    return f
    
def fitFloatVar(var_list, targets, scalars, max_error, max_avg_error, dsl):
    assert scalars[0] == 1.0, "Saw a non 1 first scalar"
    scalars = np.array(scalars)

    l = []
    for spec in dsl.FloatVarPrefOrder:
        if spec['name'] == 'id':
            l.append(checkFixedId(
                [v for v in var_list[spec['pstart']:spec['pend']] if v[0].split('_')[0] == 'f'],
                 spec['params'], targets, scalars, max_error, max_avg_error, dsl
            ))
        elif spec['name'] == 'comb':            
            l.append(checkFixedComb(
                [v for v in var_list[spec['p1start']:spec['p1end']] if v[0].split('_')[0] == 'f'],
                [v for v in var_list[spec['p2start']:spec['p2end']] if v[0].split('_')[0] == 'f'],
                spec['params'], targets, scalars, max_error, max_avg_error, dsl
            ))
        elif spec['name'] == 'lin':
            l.append(checkFitLin(
                [v for v in var_list[spec['pstart']:spec['pend']] if v[0].split('_')[0] == 'f'],
                targets, scalars, max_error, max_avg_error, dsl
            ))
        elif spec['name'] == 'lincomb':
            l.append(checkFitComb(
                [v for v in var_list[spec['p1start']:spec['p1end']] if v[0].split('_')[0] == 'f'],
                [v for v in var_list[spec['p2start']:spec['p2end']] if v[0].split('_')[0] == 'f'],
                targets, scalars, max_error, max_avg_error, dsl
            ))
        else:
            assert False, f'bad spec {spec}'
                
            
    for f in l:
        res, fit = f()
        if res is not None:
            return res, fit

    return None, None
    
                
def getCandProg(line_attrs, return_attrs, dsl, max_error, max_avg_error):

    var_list = dsl.getInitVarList(line_attrs, return_attrs)

    lines = []
    cabs = set()

    fvars, ivars, bvars, cvars, rets, has_const = None, None, None, None, None, None
    cab = []

    fassoc = {
        v: (v, 1.0) for v,_ in var_list
    }
    
    iassoc = {}
    bassoc = {}
    cassoc = {}
    
    skip_length = None

    start_struct, start_att = None, None
    
    while True:

        # First do abstraction generation logic
        if len(lines) == 1:
            fvars, ivars, bvars, cvars, rets, has_const = parseLine(lines[0][1], lines[0][0])
            cab.append(fillLine(lines[0][1], fassoc, iassoc, bassoc, cassoc, rets, dsl))            
            if has_const:
                cabs.add(tuple(cab))

            start_att = lines[0][1][0]
            start_struct = dsl.library[start_att].getStructure()    
            

        elif len(lines) > 1:
            n_line = lines[-1][1]
            n_ret = lines[-1][0]
            
            next_struct = dsl.library[n_line[0]].getStructure()
                
            nfvars, nivars, nbvars, ncvars,nrets, new_has_const = parseLine(n_line, n_ret)

            has_const = new_has_const or has_const
            finter = fvars.intersection(nfvars)
            iinter = ivars.intersection(nivars)
            binter = bvars.intersection(nbvars)
            cinter = cvars.intersection(ncvars)
            rinter = set(rets.keys()).intersection(nivars)
            
            cab.append(fillLine(n_line, fassoc, iassoc, bassoc, cassoc, rets, dsl))

            # Check that abstraction has valid structure (under some criteria)

            msg = dsl.checkValidMacro(cab, start_struct, next_struct)

            if msg == 'stop':
                break
            elif msg == 'add':                            
                cabs.add(tuple(cab))
                        
            fvars = fvars.union(nfvars)
            ivars = ivars.union(nivars)
            bvars = bvars.union(nbvars)
            cvars = cvars.union(ncvars)
        
            for nret in nrets:
                rets[nret] = len(rets)                                                    
                
        if len(line_attrs) == 0:
            break            

        # Now go on to candidate program generation logic
        
        library = dsl.getOrdLibrary()
        
        while(len(library) > 0):
            n, f = library.pop(0)

            num_attrs = len(f.getStructure())
            misses = 0.
            count = 0.
            
            for i in range(len(line_attrs[:num_attrs][0])):
                prog_line = [a[i] for a in line_attrs[:num_attrs]]
                return_line = [r for r in return_attrs[:num_attrs] if r is not None]

                def_params = [v[1][i] for v in var_list[:len(dsl.inp_params)]]

                is_fit, _ = f.checkFit(prog_line, def_params, return_line, max_error)
                
                count += 1.
                
                if not is_fit:
                   misses += 1.                                   

            if ((count - misses) / (count + 1e-8)) < dsl.err_shape_perc:
                continue
            
            if len(lines) == 0:
                skip_length = num_attrs
            
            line = [n]

            var_res = {
                v: var_list[v][1] for v in range(len(dsl.inp_params))
            }
            
            for ind, ifvar in enumerate(f.getInterface()):
                find = ind + len(dsl.inp_params)
                var_result = None
                # float
                if ifvar[0] == 'f':
                    
                    func_logic = f.getLogic()
                                        
                    scalars = []
                    targets = []
                    
                    for i in range(len(func_logic)):
                        for j in range(len(func_logic[i])):
                            if func_logic[i][j][0] == 'f' and find in [fl[0] for fl in func_logic[i][j][1]]:
                                skip = False
                                _scalars = []
                                _target = np.array([a[j+1] for a in line_attrs[i]])

                                _psum = np.zeros(_target.shape)

                                for _ind, _scale in func_logic[i][j][1]:                                    
                                    if _ind == find:
                                        _scalars.append(_scale)
                                    elif _ind not in var_res:
                                        skip = True
                                    else:
                                        _psum += var_res[_ind] * _scale
                                    
                                if skip:
                                    continue
                                
                                targets.append(_target - _psum)
                                scalars += _scalars
                                
                    targets = np.array(targets)
                                                            
                    best_var, _var_result = fitFloatVar(
                        var_list, targets, scalars, max_error, max_avg_error, dsl
                    )

                    var_result = _var_result
                    
                    if best_var is not None:
                        line.append(best_var)

                    else:
                        line.append(f"f_var_{len(var_list)}")
                        var_list.append((
                            f"f_var_{len(var_list)}",
                            targets[0]))
                        var_result = targets[0]                        
                        
                # boolean
                elif ifvar[0] == 'b':
                    func_logic = f.getLogic()
                    targets = [] 

                    for i in range(len(func_logic)):
                        for j in range(len(func_logic[i])):
                            if func_logic[i][j][0] == 'b' and \
                               func_logic[i][j][1] == 'var' and \
                               func_logic[i][j][2] == find:
                                targets = [a[j+1] for a in line_attrs[i]]
                                break
                        if len(targets) > 0:
                            break
                            
                    best_var = None

                    num_count = len(targets) * 1.
                    num_true = len([t for t in targets if t is True]) * 1.

                    if (num_true / num_count) > dsl.err_const_perc:
                        best_var = "b_1"

                    elif (num_true/ num_count) < 1-dsl.err_const_perc:
                        best_var = "b_0"
                        
                    else:
                        for var in var_list:
                            if var[0].split('_')[0] == 'b':
                                nm = 0
                                for _target, _var in zip(targets, var[1]):
                                    nm += float(_target == _var)

                                if (nm/num_count) > dsl.err_shape_perc:
                                    best_var = var[0]
                                    break

                    if best_var is None:
                        line.append(f"b_var_{len(var_list)}")
                        var_list.append((
                            f"b_var_{len(var_list)}",
                            targets
                        ))
                    else:                        
                        line.append(best_var)
                        
                elif ifvar[0] == 'c':
                    func_logic = f.getLogic()
                    targets = []
                    for i in range(len(func_logic)):
                        for j in range(len(func_logic[i])):
                            if func_logic[i][j][0] == 'c' and \
                               func_logic[i][j][1] == 'var' and \
                               func_logic[i][j][2] == find:
                                targets = [a[j+1] for a in line_attrs[i]]
                                break
                        if len(targets) > 0:
                            break

                    num_count = len(targets)
                    best_var = None

                    for const in dsl.c_vals:
                        num_const = len([t for t in targets if t == const]) * 1.

                        if (num_const / num_count) > dsl.err_const_perc:
                            best_var = f"c_{const}"
                            break
                        
                    if best_var is None:
                        for var in var_list:
                            if var[0].split('_')[0] == 'c':
                                nm = 0
                                for _target, _var in zip(targets, var[1]):
                                    nm += float(_target == _var)

                                if (nm/num_count) > dsl.err_shape_perc:
                                    best_var = var[0]
                                    break
                                
                    if best_var is None:
                        line.append(f"c_var_{len(var_list)}")
                        var_list.append((
                            f"c_var_{len(var_list)}",
                            targets
                        ))
                    else:
                        line.append(best_var)

                        
                elif ifvar[0] == 'i':
                    func_logic = f.getLogic()
                    targets = []
                    for i in range(len(func_logic)):
                        for j in range(len(func_logic[i])):
                            if func_logic[i][j][0] == 'i' and \
                               func_logic[i][j][1] == 'var' and \
                               func_logic[i][j][2] == find:
                                targets.append([a[j+1] for a in line_attrs[i]])

                    line.append(f"i_{targets[0][0]}")

                if var_result is not None:
                    var_res[find] = var_result

                
            return_values = []
            for _ in range(num_attrs):
                line_attrs.pop(0)
                return_values.append(return_attrs.pop(0))

            return_values = tuple([n for n in return_values if n is not None])
 
            lines.append((return_values, tuple(line)))
            break
        
    return cabs, skip_length
    

def parseLine(line, ret):
    fvars = set()
    ivars = set()
    bvars = set()
    cvars = set()
    rets = {}
    has_const = False
    
    for var in line[1:]:
        if var.split('_')[0] == 'f':
            prms = var.split('+')
            all_const = True
            for prm in prms:
                if 'f_var' in prm:
                    all_const = False
                    if '*' in prm:
                        fvars.add(prm.split()[0])
                    else:
                        fvars.add(prm)

            if all_const:
                has_const = True            
                        
        elif var.split('_')[0] == 'i':
            if 'i_bbox' in var:
                has_const = True
            else:
                ivars.add(var)

        elif var.split('_')[0] == 'b':
            if 'b_var' in var:
                bvars.add(var)
            else:
                has_const = True            

        elif var.split('_')[0] == 'c':
            if 'c_var' in var:
                cvars.add(var)
            else:
                has_const = True
                
    for i, r in enumerate(ret):
        rets[r] = i

    return fvars, ivars, bvars, cvars, rets, has_const

def fillLine(cline, fassoc, iassoc, bassoc, cassoc, rets, dsl):
    lab = [cline[0]]
    for var in cline[1:]:
        # constants
        if var.split('_')[0] == 'f':
            prms = var.split('+')
            expr = []

            for prm in prms:
                if 'f_var' in var:
                    n_var = prm.split()[0] if '*' in prm else prm
                    scalar = float(prm.split()[2]) if '*' in prm else 1.0
                    if n_var not in fassoc:
                        fassoc[n_var] = (f'f_var_{len(fassoc)-len(dsl.inp_params)}', scalar)
                        expr.append(f'{fassoc[n_var][0]}')
                    else:
                        expr.append(
                            f"{fassoc[n_var][0]} * {round(scalar / fassoc[n_var][1], dsl.prec)}"
                        )
                    
                else:
                    expr.append(f'{prm}')
                                        
            lab.append(' + '.join(expr))
            
        # int vars                
        elif var.split('_')[0] == 'i':            
            if 'bbox' in var:
                lab.append(var)
            elif var[2:] in rets:
                lab.append(f'i_ret_{rets[var[2:]]}')
            else:
                if var not in iassoc:
                    iassoc[var] = len(iassoc)
                lab.append(f'i_var_{iassoc[var]}')

        # b vars
        elif var.split('_')[0] == 'b':
            if 'var' not in var:
                lab.append(var)
            else:
                if var not in bassoc:
                    bassoc[var] = len(bassoc)
                lab.append(f'b_var_{bassoc[var]}')

        elif var.split('_')[0] == 'c':
            if 'var' not in var:
                lab.append(var)
            else:
                if var not in cassoc:
                    cassoc[var] = len(cassoc)
                lab.append(f'c_var_{cassoc[var]}')
                
        else:
            print(f'saw bad instance: {var}')
                        
    return tuple(lab)


def findCandAbs(
    attr_list, return_list, dsl, max_error, max_avg_error, count
):
    
    cand_abs = {}

    static_attr, static_ret = dsl.getStaticInfo(attr_list, return_list)
    
    line_count = 1
    
    while(len(attr_list) > 0):
        
        new_abs, skip = getCandProg(
            static_attr + attr_list,
            static_ret + return_list,            
            dsl,
            max_error,
            max_avg_error
        )
        
        for _ in range(skip):
            attr_list.pop(0)
            return_list.pop(0)

        for na in new_abs:
            length = 0
            for n in na:
                length += len(dsl.library[n[0]].getStructure())
            lines = list(range(line_count, line_count+length))
            lines = set([f'{count}_{l}' for l in lines])
            
            if na in cand_abs:
                cand_abs[na] = cand_abs[na].union(lines)
            else:
                cand_abs[na] = lines
                                        
        line_count += skip
            
    return cand_abs

            
class Function():
    # descr containts the tuple of the function def in strings
    def __init__(
        self,
        descr,
        dsl,
    ):
        
        self.dsl = dsl
        self.descr = descr
        self.name = None
        self.raw_structure = []
        self.structure = []
        self.num_def_params = len(dsl.inp_params)
        self.inp_params = list(dsl.inp_params)
        self.logic = []
        self.var_map = {f:i for i,f in enumerate(dsl.inp_params)}
        
        # For return variables from previous functions
        prev_returns = 0
        
        for line in descr:
            self.raw_structure.append(line[0])
            ll = self.local_logic(line) 
            struct = None

            if line[0] not in dsl.base_functions and \
               (line[0] in dsl.full_library or line[0] in dsl.library):
                if line[0] in dsl.library:
                    f = dsl.library[line[0]]
                else:
                    f = dsl.full_library[line[0]]
                struct = f.getStructure()                
                self.comp_func(ll, f, prev_returns)
            else:
                struct = [line[0]]
                self.logic.append(ll)                

            prev_returns += struct.count(dsl.ret_name)
            self.structure += struct

        self.rev_var_map = {v:k for k,v in self.var_map.items()}
                
    def comp_func(self, ll, f, prev_returns):
        f_logic = f.getLogic()
        for f_ll in f.logic:
            b_ll = []
            for var in f_ll:
                if var[0] == 'f':
                    p_ll = []
                    for ind, scale in var[1]:
                        find = ind - self.num_def_params
                        if find < 0:
                            p_ll.append((ind, scale))
                        else:                                                    
                            l = ll[find]
                            for _ind, _scale in l[1]:
                                p_ll.append((_ind, _scale * scale))

                    b_ll.append(('f', tuple(p_ll)))

                elif var[0] == 'b':
                    if var[1] == 'const':                        
                        b_ll.append(var)
                    else:
                        find = var[2] - self.num_def_params
                        b_ll.append(ll[find])

                elif var[0] == 'c':
                    if var[1] == 'const':
                        b_ll.append(var)
                    else:
                        find = var[2] - self.num_def_params
                        b_ll.append(ll[find])
                        
                elif var[0] == 'i':
                    if var[1] == 'const':
                        b_ll.append(var)

                    elif var[1] == 'ret':
                        b_ll.append((var[0], var[1], var[2] + prev_returns))
                        
                    elif var[1] == 'var':
                        find = var[2] - self.num_def_params
                        b_ll.append(ll[find])
            self.logic.append(b_ll)

                
    def local_logic(self, line):                
        ll = []
            
        for var in line[1:]:
            if var.split('_')[0] == 'i':
                vtype = var.split('_')[1]
                if self.dsl is not None and vtype in self.dsl.ivar_const:
                    ll.append(("i", "const", vtype))

                elif vtype == 'ret':                    
                    ind = int(var.split('_')[2])
                    ll.append(("i", "ret", ind))
                                        
                else:
                    if var not in self.var_map:
                        self.var_map[var] = len(self.inp_params)
                        self.inp_params.append(var)
                        
                    ind = self.var_map[var]                    
                    ll.append(("i", "var", ind))

            elif var.split('_')[0] == 'c':                                
                if 'var' not in var:
                    ll.append(('c', "const", var.split('_')[1]))
                else:
                    if var not in self.var_map:
                        self.var_map[var] = len(self.inp_params)
                        self.inp_params.append(var)
                    ind = self.var_map[var]
                    ll.append(('c', "var", ind))
                    
            elif var.split('_')[0] == 'f':
                prms = var.split('+')
                pl = []
                for prm in prms:
                    name = prm.split()[0] if "*" in prm else prm.strip()
                    a = float(prm.split()[2]) if "*" in prm else 1.0
                
                    if name not in self.var_map:
                        self.var_map[name] = len(self.inp_params)
                        self.inp_params.append(name)
                        
                    ind = self.var_map[name]
                    pl.append((ind, a))
                ll.append(("f", tuple(pl)))

            elif var.split('_')[0] == 'b':
                if 'var' in var:
                    if var not in self.var_map:
                        self.var_map[var] = len(self.inp_params)
                        self.inp_params.append(var)

                    ind = self.var_map[var]
                    ll.append(("b", 'var', ind))
                else:
                    if var.split('_')[1] == '1':
                        ll.append(("b", 'const', True))
                    else:
                        ll.append(("b", 'const', False))
            else:
                self.dsl.log_print(f"saw bad token {var}")
                
        return ll
    
                            
    # returns tuple of type of return attributes (i.e. (attach, attach))
    def getStructure(self):
        return self.structure
        
    # returns the expected input parameters and types
    def getInterface(self):
        return self.inp_params[self.num_def_params:]

    def getSplitInterface(self):
        d = {
            'i_var': 0,
            'f_var': 0,
            'b_var': 0,
            'c_var': 0,
        }

        for prm in self.inp_params[self.num_def_params:]:
            typ = prm.split('_')[0]
            d[f'{typ}_var'] += 1
            
        return d
    
    def constrData(self):
        free_vars = self.getSplitInterface()
        cd = {k: v * -1 for k,v in free_vars.items()}
        
        for l in self.getStructure():
            _cd = self.dsl.library[l].getSplitInterface()
            for k,v in _cd.items():
                cd[k] += v
                
        return cd
    
    def getLogic(self):
        return self.logic
            
    def getRead(self, line):
        params = []
        for var in line:
            if var[0] == 'f':
                fparams = []
                for ind, scale in var[1]:
                    if ind == 0:
                        fparams.append(str(scale))
                    elif scale == 1.0:
                        fparams.append(str(self.rev_var_map[ind]))
                    else:
                        fparams.append(f"{scale} * {self.rev_var_map[ind]}")

                params.append(' + '.join(fparams))
                    
            elif var[0] == 'i':
                if var[1] == 'const':
                    params.append(var[2])
                elif var[1] == 'ret':
                    params.append(f"{self.dsl.ret_name}{var[2]}")
                else:
                    params.append(self.rev_var_map[var[2]])

            elif var[0] == 'b' or var[0] == 'c':
                if var[1] == 'const':
                    params.append(var[2])
                else:
                    params.append(self.rev_var_map[var[2]])

        return params

    def checkFit(self, lines, def_params, return_vals, thresh = None):

        if thresh is None:
            thresh = self.dsl.abs_max_error
        
        structure_fit = [l[0] for l in lines] == self.getStructure()
        
        if not structure_fit:
            return False, None

        error = 0.

        param_fit = True

        params = copy(def_params)

        # NOTE: assumes all default params are floats
        float_inds = list(range(self.num_def_params)) + [i+self.num_def_params for i, var in enumerate(self.getInterface()) if var.split('_')[0] == 'f']
        
        A = []
        B = []
        for line, logic in zip(lines, self.getLogic()):
            for var, log in zip(line[1:], logic):
                _err = 0.
                if log[0] == 'f':
                    b = var
                    row = []
                    _map = {}
                    
                    for ind, scale in log[1]:
                        if ind == len(params):
                            params.append(None)
                        _map[ind] = scale

                    for ind in float_inds:
                        if ind < self.num_def_params:
                            if ind in _map:
                                b -= def_params[ind] * _map[ind]
                        elif ind in _map:
                            row.append(_map[ind])
                        else:
                            row.append(0)

                    A.append(np.array(row))
                    B.append(b)

                elif log[0] == 'b':
                    if log[1] == 'const':
                        _err = int(log[2] != var)
                    elif log[1] == 'var':
                        if log[2] == len(params):
                            params.append(var)
                        else:
                            _err = int(params[log[2]] != var)

                    if _err > 0:
                        param_fit = False

                elif log[0] == 'c':
                    if log[1] == 'const':
                        _err = int(log[2] != var)
                    elif log[1] == 'var':
                        if log[2] == len(params):
                            params.append(var)
                        else:
                            _err = int(params[log[2]] != var)
                    if _err > 0:
                        param_fit = False
                        
                elif log[0] == 'i':
                    if log[1] == 'const':
                        _err = int(log[2] != var)
                    elif log[1] == 'ret':
                        _err = int(return_vals[log[2]] != var)
                    elif log[1] == 'var':
                        if log[2] == len(params):
                            params.append(var)
                        else:
                            _err = int(params[log[2]] != var)

                    if _err > 0:
                        param_fit = False
                else:
                    assert False, 'bad param'

        if len(A) > 0:
            A = np.stack(A)
            B = np.array(B)
            sol = np.linalg.lstsq(A, B, rcond=None)[0]
        
            _error = np.abs((A @ sol) - B)

            max_error = _error.max()

            if max_error > thresh:
                param_fit = False
            
            error = _error.sum()

            fi = 0
        
            for i in range(len(params)):
                if params[i] == None:
                    params[i] = round(sol[fi].item(), 2)
                    fi += 1

            assert fi == len(sol)
            
        else:
            error = 0    
                
        return param_fit, (params[self.num_def_params:], error)
    
    def printInfo(self):        
        self.dsl.log_print(f"Sub-Functions: {self.raw_structure}")
        self.dsl.log_print(f"Parameters: {self.getInterface()}")
        self.dsl.log_print(f"Logic: ")
        ret_num = 0
        for fn, line in zip(self.structure, self.getLogic()):
            mf = make_function(fn, self.getRead(line))
            ret = ""
            if fn in self.dsl.ret_fns:
                ret = f"{self.dsl.ret_name}{ret_num} = "
                ret_num += 1

            self.dsl.log_print(ret + mf)
        self.dsl.log_print(f"Desc: ('{self.name}', {self.descr}),")
        self.dsl.log_print("")


class DSL():
    def __init__(self, config):
        # NOTE: assumes all base functions are one line
        self.base_functions = [c[0] for c in config['base_functions']]
        self.ret_name = config['ret_name']                
        self.weights = config['weights']
        self.out_file = config['out_file']
        self.abs_beam_size = config['abs_beam_size']
        self.cand_max_changes = config['cand_max_changes']
        self.cand_shape_num = config['cand_shape_num']
        self.cand_cluster_size = config['cand_cluster_size']
        self.cand_cluster_num = config['cand_cluster_num']
        self.cand_abs_num = config['cand_abs_num']
        self.cand_max_error = config['cand_max_error']
        self.cand_max_avg_error = config['cand_max_avg_error']
        self.abs_shape_num = config['abs_shape_num']
        self.abs_order_num = config['abs_order_num']
        self.abs_max_error = config['abs_max_error']
        self.abs_add_threshold = config['abs_add_threshold']
        self.num_rounds = config['num_rounds']
        self.order_thresh = config['order_thresh']
        self.ivar_const = config['ivar_const']
        self.ret_fns = config['ret_fns']
        self.checkValidMacro = config['valid_macro_fn']
        self.inp_params = config['inp_params']

        self.func_change_thresh = config['func_change_thresh']
        self.preProcAttrs = config['preProcAttrs']
        
        self.use_parallel = config['use_parallel']
        self.num_cores = config['num_cores']
        
        self.err_shape_perc = config['err_shape_perc']
        self.err_const_perc = config['err_const_perc']
        self.c_vals = config['c_vals']
        self.prec = config['prec']
        self.scale_prec = config['scale_prec']
        
        self.cluster_eps = config['cluster_eps']

        self.getStaticInfo = config['getStaticInfo']
        self.getInitVarList = config['getInitVarList']
        self.FloatVarPrefOrder = config['FloatVarPrefOrder']
        
        self.library = {            
            fn[0]: Function((fn,), self) for fn in config['base_functions']
        }        
        
        for k in self.library.keys():
            self.library[k].name = k
            self.library[k].name = k        
            
        self.full_library = copy(self.library)

        
    def log_print(self, s):
        with open(self.out_file, 'a') as f:
            f.write(f"{s}\n")
        print(s)

        
    def printInfo(self):
        removed = set()
        
        for n,f in self.library.items():
            self.log_print(f"Library function {n}:")
            f.printInfo()
            for sf in f.raw_structure:
                if sf not in self.library:
                    removed.add(sf)

        lremoved = deepcopy(removed)
                    
        queue = copy(removed)
        while(len(queue) > 0):
            rf = queue.pop()
            f = self.full_library[rf]
            self.log_print(f"Removed function {rf}:")
            f.printInfo()
            for sf in f.raw_structure:
                if sf not in self.library and sf not in removed:
                    removed.add(sf)
                    lremoved.add(sf)
                    queue.add(sf)

        self.log_print("Library Description for Modeling:\n")        
        fkeys = list(set(self.library.keys()).union(lremoved) - set(self.base_functions))
        fkeys = [(int(f.split('_')[1]), f) for f in fkeys]
        fkeys.sort()
        fkeys = [f for _,f in fkeys]
        self.log_print("ADD_FUNCS = [")
        for fk in fkeys:
            self.log_print(f"    ('{fk}', {self.full_library[fk].descr}),")
        self.log_print("]")

        self.log_print("RM_FUNCS = [")
        for rk in lremoved:
            self.log_print(f"    '{rk}',")
        self.log_print("]")
                    
                    
    def scoreProg(self, data):
        return self.weights['func'] * (len(self.library) - len(self.base_functions)) \
            + self.weights['lines'] * (data['lines']-1) \
            + self.weights['i_var'] * data['i_var'] \
            + self.weights['f_var'] * data['f_var'] \
            + self.weights['b_var'] * data['b_var'] \
            + self.weights['c_var'] * data['c_var'] \
            + self.weights['f_error'] * data['f_error']
            
    def calcFuncFreq(self, best_progs):
        total = 0.
        freqs = {k: 0 for k in self.library.keys()}
        for bp in best_progs:
            for _, line in bp:
                freqs[line[0]] += 1.
                total += 1.

        freqs = {f: freqs[f] / (total+1e-8) for f in freqs}
        return freqs
        
    def getObjFunc(self, nodes):
        if self.use_parallel:
            with Pool(self.num_cores) as p:
                prog_scores = p.map(self.getApproxBestProg, nodes)
        else:
            prog_scores = []
            for n in nodes:
                prog_scores.append(self.getApproxBestProg(n))

        res = {'funcs': len(self.library)}
        for key in ('lines', 'i_var', 'f_var', 'c_var', 'b_var', 'f_error'):
            res[key] = round(np.array([p[1][key] for p in prog_scores if p is not None]).mean(), 3)
        func_freq = self.calcFuncFreq([p[2] for p in prog_scores if p is not None])

        prog_scores = [p[0] for p in prog_scores if p is not None]
        
        return (
            1. * sum(prog_scores) / len(prog_scores),
            res,
            func_freq
        )
                
    def findBestOrder(self, node):
        
        orders = node.orders
        node.orders = []
        
        res = []
        
        for o in orders[:self.abs_order_num]:
            line_attrs = o.param_list
            ret_attrs = o.ret_list
            score, _, _  = self.getApproxBestOps(line_attrs, ret_attrs)                
            res.append((score, o))

        def key(a): 
            return a[0]

        res.sort(key=key)

        best_score = res[0][0]
        best_orders = [r[1] for r in res if r[0] < (best_score + self.order_thresh)]

        node.orders = best_orders
        
    def updateBestOrders(self, nodes):        
        
        if self.use_parallel:
            with Pool(self.num_cores) as p:
                list(tqdm(p.imap(self.findBestOrder, nodes), total = len(nodes)))
        else:
            best_orders = []
            for node in tqdm(nodes):
                self.findBestOrder(node)        

    def getApproxBestProg(self, node):

        best_score = 1e8
        best_program = None
        best_bd = None

        for o in node.orders[:self.abs_order_num]:
            line_attrs = o.param_list
            ret_attrs = o.ret_list

            score, program, bd  = self.getApproxBestOps(line_attrs, ret_attrs)
                
            if score < best_score:
                best_score = score
                best_program = program
                best_bd = bd
                
        return (best_score, best_bd, best_program)


    # do beam search, keeping top k candidates with score per_line
    def getApproxBestOps(self, line_attrs, ret_attrs):
        # tuples of (cur_score, cur_lines, cur_struct, cur_vars, cur_error, remaining_line_attr)
        best_prog = None
        best_score = 1e8
        best_bd = None

        init_prog, next_lines, next_rets, def_params = self.preProcAttrs(ret_attrs, line_attrs)
        
        candidates = [
            (
                0, # score - 0
                init_prog, # program 1
                {'lines':1, 'struct':0, 'i_var':0, 'f_var':0, 'c_var': 0, 'b_var': 0, 'f_error': 0.}, # data for scoring 2 
                next_lines, # next lines  3
                next_rets # next ret  4
            )
        ]
        
        next_candidates = []
        
        while(len(candidates) > 0):            
            
            cand = candidates.pop(-1)

            if len(cand[3]) == 0:
                score = self.scoreProg(cand[2])
                if score < best_score:
                    best_score = score
                    best_prog = cand[1]
                    best_bd = cand[2]
                    
                continue

            for n, f in list(self.library.items()):
                num_attrs = len(f.getStructure())
                ret_val = [r for r in cand[4][:num_attrs] if r is not None]
                is_fit, res = f.checkFit(
                    cand[3][:num_attrs],
                    def_params,
                    ret_val                    
                )
                
                if is_fit:
                    nlines = cand[1] + [(tuple(ret_val), tuple([n] + res[0]))]
                    
                    split_inter = f.getSplitInterface()
                    
                    _data = cand[2]
                    
                    ndata = {
                        'lines': len(nlines),
                        'struct': _data['struct'] + num_attrs,
                        'f_error': _data['f_error'] + res[1],
                        'i_var': _data['i_var'] + split_inter['i_var'],
                        'f_var': _data['f_var'] + split_inter['f_var'],
                        'b_var': _data['b_var'] + split_inter['b_var'],
                        'c_var': _data['c_var'] + split_inter['c_var']
                    }
                    
                    nattrs = cand[3][num_attrs:]
                    nrets = cand[4][num_attrs:]
                    
                    nscore = 1.0 * self.scoreProg(ndata) / ndata['struct']
                    
                    next_candidates.append((
                        nscore, nlines, ndata, nattrs, nrets
                    ))

            if len(candidates) == 0:
                next_candidates.sort()
                candidates = next_candidates[:self.abs_beam_size]
                next_candidates = []
                
        return best_score, best_prog, best_bd

    def getFunctionChanges(self, best_freq, new_freq):

        changed_funcs = set()
        
        for key in best_freq:
            if key in self.base_functions:
                continue
            bfreq= best_freq[key]
            nfreq= new_freq[key]
            # Frequency decreased by more than half
            if ((nfreq+1e-8) / (bfreq+1e-8)) < self.func_change_thresh:
                changed_funcs.add(key)

        return changed_funcs
    
    def heurScore(self, f, freq):
        cd = f.constrData()
        
        func_comp = (self.weights['lines'] * (len(f.getStructure()) - 1) ) + \
            self.weights['i_var'] * cd['i_var'] + \
            self.weights['f_var'] * cd['f_var'] + \
            self.weights['b_var'] * cd['b_var'] + \
            self.weights['c_var'] * cd['c_var']
            
        return func_comp * freq

    def getOrdLibrary(self):
        olib = [
            (
                self.heurScore(f, 1.),
                n,
                f
            ) for n,f in self.library.items()
        ]
        olib.sort(reverse=True)
        return [(ol[1], ol[2]) for ol in olib]

    def scoreGain(self, gain):
        return self.weights['lines'] * gain['line'] + \
            self.weights['i_var'] * gain['i_var'] + \
            self.weights['f_var'] * gain['f_var'] + \
            self.weights['b_var'] * gain['b_var'] + \
            self.weights['c_var'] * gain['c_var']
    
    def scoreCandAbs(self, func, cov):
        dsl_func = Function(func, self)
        gain = calcGain(dsl_func, self)
        score_gain = self.scoreGain(gain)
        
        freq = len(cov) / (len(dsl_func.getStructure()) * self.cand_cluster_num * 1.)

        return score_gain * freq
    
    def _integration(self, cand_abs, samp_nodes):
        global NPROG_COUNT
        # func_dist -> map {fn: #function uses / #all function uses}
        best_score, best_res, best_func_dist = self.getObjFunc(samp_nodes)
        
        self.log_print(f"Starting score: {best_score} {best_res}")
        
        group_cand_abs = []
        for group in cand_abs:
            group_cand_abs.append([None, group])
        
        try_num = 0
        while len(group_cand_abs) > 0 and try_num < self.cand_abs_num:

            for i in range(len(group_cand_abs)):                
                for j in range(len(group_cand_abs[i][1])):
                    group_cand_abs[i][1][j][0] = self.scoreCandAbs(
                        group_cand_abs[i][1][j][1],
                        group_cand_abs[i][1][j][2]
                    )
                group_cand_abs[i][1].sort(reverse=True)
                group_cand_abs[i][0] = group_cand_abs[i][1][0][0]
                     
            group_cand_abs.sort(reverse=True)
                
            _, cab, cab_cov = group_cand_abs[0][1].pop(0)

            if len(group_cand_abs) > 0 and \
               len(group_cand_abs[0][1]) == 0:
                group_cand_abs.pop(0)
                
            try_num += 1
                        
            if VERBOSE:
                self.log_print(f"Trying {cab}")

            self.library['temp'] = Function(cab, self)
                        
            new_score, new_res, new_func_dist = self.getObjFunc(samp_nodes)
            
            fcab = self.library.pop('temp')
            
            if best_score - new_score > self.abs_add_threshold:
                name = f'nfunc_{NPROG_COUNT}'
                NPROG_COUNT += 1
                fcab.name = name
                self.log_print(f"Adding {name} to library with score {new_score} {new_res}")
                fcab.printInfo()

                # update best version
                best_score = new_score
                best_res = new_res
                new_func_dist[name] = new_func_dist.pop('temp')
                best_func_dist = new_func_dist
                
                self.library[name] = fcab
                self.full_library[name] = fcab

                updateFreq(group_cand_abs, cab_cov)
                
            else:
                added = False
                # See if function distribution changes negaitvely -> return fns to try removing
                fns_changes = self.getFunctionChanges(best_func_dist, new_func_dist)
                if len(fns_changes) > 0:
                    self.log_print(f"Trying to remove {fns_changes}")
                    old_fns = {}
                    for fn in fns_changes:
                        old_fns[fn] = self.library.pop(fn)
                        
                    self.library['temp'] = Function(cab, self)
                    new_score, new_res, new_func_dist = self.getObjFunc(samp_nodes)            
                    fcab = self.library.pop('temp')
                    
                    if best_score - new_score > self.abs_add_threshold:
                        added = True
                        name = f'nfunc_{NPROG_COUNT}'
                        NPROG_COUNT += 1
                        fcab.name = name
                        self.library[name] = fcab
                        self.full_library[name] = fcab
                        self.log_print(f"Adding {name} to library with score {new_score} {new_res}")
                        fcab.printInfo()
                        best_score = new_score
                        best_res = new_res
                        new_func_dist[name] = new_func_dist.pop('temp')
                        best_func_dist = new_func_dist
                        updateFreq(group_cand_abs, cab_cov)

                        self.log_print(f"Trying to add back in {fns_changes}")
                        for fn in old_fns:
                            self.log_print(fn)
                            self.library['temp'] = old_fns[fn]
                            new_score, new_res, new_func_dist = self.getObjFunc(samp_nodes)
                            fcab = self.library.pop('temp')
                            if best_score - new_score > self.abs_add_threshold:
                                self.log_print(f"Adding {fn} back in to library with score {new_score} {new_res}")
                                best_score = new_score
                                best_res = new_res
                                new_func_dist[fn] = new_func_dist.pop('temp')
                                best_func_dist = new_func_dist
                                self.library[fn] = old_fns[fn]
                            elif VERBOSE:
                                self.log_print(f"Removing {fn} with score {new_score} {new_res}")
                        
                    else:
                        self.log_print(f"Adding back in {fns_changes}")
                        for fn in old_fns:
                            self.library[fn] = old_fns[fn]                                                
                            
                # Remove all related functions from future consideration this round
                if not added:
                    if len(group_cand_abs) > 0:
                        group_cand_abs.pop(0)                
                    if VERBOSE:                                    
                        self.log_print(f"Skipping {cab} with score: {new_score} {new_res}")
                                    
        self.log_print(f"After adding score: {best_score}")
        
        fns = list(self.library.keys())
        for fn in fns:
            if fn in self.base_functions:
                continue
            
            func = self.library.pop(fn)
            new_score, new_res, _ = self.getObjFunc(samp_nodes)

            if best_score - new_score > self.abs_add_threshold:
                self.log_print(f"Removing {fn} from library with score {new_score} {new_res}")
                best_score = new_score
                best_res = new_res
            else:
                self.library[fn] = func

        self.log_print(f"Final score: {best_score} {best_res}")
        
        return best_score
        
def distance_sample(dsl, all_data):
    ret_list = all_data[0].ret_list
    
    q = all_data[sample(list(range(len(all_data))), 1)[0]].float_params

    dists = []
    for d in all_data:
        dists.append(np.linalg.norm(q-d.float_params))

    dists = np.array(dists)
    probs = 1 - (dists/(dists.max() + 1e-8))
    probs = probs / probs.sum()
    samp_inds = np.random.choice(len(all_data), dsl.cand_cluster_size, replace=False,p=probs)

    data = []
    for i in samp_inds:
        data.append(all_data[i])
        
    attr_list = []
    for i in range(len(all_data[0].param_list)):
        attr = [d.param_list[i] for d in data]
        attr_list.append(attr)

    assert len(ret_list) == len(attr_list)

    return deepcopy(attr_list), deepcopy(ret_list)
        

def get_order_maps(nodes):
    sigs_to_attrs = {}    
    inds_to_nodes = {}
    
    for node in tqdm(nodes):
        inds_to_nodes[node.ind] = node
        
        for o in node.orders:
            if o.sig in sigs_to_attrs:
                sigs_to_attrs[o.sig].append(o)
            else:
                sigs_to_attrs[o.sig] = [o]

    return sigs_to_attrs, inds_to_nodes

# Proposal phase of ShapeMOD
def proposal(nodes, dsl):
    dsl.log_print("In Proposal phase")
    t = time.time()

    sigs_to_attrs, inds_to_nodes = get_order_maps(nodes)  
     
    cand_abs = {}
    count = 0

    ind_list = list(inds_to_nodes.keys())
    
    while count < (dsl.cand_cluster_num):
        ind = sample(ind_list, 1)[0]
        node = inds_to_nodes[ind]
        sigs = [o.sig for o in node.orders]
        
        if len(sigs) == 0:
            continue
        
        sig = sample(sigs, 1)[0]
        attr_list = sigs_to_attrs[sig]
                
        if len(attr_list) < dsl.cand_cluster_size:
            continue
        
        prog_attr_list, ret_list = distance_sample(dsl, attr_list)
        
        cluster_cand_abs = findCandAbs(
            prog_attr_list,
            ret_list,
            dsl,
            dsl.cand_max_error,
            dsl.cand_max_avg_error,            
            count
        )
        
        for cand in cluster_cand_abs:
            if cand in cand_abs:
                cand_abs[cand] = cand_abs[cand].union(cluster_cand_abs[cand])
            else:
                cand_abs[cand] = cluster_cand_abs[cand]

        if (count+1) % 100 == 0:
            dsl.log_print(f"On count: {count}")

        count += 1

    dsl.log_print(f"Finished finding candidate abstractions in time: {time.time() - t}")

    st = time.time()
    
    ret, abs_count = genCandAbs(cand_abs, count, dsl.cand_max_changes, dsl)                
    dsl.log_print(f"Finished generalizing candidate abstractions, found {abs_count} in time: {time.time() - st}")
    
    dsl.log_print(f"Finished wake in time: {time.time() - t}")
    
    return ret

# Integration phase of ShapeMOD
def integration(nodes, dsl, cand_abs, ri):
    dsl.log_print(f"In Integration phase")

    t = time.time()

    samp_nodes = sample(nodes, min(len(nodes), dsl.abs_shape_num)) 

    score = dsl._integration(cand_abs, samp_nodes)
        
    dsl.log_print(f"Library at end of ROUND {ri} ({score})")

    dsl.printInfo()
               
    dsl.log_print(f"Finished sleep in time {time.time() - t}")

    dsl.log_print("Finding Updated Best Orders")

    dsl.updateBestOrders(nodes)

class OrderedProg:
    def __init__(self, sig, param_list, ret_list, float_params):
        self.sig = sig
        self.param_list = param_list
        self.ret_list = ret_list
        self.float_params = np.array(float_params)

class ProgNode:
    def __init__(self, ind, orders):
        self.ind = ind
        self.orders = orders        

def loadInput(input_data):
    return pickle.load(open(input_data, 'rb'))    

def writeDSL(dsl, outdir):

    os.system(f'rm {outdir}/shapemod_dsl.py')
    
    with open(f'{outdir}/shapemod_dsl.py', 'w') as F:
        removed = set()
        
        for n,f in dsl.library.items():
            for sf in f.raw_structure:
                if sf not in dsl.library:
                    removed.add(sf)

        lremoved = deepcopy(removed)
        
        fkeys = list(set(dsl.library.keys()).union(lremoved) - set(dsl.base_functions))
        fkeys = [(int(f.split('_')[1]), f) for f in fkeys]
        fkeys.sort()
        fkeys = [f for _,f in fkeys]
        F.write("ADD_FUNCS = [\n")
        for fk in fkeys:
            F.write(f"    ('{fk}', {dsl.full_library[fk].descr}),\n")
        F.write("]\n")

        F.write("RM_FUNCS = [\n")
        for rk in lremoved:
            F.write(f"    '{rk}',\n")
        F.write("]\n")

def make_line(ret, val):
    if len(ret) == 0:
        return make_function(val[0], val[1:])
    else:
        return assign(','.join(list(ret)), make_function(val[0], val[1:]))
        
def writeProgram(dsl, node, outpath):

    res = []

    for o in node.orders[:dsl.abs_order_num]:
        line_attrs, ret_attrs = o.param_list, o.ret_list
        score, program, _  = dsl.getApproxBestOps(line_attrs, ret_attrs)
        res.append((score, program, (ret_attrs, line_attrs)))

    res.sort()    
    best_prog = res[0][1]

    shapemod_prog = []
    orig_prog = []
    
    for line in best_prog:
        ret, val = line[0], line[1]
        line = make_line(ret, val)
        shapemod_prog.append(line)

    for ret, val in zip(res[0][2][0], res[0][2][1]):
        line = make_line(tuple([ret]) if ret is not None else [], val)
        orig_prog.append(line)
        
    with open(f'{outpath}_orig.txt', 'w') as f:
        for line in orig_prog:
            f.write(f'{line}\n')

    with open(f'{outpath}_shapemod.txt', 'w') as f:
        for line in shapemod_prog:
            f.write(f'{line}\n')
            
def writePrograms(dsl, nodes, outdir):
    os.system(f'rm {outdir}/progs/ -r')
    os.system(f'mkdir {outdir}/progs')

    for node in tqdm(nodes):
        writeProgram(dsl, node, f'{outdir}/progs/{node.ind}')
    
# Main entrypoint for ShapeMOD Algorithm        
def shapeMOD(config):
    
    raw_nodes = loadInput(config['input_data'])
    nodes = [n for n in raw_nodes if len(n.orders) > 0][:config['cand_shape_num']]

    dsl = DSL(config)
    
    start_time = time.time()
                                                   
    for i in range(1, 1+dsl.num_rounds):
        dsl.log_print(f"Starting ROUND {i}")
        
        cand_abs = proposal(nodes, dsl)
        integration(nodes, dsl, cand_abs, i)    
        writeDSL(dsl, config['outdir'])
        
    writePrograms(dsl, nodes, config['outdir'])
        

def loadConfig(config_path):
    config_mod = importlib.import_module(config_path)
    config = config_mod.loadConfig()    
    return config
    
# Takes in a config, a dataset of programs, and writes results of ShapeMOD algorithm to outdir
if __name__ == '__main__':
    config_path = sys.argv[1]
    data_path = sys.argv[2]
    outdir = sys.argv[3]
    
    config = loadConfig(config_path)
    config['input_data'] = data_path
    config['outdir'] = outdir

    os.system(f'mkdir {outdir}')
    
    shapeMOD(config)
    
