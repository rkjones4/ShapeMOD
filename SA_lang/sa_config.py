import numpy as np
import multiprocessing

"""
Config file specifying how ShapeMOD should interact with the ShapeAssembly language
This file can be extended to work for other domains
"""

"""
Given a current version of a macro (cab), determines whether or not that macro should be added as a candidate.
Also takes in the first line of the program (start_struct) and the upcoming program lines (next_struct)
Not strictly needed (e.g. can always return "add")
A return value of:
  'add' -> means macro will be added to candidate
  'stop' -> means that creating a macro from current point will halt
  'skip' -> means that the macro won't be added to candidates, but another line will be added to the macro
Check ShapeMOD appendix for conditions which SA considers to have a valid macro.
"""
def checkValidMacro(cab, start_struct, next_struct):
    ncl = 0
    for c in cab:
        if c[0] == 'Cuboid':
            ncl += 1

    if start_struct[0] != "Cuboid" and ncl > 0:
        return "stop"
    if len(cab) > 4:
        return "stop"            
    if next_struct[-1] != 'Cuboid':
        return "add"
    else:
        return "skip"

"""
Given a list of program lines (attr_list) and return values (return_list), determine if any information should be passed in to *all* macros for this program
In SA this the bounding box line, the first line of each program --> as we assume this is given we also remove it from the input program (so no macros tries to recreate it)
"""
def getStaticInfo(attr_list, return_list):
    bb_attr = attr_list.pop(0)
    bb_ret = return_list.pop(0)
    
    return [bb_attr], [bb_ret]

"""
Given a list of program lines (line_attrs) and return values (return_attrs), gets information for finding best program
1. init program is what lines are 'given', for SA this is the bounding box line
2. next lines are the lines to find a program for
3. next rets are the return values to find a program for
4. def params are the given params that future macros have access to (by default)
"""
def preProcAttrs(ret_attrs, line_attrs):
    init_prog = [((ret_attrs[0],), line_attrs[0])]
    next_lines = line_attrs[1:]
    next_rets = ret_attrs[1:]
    def_params = [1., line_attrs[0][1], line_attrs[0][2], line_attrs[0][3]]
    return init_prog, next_lines, next_rets, def_params

"""
Given a list of program lines, and return values, return the values of the 'given' parameters for each program instance.
For shapeAssembly this is the float constant (1.0) and the bounding box values
"""
def getInitVarList(line_attrs, return_attrs):
    bbox_line = line_attrs.pop(0)
    return_attrs.pop(0)

    var_list = [
        ("f_1", np.array([1.0 for b in bbox_line])),
        ("f_bb_x", np.array([b[1] for b in bbox_line])),
        ("f_bb_y", np.array([b[2] for b in bbox_line])),
        ("f_bb_z", np.array([b[3] for b in bbox_line])),        
    ]
    return var_list

"""
Preference ordering over what expressions to propose for float variables, check fitFloatVar in ShapeMOD.py to see how these params are used
"""
FloatVarPrefOrder = [
    {'name': 'id', 'pstart': 0, 'pend': 1, 'params': [0.0, 0.5, 1.0]}, # Check constants 
    {'name': 'id', 'pstart': 1, 'pend': 4, 'params': [1.0]}, # Check using a bounding box parameter
    {'name': 'id', 'pstart': 4, 'pend': None, 'params': [1.0]}, # Check using a previously defined variable
    {'name': 'comb', 'p1start': 0, 'p1end': 1, 'p2start': 1, 'p2end': 4, 'params': [(1, 1), (1, -1), (-1, 1)]}, # Check a simple linear combination of 1 and bounding box param
    {'name': 'comb', 'p1start': 0, 'p1end': 1, 'p2start': 4, 'p2end': None, 'params': [(1, 1), (1, -1), (-1, 1)]}, # Check a simple linear combination of 1 and a preiously defined variable
    {'name': 'comb', 'p1start': 1, 'p1end': 4, 'p2start': 4, 'p2end': None, 'params': [(1, 1), (1, -1), (-1, 1)]}, # # Check a simple linear combination of a bounding box param and bounding box param
    {'name': 'comb', 'p1start': 4, 'p1end': None, 'p2start': 5, 'p2end': None, 'params': [(1, 1), (1, -1), (-1, 1)]}, # Check a simple linear combination of two previously defined variables
    {'name': 'lin', 'pstart': 4, 'pend': None}, #Check a multiplier on any previously defined variable
]

# Base functions of the ShapeAssembly language
base_functions = [
    ("Cuboid", "f_var_0", "f_var_1", "f_var_2", "b_var_0"),
    ("attach", "i_var_0", "f_var_0", "f_var_1", "f_var_2", "f_var_3", "f_var_4", "f_var_5"),
    ("squeeze", "i_var_0", "i_var_1", "c_var_0", "f_var_0", "f_var_1"),
    ("reflect", "c_var_0"),
    ("translate", "c_var_0", "f_var_0", "f_var_1"),
]

# Weights in the ShapeMOD algorithm, determine objective function value
weights = {
    'func': 1., # functions in DSL
    'lines': 8., # lines in best program
    'i_var': 8., # i vars in best program
    'f_var': 1., # f vars in best program
    'b_var': .25, # b vars in best program
    'c_var': .5,  # c vars in best program
    'f_error': 10. # float error in best program
}

def loadConfig():
    FAST_MODE = False # Can be set to True to run a 'smoke' mode

    out_file = 'out.txt' # where results will be written to
                                    
    config = {
        # Language Specific Parameters
        'base_functions': base_functions,       
        'valid_macro_fn': checkValidMacro,
        'getStaticInfo': getStaticInfo,
        'getInitVarList': getInitVarList,
        'FloatVarPrefOrder': FloatVarPrefOrder,
        'preProcAttrs': preProcAttrs,
        'inp_params': ('f_1', 'f_bb_x', 'f_bb_y', 'f_bb_z'), #Names of parameters that every macro should have access to (default would just be f_1 at minimum)
        'c_vals': ['X', 'Y', 'Z', 'left', 'right', 'bot', 'top', 'back', 'front'], # All valid values for the c vars
        'ret_name': 'cube', # The name of returned vals, assumes that only one 'type' of thing is returned by the language
        'ivar_const': ['bbox'], # If any i variables can be treated as constants by the macros
        'ret_fns': ['Cuboid'], # What functions return values, assumes that one return object is returned by this function

        # ShapeMOD Hyperparameters
        'out_file': out_file,
        'weights': weights,
        'abs_beam_size': 10, # beam size while finding best program
        'cand_max_changes': 2, # number of generalizing steps
        'cand_shape_num': 100000, # number of shape instances to run proposal phase over
        'cand_cluster_size': 20,  # size of cluster in proposal phase
        'cand_cluster_num': 10000, # number of rounds of the proposal phase
        'cand_abs_num': 20,  # number of candidate abstractions to consider each integration phaes
        'cand_max_error': 0.05, # max float error (for any program in cluster) allowable in the proposal round
        'cand_max_avg_error': 0.02, # maximum float error (averaged over cluster) allowable in the proposal round
        'abs_shape_num': 1000, # number of shapes to run integration phase over
        'abs_order_num': 1000, # number of program reorderigns to check in the integration phase
        'abs_max_error': 0.06, # maximum float error allowable in best programs found during integration
        'abs_add_threshold': 0.05, # only add a candidate macro to the library if its improves objective score by atleast this much 
        'num_rounds': 5, # number of rounds to run ShapeMOD for
        'order_thresh': 1., # any orderings that are within this threshold will be considered part of the best orderings for the next proposal round
        'func_change_thresh': 0.5, # if prev macro changes usage by atleast this amount, we will try removing it from the library, to see if the newly proposed macro might be better
        'err_shape_perc': .7, # for new variables / expressions must explain atleast this percentage of the cluster
        'err_const_perc': .9, # for constants must explain atleast this percentage of the cluster
        'prec': 2, # what final program values get rounded to (places)
        'scale_prec': 20, # number of slots from 0 to 1 that proposed terms in float expressions can take (e.g. if 20 then must be a multiple of 0.05)
        'cluster_eps': .05, # rounding precision used to group related proposed candidate macros

        # whether to find best programs in parallel
        'use_parallel': False, 
        'num_cores': multiprocessing.cpu_count(),                        
    }

    if FAST_MODE:
        config['cand_shape_num'] = 50
        config['cand_cluster_size'] = 2
        config['cand_cluster_num'] = 10
        config['cand_abs_num'] = 10
        config['abs_order_num'] = 10
        config['abs_shape_num'] = 10
        config['num_rounds'] = 3
    
    return config

