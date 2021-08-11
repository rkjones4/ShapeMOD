ADD_FUNCS = [
    ('nfunc_1', (('Cuboid', 'f_bb_x * 1.0', 'f_var_0', 'f_bb_z * 1.0', 'b_1'), ('attach', 'i_bbox', 'f_1 * 0.5', 'f_1 * 1.0', 'f_1 * 0.5', 'f_1 * 0.5', 'f_1 * 0.0', 'f_1 * 0.5'), ('Cuboid', 'f_var_1', 'f_bb_y * 1.0 + f_var_0 * -1.0', 'f_var_2', 'b_var_0'), ('squeeze', 'i_bbox', 'i_ret_0', 'c_bot', 'f_1 * 0.5', 'f_1 * 0.5'))),
    ('nfunc_2', (('Cuboid', 'f_bb_x * 1.0', 'f_var_0', 'f_bb_z * 1.0', 'b_1'), ('attach', 'i_bbox', 'f_1 * 0.5', 'f_1 * 1.0', 'f_1 * 0.5', 'f_1 * 0.5', 'f_1 * 0.0', 'f_1 * 0.5'), ('Cuboid', 'f_var_1', 'f_var_2', 'f_var_3', 'b_1'), ('attach', 'i_bbox', 'f_1 * 0.5', 'f_1 * 0.0', 'f_1 * 0.5', 'f_1 * 0.5', 'f_1 * 1.0', 'f_1 * 0.5'))),
    ('nfunc_5', (('attach', 'i_bbox', 'f_1 * 0.5', 'f_1 * 1.0', 'f_1 * 0.5', 'f_var_0', 'f_1 * 0.0', 'f_var_1'),)),
    ('nfunc_7', (('attach', 'i_bbox', 'f_1 * 0.5', 'f_1 * 0.0', 'f_1 * 0.5', 'f_var_0', 'f_1 * 1.0', 'f_var_1'),)),
    ('nfunc_8', (('Cuboid', 'f_var_0', 'f_var_1', 'f_var_2', 'b_1'), ('squeeze', 'i_bbox', 'i_var_0', 'c_bot', 'f_var_3', 'f_var_4'), ('reflect', 'c_var_0'))),
    ('nfunc_9', (('Cuboid', 'f_var_0', 'f_var_1', 'f_var_2', 'b_1'), ('attach', 'i_var_0', 'f_var_3', 'f_1 * 0.5', 'f_var_4', 'f_var_5', 'f_var_6', 'f_var_7'))),
    ('nfunc_10', (('Cuboid', 'f_var_0', 'f_var_1', 'f_var_2', 'b_var_0'), ('squeeze', 'i_bbox', 'i_var_0', 'c_bot', 'f_var_3', 'f_var_4'))),
    ('nfunc_12', (('nfunc_2', 'f_var_0', 'f_var_1', 'f_var_2', 'f_var_3'), ('attach', 'i_ret_0', 'f_1 * 0.5', 'f_1 * 1.0', 'f_1 * 0.5', 'f_1 * 0.5', 'f_var_4', 'f_1 * 0.5'))),
    ('nfunc_17', (('Cuboid', 'f_var_0', 'f_var_1', 'f_var_2', 'b_1'), ('nfunc_5', 'f_var_3', 'f_var_4'))),
    ('nfunc_18', (('nfunc_8', 'f_var_0', 'f_bb_y * 1.0', 'f_var_1', 'i_bbox', 'f_var_2', 'f_var_3', 'c_var_0'),)),
    ('nfunc_19', (('nfunc_10', 'f_var_0', 'f_bb_y * 1.0', 'f_var_1', 'b_var_0', 'i_bbox', 'f_var_2', 'f_var_3'), ('nfunc_10', 'f_var_4', 'f_bb_y * 1.0', 'f_var_5', 'b_var_1', 'i_bbox', 'f_var_6', 'f_var_7'))),
    ('nfunc_20', (('nfunc_18', 'f_var_0', 'f_var_0 * 1.0', 'f_var_0 * 0.4', 'f_var_1', 'c_X'), ('nfunc_18', 'f_var_0 * 1.0', 'f_var_0 * 1.0', 'f_var_2', 'f_1 * 1.0  + f_var_1 * -1.0', 'c_X'))),
]
RM_FUNCS = [
    'nfunc_2',
    'nfunc_8',
]
