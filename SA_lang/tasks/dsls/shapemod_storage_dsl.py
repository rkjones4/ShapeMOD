ADD_FUNCS = [
    ('nfunc_0', (('squeeze', 'i_bbox', 'i_bbox', 'c_bot', 'f_var_0', 'f_var_1'),)),
    ('nfunc_1', (('Cuboid', 'f_bb_x * 1.0', 'f_var_0', 'f_bb_z * 1.0', 'b_1'), ('attach', 'i_bbox', 'f_1 * 0.5', 'f_1 * 1.0', 'f_1 * 0.5', 'f_1 * 0.5', 'f_1 * 0.0', 'f_1 * 0.5'), ('Cuboid', 'f_var_1', 'f_var_2', 'f_var_3', 'b_1'), ('squeeze', 'i_bbox', 'i_ret_0', 'c_bot', 'f_1 * 0.5', 'f_var_4'))),
    ('nfunc_2', (('Cuboid', 'f_var_0', 'f_var_1', 'f_var_2', 'b_1'), ('attach', 'i_bbox', 'f_1 * 0.5', 'f_1 * 0.0', 'f_1 * 0.5', 'f_1 * 0.5', 'f_1 * 1.0', 'f_var_3'))),
    ('nfunc_3', (('Cuboid', 'f_var_0', 'f_var_1', 'f_bb_z * 1.0', 'b_1'), ('attach', 'i_bbox', 'f_1 * 0.5', 'f_1 * 0.0', 'f_1 * 0.5', 'f_var_2', 'f_1 * 1.0', 'f_1 * 0.5'))),
    ('nfunc_4', (('attach', 'i_bbox', 'f_1 * 0.5', 'f_1 * 1.0', 'f_1 * 0.5', 'f_var_0', 'f_1 * 0.0', 'f_var_1'),)),
    ('nfunc_5', (('Cuboid', 'f_var_0', 'f_bb_y * 1.0', 'f_var_1', 'b_1'),)),
    ('nfunc_7', (('Cuboid', 'f_var_0', 'f_var_1', 'f_var_2', 'b_1'), ('attach', 'i_bbox', 'f_1 * 0.5', 'f_1 * 1.0', 'f_1 * 0.5', 'f_var_3', 'f_bb_x * 1.0  + f_var_0 * -1.0', 'f_var_4'), ('attach', 'i_var_0', 'f_1 * 0.0', 'f_1 * 0.5', 'f_1 * 0.5', 'f_1 * 1.0', 'f_var_5', 'f_var_4 * 1.0'))),
    ('nfunc_9', (('Cuboid', 'f_var_0', 'f_var_1', 'f_var_2', 'b_1'), ('attach', 'i_var_0', 'f_var_3', 'f_var_4', 'f_var_5', 'f_var_6', 'f_var_7', 'f_var_8'))),
    ('nfunc_10', (('nfunc_5', 'f_var_0', 'f_var_1'), ('nfunc_0', 'f_var_2', 'f_var_3'))),
    ('nfunc_11', (('nfunc_9', 'f_bb_x * 1.0', 'f_var_0', 'f_var_1', 'i_bbox', 'f_1 * 0.5', 'f_1 * 1.0', 'f_1 * 0.5', 'f_1 * 0.5', 'f_1 * 0.0', 'f_var_2'),)),
    ('nfunc_12', (('nfunc_2', 'f_var_0', 'f_var_1', 'f_var_2', 'f_var_3'), ('attach', 'i_var_0', 'f_1 * 0.0', 'f_1 * 0.5', 'f_1 * 0.5', 'f_var_4', 'f_1 * 0.0', 'f_1 * 0.5'), ('reflect', 'c_Y'))),
    ('nfunc_14', (('nfunc_10', 'f_var_0', 'f_var_1', 'f_var_2', 'f_var_3'), ('reflect', 'c_X'))),
    ('nfunc_15', (('nfunc_10', 'f_var_0', 'f_var_1', 'f_var_2', 'f_var_3'), ('nfunc_10', 'f_var_4', 'f_var_5', 'f_var_6', 'f_var_7'))),
]
RM_FUNCS = [
    'nfunc_5',
]
