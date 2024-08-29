from skrgbd.utils import SimpleNamespace as SNS


sl_default = SNS(
    occ_thres=1e-3,
    ss=None,
)

sl_configs = {
    ('tis_right', 'rgb'): sl_default.copy_with(ss=4),
    ('tis_left', 'rgb'): sl_default.copy_with(ss=4),
    ('kinect_v2', 'ir'): sl_default.copy_with(ss=16),
    ('real_sense', 'ir'): sl_default.copy_with(ss=8),
    ('phone_right', 'ir'): sl_default.copy_with(ss=32),
    ('phone_left', 'ir'): sl_default.copy_with(ss=32),
}

reproj_configs = {
    'kinect_v2': SNS(max_rel_edge_len=3e-2),
    'real_sense': SNS(max_rel_edge_len=3e-2),
    'phone_right': SNS(max_rel_edge_len=6e-2),
    'phone_left': SNS(max_rel_edge_len=6e-2),
}
