from skrgbd.utils import SimpleNamespace as SNS


default = SNS(
    adam=SNS(
        iters_n=500,
        lr=1e-4,
    ),
    aligner=SNS(
        angle_gamma=4,
        max_angle_deg=70,
    ),
    check_img_alignment=SNS(
        occ_thres=1e-3,
    ),
    init_cams=SNS(
        pad=50,
    ),
    lbfgs=SNS(
        iters_n=10,
        lr=1,
    ),
    load_cam_data=SNS(),
    load_cam_data_ref=SNS(
        cam_name='tis_right',
        mode='rgb',
        light='hdr',
        view_ids=range(100),
    ),
    load_sl_data=SNS(
        angle_gamma=4,
        max_angle_deg=70,
        occ_thres=1e-3,
        pad=50,
        samples_n=2**14,
    ),
    seed=481953483,
)

default_ir = SNS(
    occ_thres=1e-3,
    ss=4,
)

configs = {
    ('stl_right', 'partial'): default,
    ('tis_right', 'rgb'): default.copy_with(
        load_cam_data_ref=SNS(cam_name='stl_right', mode='partial', light='maxwhite_00_0000', view_ids=range(27)),
    ),
    ('tis_left', 'rgb'): default,
    ('kinect_v2', 'rgb'): default.copy_with(load_cam_data_ref=SNS(scale_factor=.43)),
    ('real_sense', 'rgb'): default.copy_with(load_cam_data_ref=SNS(scale_factor=.57)),
    ('phone_right', 'rgb'): default.copy_with(load_cam_data=SNS(scale_factor=.44)),
    ('phone_left', 'rgb'): default.copy_with(load_cam_data=SNS(scale_factor=.44)),
    ('real_sense', 'ir'): default_ir,
    ('real_sense', 'ir_right'): default_ir,
    ('kinect_v2', 'ir'): default_ir,
    ('phone_right', 'ir'): default_ir.copy_with(ss=8),
    ('phone_left', 'ir'): default_ir.copy_with(ss=8),
}
