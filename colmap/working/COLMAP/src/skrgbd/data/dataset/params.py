cam_pos_ids = range(100)
stl_view_ids = range(27)
stl_val_view_ids = range(5)

light_setups = [
    'flash@best', 'flash@fast', 'ambient@best', 'ambient_low@fast', 'hard_left_bottom_close@best',
    'hard_left_bottom_far@best', 'hard_left_top_close@best', 'hard_left_top_far@best', 'hard_right_bottom_close@best',
    'hard_right_top_close@best', 'hard_right_top_far@best', 'soft_left@best', 'soft_right@best', 'soft_top@best']

kinect_light_setups = [
    'flash', 'ambient', 'ambient_low', 'hard_left_bottom_close', 'hard_left_bottom_far', 'hard_left_top_close',
    'hard_left_top_far', 'hard_right_bottom_close', 'hard_right_top_close', 'hard_right_top_far', 'soft_left',
    'soft_right', 'soft_top']

stl_light_setups = [
    'maxwhite_00_0000', 'backgrnd_00_0000',
    'codes____00_0000', 'codes____00_0001', 'codes____00_0002', 'codes____00_0003', 'codes____00_0004',
    'codes____00_0005', 'codes____00_0006', 'codes____00_0007', 'codes____00_0008',
    'lines____00_0000', 'lines____00_0001', 'lines____00_0002', 'lines____00_0003', 'lines____00_0004', 'lines____00_0005',
]

sensor_to_cam_mode = {
    'real_sense_rgb': ('real_sense', 'rgb'),
    'real_sense_ir': ('real_sense', 'ir'),
    'real_sense_ir_right': ('real_sense', 'ir_right'),
    'kinect_v2_rgb': ('kinect_v2', 'rgb'),
    'kinect_v2_ir': ('kinect_v2', 'ir'),
    'tis_left': ('tis_left', 'rgb'),
    'tis_right': ('tis_right', 'rgb'),
    'phone_left_rgb': ('phone_left', 'rgb'),
    'phone_left_ir': ('phone_left', 'ir'),
    'phone_right_rgb': ('phone_right', 'rgb'),
    'phone_right_ir': ('phone_right', 'ir'),
    'stl_left': ('stl_left', 'rgb'),
    'stl_right': ('stl_right', 'rgb'),
}

# Substitution for
# from skrgbd.calibration.trajectories.camera_sphere import CameraCalibrationSphere
# cam_trajectory = CameraCalibrationSphere()

cam_trajectory_spheres = {
 0:  'sphere@1.0,0.0',
 1:  'sphere@0.916,0.0',
 2:  'sphere@0.832,0.0',
 3:  'sphere@0.749,0.0',
 4:  'sphere@0.665,0.0',
 5:  'sphere@0.581,0.0',
 6:  'sphere@0.497,0.0',
 7:  'sphere@0.413,0.0',
 8:  'sphere@0.329,0.0',
 9:  'sphere@0.246,0.0',
 10: 'sphere@0.162,0.0',
 11: 'sphere@0.0779,0.0',
 12: 'sphere@0.0,0.115',
 13: 'sphere@0.0847,0.115',
 14: 'sphere@0.169,0.115',
 15: 'sphere@0.254,0.115',
 16: 'sphere@0.339,0.115',
 17: 'sphere@0.423,0.115',
 18: 'sphere@0.508,0.115',
 19: 'sphere@0.593,0.115',
 20: 'sphere@0.678,0.115',
 21: 'sphere@0.762,0.115',
 22: 'sphere@0.847,0.115',
 23: 'sphere@0.932,0.115',
 24: 'sphere@1.0,0.23',
 25: 'sphere@0.914,0.23',
 26: 'sphere@0.828,0.23',
 27: 'sphere@0.742,0.23',
 28: 'sphere@0.655,0.23',
 29: 'sphere@0.569,0.23',
 30: 'sphere@0.483,0.23',
 31: 'sphere@0.397,0.23',
 32: 'sphere@0.311,0.23',
 33: 'sphere@0.225,0.23',
 34: 'sphere@0.139,0.23',
 35: 'sphere@0.0525,0.23',
 36: 'sphere@0.0,0.344',
 37: 'sphere@0.0882,0.344',
 38: 'sphere@0.176,0.344',
 39: 'sphere@0.265,0.344',
 40: 'sphere@0.353,0.344',
 41: 'sphere@0.441,0.344',
 42: 'sphere@0.529,0.344',
 43: 'sphere@0.617,0.344',
 44: 'sphere@0.706,0.344',
 45: 'sphere@0.794,0.344',
 46: 'sphere@0.882,0.344',
 47: 'sphere@0.97,0.344',
 48: 'sphere@1.0,0.459',
 49: 'sphere@0.909,0.459',
 50: 'sphere@0.818,0.459',
 51: 'sphere@0.727,0.459',
 52: 'sphere@0.636,0.459',
 53: 'sphere@0.545,0.459',
 54: 'sphere@0.454,0.459',
 55: 'sphere@0.363,0.459',
 56: 'sphere@0.272,0.459',
 57: 'sphere@0.181,0.459',
 58: 'sphere@0.0902,0.459',
 59: 'sphere@0.0,0.574',
 60: 'sphere@0.0946,0.574',
 61: 'sphere@0.189,0.574',
 62: 'sphere@0.284,0.574',
 63: 'sphere@0.378,0.574',
 64: 'sphere@0.473,0.574',
 65: 'sphere@0.567,0.574',
 66: 'sphere@0.662,0.574',
 67: 'sphere@0.756,0.574',
 68: 'sphere@0.851,0.574',
 69: 'sphere@0.946,0.574',
 70: 'sphere@1.0,0.689',
 71: 'sphere@0.901,0.689',
 72: 'sphere@0.802,0.689',
 73: 'sphere@0.703,0.689',
 74: 'sphere@0.604,0.689',
 75: 'sphere@0.505,0.689',
 76: 'sphere@0.405,0.689',
 77: 'sphere@0.306,0.689',
 78: 'sphere@0.207,0.689',
 79: 'sphere@0.108,0.689',
 80: 'sphere@0.00907,0.689',
 81: 'sphere@0.0,0.804',
 82: 'sphere@0.105,0.804',
 83: 'sphere@0.21,0.804',
 84: 'sphere@0.314,0.804',
 85: 'sphere@0.419,0.804',
 86: 'sphere@0.524,0.804',
 87: 'sphere@0.629,0.804',
 88: 'sphere@0.734,0.804',
 89: 'sphere@0.838,0.804',
 90: 'sphere@0.943,0.804',
 91: 'sphere@1.0,0.919',
 92: 'sphere@0.888,0.919',
 93: 'sphere@0.776,0.919',
 94: 'sphere@0.664,0.919',
 95: 'sphere@0.552,0.919',
 96: 'sphere@0.44,0.919',
 97: 'sphere@0.328,0.919',
 98: 'sphere@0.216,0.919',
 99: 'sphere@0.104,0.919'}

for pos_i, pos_name in list(cam_trajectory_spheres.items()):
    cam_trajectory_spheres[pos_name] = pos_i


cam_trajectory_tabletop = dict()
for pos_i in range(100):
    pos_name = f'table@{pos_i:03}'
    cam_trajectory_tabletop[pos_name] = pos_i
    cam_trajectory_tabletop[pos_i] = pos_name


cam_trajectory_human_sphere = {
    0: 'human_sphere@0.7,0.0',
    1: 'human_sphere@0.393,0.1',
    2: 'human_sphere@0.636,0.1',
    3: 'human_sphere@0.814,0.2',
    4: 'human_sphere@0.571,0.2',
    5: 'human_sphere@0.328,0.2',
    6: 'human_sphere@0.447,0.3',
    7: 'human_sphere@0.689,0.3',
    8: 'human_sphere@0.768,0.4',
    9: 'human_sphere@0.525,0.4',
    10: 'human_sphere@0.282,0.4',
    11: 'human_sphere@0.28,0.5',
    12: 'human_sphere@0.523,0.5',
    13: 'human_sphere@0.766,0.5',
    14: 'human_sphere@0.764,0.6',
    15: 'human_sphere@0.521,0.6',
    16: 'human_sphere@0.279,0.6',
    17: 'human_sphere@0.277,0.7',
    18: 'human_sphere@0.52,0.7',
    19: 'human_sphere@0.762,0.7',
    20: 'human_sphere@0.761,0.8',
    21: 'human_sphere@0.518,0.8',
    22: 'human_sphere@0.275,0.8',
    23: 'human_sphere@0.273,0.9',
    24: 'human_sphere@0.516,0.9',
    25: 'human_sphere@0.759,0.9',
    26: 'human_sphere@1.0,1.0',
    27: 'human_sphere@0.757,1.0',
    28: 'human_sphere@0.514,1.0',
    29: 'human_sphere@0.272,1.0',
}

for pos_i, pos_name in list(cam_trajectory_human_sphere.items()):
    cam_trajectory_human_sphere[pos_name] = pos_i
