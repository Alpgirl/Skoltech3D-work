from skrgbd.utils import SimpleNamespace
from skrgbd.data.dataset.params import (
    cam_trajectory_spheres, cam_trajectory_tabletop, cam_trajectory_human_sphere, sensor_to_cam_mode)


class ScenePaths(SimpleNamespace):
    def __init__(self, scene_name=None, data_root=None, aux_root=None, raw_scans_root=None, trajectory='spheres'):
        self.scene_name = scene_name
        self.trajectory = trajectory

        trajectory = {'spheres': cam_trajectory_spheres,
                      'table': cam_trajectory_tabletop,
                      'human_sphere': cam_trajectory_human_sphere}[trajectory]

        if trajectory is cam_trajectory_spheres:
            self.stl = StlPaths(scene_name, data_root, aux_root, raw_scans_root)
        self.tis_left = TISPaths('tis_left', data_root, scene_name, raw_scans_root, trajectory)
        self.tis_right = TISPaths('tis_right', data_root, scene_name, raw_scans_root, trajectory)
        self.kinect_v2 = KinectPaths(data_root, scene_name, raw_scans_root, aux_root, trajectory)
        self.phone_left = PhonePaths('phone_left', data_root, scene_name, raw_scans_root, aux_root, trajectory)
        self.phone_right = PhonePaths('phone_right', data_root, scene_name, raw_scans_root, aux_root, trajectory)
        self.real_sense = RealSensePaths(data_root, scene_name, raw_scans_root, aux_root, trajectory)

        self.stl_right = SimpleNamespace()
        self.stl_right.rgb = SimpleNamespace()
        self.stl_right.rgb.pinhole_intrinsics = f'{data_root}/calibration/stl_right/rgb/cameras.txt'

        self.reprojected = SimpleNamespace()
        self.reprojected.depth = dict()

        src_device = 'stl'
        src_variant = 'clean_reconstruction'
        dst_variant = 'undistorted'
        for dst_sensor in ['tis_right', 'kinect_v2_ir']:
            self.reprojected.depth[(src_device, src_variant), (dst_sensor, dst_variant)] = ReprojectedDepthPaths(
                src_device, src_variant, dst_sensor, dst_variant, data_root, scene_name)

        dst_variant = 'undistorted'
        dst_sensor = 'tis_right'
        src_variant = 'undistorted'
        for src_device in ['kinect_v2']:
            self.reprojected.depth[(src_device, src_variant), (dst_sensor, dst_variant)] = ReprojectedDepthPaths(
                src_device, src_variant, dst_sensor, dst_variant, data_root, scene_name)


class Pathfinder(ScenePaths):
    def __init__(self, data_root=None, aux_root=None, raw_scans_root=None, trajectory='spheres'):
        ScenePaths.__init__(self, 'scene_name', data_root, aux_root, raw_scans_root, trajectory)
        self.data_root = data_root
        self.aux_root = aux_root
        self.raw_scans_root = raw_scans_root

    def set_dirs(self, data_root=None, aux_root=None, raw_scans_root=None):
        data_root = data_root if data_root else self.data_root
        aux_root = aux_root if aux_root else self.aux_root
        raw_scans_root = raw_scans_root if raw_scans_root else self.raw_scans_root
        self.__init__(data_root, aux_root, raw_scans_root, self.trajectory)

    def __getitem__(self, attr_name):
        if attr_name in {'tis_left', 'tis_right', 'kinect_v2', 'phone_left', 'phone_right', 'real_sense'}:
            cam_name = attr_name
            return getattr(self, cam_name)
        else:
            scene_name = attr_name
            return ScenePaths(scene_name, self.data_root, self.aux_root, self.raw_scans_root, self.trajectory)


class StlPaths(SimpleNamespace):
    def __init__(self, scene_name, data_root=None, aux_root=None, raw_scans_root=None):
        self.partial = SimpleNamespace()
        self.partial.raw = f'{raw_scans_root}/{scene_name}/stl/{scene_name}_folder'
        self.partial.aligned = IndexedPath(lambda scan_i: f'{data_root}/{scene_name}/stl/partial/aligned/{scan_i:04}.ply')
        self.partial.aligned.refined_board_to_world = f'{aux_root}/{scene_name}/stl/partial/refined_board_to_world.pt'
        self.partial.cleaned = IndexedPath(lambda scan_i: f'{data_root}/{scene_name}/stl/partial/cleaned/{scan_i:04}.ply')

        self.validation = SimpleNamespace()
        self.validation.raw = f'{raw_scans_root}/{scene_name}/stl/{scene_name}_check_folder'
        self.validation.aligned = IndexedPath(lambda scan_i: f'{data_root}/{scene_name}/stl/validation/aligned/{scan_i:04}.ply')
        self.validation.aligned.refined_board_to_world = f'{aux_root}/{scene_name}/stl/validation/refined_board_to_world.pt'

        self.reconstruction = SimpleNamespace()
        self.reconstruction.pre_cleaned = f'{data_root}/{scene_name}/stl/reconstruction/pre_cleaned.ply'
        self.reconstruction.cleaned = f'{data_root}/{scene_name}/stl/reconstruction/cleaned.ply'

        self.occluded_space = f'{data_root}/{scene_name}/stl/occluded_space.ply'

        self.alignment_mesh = f'{aux_root}/{scene_name}/stl/alignment_mesh.ply'


class ImageSensorPaths(SimpleNamespace):
    def __init__(self, camera_name, modality, data_root=None, scene_name=None, ext='png', light_dependent=True):
        sensor_calib_root = f'{data_root}/calibration/{camera_name}/{modality}'
        self.calibrated_intrinsics = f'{sensor_calib_root}/intrinsics.yaml'
        self.calibrated_extrinsics = f'{sensor_calib_root}/images.txt'
        self.pinhole_intrinsics = f'{sensor_calib_root}/cameras.txt'
        self.pinhole_pxs_in_raw = f'{sensor_calib_root}/pinhole_pxs_in_raw.pt'

        sensor_data_root = f'{data_root}/{scene_name}/{camera_name}/{modality}'
        if light_dependent:
            self.undistorted = IndexedPath(
                lambda light_setup_pos_i:
                f'{sensor_data_root}/undistorted/{light_setup_pos_i[0]}/{light_setup_pos_i[1]:04}.{ext}')
        else:
            self.undistorted = IndexedPath(
                lambda pos_i:
                f'{sensor_data_root}/undistorted/{pos_i:04}.{ext}')
        self.refined_extrinsics = f'{sensor_data_root}/images.txt'

        self.mvsnet_input = SimpleNamespace()
        self.mvsnet_input.pair_txt = f'{sensor_data_root}/mvsnet_input/pair.txt'
        self.mvsnet_input.cam_txt = IndexedPath(
            lambda pos_i:
            f'{sensor_data_root}/mvsnet_input/cams/{pos_i:04}.txt')


class DepthSensorPaths(SimpleNamespace):
    def __init__(self, camera_name, data_root=None, aux_root=None, scene_name=None, light_dependent=False):
        self.calibrated_intrinsics = f'{data_root}/calibration/{camera_name}/ir/intrinsics.yaml'
        self.calibrated_extrinsics = f'{data_root}/calibration/{camera_name}/ir/images.txt'
        self.pinhole_intrinsics = f'{data_root}/calibration/{camera_name}/ir/cameras.txt'
        self.undist_data = f'{aux_root}/{scene_name}/{camera_name}/depth/undist_data.pt'
        self.undistortion_model = f'{data_root}/calibration/{camera_name}/depth/undistortion.pt'
        if not light_dependent:
            self.undistorted = IndexedPath(
                lambda pos_i:
                f'{data_root}/{scene_name}/{camera_name}/depth/undistorted/{pos_i:04}.png')
        else:
            self.undistorted = IndexedPath(
                lambda light_setup_pos_i:
                f'{data_root}/{scene_name}/{camera_name}/depth/undistorted/{light_setup_pos_i[0]}/{light_setup_pos_i[1]:04}.png')
        self.refined_extrinsics = f'{data_root}/{scene_name}/{camera_name}/ir/images.txt'


class TISPaths(SimpleNamespace):
    def __init__(self, camera_name, data_root=None, scene_name=None, raw_scans_root=None, trajectory=None):
        self.rgb = ImageSensorPaths(camera_name, 'rgb', data_root, scene_name)
        self.rgb.raw = IndexedPath(
            lambda light_setup_pos_i:
            f'{raw_scans_root}/{scene_name}/{camera_name}/{trajectory[light_setup_pos_i[1]]}@{light_setup_pos_i[0]}.png')

        # Kept for backwards compatibility
        if camera_name == 'tis_right':
            self.mvsnet_input = SimpleNamespace()
            self.mvsnet_input.pair_txt = f'{data_root}/{scene_name}/{camera_name}/mvsnet_input/pair.txt'
            self.mvsnet_input.cam_txt = IndexedPath(
                lambda pos_i:
                f'{data_root}/{scene_name}/{camera_name}/mvsnet_input/cams/{pos_i:04}.txt')


class KinectPaths(SimpleNamespace):
    def __init__(self, data_root=None, scene_name=None, raw_scans_root=None, aux_root=None, trajectory=None):
        self.rgb = ImageSensorPaths('kinect_v2', 'rgb', data_root, scene_name)
        self.ir = ImageSensorPaths('kinect_v2', 'ir', data_root, scene_name, light_dependent=False)
        self.depth = DepthSensorPaths('kinect_v2', data_root, aux_root, scene_name)

        self.rgb.raw = IndexedPath(
            lambda light_setup_pos_i:
            f'{raw_scans_root}/{scene_name}/kinect_v2/{trajectory[light_setup_pos_i[1]]}@{light_setup_pos_i[0]}.png')
        self.ir.raw = IndexedPath(
            lambda pos_i:
            f'{raw_scans_root}/{scene_name}/kinect_v2/{trajectory[pos_i]}_ir.png')
        self.depth.raw = IndexedPath(
            lambda pos_i:
            f'{raw_scans_root}/{scene_name}/kinect_v2/{trajectory[pos_i]}_depth.png')


class PhonePaths(SimpleNamespace):
    def __init__(self, camera_name, data_root=None, scene_name=None, raw_scans_root=None, aux_root=None, trajectory=None):
        self.rgb = ImageSensorPaths(camera_name, 'rgb', data_root, scene_name, ext='jpg')
        self.ir = ImageSensorPaths(camera_name, 'ir', data_root, scene_name, light_dependent=False)
        self.depth = DepthSensorPaths(camera_name, data_root, aux_root, scene_name)

        self.rgb.raw = IndexedPath(
            lambda light_setup_pos_i:
            f'{raw_scans_root}/{scene_name}/{camera_name}/{trajectory[light_setup_pos_i[1]]}@{light_setup_pos_i[0]}.jpg')
        self.ir.raw = IndexedPath(
            lambda pos_i:
            f'{raw_scans_root}/{scene_name}/{camera_name}/{trajectory[pos_i]}_ir.png')
        self.depth.raw = IndexedPath(
            lambda pos_i:
            f'{raw_scans_root}/{scene_name}/{camera_name}/{trajectory[pos_i]}_depth.png')


class RealSensePaths(SimpleNamespace):
    def __init__(self, data_root=None, scene_name=None, raw_scans_root=None, aux_root=None, trajectory=None):
        self.rgb = ImageSensorPaths('real_sense', 'rgb', data_root, scene_name)
        self.ir = ImageSensorPaths('real_sense', 'ir', data_root, scene_name)
        self.ir_right = ImageSensorPaths('real_sense', 'ir_right', data_root, scene_name)
        self.depth = DepthSensorPaths('real_sense', data_root, aux_root, scene_name, light_dependent=True)

        self.rgb.raw = IndexedPath(
            lambda light_setup_pos_i:
            f'{raw_scans_root}/{scene_name}/real_sense/{trajectory[light_setup_pos_i[1]]}@{light_setup_pos_i[0]}.png')
        self.ir.raw = IndexedPath(
            lambda light_setup_pos_i:
            f'{raw_scans_root}/{scene_name}/real_sense/{trajectory[light_setup_pos_i[1]]}@{light_setup_pos_i[0]}_ir.png')
        self.ir_right.raw = IndexedPath(
            lambda light_setup_pos_i:
            f'{raw_scans_root}/{scene_name}/real_sense/{trajectory[light_setup_pos_i[1]]}@{light_setup_pos_i[0]}_irr.png')
        self.depth.raw = IndexedPath(
            lambda light_setup_pos_i:
            f'{raw_scans_root}/{scene_name}/real_sense/{trajectory[light_setup_pos_i[1]]}@{light_setup_pos_i[0]}_depth.png')


class ReprojectedDepthPaths(SimpleNamespace):
    def __init__(self, src_device, src_variant, dst_sensor, dst_variant, data_root=None, scene_name=None, light_dependent=False):
        self.data_root = data_root
        self.scene_name = scene_name
        self.light_dependent = light_dependent
        self.device_from = f'{src_device}.{src_variant}'
        self.sensor_to = f'{dst_sensor}.{dst_variant}'

    def __getitem__(self, item):
        if not self.light_dependent:
            pos_i = item
            return f'{self.data_root}/{self.scene_name}/reprojected/depth/{self.device_from}@{self.sensor_to}/{pos_i:04}.png'
        else:
            light_setup, pos_i = item
            return f'{self.data_root}/{self.scene_name}/reprojected/depth/{self.device_from}@{self.sensor_to}/{light_setup}/{pos_i:04}.png'


class IndexedPath:
    def __init__(self, i_to_path):
        self.i_to_path = i_to_path

    def __getitem__(self, i):
        return self.i_to_path(i)


pathfinder = Pathfinder()
