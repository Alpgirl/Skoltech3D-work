class ScenePaths:
    def __init__(self, scene_name, *, data_dir=None, raw_dir=None, aux_dir=None, raw_calib_dir=None, addons_dir=None):
        self.scene_name = scene_name
        self.data_dir = data_dir
        self.raw_dir = raw_dir
        self.aux_dir = aux_dir
        self.raw_calib = None if (raw_calib_dir is None) else RawCalib(raw_calib_dir)
        self.addons_dir = addons_dir

    # Images
    # ------
    def img(self, cam, mode, pos_i, light=None, var='undist'):
        if var == 'raw':                                   base_dir = self.raw_dir
        elif light == 'hdr':                               base_dir = self.addons_dir
        elif cam.startswith('stl') and (var == 'undist'):  base_dir = self.addons_dir
        else:                                              base_dir = self.data_dir

        base_dir = f'{base_dir}/{self.scene_name}/{cam}/{mode}'

        if var == 'raw':       base_dir = f'{base_dir}/raw'
        elif var == 'undist':  base_dir = f'{base_dir}/undistorted'

        if light is not None:  base_dir = f'{base_dir}/{light}'

        if cam.startswith('phone') and (mode == 'rgb'):  ext = 'jpg'
        elif cam.startswith('stl') and (var == 'raw'):   ext = 'bmp'
        else:                                            ext = 'png'

        return f'{base_dir}/{pos_i:04}.{ext}'

    # Structured light
    # ----------------
    def sl_raw(self, var='ref'):
        if var == 'ref':    return f'{self.raw_dir}/{self.scene_name}/stl/{self.scene_name}_folder'
        elif var == 'val':  return f'{self.raw_dir}/{self.scene_name}/stl/{self.scene_name}_check_folder'

    def sl_part(self, scan_i, var='ref'):
        if var == 'ref':    return f'{self.data_dir}/{self.scene_name}/stl/partial/aligned/{scan_i:04}.ply'
        elif var == 'val':  return f'{self.data_dir}/{self.scene_name}/stl/validation/aligned/{scan_i:04}.ply'

    def sl_part_stats(self, scan_i, var='ref'):
        if var == 'ref':    return f'{self.aux_dir}/{self.scene_name}/stl/partial/stats/{scan_i:04}.ply'
        elif var == 'val':  return f'{self.aux_dir}/{self.scene_name}/stl/validation/stats/{scan_i:04}.ply'

    def sl_board_to_w_refined(self, var='ref'):
        if var == 'ref':    return f'{self.aux_dir}/{self.scene_name}/stl/partial/board_to_w_refined.pt'
        elif var == 'val':  return f'{self.aux_dir}/{self.scene_name}/stl/validation/board_to_w_refined.pt'

    def sl_full(self, var='cleaned'):
        if var == 'cleaned':        return f'{self.data_dir}/{self.scene_name}/stl/reconstruction/cleaned.ply'
        elif var == 'pre_cleaned':  return f'{self.aux_dir}/{self.scene_name}/stl/reconstruction/pre_cleaned.ply'
        elif var == 'raw':          return f'{self.aux_dir}/{self.scene_name}/stl/reconstruction/raw.ply'

    def sl_occ(self):
        return f'{self.data_dir}/{self.scene_name}/stl/occluded_space.ply'

    # Calibration
    # -----------
    def cam_model(self, cam, mode, var='pinhole'):
        if var == 'pinhole':    return f'{self.data_dir}/calibration/{cam}/{mode}/cameras.txt'
        elif var == 'generic':  return f'{self.data_dir}/calibration/{cam}/{mode}/intrinsics.yaml'
        elif var == 'pt':       return f'{self.data_dir}/calibration/{cam}/{mode}/cam_model.pt'

    def undist_model(self, cam, mode, var=None):
        if var == 'wall':  basedir = self.aux_dir
        else:              basedir = self.data_dir
        basedir = f'{basedir}/calibration/{cam}/{mode}'

        if mode in {'rgb', 'ir', 'ir_right'}:
            return f'{basedir}/pinhole_pxs_in_raw.pt'
        elif mode == 'depth':
            if var is None:
                return f'{basedir}/undistortion.pt'
            if var == 'wall':
                return f'{basedir}/undistortion_wall.pt'

    def cam_poses(self, cam, mode, var='ref'):
        if var == 'calib':
            return f'{self.aux_dir}/calibration/{cam}/{mode}/images.txt'
        else:
            if cam.startswith('stl'):  base_dir = self.addons_dir
            else:
                if var == 'aux':       base_dir = self.aux_dir
                else:                  base_dir = self.data_dir
        return f'{base_dir}/{self.scene_name}/{cam}/{mode}/images.txt'

    # Depth calibration
    # -----------------
    def depth_undist_data(self, cam):
        return f'{self.aux_dir}/{self.scene_name}/{cam}/depth/undist_data.pt'

    # Aux
    # ---

    # Addons
    # ------
    def proj_depth(self, src, svar, dst, dvar, pos_i, light=None):
        base_dir = f'{self.addons_dir}/{self.scene_name}/proj_depth/{src}.{svar}@{dst}.{dvar}'
        if light is not None:  base_dir = f'{base_dir}/{light}'
        return f'{base_dir}/{pos_i:04}.png'

    def mvsnet_cam(self, cam, mode, pos_i):
        return f'{self.addons_dir}/{self.scene_name}/{cam}/{mode}/mvsnet_input/{pos_i:08}_cam.txt'

    def mvsnet_pair(self, cam, mode):
        return f'{self.addons_dir}/{self.scene_name}/{cam}/{mode}/mvsnet_input/pair.txt'

    def idr_cams(self, cam, mode):
        return f'{self.addons_dir}/{self.scene_name}/{cam}/{mode}/idr_input/cameras.npz'

    # Redundant data
    # --------------
    def sl_img(self, cam, scan_i, light='maxwhite_00_0000', var='ref'):
        cam_id = {'stl_left': 'a', 'stl_right': 'b'}[cam]
        return f'{self.sl_raw(var)}/scan_res_{scan_i:04}/Debug/Pictures/cam{cam_id}_{light}.bmp'


class RawCalib:
    def __init__(self, raw_calib_dir):
        self.scan0_to_world = f'{raw_calib_dir}/rv_calib_to_stl_right.pt'
        self.sl_right_poses = f'{raw_calib_dir}/stl_right@stl_sphere_to_zero.pt'
