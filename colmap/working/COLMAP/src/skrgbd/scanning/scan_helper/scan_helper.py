from inspect import cleandoc
import datetime
from IPython.display import display, clear_output
import os
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
import signal
from time import sleep
import subprocess
import warnings

import numpy as np

from skrgbd.periphery.periphery import Periphery
from skrgbd.devices.phone import scan_root as phone_scan_root
from skrgbd.devices.phone import LeftPhone, RightPhone, port as phone_port, connect_adb_over_wifi
from skrgbd.devices.rig import Rig
from skrgbd.devices.robot.robot_on_sphere import RobotOnSphere, RobotOnSTLSphere
from skrgbd.devices.robot.robot_on_table import RobotOnTable
from skrgbd.devices.robot.robot_on_human_sphere import RobotOnHumanSphere
from skrgbd.devices.rv.communication import RVClient

from skrgbd.utils.camera_utils import auto_white_balance
from skrgbd.utils.logging import logger, tqdm
from skrgbd.scanning.scan_helper.extensions.task import Noop
from skrgbd.utils.parallel import ThreadSet, PropagatingThread as Thread

from skrgbd.scanning.scan_helper.extensions import log_context, _Lights, _Robot, _Kinect, _Phone, _RealSense, _Tis
from skrgbd.scanning.scan_helper.scan_meta import ScanMeta


warnings.filterwarnings('ignore', message='The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. ')


stl_scanning_velocity = .1          # Robot velocity during scanning with structured light scanner, in m/s.
robot_shaking_timeout_stl = 1.5     # Timeout after moving the robot and before scanning with structured light, in seconds
scanning_velocity = .2              # Robot velocity during scanning with cameras, in m/s.
robot_shaking_timeout = 1           # Timeout after moving the robot and before scanning with cameras, in seconds.
warmup_shoot_interval = 1.5         # Time interval between shots during warmup, in seconds.


class ScanHelper:
    r"""
    Parameters
    ----------
    data_dir : str
        Root directory where scan subdirectories will be saved.
    logfile : str
    """
    name = log_context
    stl_scans_root = '/mnt/data/sk3d/stl_shared_folder/scans'

    def __init__(self, data_dir, logfile, trajectories='spheres', thread='both'):
        self.thread = thread
        logger.prepare_for_scanning(logfile)
        logger.info(f'{self.name}: Init {self.name} for thread {thread}')

        self._check_env()
        self.robot = None
        self.trajectories = trajectories
        if trajectories == 'spheres':
            self.stl_trajectory_class = RobotOnSTLSphere
            self.camera_trajectory_class = RobotOnSphere
        elif trajectories == 'tabletop':
            self.stl_trajectory_class = None
            self.camera_trajectory_class = RobotOnTable
        elif trajectories == 'human_sphere':
            self.stl_trajectory_class = None
            self.camera_trajectory_class = RobotOnHumanSphere
        else:
            raise ValueError(f'Unknown trajectories {trajectories}')

        if thread in {'both', 'cameras'}:
            self.rig = Rig()
        if thread in {'both', 'stl'}:
            if trajectories in {'tabletop', 'human_sphere'}:
                raise ValueError('SL scanning is only implemented for the spheres trajectory')
            self.rv = RVClient()

        if thread in {'both', 'cameras'}:
            self.rig.init_cameras().join()
            self.periphery = Periphery(self.rig.phones)
            self.light_setups = None
            self.set_light_setups()

        self.data_dir = Path(data_dir)
        self._started = False
        self.start()
        self._do_pause = False

    def __del__(self):
        self.stop()

    def _check_env(self):
        if self.thread in {'both', 'cameras'}:
            usbfs_memory_mb = subprocess.check_output('cat /sys/module/usbcore/parameters/usbfs_memory_mb'.split()).decode().split('\n')[0]
            if usbfs_memory_mb != '1024':
                raise RuntimeError(f'USBFS memory is {usbfs_memory_mb}, set it to 1024')

            connect_adb_over_wifi()
            sleep(3)
            adb_devices = subprocess.check_output(f'adb devices | grep {phone_port}', shell=True).decode().split('\n')
            adb_devices = [_.split('\t')[0] for _ in adb_devices]
            if (LeftPhone.serial not in adb_devices) or (RightPhone.serial not in adb_devices):
                raise RuntimeError(f'Wrong adb devices: {adb_devices}.')

    def start(self):
        if self.thread not in {'both', 'cameras'}:
            return
        if not self._started:
            self.rig.start_cameras().join()
            self.setup_cameras()
            self.start_warmup()
            self._started = True

    def stop(self):
        if self.thread not in {'both', 'cameras'}:
            return
        if self._started:
            self.stop_warmup()
            self.rig.stop_cameras().join()
            self._started = False

    def setup_cameras(self):
        if self.thread not in {'both', 'cameras'}:
            raise RuntimeError('Not available in STL thread')
        self.rig.realsense.laser_on()
        for camera in self.rig.cameras:
            camera.setup('soft_left', 'best')

    def start_warmup(self):
        if self.thread not in {'both', 'cameras'}:
            raise RuntimeError('Not available in STL thread')
        logger.debug(f'{self.name}: Start warmup')
        for phone in self.rig.phones:
            phone.set_af(False)
        for camera in self.rig.cameras:
            camera.start_warmup(warmup_shoot_interval)

    def stop_warmup(self):
        if self.thread not in {'both', 'cameras'}:
            raise RuntimeError('Not available in STL thread')
        logger.debug(f'{self.name}: Stop warmup')
        threads = ThreadSet([Thread(target=camera.stop_warmup) for camera in self.rig.cameras])
        threads.start_and_join()

    def set_white_balance(self):
        if self.thread not in {'both', 'cameras'}:
            raise RuntimeError('Not available in STL thread')
        logger.info(f'{self.name}: Set white balance')
        self.periphery.soft_right.on()
        cameras = [self.rig.realsense, self.rig.tis_left, self.rig.tis_right]
        phones = self.rig.phones
        for phone in self.rig.phones:
            phone.set_af(False)
        ThreadSet([Thread(target=camera.stop_warmup) for camera in (cameras + phones)]).start_and_join()
        ThreadSet([Thread(target=phone.light_display) for phone in phones]).start_and_join()
        ThreadSet([Thread(target=phone.switch_auto_whitebalance) for phone in phones]).start_and_join()
        auto_white_balance(cameras)
        ThreadSet([Thread(target=phone.switch_auto_whitebalance) for phone in phones]).start_and_join()
        ThreadSet([Thread(target=phone.dim_display) for phone in phones]).start_and_join()
        for camera in (cameras + phones):
            camera.start_warmup(warmup_shoot_interval)
        self.periphery.soft_right.off()
        logger.info(f'{self.name}: Set white balance DONE')

    def pick_camera_settings(self, scene_name, light_setups=None):
        if self.thread not in {'both', 'cameras'}:
            raise RuntimeError('Not available in STL thread')
        try:
            self.load_camera_settings(scene_name)
        except FileNotFoundError:
            pass

        logger.info(f'{self.name}: Pick camera settings for {scene_name}')
        self.periphery.kinect_emitter.off()
        tis = self.rig.tis_right
        phone = self.rig.phone_right
        realsense = self.rig.realsense
        cameras = [tis, phone]
        robot = self.camera_trajectory_class()

        self.periphery.lights_off()
        robot.move_for_cam_settings(scanning_velocity)
        ThreadSet([Thread(target=camera.stop_warmup) for camera in [*cameras, realsense]]).start_and_join()
        for phone in self.rig.phones:
            phone.set_af(False)
        phone.set_low_resolution()
        ThreadSet([Thread(target=ir_emitter.off)
                   for ir_emitter in [self.periphery.left_phone_ir, self.periphery.right_phone_ir]]).start_and_join()

        if self.trajectories in {'spheres', 'human_sphere'}:
            logger.info(f'{self.name}: Set foreground mask')
            while True:
                if self.trajectories == 'spheres':
                    input(
                        'Выключите прожектор structured light сканера из розетки,'
                        ' включите свет в комнате, подготовьте место для сканирования объекта, но не ставьте сам объект.')
                else:
                    input('Включите свет в комнате, оставьте место для сканирования пустым.')
                clear_output()
                ThreadSet([Thread(target=camera.set_bg_frame) for camera in [*cameras, realsense]]).start_and_join()

                if self.trajectories == 'spheres':
                    input(cleandoc(r"""Расположите сканируемый объект так чтобы
                        * Его наиболее интересная сторона была обращена к камерам,
                        * Его центр находился над меткой на столе."""))
                else:
                    input('Поставьте человека в позу для сканирования.')
                clear_output()
                ThreadSet([Thread(target=camera.set_fg_masks) for camera in [*cameras, realsense]]).start_and_join()

                def check_foreground():
                    images = []
                    for camera in cameras:
                        images.append(camera._fg_masks)
                    images.append(realsense._fg_masks['image'])
                    images.append(realsense._fg_masks['ir'])
                    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
                    for im, ax in zip(images, axes.T.ravel()):
                        ax.imshow(im, cmap='gray', interpolation='none', vmin=0, vmax=1)
                    figure.tight_layout()
                    plt.close()
                    display(figure)

                check_foreground()
                good_masks = input('Оставьте поле пустым и нажмите Enter чтобы продолжить'
                                   ' или введите что-нибудь чтобы перезапустить настройку фона.')
                clear_output()
                if good_masks == '':
                    break
            input('ВЫключите свет в комнате.')
        elif self.trajectories == 'tabletop':
            ThreadSet([Thread(target=camera.set_fg_masks, kwargs=dict(method='full'))
                       for camera in [*cameras, realsense]]).start_and_join()
        clear_output()

        def pick_settings(light, preset):
            logger.info(f'{self.name}: Pick settings for {light} at {preset}')
            self.periphery.on_only(light)
            threads = [Thread(target=camera.pick_settings, args=[light, preset]) for camera in cameras]
            threads.append(Thread(target=realsense.pick_settings, args=[light, preset], kwargs={'ir_frames_n': 8}))
            ThreadSet(threads).start_and_join()

            logger.debug(f'{self.name}: Save camera setting previews for {light} at {preset}')
            previews_dir = self.make_scene_dir(scene_name) / 'settings_previews'
            threads = [Thread(target=camera.setup, args=[light, preset]) for camera in [*cameras, realsense]]
            ThreadSet(threads).start_and_join()

            tis.save_image(f'{previews_dir}/tis.{light}@{preset}', blocking=False)
            realsense.save_image(f'{previews_dir}/realsense.{light}@{preset}', modalities={'image', 'ir', 'ir_right'},
                                 blocking=False, compressed=True)
            phone.save_light_picking_frame(f'{previews_dir}/phone.{light}@{preset}.jpg')
            logger.info(f'{self.name}: Pick settings for {light} at {preset} DONE')

        logger.debug(f'{self.name}: Pick settings')
        if light_setups is None:
            light_setups = []
            if 'flash' in self.light_setups:
                light_setups.extend([('flash', 'best'), ('flash', 'fast')])
            if 'ambient' in self.light_setups:
                light_setups.append(('ambient', 'best'))
            if 'ambient_low' in self.light_setups:
                light_setups.append(('ambient_low', 'fast'))
            for light in ['hard_right_top_close', 'hard_right_bottom_close', 'hard_left_bottom_close',
                          'hard_right_top_far',
                          'hard_left_top_far', 'hard_left_bottom_far', 'soft_left', 'soft_right', 'soft_top']:
                if light in self.light_setups:
                    light_setups.append((light, 'best'))

        for (light, preset) in tqdm(light_setups, eta_format=True):
            pick_settings(light, preset)

            for camera in cameras + [realsense]:
                camera.copy_light_preset(('hard_right_top_close', 'best'), ('hard_left_top_close', 'best'))
            self.rig.tis_left.exposure_presets = self.rig.tis_right.exposure_presets
            self.rig.tis_left.gain_presets = self.rig.tis_right.gain_presets
            self.rig.phone_left.exposure_presets = self.rig.phone_right.exposure_presets
            self.rig.phone_left.iso_presets = self.rig.phone_right.iso_presets

            self._save_camera_settings(scene_name)

        self.periphery.lights_off()
        phone.set_high_resolution()
        ThreadSet([Thread(target=camera.start_warmup, args=[warmup_shoot_interval])
                   for camera in [*cameras, realsense]]).start_and_join()
        robot.rest(scanning_velocity)
        logger.info(f'{self.name}: Pick camera settings for {scene_name} DONE')

    def load_camera_settings(self, scene_name):
        if self.thread not in {'both', 'cameras'}:
            raise RuntimeError('Not available in STL thread')
        scene_dir = self.make_scene_dir(scene_name)
        for camera in [self.rig.tis_left, self.rig.tis_right, self.rig.phone_left, self.rig.phone_right, self.rig.realsense]:
            settings_pkl = scene_dir / f'{camera.name}_light_settings.pkl'
            if not settings_pkl.exists():
                if self.trajectories == 'tabletop':
                    settings_pkl = self.make_scene_dir('tabletop_debug') / f'{camera.name}_light_settings.pkl'
                elif self.trajectories == 'human_sphere':
                    settings_pkl = self.make_scene_dir('people_debug') / f'{camera.name}_light_settings.pkl'
                else:
                    assert False
            camera.load_light_presets(settings_pkl)

    def set_light_setups(self, scene_name=None):
        if self.trajectories == 'spheres':
            self.light_setups = [
                'flash', 'soft_left', 'soft_right', 'soft_top', 'hard_left_bottom_far', 'hard_left_bottom_close',
                'hard_left_top_close', 'hard_right_top_far', 'hard_right_top_close', 'hard_left_top_far',
                'hard_right_bottom_close', 'ambient', 'ambient_low'
            ]
        elif self.trajectories == 'tabletop':
            self.light_setups = [
                'flash', 'soft_left', 'soft_right', 'soft_top', 'hard_left_bottom_far',
                'hard_right_top_far', 'hard_right_top_close', 'hard_left_top_far',
                'hard_right_bottom_close', 'ambient', 'ambient_low'
            ]
        elif self.trajectories == 'human_sphere':
            self.light_setups = [
                'flash',  'soft_left', 'soft_right', 'soft_top', 'hard_right_bottom_close', 'hard_left_bottom_far',
                'hard_left_top_far', 'hard_right_top_far', 'ambient',
            ]
        else:
            assert False

    def _save_camera_settings(self, scene_name):
        scene_dir = self.make_scene_dir(scene_name)
        for camera in [self.rig.tis_left, self.rig.tis_right, self.rig.phone_left, self.rig.phone_right, self.rig.realsense]:
            camera.save_light_presets(scene_dir / f'{camera.name}_light_settings.pkl')

    def scan_stl(self, scene_name):
        if self.thread not in {'both', 'stl'}:
            raise RuntimeError('Not available in Camera thread')
        return self._scan_stl(scene_name, False)

    def scan_stl_check(self, scene_name):
        if self.thread not in {'both', 'stl'}:
            raise RuntimeError('Not available in Camera thread')
        if input('Объект был покрашен и краска удалена? Напишите YES или функция вернется с ошибкой.') != 'YES':
            raise RuntimeError
        return self._scan_stl(scene_name, True)

    def _scan_stl(self, scene_name, check):
        r"""Scans one scene with the structured light scanner.

        Parameters
        ----------
        scene_name : str
            The name of the scene.
        check : bool

        Returns
        -------
        stl_export : Thread
            Thread object representing the saving of the scans to disk.

        Notes
        -------
        The timeline of structured light scanning is illustrated below.

        -----------------> ------------------------------------------> -----------------> ...
        get-scan-images--> calculate-geometry------------------------> get-scan-images--> ...
                           move-to-next-point--> wait-out-shaking-->                      ...
        """
        subprocess.Popen("""/usr/bin/dconf write /org/gnome/mutter/overlay-key "''" """, shell=True)
        try:
            input('Выключите свет в комнате, и удостоверьтесь что лампы установки не горят.')
            input('Включите проектор structured light сканера в розетку и нажмите Enter.')
            clear_output()

            if check:
                project_name = f'{scene_name}_check'
            else:
                project_name = scene_name

            logger.info(f'{self.name}: Scan with structured light scanner {project_name}')
            self.robot = self.stl_trajectory_class()
            if self.trajectories == 'spheres':
                self.robot.move_to((.5, 0), scanning_velocity)
            else:
                assert False

            rv_scene_dir = Path(f'{self.stl_scans_root}/{project_name}_folder')
            rv_scene_proj = Path(f'{self.stl_scans_root}/{project_name}.scanproj')
            stl_dir = self.make_scene_dir(scene_name) / 'stl'
            moved_scene_dir = stl_dir / f'{project_name}_folder'
            moved_scene_proj = stl_dir / f'{project_name}.scanproj'
            if rv_scene_dir.exists() or rv_scene_proj.exists() or moved_scene_dir.exists() or moved_scene_proj.exists():
                resp = input(f'The structured light scan with the name {project_name} already exists.'
                             ' Do you want to delete it? Type YES or the scanning will be aborted.')
                if resp == 'YES':
                    if rv_scene_dir.exists():
                        logger.debug(f'{self.name}: Delete {rv_scene_dir}')
                        shutil.rmtree(rv_scene_dir)
                    if rv_scene_proj.exists():
                        logger.debug(f'{self.name}: Delete {rv_scene_proj}')
                        os.remove(rv_scene_proj)
                    if moved_scene_dir.exists():
                        logger.debug(f'{self.name}: Delete {moved_scene_dir}')
                        shutil.rmtree(moved_scene_dir)
                    if moved_scene_proj.exists():
                        logger.debug(f'{self.name}: Delete {moved_scene_proj}')
                        os.remove(moved_scene_proj)
                else:
                    raise RuntimeError(f'Structured light scan with the name {project_name} already exists.')

            self.rv.init_project(project_name)
            if not check:
                input(cleandoc(r"""
                Включите проекцию Cross на проекторе и расположите сканируемый объект так чтобы
                    * Его наиболее интересная сторона была обращена к сканеру,
                    * Его центр находился над меткой на столе,
                    * Его центр находился на перекрестье проектора."""))
                clear_output()

            input(cleandoc(r"""
            На вкладке Clipping задайте оптимальную область реконструкции:
                * Выставьте значение Far plane равным 210,
                * Выставьте значение Near plane равным -70.

            На вкладке Scanning подберите экспозицию сканирования.
                * Выставьте первую Exposure так чтобы хорошо было видно бОльшую часть объекта.
                * Если при такой экспозиции остаются существенные пересвеченные либо недоосвещенные части,
                  включите Second exposure и подберите значение так чтобы именно эти части было видно хорошо.

            Откройте вкладку Scanning и нажмите Enter."""))
            clear_output()

            input('Уберите трафарет и посторонние предметы из зоны сканирования, после чего нажмите Enter чтобы запустить сканирование.')
            clear_output()
            print('Смотрите, каким получается первый скан: бОльшая часть объекта в кадре должна отсканироваться.'
                  ' При проверочном сканировании объектов, с которых была удалена краска, отсканироваться должна просто какая-то часть объекта.')

            def scan(point_id, point_pos):
                sleep(robot_shaking_timeout_stl)
                self.rv.wait_for_scanning_end()
                self.rv.scan()
            trajectory = self.robot.generate_trajectory_points()
            if check:
                trajectory = trajectory[[0, 5, 15, -4, -1]]
            self.robot.move_over_points(points=trajectory, velocity=stl_scanning_velocity, closure=scan)

            logger.info(f'{self.name}: Save structured light scans to disk for {project_name}')

            def export():
                self.rv.wait_for_scanning_end()
                self.rv.export_scans()
                self.move_stl_scan_to_scene_dir(scene_name, project_name)
                logger.info(f'{self.name}: Save structured light scans to disk for {project_name} DONE')
            stl_export = Thread(target=export)
            stl_export.start()
            self.robot.rest(scanning_velocity)
            logger.info(f'{self.name}: Scan with structured light scanner {project_name} DONE')
            return stl_export
        finally:
            subprocess.Popen("""/usr/bin/dconf write /org/gnome/mutter/overlay-key "'Super L'" """, shell=True)

    def scan_cameras(self, scene_name, trajectory_points=None):
        r"""Scans one scene with the cameras.

        Parameters
        ----------
        scene_name : str
        trajectory_points : iterable of tuple of numbers
            Sequence of specific trajectory points to rescan at.
        """
        if self.thread not in {'both', 'cameras'}:
            raise RuntimeError('Not available in STL thread')
        self.periphery.lights_off()
        if self.trajectories in {'spheres',}:
            input('Выключите проектор structured light сканера из розетки, выключите свет в комнате и нажмите Enter.')
        else:
            input('Выключите свет в комнате и нажмите Enter.')
        clear_output()

        logger.info(f'{self.name}: Scan with cameras {scene_name}')
        self.stop_warmup()
        ThreadSet([Thread(target=phone.set_high_resolution) for phone in self.rig.phones]).start_and_join()
        for phone in self.rig.phones:
            phone.set_af(True)

        self._scan_cameras(scene_name, trajectory_points)
        if (not self._do_pause) and (trajectory_points is None):
            while True:
                sleep(3)
                while (self.rig.phone_left._files_being_saved != 0) or (self.rig.phone_right._files_being_saved != 0):
                    sleep(1)
                rescan_points = self.check_camera_data(scene_name)
                if len(rescan_points) == 0:
                    break
                logger.info(f'{self.name}: Rescan with cameras {scene_name} at {len(rescan_points)} points')
                self._scan_cameras(scene_name, rescan_points)

        for phone in self.rig.phones:
            phone.set_af(False)
        logger.info(f'{self.name}: Scan with cameras {scene_name} DONE')
        self.start_warmup()
        self.robot.rest(scanning_velocity)

    def _scan_cameras(self, scene_name, trajectory_points=None):
        r"""Scans one scene with the cameras.

        Parameters
        ----------
        scene_name : str
        trajectory_points : iterable of tuple of numbers
            Sequence of specific trajectory points to rescan at.
        """
        self._do_pause = False
        scene_dir = self.make_scene_dir(scene_name)
        for phone in self.rig.phones:
            phone.prepare_scan_dir(scene_name)
        phone_scene_dir = f'{phone_scan_root}/{scene_name}'

        self.robot = self.camera_trajectory_class()
        robot = _Robot(self.robot)

        lights = _Lights(self.periphery)
        kinect = _Kinect(self.rig.kinect, self.periphery, scene_dir, self.trajectories)
        realsense = _RealSense(self.rig.realsense, scene_dir)
        tis_left = _Tis(self.rig.tis_left, scene_dir)
        tis_right = _Tis(self.rig.tis_right, scene_dir)
        phone_left = _Phone(self.rig.phone_left, self.periphery.left_phone_ir, phone_scene_dir)
        phone_right = _Phone(self.rig.phone_right, self.periphery.right_phone_ir, phone_scene_dir)
        phone_left.phone_id = 1
        phone_right.phone_id = 2

        phone_left.ir_off()
        phone_right.ir_off()

        scan_meta = ScanMeta(scene_dir)

        if trajectory_points is None:
            trajectory = robot.generate_trajectory_points()
        else:
            trajectory = np.array(trajectory_points)
            point_ids = [f'{self.robot.get_point_id(*point)}' for point in trajectory]
            scan_meta.reset_scanned_with_cameras(point_ids)
        trajectory = [point for point in trajectory
                      if not scan_meta.is_scanned_with_cameras(f'{self.robot.get_point_id(*point)}')]

        def eta_format(now_t, remaining_t, cam_settings_t=27):
            cam_settings_t = datetime.timedelta(minutes=cam_settings_t)
            eta = now_t + remaining_t
            eta_with_settings = eta + cam_settings_t
            eta = eta.strftime('%H:%M')
            eta_with_settings = eta_with_settings.strftime('%H:%M')
            return f'ETA: {eta}, ETA w/settings: {eta_with_settings}'
        trajectory = tqdm(trajectory, eta_format=eta_format)

        def sigint_ignorer(signum, frame):
            self._do_pause = True
            logger.info(f'{self.name}: Pausing...')

        signal.signal(signal.SIGINT, sigint_ignorer)
        # For each trajectory point
        for point in trajectory:
            point_id = f'{self.robot.get_point_id(*point)}'
            point_img_name = f'{point_id}'

            # Move to the point
            move = robot.move_to(point, scanning_velocity).start()
            # Wait out robot shaking
            shake = move.wait_after(robot_shaking_timeout).start()
            # Turn off Kinect IR emitter
            kinect_ir_off = kinect.ir_off().start()

            shoot = {kinect: Noop(), realsense: Noop(), phone_left: Noop(),
                     phone_right: Noop(), tis_left: Noop(), tis_right: Noop()}

            # For each light source
            for light in self.light_setups:
                light_img_name = f'{point_id}@{light}'
                if light == 'flash':
                    if self.trajectories == 'human_sphere':
                        imaging_presets = ['fast']
                    else:
                        imaging_presets = ['best', 'fast']
                elif light == 'ambient_low':
                    imaging_presets = ['fast']
                else:
                    imaging_presets = ['best']

                # Setup the cameras for the first imaging preset
                first_preset = imaging_presets[0]
                setup_cameras = dict()
                for camera in [kinect, realsense, tis_left, tis_right, phone_left, phone_right]:
                    setup_cameras[camera] = camera.setup(light, first_preset).after(shoot[camera]).start()

                # Wait util scanning with the previous lighting is done
                for task in shoot.values():
                    task.wait()
                shoot = dict()

                # Switch the lights
                setup_light = lights.setup_light(light).after(move).start()

                # Shoot with Kinect without imaging presets since it does not allow to setup the camera
                wait = setup_light.wait_after(kinect.light_delay[light], kinect.name).start()
                shoot[kinect] = kinect.save_photo(light_img_name).after(shake, setup_cameras[kinect], wait).start()

                # For each imaging preset
                for preset in imaging_presets:
                    preset_img_name = f'{point_id}@{light}@{preset}'

                    # Setup the cameras for the imaging preset if its not the first one,
                    # since for the first one we've already done this
                    if preset != first_preset:
                        for camera in [realsense, tis_left, tis_right, phone_left, phone_right]:
                            setup_cameras[camera] = camera.setup(light, preset).after(shoot[camera]).start()

                    # Shoot with tises and phones
                    for camera in [tis_left, tis_right, phone_left, phone_right]:
                        wait = setup_light.wait_after(camera.light_delay[light], camera.name).start()
                        shoot[camera] = camera.save_photo(preset_img_name).after(
                            shake, setup_cameras[camera], wait
                        ).start()

                    # Shoot with RealSense after IR emitters are turned off
                    wait = setup_light.wait_after(realsense.light_delay[light], realsense.name).start()
                    shoot[realsense] = realsense.save_all(preset_img_name).after(
                        shake, setup_cameras[realsense], wait, kinect_ir_off
                    ).start()

            # Save the depth and ir from phones
            shoot[realsense].wait()
            for phone in [phone_left, phone_right]:
                phone.ir_on().start().wait_after(.3).start().wait()
                phone.save_ir_depth(point_img_name).start().wait()
                phone.ir_off().start().wait()

            # Turn on Kinect IR emitter and save Kinect depth and ir
            kinect_ir_on = kinect.ir_on().start()
            shoot[kinect] = kinect.save_ir_depth(point_img_name).after(
                kinect_ir_on.wait_after(.6).start(), shoot[kinect]).start()

            # Wait util scanning is done
            for task in shoot.values():
                task.wait()
            lights.off()
            scan_meta.add_scanned_with_cameras(point_id)
            if self._do_pause:
                break

        signal.signal(signal.SIGINT, signal.default_int_handler)
        lights.off()

    def check_camera_data(self, scene_name):
        logger.info(f'{self.name}: Check camera data for {scene_name}')
        missing_trajectory_points = []
        robot = self.camera_trajectory_class(simulation=True)
        trajectory = robot.generate_trajectory_points()
        scene_dir = self.make_scene_dir(scene_name)
        phone_scene_dir = f'{phone_scan_root}/{scene_name}'

        for point in trajectory:
            point_is_missing = False
            point_id = f'{robot.get_point_id(*point)}'
            point_img_name = f'{point_id}'

            files_on_pc = []
            files_on_phone_left = []
            files_on_phone_right = []
            # Phones IR, depth
            files_on_phone_left.append(f'{phone_scene_dir}/{self.rig.phone_left.name}/{point_img_name}_ir.bin')
            files_on_phone_left.append(f'{phone_scene_dir}/{self.rig.phone_left.name}/{point_img_name}_depth.bin')
            files_on_phone_right.append(f'{phone_scene_dir}/{self.rig.phone_right.name}/{point_img_name}_ir.bin')
            files_on_phone_right.append(f'{phone_scene_dir}/{self.rig.phone_right.name}/{point_img_name}_depth.bin')
            # Kinect IR, depth
            files_on_pc.append(f'{scene_dir}/{self.rig.kinect.name}/{point_img_name}_ir.png')
            files_on_pc.append(f'{scene_dir}/{self.rig.kinect.name}/{point_img_name}_depth.png')
            for light in self.light_setups:
                light_img_name = f'{point_id}@{light}'
                if light == 'flash':
                    if self.trajectories == 'human_sphere':
                        imaging_presets = ['fast']
                    else:
                        imaging_presets = ['best', 'fast']
                elif light == 'ambient_low':
                    imaging_presets = ['fast']
                else:
                    imaging_presets = ['best']
                # Kinect RGB
                files_on_pc.append(f'{scene_dir}/{self.rig.kinect.name}/{light_img_name}.png')
                for preset in imaging_presets:
                    preset_img_name = f'{point_id}@{light}@{preset}'
                    # TIS
                    for camera in [self.rig.tis_left, self.rig.tis_right]:
                        files_on_pc.append(f'{scene_dir}/{camera.name}/{preset_img_name}.png')
                    # RS RGB, IR, depth
                    files_on_pc.append(f'{scene_dir}/{self.rig.realsense.name}/{preset_img_name}.png')
                    files_on_pc.append(f'{scene_dir}/{self.rig.realsense.name}/{preset_img_name}_ir.png')
                    files_on_pc.append(f'{scene_dir}/{self.rig.realsense.name}/{preset_img_name}_irr.png')
                    files_on_pc.append(f'{scene_dir}/{self.rig.realsense.name}/{preset_img_name}_depth.png')
                    # Phones RGB
                    files_on_phone_left.append(f'{phone_scene_dir}/{self.rig.phone_left.name}/{preset_img_name}.jpg')
                    files_on_phone_right.append(f'{phone_scene_dir}/{self.rig.phone_right.name}/{preset_img_name}.jpg')

            logger.debug(f'{self.name}: Check {point_id} on PC')
            for file in files_on_pc:
                if not Path(file).is_file():
                    missing_trajectory_points.append(point.tolist())
                    point_is_missing = True
                    logger.debug(f'{self.name}: Missing {file}')
                    break
            if point_is_missing:
                continue
            logger.debug(f'{self.name}: Check Kinect depth is not empty at {point_id}')
            depthmap = Path(f'{scene_dir}/{self.rig.kinect.name}/{point_img_name}_depth.png')
            depthmap_size = depthmap.stat().st_size
            if depthmap_size < 100_000:
                missing_trajectory_points.append(point.tolist())
                logger.debug(f'{self.name}: Corrupted {depthmap}')
                continue
            logger.debug(f'{self.name}: Check {point_id} on phone_left')
            if not self.rig.phone_left.files_exist(files_on_phone_left):
                missing_trajectory_points.append(point.tolist())
                logger.debug(f'{self.name}: Missing files on phone_left')
                continue
            logger.debug(f'{self.name}: Check {point_id} on phone_right')
            if not self.rig.phone_right.files_exist(files_on_phone_right):
                missing_trajectory_points.append(point.tolist())
                logger.debug(f'{self.name}: Missing files on phone_right')
                continue
        logger.info(f'{self.name}: Check camera data for {scene_name} DONE')
        return missing_trajectory_points

    def move_stl_scan_to_scene_dir(self, scene_name, project_name):
        if self.thread not in {'both', 'stl'}:
            raise RuntimeError('Not available in Camera thread')
        stl_dir = self.make_scene_dir(scene_name) / 'stl'
        stl_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(f'{self.stl_scans_root}/{project_name}_folder', stl_dir)
        shutil.move(f'{self.stl_scans_root}/{project_name}.scanproj', stl_dir)

    def make_scene_dir(self, scene_name):
        if ' ' in scene_name:
            raise RuntimeError(f'Имя сцены "{scene_name}" не должно содержать пробелы. Замените пробелы на "_".')
        scene_dir = self.data_dir / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        return scene_dir

    def log_working_with_scene(self, scene_name):
        logger.info(f'{self.name}: Start working with {scene_name}')

    def log_spray_removal(self, scene_name):
        logger.info(f'{self.name}: Remove spray from {scene_name}')

    def reset_camera_scanning_status(self, scene_name):
        scene_dir = self.make_scene_dir(scene_name)
        scan_meta = ScanMeta(scene_dir)
        scan_meta.reset_scanned_with_cameras()
