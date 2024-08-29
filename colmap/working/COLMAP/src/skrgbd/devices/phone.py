import socket
from io import BytesIO
import os
from pathlib import Path
import pickle
from time import sleep
import subprocess
import uuid

from ppadb.device import Device
from PIL import Image
import numpy as np
import torch

from skrgbd.utils.signal import announce_sensor_freeze
from skrgbd.devices.camera import Camera
from skrgbd.data.image_utils import equalize_image
from skrgbd.utils.parallel import get_timestamp
from skrgbd.utils.logging import logger, tqdm
from skrgbd.utils.parallel import ThreadSet, PropagatingThread as Thread
from skrgbd.image_processing.auto_exposure import pick_camera_settings
from skrgbd.image_processing.matting import segment_fg


scan_root = '/sdcard/Scans'
calibration_root = '/sdcard/Calibration'
port = 5555


class _HuaweiDevice_Base(Device, Camera):
    fast_exposure = 33_333
    best_iso = 100

    exp_min = 20
    exp_max = 500_000
    iso_min = 100
    iso_max = 400_000

    def get_photo(self): ...

    # Setup procedures
    # ----------------
    def __init__(self, client, data_only=False):
        logger.debug(f'{self.name}: Init device')
        if data_only:
            self.serial = self.usb_serial
            Device.__init__(self, client, self.serial)
            logger.debug(f'{self.name}: Init device DONE')
            return

        Device.__init__(self, client, self.serial)
        Camera.__init__(self)
        self._remount()
        self.kill_camera()
        self.auto_whitebalance = True

        self.exposure = -1
        self.iso = -1
        self.flash_is_on = False
        logger.debug(f'{self.name}: Init device DONE')

        self.exposure_presets = dict()
        self.iso_presets = dict()
        self._bg_frame = None
        self._fg_masks = None
        self.setup_light_presets()

    def start(self):
        r"""Setups the phone for scanning and starts the camera app."""
        logger.debug(f'{self.name}: Start')
        self.unlock_screen()
        # self._setup_battery()
        self._setup_screen()
        self.clean()
        self._setup_tof()
        self.start_camera()
        self._setup_camera()
        sleep(2)
        self.switch_auto_whitebalance()
        logger.debug(f'{self.name}: Start DONE')

    def stop(self):
        r"""Stops the phone after scanning."""
        logger.debug(f'{self.name}: Stop')
        self.stop_camera()
        self.light_display()
        self.lock_screen()
        logger.debug(f'{self.name}: Stop DONE')

    def clean(self):
        r"""Cleans photos and ToF data."""
        logger.debug(f'{self.name}: Clean')
        self.persistent_shell('rm -rf /data/vendor/camera/img')
        self.persistent_shell('mkdir /data/vendor/camera/img')

        self.persistent_shell('rm -rf /sdcard/DCIM/Camera')
        self.persistent_shell('mkdir -p /sdcard/DCIM/Camera')

        self.persistent_shell('rm -rf data/log/android_logs/*')
        self.persistent_shell('rm -rf data/vendor/log/isp-log/*')
        logger.debug(f'{self.name}: Clean DONE')

    def start_camera(self):
        r"""Starts the camera app.

        Notes
        -----
        We use a third-party camera app OpenCamera
        * It does not shake the images randomly as the stock camera does,
        * It does not require waking it up so we can take photos faster.

        After install, setup OpenCamera once:
        * Choose the STD mode,
        * Set the focus to locked,
        * Disable the flashlight.
        """
        # self.persistent_shell('am start -n com.huawei.camera/com.huawei.camera')
        self.persistent_shell('am start net.sourceforge.opencamera/.MainActivity')
        self.auto_whitebalance = True

    def stop_camera(self):
        r"""Closes camera app gracefully."""
        self.go_home()

    def pause_camera(self):
        r"""Stops camera without closing the OpenCamera app."""
        raise DeprecationWarning('Only for landscape')
        self.shell('input tap 200 100')

    def unpause_camera(self):
        self.shell('input keyevent 4')

    def kill_camera(self):
        r"""Closes camera app forcefully."""
        # self.persistent_shell('am force-stop com.huawei.camera/com.huawei.camera')
        self.persistent_shell('am force-stop net.sourceforge.opencamera')

    def set_low_resolution(self):
        r"""Sets photo resolution to low 320x640."""
        logger.debug(f'{self.name}: Set camera resolution to low')
        # Open fast settings in OpenCamera
        self.shell('input tap 100 400')
        # Switch resolutions
        orientation = self.orientation
        for i in range(20):
            if orientation == 'landscape':
                self.shell('input tap 1000 200')
            else:
                self.shell('input tap 1200 1100')
        # Close fast settings
        self.shell('input tap 100 400')
        logger.debug(f'{self.name}: Set camera resolution to low DONE')

    def set_high_resolution(self):
        r"""Sets photo resolution to full 40M."""
        logger.debug(f'{self.name}: Set camera resolution to high')
        # Open fast settings in OpenCamera
        self.shell('input tap 100 400')
        # Switch resolutions
        orientation = self.orientation
        for i in range(20):
            if orientation == 'landscape':
                self.shell('input tap 320 200')
            else:
                self.shell('input tap 1200 400')
        # Close fast settings
        self.shell('input tap 100 400')
        logger.debug(f'{self.name}: Set camera resolution to high DONE')

    def setup(self, light, preset):
        r"""Sets up the cameras for the light source with the specific imaging preset.

        Parameters
        ----------
        light : {'flash', 'ambient_low', 'room_lights', default}
        preset : {'best', 'fast', 'calibration'}
        """
        logger.debug(f'{self.name}: Setup for {preset} at {light}')
        self.set_exposure(self.exposure_presets[(light, preset)])
        self.set_iso(self.iso_presets[(light, preset)])
        logger.debug(f'{self.name}: Setup for {preset} at {light} DONE')

    def setup_light_presets(self):
        light, preset = 'room_lights', 'calibration'
        self.exposure_presets[(light, preset)] = 10_000
        self.iso_presets[(light, preset)] = self.best_iso

        preset = 'best'
        for light in [
            'soft_left', 'soft_right', 'soft_top', 'hard_left_bottom_far', 'hard_left_bottom_close',
            'hard_left_top_close', 'hard_right_top_far', 'hard_right_top_close', 'hard_left_top_far',
            'hard_right_bottom_close', 'ambient'
        ]:
            self.exposure_presets[(light, preset)] = 20_000
            self.iso_presets[(light, preset)] = self.best_iso
        light = 'flash'
        self.exposure_presets[(light, preset)] = 150_000
        self.iso_presets[(light, preset)] = self.best_iso

        preset = 'fast'
        light = 'flash'
        self.exposure_presets[(light, preset)] = self.fast_exposure
        self.iso_presets[(light, preset)] = 500
        light = 'ambient_low'
        self.exposure_presets[(light, preset)] = self.fast_exposure
        self.iso_presets[(light, preset)] = 650

        light, preset = 'room_lights', 'background'
        self.exposure_presets[(light, preset)] = 100_000
        self.iso_presets[(light, preset)] = self.best_iso

    def save_light_presets(self, file):
        light_presets = dict(exposure_presets=self.exposure_presets, iso_presets=self.iso_presets)
        pickle.dump(light_presets, open(file, 'wb'))

    def load_light_presets(self, file):
        light_presets = pickle.load(open(file, 'rb'))
        self.exposure_presets = light_presets['exposure_presets']
        self.iso_presets = light_presets['iso_presets']

    def copy_light_preset(self, src, dst):
        self.exposure_presets[dst] = self.exposure_presets[src]
        self.iso_presets[dst] = self.iso_presets[src]

    def set_exposure(self, exposure):
        r"""
        Parameters
        ----------
        exposure : float
            Exposure in us in range 20-500_000.
        """
        exposure = int(np.clip(exposure, self.exp_min, self.exp_max))
        if exposure == self.exposure:
            return
        logger.debug(f'{self.name}: Set exposure to {exposure}')
        self.exposure = exposure
        self.persistent_shell(f'setprop vendor.manual_ae.expotime.value {round(exposure)}')

    def set_iso(self, iso):
        r"""
        Parameters
        ----------
        iso : float
            ISO in range 100-409_600.
        """
        iso = int(np.clip(iso, self.iso_min, self.iso_max))
        if iso == self.iso:
            return
        logger.debug(f'{self.name}: Set ISO to {iso}')
        self.iso = iso
        self.persistent_shell(f'setprop vendor.manual_ae.gain.value {round(iso * 5.12)}')

    def switch_auto_whitebalance(self):
        r"""Switches auto white balance in OpenCamera."""
        self.shell('input tap 200 1100')
        self.auto_whitebalance = not self.auto_whitebalance
        if self.auto_whitebalance:
            self.persistent_shell('setprop vendor.camera.manual_ae.flag 0')  # enable auto exposure
        else:
            self.persistent_shell('setprop vendor.camera.manual_ae.flag 1')  # disable auto exposure

    def auto_white_balance(self):
        r"""Adjusts the white balance by switching AWB on, waiting, and switching it off."""
        self.switch_auto_whitebalance()
        sleep(2)
        self.switch_auto_whitebalance()

    def pick_settings(self, light, preset, ):
        logger.debug(f'{self.name}: Pick settings for RGB at {light} at {preset}')
        exp_low, exp_high = self.exp_min, 300_000
        gain_low, gain_high = self.best_iso, 20_000

        i, j = self._fg_masks.nonzero()
        bbox_i = slice(i.min(), i.max())
        bbox_j = slice(j.min(), j.max())
        fg_mask = torch.from_numpy(self._fg_masks[bbox_i, bbox_j])

        def get_img():
            img = self.get_light_picking_frame()
            img = torch.from_numpy(img)
            return img[bbox_i, bbox_j]

        exposure, gain = pick_camera_settings(
            preset, get_img, fg_mask, self.set_exposure, exp_low, exp_high, self.fast_exposure,
            self.set_iso, gain_low, gain_high, self.best_iso)
        self.exposure_presets[(light, preset)] = exposure
        self.iso_presets[(light, preset)] = gain
        logger.debug(f'{self.name}: Pick settings for RGB at {light} at {preset} DONE exp={exposure}, iso={gain}')

    def set_bg_frame(self):
        self.setup('room_lights', 'background')
        self._bg_frame = self.get_light_picking_frame()

    def set_fg_masks(self, method='nn', threshold=25):
        self.setup('room_lights', 'background')
        fg_frame = self.get_light_picking_frame()
        if method == 'classic':
            self._fg_masks = np.abs(fg_frame.astype(np.int16) - self._bg_frame.astype(np.int16)) > threshold
        elif method == 'nn':
            fg_frame = torch.from_numpy(fg_frame).float().div(255).unsqueeze(0).expand(3, -1, -1)
            bg_frame = torch.from_numpy(self._bg_frame).float().div(255).unsqueeze(0).expand(3, -1, -1)
            self._fg_masks = segment_fg(fg_frame, bg_frame).numpy()
        elif method == 'full':
            self._fg_masks = np.ones_like(fg_frame, dtype=bool)
        else:
            raise ValueError(f'Unknown method {method}')

    def get_light_picking_frame(self):
        image = self.get_photo()
        image = np.asarray(Image.fromarray(image).convert('L'))  # convert to luma
        return image

    def save_light_picking_frame(self, path):
        image = self.get_photo()
        Image.fromarray(image).save(path)

    def _remount(self):
        r"""Remounts system folders in rw mode."""
        self.persistent_shell('mount -o rw,remount /odm')
        self.persistent_shell('mount -o rw,remount /vendor')

    def _setup_battery(self):
        r"""Tricks the phone charger so that it takes larger current from the USB."""
        self.persistent_shell('echo 4 > /sys/devices/platform/huawei_charger/vr_charger_type')  # reset to default
        self.persistent_shell('echo 1 > /sys/devices/platform/huawei_charger/vr_charger_type')  # corresponds to 1.5A max
        sleep(1)
        if self.persistent_shell('cat /sys/class/power_supply/USB/current_max')[:-1] != '1400000':
            raise RuntimeError('Could not increase the charging current')

    def _setup_screen(self):
        r"""Setups the screen for scanning"""
        self.persistent_shell('settings put system screen_brightness_mode 0')  # disable auto brightness
        self.persistent_shell('settings put system screen_off_timeout 600000')  # set screen off timeout to 10 minutes
        self.dim_display()

    def _setup_tof(self):
        r"""Does some magical stuff with the ToF sensor."""
        self.persistent_shell('setprop persist.vendor.camera.itof.mode 1')

        # set dump preferences
        self.persistent_shell('setprop vendor.camera.itofraw.dump 1')
        self.persistent_shell('setprop vendor.camera.itofraw.dump.count 1')
        self.persistent_shell('setprop vendor.camera.itofraw.dump.time 0')
        self.persistent_shell('setprop vendor.camera.itofraw.dump.temperature 1')
        self.persistent_shell('setprop vendor.camera.itofresult.dump 1')
        self.persistent_shell('setprop vendor.camera.itofresult.dump.count 1')
        self.persistent_shell('setprop vendor.camera.itofresult.dump.time 0')

        # set additional ToF preferences
        self.persistent_shell('setprop vendor.disable.tof.check 1')
        self.persistent_shell('setprop vendor.camera.moca.onivp true')
        self.persistent_shell('setprop vendor.camera.moca.depth2rgb 1')

    def _setup_camera(self):
        r"""Setups the RGB camera."""
        logger.debug(f'{self.name}: Setup Huawei camera')
        # self.persistent_shell('setprop vendor.camera.manual_af.flag 0')  # enable autofocus
        self.persistent_shell('setprop vendor.camera.manual_af.flag 1')  # disable autofocus
        self.persistent_shell(f'setprop vendor.manual_af.vcmcode.value {self.focus}')
        self.persistent_shell('setprop vendor.debug.smartae.enable 0')  # disable auto exposure etc
        self.persistent_shell('setprop vendor.camera.manual_ae.flag 1')  # disable auto exposure
        self.set_exposure(5_000)
        self.set_iso(100)
        logger.debug(f'{self.name}: Setup Huawei camera DONE')

    def set_af(self, af):
        af = 0 if af else 1
        self.persistent_shell(f'setprop vendor.camera.manual_af.flag {af}')

    # Utils
    # -----
    def go_home(self):
        self.persistent_shell('am start -a android.intent.action.MAIN -c android.intent.category.HOME')

    def dim_display(self):
        self.persistent_shell('settings put system screen_brightness 0')

    def light_display(self):
        self.persistent_shell('settings put system screen_brightness 32')

    def lock_screen(self):
        if self.is_screen_on():
            self.shell('input keyevent 26')

    def unlock_screen(self):
        self.shell('input keyevent 224')

    def flash_on(self):
        if self.flash_is_on:
            return
        logger.debug(f'{self.name}: Flash ON')
        self.persistent_shell("dmesg -C && echo 255 > /sys/class/leds/torch/brightness"
                   " && (dmesg -w | grep -q 'hw_lm3644_torch_brightness_set brightness')")
        self.flash_is_on = True
        logger.debug(f'{self.name}: Flash ON DONE')

    def flash_off(self):
        if not self.flash_is_on:
            return
        logger.debug(f'{self.name}: Flash OFF')
        self.persistent_shell("dmesg -C && echo 0 > /sys/class/leds/torch/brightness"
                   " && (dmesg -w | grep -q 'hw_lm3644_torch_brightness_set brightness')")
        self.flash_is_on = False
        logger.debug(f'{self.name}: Flash OFF DONE')

    # Checks and properties
    # ------
    def is_screen_on(self):
        return self.persistent_shell('dumpsys display | grep "mScreenState" | cut -d"=" -f2') == 'ON\n'

    @property
    def battery_level(self):
        return int(self.persistent_shell('cat /sys/class/power_supply/Battery/capacity')[:-1])

    @property
    def disk_space(self):
        return self.persistent_shell('df -h /data | tail -n1 | cut -F4')[:-1]

    @property
    def orientation(self):
        orientation = int(self.persistent_shell("dumpsys input | grep 'SurfaceOrientation' | awk '{ print $2 }'")[0])
        return {0: 'landscape', 2: 'landscape', 1: 'portrait', 3: 'portrait'}[orientation]

    def persistent_shell(self, cmd, timeout=8):
        while True:
            try:
                ret = super().shell(cmd, timeout=timeout)
                break
            except socket.timeout as exception:
                logger.warn(f'{self.name}: {cmd} raised {type(exception).__name__}({exception}). Retrying.')
        return ret


class _HuaweiDevice_Scan(_HuaweiDevice_Base):
    def __init__(self, client, data_only=False):
        super().__init__(client, data_only)
        self._files_being_saved = 0

    # Not implemented
    def save_rgb(self, path, frame): raise NotImplementedError
    def save_rgb_calib(self, root_dir, image_name, frame): raise NotImplementedError
    def save_depth_npy(self, path, frame): raise NotImplementedError
    def save_depth_png(self, path, frame): raise NotImplementedError
    def save_ir_npy(self, path, frame): raise NotImplementedError
    def save_ir_png(self, path, frame): raise NotImplementedError
    def save_ir_calib(self, root_dir, image_name, frame): raise NotImplementedError

    # Scanning procedures
    # -------------------
    def clean_scans(self):
        self.persistent_shell(f'rm -rf {scan_root} && mkdir -p {scan_root}')

    def prepare_scan_dir(self, scene_name):
        ret = self.persistent_shell(f'mkdir -p {scan_root}/{scene_name}/{self.name}')
        if ret != '':
            raise RuntimeError(ret)

    def snap_frame(self, equalize_ir=False, compressed=False):
        self.take_photo()
        return dict()

    def take_photo(self):
        r"""Takes the photo and waits until the photo is taken.

        The idea behind the waiting trick is that the photo is started to be compressed into JPEG when it is taken,
        which produces the following message in dmesg.
        So we clear dmesg first, then shoot, then wait until the message appears.
        """
        logger.debug(f'{self.name}: Taking image')
        try:
            ret = self.shell("dmesg -C && input keyevent 27 && (dmesg -w | grep -q 'CAMERA]INFO: hjpeg_power_on enter')")
            if ret != '\x1b[0m':
                raise RuntimeError(ret)
        except socket.timeout:
            announce_sensor_freeze()
            raise RuntimeError(f'{self.name} not responding')

        logger.debug(f'{self.name}: Taking image DONE')

    # IO procedures
    # -------------
    def save_image(self, path, modalities=None, compressed=False, blocking=True):
        r"""Saves rgb image to {path}.jpg, depthmap to {path}_depth.bin, and IR image to {path}_ir.bin.

        Parameters
        ----------
        path : str
        modalities : iterable of {'image', 'depth', 'ir'}
        blocking : bool
            If False, writes the files in background.
        """
        if compressed:
            raise NotImplementedError
        if modalities is None:
            modalities = {'image', 'depth', 'ir'}

        threads = []
        if 'image' in modalities:
            threads.append(Thread(target=self.save_photo, args=[path, blocking]))
        if 'depth' in modalities:
            threads.append(Thread(target=self.save_depth, args=[path]))
        if 'ir' in modalities:
            threads.append(Thread(target=self.save_ir, args=[path]))
        threads = ThreadSet(threads)
        threads.start_and_join()

    def pull_scans(self, data_dir, rm=True):
        if rm:
            self.pull_dir_rm(scan_root, data_dir)
        else:
            self.pull_dir(scan_root, data_dir)

    def pull_dir(self, dir_on_phone, dir_on_host, rm=False):
        while self._files_being_saved != 0:
            logger.debug(f'{self.name}: Wait for {self._files_being_saved} to be saved')
            sleep(1)

        dir_on_host = Path(dir_on_host)
        dir_on_host.mkdir(parents=True, exist_ok=True)
        for sub in self.persistent_shell(f'ls {dir_on_phone}')[:-1].split():
            logger.info(f'{self.name}: Pull {dir_on_phone}/{sub} to {dir_on_host}')
            progress = tqdm(total=100)
            p = subprocess.Popen(
                f'adb -s {self.serial} pull {dir_on_phone}/{sub} {dir_on_host}/'.split(), stdout=subprocess.PIPE)
            progress_percents = 0
            new_percents = 0
            while True:
                line = p.stdout.readline()
                if not line:
                    break
                try:
                    new_percents = int(line[1:4])
                except ValueError:
                    pass
                if new_percents != progress_percents:
                    progress.update(new_percents - progress_percents)
                    progress_percents = new_percents
            logger.info(f'{self.name}: Pull {dir_on_phone}/{sub} to {dir_on_host} DONE')
            if rm:
                ret = self.persistent_shell(f'rm -rf {dir_on_phone}/{sub}')
                if ret != '':
                    raise RuntimeError(ret)

    def pull_dir_rm(self, src_dir, dst_dir):
        while True:
            ret = self.persistent_shell(f'find {src_dir} -type f | head -c1')
            if ret == '':
                return
            elif len(ret) != 1:
                raise RuntimeError(ret)
            self.del_pulled(src_dir, dst_dir)
            self.pull_dir(src_dir, dst_dir)

    def del_pulled(self, src_dir, dst_dir, batch_size=100):
        command = f'find {src_dir} -type f -print0 | xargs -0 ls -l | cut -F "5,8" | sed "s|{src_dir}/||"'
        ret = self.persistent_shell(command)
        files_to_remove = []

        for size_file in ret.splitlines():
            size, file = size_file.split()
            file_on_pc = Path(f'{dst_dir}/{file}')
            if file_on_pc.exists() and (file_on_pc.stat().st_size == int(size)):
                files_to_remove.append(file)

        for batch_start in range(0, len(files_to_remove), batch_size):
            batch = files_to_remove[batch_start: batch_start + batch_size]
            ret = self.persistent_shell(f'cd {src_dir} && rm -rf ' + ' '.join(batch))
            if ret != '':
                raise RuntimeError(ret)

    def files_exist(self, files):
        files = ' '.join(f'"{file}"' for file in files)
        command = (f'for file in {files}; do'
                    '  if [ ! -f "$file" ]; then printf 1; break; fi'
                    ' done')
        return self.persistent_shell(command) == ''

    # --- private
    def save_photo(self, path, blocking=False):
        r"""Saves photo to {path}.jpg.

        Parameters
        ----------
        path : str
        blocking : bool
            If False, writes the files in background.
        """
        # Mark the snap time
        snap_time = self.persistent_shell("date +'%s'")[:-1]
        # Take the photo and wait until the photo is taken.
        self.take_photo()
        if blocking:
            self._save_photo(snap_time, path)
        else:
            Thread(target=self._save_photo, args=[snap_time, path]).start()

    def _save_photo(self, snap_time, path):
        r"""Waits for the photo to be saved and moves it to the scan directory.

        Parameters
        ----------
        snap_time : int
            The time since epoch when the photo was shot.
        path : str
        """
        self._files_being_saved += 1
        dst = f'{path}.jpg'
        self.shell('sleep 3')
        src = self.wait_for_photo(snap_time)
        if src == '':
            raise RuntimeError(f'No photo was saved for {dst}')
        self.shell(f'mkdir -p "$(dirname "{dst}")" && mv {src} {dst}')
        logger.debug(f'{self.name}: Saved {src} to {dst}')
        self._files_being_saved -= 1

    def wait_for_photo(self, snap_time, timeout=10):
        r"""Waits for the photo to be saved.

        Parameters
        ----------
        snap_time : int
            The time since epoch when the photo was shot.

        Returns
        -------
        file : str
            Filename of the photo.
        """
        file = self.shell(fr"""
            start_time=$(date +'%s')
            snap_time={snap_time}
            oldest_file_since_snap=''
            oldest_file_ctime=$(({snap_time} + 600))

            while [ -z $oldest_file_since_snap ]; do
              if [ $(date +'%s') -ge $((start_time + {timeout})) ]; then break; fi
              while read -r file; do
                ctime=$(stat -c '%Z' "$file")
                if [ $ctime -ge $snap_time ] && [ $ctime -lt $oldest_file_ctime ]; then
                  oldest_file_ctime=$ctime
                  oldest_file_since_snap="$file"
                fi
              done <<< $(find /sdcard/DCIM/Camera -name '*.jpg')
            done
            echo $oldest_file_since_snap
        """)[:-1]
        return file

    def save_depth(self, path):
        r"""Saves depthmap to {path}_depth.bin.

        Parameters
        ----------
        path : str
        """
        logger.debug(f'{self.name}: Saving depth')
        dst = f'{path}_depth.bin'
        self.save_binary('/data/vendor/camera/img', 'depth_0.bin', dst, size=86400)
        logger.debug(f'{self.name}: Saved depth to {dst}')

    def save_ir(self, path):
        r"""Saves ir image to {path}_ir.bin.

        Parameters
        ----------
        path : str
        """
        logger.debug(f'{self.name}: Saving ir')
        dst = f'{path}_ir.bin'
        self.save_binary('/data/vendor/camera/img', 'confidence_0.bin', dst, size=86400)
        logger.debug(f'{self.name}: Saved ir to {path}_ir.bin')

    def save_binary(self, src_dir, src_filename, dst, size):
        r"""Waits for a binary file to be complete and moves it to the scan directory.

        Parameters
        ----------
        src_dir : str
        src_filename : str
        dst : str
        size : int
            Correct size of the source file in bytes.

        Notes
        -----
        The algorithm is:
        1. Wait for the source file to be created.
        2. Wait for the size of the source file to be correct.
        3. Preserve the source file by renaming it, keeping it on the same partition.
        4. If the renamed file has a wrong size, repeat from step 1.
        5. Otherwise, move it to the destination.
        """
        src = f'{src_dir}/{src_filename}'
        tmp = f'{src_dir}/{uuid.uuid4()}'
        ret = self.shell(fr"""
            while :; do
                while [ ! -e {src} ]; do :; done
                while [ $(stat -c %s {src}) -lt {size} ]; do :; done
                mv {src} {tmp}
                if [ $(stat -c %s {tmp}) -eq {size} ]; then break; fi
            done
            mkdir -p "$(dirname "{dst}")"
            mv {tmp} {dst}
        """)
        if ret != '':
            raise RuntimeError(ret)

    # Calibration procedures
    # --------------------
    def clean_calibration(self):
        ret = self.persistent_shell(f'rm -rf {calibration_root}')
        if ret != '':
            raise RuntimeError(ret)

    def save_calib_data(self, root_dir, image_name, modalities, blocking=True):
        r"""Saves images to /sdcard/Calibration/{subdir}/{camera_name}_{modality}/{image_name}.jpg or _ir.bin,
        where subdir is the stem of root_dir.

        Parameters
        ----------
        root_dir : str
        image_name : str
        modalities : iterable of {'image', 'ir'}
        blocking : bool
            If False, write the files in background.
        """
        if modalities is None:
            modalities = {'image', 'ir'}
        subdir = Path(root_dir).stem
        threads = []
        if 'image' in modalities:
            path = f'{calibration_root}/{subdir}/{self.name}_rgb/{image_name}'
            threads.append(Thread(target=self.save_photo, args=[path, blocking]))
        if 'ir' in modalities:
            path = f'{calibration_root}/{subdir}/{self.name}_ir/{image_name}'
            threads.append(Thread(target=self.save_ir, args=[path]))
        threads = ThreadSet(threads)
        threads.start_and_join()

    def pull_calibration(self, calib_dir):
        self.pull_dir(calibration_root, calib_dir)

    # Streaming procedures
    # --------------------
    def start_streaming(self, modality, im_slice=None, figsize_base=5, figsize=None, ticks=True, frames=True, name=True):
        if im_slice is None:
            im_slice = slice(None, None, None)
            if figsize is None:
                figsize = (figsize_base, figsize_base * 128 / 153)
        else:
            if modality == 'image':
                im = self.get_photo()[im_slice]
            else:
                im = self.get_ir()[im_slice]
            h, w = im.shape[:2]
            figsize = (figsize_base, figsize_base * h / w)

        if modality.startswith('ir'):
            kwargs = dict(cmap='gray', vmin=0)
        else:
            kwargs = dict()

        if modality == 'image':
            get_image = lambda: self.get_photo()[im_slice]
        else:
            get_image = lambda: self.get_ir()[im_slice]
        return super().start_streaming(modality, get_image, figsize, ticks, frames, name, **kwargs)

    def get_ir(self, h=180, w=240):
        r"""Reads the IR image from the phone.
        The Camera or another imaging app has to be active.

        Returns
        -------
        ir : np.ndarray
            of shape [h, w]
        """
        ir = np.empty([1])
        while ir.size < h * w:
            ir = self.read_file('/data/vendor/camera/img/confidence_0.bin')
            ir = np.frombuffer(ir, dtype=np.uint16)
        ir = ir.reshape(h, w)
        return ir

    def get_photo(self):
        r"""Takes a photo and reads it.
        The Camera or another imaging app has to be active.

        Returns
        -------
        photo : np.ndarray
            of shape [h, w, 3]
        """
        while True:
            try:
                snap_time = self.persistent_shell("date +'%s'")[:-1]
                self.take_photo()
                sleep(1)
                last_photo = self.wait_for_photo(snap_time, timeout=4)
                if last_photo == '':
                    continue
                photo = self.read_file(last_photo)
                photo = BytesIO(photo)
                photo = np.asarray(Image.open(photo))
                break
            except OSError as exception:
                logger.warn(f'{self.name}: get_photo raised {type(exception).__name__}({exception})')
        return photo

    def read_file(self, file):
        conn = self.create_connection()
        with conn:
            cmd = f'shell:cat {file}'
            conn.send(cmd)
            result = conn.read_all()
        if result and len(result) > 5 and result[5] == 0x0d:
            result = result.replace(b'\r\n', b'\n')
        return bytes(result)

    # Warmup procedures
    # --------------------
    def start_warmup(self, shoot_interval=0, clean_at_warmup_images_n=500):
        r"""Keeps the camera warm, taking images in a background thread.
        Call `stop_warmup` to stop.

        Parameters
        ----------
        shoot_interval : float
            Time interval in seconds between subsequent image captures.
        clean_at_warmup_images_n : int
            Pause shooting and clean the photos if this number of photos have been shot.
        """
        self._stop_warmup = False

        def target():
            prev_shoot_time = 0
            warmup_images_n = 0
            while not self._stop_warmup:
                sleep(max(.01, prev_shoot_time + shoot_interval - get_timestamp()))
                prev_shoot_time = get_timestamp()
                self.take_photo()
                warmup_images_n += 1
                if warmup_images_n >= clean_at_warmup_images_n:
                    sleep(5)
                    self.clean()
                    warmup_images_n = 0
        self._warmup_thread = Thread(target=target)
        self._warmup_thread.start()

    def stop_warmup(self):
        super().stop_warmup()
        sleep(5)
        self.clean()


HuaweiPhone = _HuaweiDevice_Scan


class LeftPhone(HuaweiPhone):
    usb_serial = '0000000000000001'
    serial = f'192.168.1.3:{port}'
    name = 'phone_left'
    focus = 165  # focus at 55-70 cm


class RightPhone(HuaweiPhone):
    usb_serial = '0000000000000002'
    serial = f'192.168.1.4:{port}'
    name = 'phone_right'
    focus = 180  # focus at 55-70 cm


def set_serials():
    command = ("""for transport_id in $(adb devices -l | grep usb | head -n3 | tail -n2 | cut -d' ' -f11 | cut -d':' -f2);"""
               """do adb -t $transport_id shell sh /sdcard/set_serial.sh; done""")
    return subprocess.Popen(command, shell=True)


def setup_adb_over_wifi():
    command = (
        """for transport_id in $(adb devices -l | grep usb | head -n3 | tail -n2 | cut -d' ' -f11 | cut -d':' -f2);"""
        f"""do adb -t $transport_id tcpip {port} &&"""
        """sleep 5;"""
        f"""adb connect {LeftPhone.serial};"""
        f"""adb connect {RightPhone.serial}"""
    )
    return subprocess.Popen(command, shell=True)


def connect_adb_over_wifi():
    command = (
        f"""adb connect {LeftPhone.serial};"""
        f"""adb connect {RightPhone.serial}"""
    )
    return subprocess.Popen(command, shell=True).wait()


def visualize_ir(ir, equalize=False):
    ir = ir.astype(np.float32)
    ir = ir / ir.max()
    if equalize:
        ir = equalize_image(ir, 7 * 4 + 1)
    ir = np.clip(ir * 255, 0, 255).astype(np.uint8)
    return ir


def convert_bin_to_img(scans_dir, calibration=False):
    r"""Converts binary ir and depth files from the phones to png.

    Parameters
    ----------
    scans_dir : str
    """
    scans_dir = Path(scans_dir)
    files = list(scans_dir.rglob('*_ir.bin')) + list(scans_dir.rglob('*_depth.bin'))
    for file in tqdm(files):
        logger.debug(f'ConvertPhoneBins: Convert {file}')
        data = read_bin(file)
        data = np.rot90(data, 2)
        if calibration:
            data = visualize_ir(data, equalize=True)
            Image.fromarray(data).save(str(file)[:-len('_ir.bin')] + '.png')
        else:
            Image.fromarray(data).save(str(file.with_suffix('.png')))
        logger.debug(f'ConvertPhoneBins: Delete {file}')
        os.remove(file)


def read_bin(file_bin, w=180, h=240):
    r"""Reads binary depth or ir file saved on the phones.

    Parameters
    ----------
    file_bin : str
    w : int
    h : int

    Returns
    -------
    data : np.ndarray
        of shape [w, h] uint16, depth is in mm.
    """
    with open(file_bin, 'rb') as file:
        data = file.read()
    data = np.frombuffer(data, dtype=np.uint16).reshape(w, h)
    return data
