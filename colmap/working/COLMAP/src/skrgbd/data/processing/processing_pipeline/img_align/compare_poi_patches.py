import subprocess

from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

from skrgbd.data.processing.alignment.image_alignment import Camera
from skrgbd.data.processing.processing_pipeline.img_align.configs import configs
from skrgbd.data.processing.processing_pipeline.img_align.helper import Helper
from skrgbd.data.processing.depth_utils.occluded_mesh_rendering import MeshRenderer as OccMeshRenderer
from skrgbd.utils.logging import tqdm


class PatchHelper:
    def __init__(self, scene_paths, cam_name, mode, logdir, show_progress=True):
        self.helper = Helper(scene_paths)
        self.cam_name = cam_name
        self.mode = mode
        self.logdir = logdir

        self.config = configs[cam_name, mode]
        self.show_progress = tqdm if show_progress else (lambda x: x)

        self.view_ids = range(100)
        self.procs = []
        self.uv_np = np.zeros(2, dtype=np.float32)

        self.ax = None
        self.cams = None
        self.crop_jis = None
        self.grid = None
        self.imgs = None
        self.init_view_i = None
        self.ir_vmax = float('inf')
        self.patch_w = None
        self.patch_w_img = None
        self.poi = None
        self.renderer = None
        self.scaling_factor = None
        self.slots = None

    def __del__(self):
        self.terminate_children()

    def terminate_children(self):
        while len(self.procs) > 0:
            proc = self.procs.pop()
            proc.terminate()

    def init_cams(self):
        cam_model = self.helper.load_cam_model(self.cam_name, self.mode)
        w2c = self.helper.load_cam_poses(self.cam_name, self.mode, 'ref')

        if (self.mode == 'rgb') and ('scale_factor' in self.config.load_cam_data) \
                                and (self.config.load_cam_data.scale_factor is not None):
            scale_factor = self.config.load_cam_data.scale_factor
            wh = cam_model.size_wh.mul(scale_factor).round_().to(cam_model.size_wh)
            cam_model = cam_model.resize_(wh)

        cams = dict()
        for view_i in self.show_progress(self.view_ids):
            cam = Camera(cam_model, w2c[view_i])
            cams[view_i] = cam; del cam
        self.cams = cams

    def init_scan(self, occ_thres=1e-3):
        rec = self.helper.load_rec()
        occ = self.helper.load_occ()
        rec = rec.compute_triangle_normals()
        self.renderer = OccMeshRenderer(rec, occ, occ_thres); del rec, occ

    def init_crops(self):
        crop_stats = f'{self.logdir}/crop_stats.pt'
        crop_stats = torch.load(crop_stats)
        crop_jis = dict()
        for view_i in self.view_ids:
            crop_jis[view_i] = torch.tensor([crop_stats[view_i]['j_min'], crop_stats[view_i]['i_min']])
        self.crop_jis = crop_jis

    def init_imgs(self):
        imgs = dict()
        for var, file_var in [('dst', f'{self.cam_name}_dst'),
                              ('repr', f'{self.cam_name}_repr'),
                              ('ref', f'{self.config.load_cam_data_ref.cam_name}_repr')]:
            imgs[var] = dict()
            for view_i in self.show_progress(self.view_ids):
                img = self.load_img(view_i, file_var)
                imgs[var][view_i] = img
        self.imgs = imgs

    def load_img(self, view_i, var):
        img = f'{self.logdir}/{view_i:04}_{var}.jpg'
        img = Image.open(img)
        img = np.array(img, copy=True)
        return img

    def remap(self, var='ref_to_repr'):
        refs = []
        reprs = []
        for view_i in self.show_progress(self.view_ids):
            refs.append(self.imgs['ref'][view_i].reshape(-1, 3))
            reprs.append(self.imgs['repr'][view_i].reshape(-1, 3))
        refs = np.concatenate(refs)
        reprs = np.concatenate(reprs)

        if var == 'ref_to_repr':
            ref_to_repr = np.linalg.lstsq(refs, reprs, rcond=None)[0]
            for view_i in self.show_progress(self.view_ids):
                mapped = self.imgs['ref'][view_i] @ ref_to_repr
                mapped = np.clip(np.round(mapped), 0, 255).astype(np.uint8)
                self.imgs['ref'][view_i] = mapped
            self.imgs['repr_for_ref'] = self.imgs['repr']
        elif var == 'repr_to_ref':
            repr_to_ref = np.linalg.lstsq(reprs, refs)[0]
            self.imgs['repr_for_ref'] = dict()
            for view_i in self.show_progress(self.view_ids):
                mapped = self.imgs['repr'][view_i] @ repr_to_ref
                mapped = np.clip(np.round(mapped), 0, 255).astype(np.uint8)
                self.imgs['repr_for_ref'][view_i] = mapped

    def init_ir_imgs(self):
        imgs = dict()
        self.imgs = imgs
        for var, load_img in [('shading', self.load_img), ('ir', self.load_ir_img)]:
            imgs[var] = dict()
            for view_i in self.show_progress(self.view_ids):
                img = load_img(view_i, var)
                h, w = img.shape
                img = np.broadcast_to(img[..., None], (h, w, 3))
                imgs[var][view_i] = img

    def load_ir_img(self, view_i, _):
        img = self.helper.load_ir_img(self.cam_name, self.mode, view_i)
        j_min, i_min = self.crop_jis[view_i]
        h, w = self.imgs['shading'][view_i].shape[:2]
        img = img[i_min: i_min + h, j_min: j_min + w]
        img = img.numpy()
        if self.cam_name in {'kinect_v2', 'phone_left', 'phone_right'}:
            self.ir_vmax = min(self.ir_vmax, img.max())
        return img

    def init_grid(self, patch_w_img=100, scaling_factor=1):
        patch_w = patch_w_img * scaling_factor

        row_starts = [0, 12, 24, 36, 48, 59, 70, 81, 91]
        row_ends = row_starts[1:] + [100]
        row_starts = np.array(row_starts)
        row_ends = np.array(row_ends)
        row_lens = row_ends - row_starts
        row_shifts = (12 - row_lens) * patch_w // 2
        row_ranges = []
        flip = True
        for start, end in reversed(list(zip(row_starts, row_ends))):
            row_range = range(start, end)
            if flip:
                row_range = reversed(row_range)
            row_ranges.append(list(row_range))
            flip = not flip
        row_shifts = list(reversed(row_shifts))

        grid = np.zeros([9 * patch_w, 12 * patch_w, 3], dtype=np.uint8)
        slots = dict()

        for row_i, (row_range, shift) in enumerate(zip(row_ranges, row_shifts)):
            for view_j, view_i in enumerate(row_range):
                slots[view_i] = grid[row_i * patch_w: (row_i + 1) * patch_w, view_j * patch_w + shift: (view_j + 1) * patch_w + shift]

        self.scaling_factor = scaling_factor
        self.patch_w_img = patch_w_img
        self.patch_w = patch_w
        self.grid = grid
        self.slots = slots

    def run(self, var='repr', init_view_i=53):
        self.init_view_i = init_view_i
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        img = self.imgs[var][init_view_i]
        if (var == 'ir') and (self.cam_name in {'kinect_v2', 'phone_left', 'phone_right'}):
            img = np.clip(img / self.ir_vmax, 0, 1)
        ax.imshow(img)
        self.ax = ax
        fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.tight_layout()

    def onclick(self, event):
        if self.mode == 'rgb':
            vars = ['dst', 'repr', 'ref', 'repr_for_ref']
        else:
            vars = ['ir', 'shading']

        self.uv_np[:] = [event.xdata, event.ydata]
        try:
            self.calc_poi()
            for var in vars:
                self.fill_grid(var)
                Image.fromarray(self.grid).save(f'/tmp/tmp.{var}.png')
        except Exception:
            self.ax.scatter([self.uv_np[0]], [self.uv_np[1]], marker='x', c='red', s=10)
            return

        self.ax.scatter([self.uv_np[0]], [self.uv_np[1]], marker='o', c='yellow', s=8**2)
        self.ax.scatter([self.uv_np[0]], [self.uv_np[1]], marker='o', c='black', s=5**2)
        self.ax.scatter([self.uv_np[0]], [self.uv_np[1]], marker='o', c='yellow', s=2**2)

        self.terminate_children()
        if self.mode == 'rgb':
            proc = subprocess.Popen(['feh', '--force-aliasing', '--keep-zoom-vp', '-D.1', '-dg', '1244x1387+0+0',
                                     '/tmp/tmp.repr_for_ref.png', '/tmp/tmp.ref.png'], env=dict(DISPLAY=':77'))
            self.procs.append(proc)
            proc = subprocess.Popen(['feh', '--force-aliasing', '--keep-zoom-vp', '-D.1', '-dg', '1244x1387+2000+0',
                                     '/tmp/tmp.repr.png', '/tmp/tmp.dst.png'], env=dict(DISPLAY=':77'))
            self.procs.append(proc)
        else:
            win_w = self.grid.shape[1]
            screen_w = 2488
            shift = (screen_w - win_w) // 2
            proc = subprocess.Popen(['feh', '--force-aliasing', '--keep-zoom-vp', '-D.1', '-dg', f'{win_w}x1387+{shift}+0',
                                     '/tmp/tmp.ir.png', '/tmp/tmp.shading.png'], env=dict(DISPLAY=':77'))
            self.procs.append(proc)

    def calc_poi(self):
        uv = torch.from_numpy(self.uv_np)
        ref_cam = self.cams[self.init_view_i]
        ref_uv = uv + self.crop_jis[self.init_view_i]
        ray_cam = ref_cam.cam_model.unproject(ref_uv.unsqueeze(1)).squeeze(1)
        casted_rays = self.renderer.make_rays_from_cam(ref_cam.c2w_t, ref_cam.c2w_rotm, ray_cam.unsqueeze(0)); del ref_cam, ray_cam
        render = self.renderer.render_rays(casted_rays.float(), cull_back_faces=True)
        self.renderer.h = self.renderer.w = 1
        self.poi = self.renderer.get_xyz(render, casted_rays).view(3); del render, casted_rays

    def fill_grid(self, var):
        pad = self.patch_w_img
        for view_i in self.show_progress(self.view_ids):
            cam = self.cams[view_i]
            poi_cam = cam.w2c_rotm @ self.poi + cam.w2c_t
            uv = cam.cam_model.project(poi_cam.unsqueeze(1)).squeeze(1)
            uv = uv - self.crop_jis[view_i] + pad

            ji_min = (uv - self.patch_w_img / 2).round().int()
            ji_max = ji_min + self.patch_w_img
            img = self.imgs[var][view_i]
            img = np.pad(img, [(pad, pad), (pad, pad), (0, 0)])
            img = img[ji_min[1]:ji_max[1], ji_min[0]:ji_max[0]]
            if (var == 'ir') and (self.cam_name in {'kinect_v2', 'phone_left', 'phone_right'}):
                img = img / self.ir_vmax
                img = np.clip(np.round(img * 255), 0, 255).astype(np.uint8)
            h, w = img.shape[:2]
            self.slots[view_i].reshape(h, self.scaling_factor, w, self.scaling_factor, 3)[:] = img.reshape(h, 1, w, 1, 3)
