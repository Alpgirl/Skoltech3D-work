from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

from skrgbd.data.processing.alignment.image_alignment import Aligner, Camera, FeatureExtractor, init_cams, Scan
from skrgbd.data.processing.processing_pipeline.img_align.configs import configs
from skrgbd.data.processing.processing_pipeline.img_align.helper import Helper
from skrgbd.utils.logging import logger, tqdm
from skrgbd.data.processing.depth_utils.mesh_rendering import MeshRenderer
from skrgbd.data.io.poses import save_poses
from skrgbd.data.dataset.scene_paths import ScenePaths


def main():
    description = 'Aligns RGB images to the SL scan.'
    parser = ArgumentParser(description=description)
    parser.add_argument('--addons-dir', type=str, required=True)
    parser.add_argument('--aux-dir', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--raw-dir', type=str, required=True)
    parser.add_argument('--fe-weights-pth', type=str, required=True)
    parser.add_argument('--cam', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--light', type=str, required=True)
    parser.add_argument('--poses-var', type=str, default='calib')
    parser.add_argument('--scene-name', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    f'RGB align {args.scene_name}' >> logger.debug
    scene_paths = ScenePaths(args.scene_name, addons_dir=args.addons_dir,
                             aux_dir=args.aux_dir, data_dir=args.data_dir, raw_dir=args.raw_dir)
    align_rgb(scene_paths, args.fe_weights_pth, f'{args.log_dir}/{args.scene_name}',
              cam_name=args.cam, mode=args.mode, light=args.light, poses_var=args.poses_var, device=args.device)


def align_rgb(scene_paths, fe_weights_pth, log_dir, cam_name='tis_right', mode='rgb', light='hdr',
              poses_var='calib', view_ids_batches=(range(100),), device='cpu', precomp_device='cuda', dtype=torch.float):
    r"""Aligns RGB images to the SL scan.

    Parameters
    ----------
    scene_paths : ScenePaths
    fe_weights_pth : str
    log_dir : str
    cam_name : str
    mode : str
    light : str
    poses_var : {'calib', 'ref'}
    view_ids_batches : iterable of iterable of int
    device : torch.device
    precomp_device : torch.device
    dtype : torch.dtype
    """
    f'Load config for {cam_name}.{mode}' >> logger.debug
    config = configs[cam_name, mode]

    f'Random seed {config.seed}' >> logger.debug
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    'Init feature extractor' >> logger.debug
    fe = FeatureExtractor(precomp_device, fe_weights_pth).to(precomp_device, dtype)

    'Load reference cams' >> logger.debug
    helper = Helper(scene_paths)
    cams, imgs = load_cam_data(helper, var='ref', device=precomp_device, dtype=dtype, **config.load_cam_data_ref)

    'Load scan data' >> logger.debug
    scan, verts = load_sl_data(helper, cams, imgs, fe, precomp_device, dtype, **config.load_sl_data); del cams, imgs, fe
    scan = scan.to(device)
    verts = verts.T.to(device, dtype)

    'Init aligner' >> logger.debug
    aligner = Aligner(scan, **config.aligner); del scan
    aligner.term_weights = dict(feat_0=1, feat_1=1, ref_feat_0=1, ref_feat_1=1)

    def empty_cache():
        for d in precomp_device, device:
            if torch.device(d).type == 'cuda':
                with torch.cuda.device(d):
                    torch.cuda.empty_cache()

    poses = dict()
    for batch_i, view_ids in enumerate(view_ids_batches):
        'Re-init feature extractor' >> logger.debug
        empty_cache()
        fe = FeatureExtractor(precomp_device, fe_weights_pth).to(precomp_device, dtype)

        f'Load cam data for batch {batch_i}' >> logger.debug
        cams, imgs = load_cam_data(helper, cam_name, mode, light, view_ids,
                                   poses_var, device, dtype, **config.load_cam_data)

        'Init cams for alignment' >> logger.debug
        cams = init_cams(cams, imgs, fe, verts, **config.init_cams); del imgs, fe

        'Align' >> logger.debug
        empty_cache()
        aligner.set_cams(cams)
        log_writer = SummaryWriter(log_dir + f'.{batch_i}.1', flush_secs=5)
        aligner.align(log_writer, optim='adam', **config.adam)
        log_writer.close()
        log_writer = SummaryWriter(log_dir + f'.{batch_i}.2', flush_secs=5)
        aligner.align(log_writer, optim='lbfgs', start_iter_i=config.adam.iters_n, **config.lbfgs)
        log_writer.close()
        aligner.reset_cams()

        with torch.no_grad():
            for view_i, cam in zip(view_ids, cams):
                pose = dict(img_i=view_i + 1)  # COLMAP uses one-based indices
                pose['w2c'] = cam.w2c.to('cpu', torch.double)
                pose['filename'] = Path(scene_paths.img(cam_name, mode, view_i, light, 'undist')).name
                poses[view_i] = pose; del pose
        del cams
    del verts

    images_txt = scene_paths.cam_poses(cam_name, mode, 'ref')
    f'Save refined poses to {images_txt}' >> logger.debug
    poses = [poses[view_i] for view_i in sorted(poses.keys())]
    Path(images_txt).parent.mkdir(exist_ok=True, parents=True)
    save_poses(images_txt, poses)
    'Done' >> logger.debug

    # Return aligner for debugging
    return aligner


def load_cam_data(helper, cam_name, mode, light, view_ids, var, device, dtype, scale_factor=None, scale_mode='bilinear'):
    r"""Loads camera images with the respective camera model and poses.

    Parameters
    ----------
    helper : Helper
    cam_name : str
    mode : str
    light : str
    view_ids : iterable of int
        of shape [cams_n].
    var : str
    device : torch.device
    dtype : torch.dtype
    scale_factor : float
    scale_mode : str

    Returns
    -------
    cams : list of Camera
        of shape [cams_n].
    imgs : list of torch.Tensor
        of shape [cams_n, 3, height, width].
    """
    f'Load cam and poses for {cam_name}.{mode}' >> logger.debug
    cam_model = helper.load_cam_model(cam_name, mode).to(device, dtype)
    w2c = helper.load_cam_poses(cam_name, mode, var).to(device, dtype)

    if scale_factor is not None:
        wh = cam_model.size_wh.mul(scale_factor).round_().to(cam_model.size_wh)
        f'Rescale from {cam_model.size_wh.tolist()} to {wh.tolist()}' >> logger.debug
        cam_model = cam_model.resize_(wh)
        hw = wh.flip(0).tolist(); del wh
        align_corners = False if (mode != 'nearest') else None

    f'Load imgs for {cam_name}.{mode} at {light}' >> logger.debug
    cams = []
    imgs = []
    for view_i in tqdm(view_ids):
        cam = Camera(cam_model, w2c[view_i])
        cams.append(cam); del cam
        img = helper.load_img(cam_name, mode, view_i, light)
        if scale_factor is not None:
            img = torch.nn.functional.interpolate(img.unsqueeze(0), hw,
                                                  mode=scale_mode, align_corners=align_corners).squeeze(0)
        imgs.append(img); del img
    del cam_model, w2c

    return cams, imgs


def load_sl_data(helper, cams, imgs, fe, device, dtype, angle_gamma=4, max_angle_deg=70, occ_thres=1e-3, pad=50, samples_n=2**14):
    r"""Loads the SL scan with the reference features.

    Parameters
    ----------
    helper : Helper
    cams : iterable of Camera
        of shape [cams_n].
    imgs : iterable of torch.Tensor
        of shape [cams_n, 3, height, width].
    fe : FeatureExtractor
    device : torch.device
    dtype : torch.dtype
    angle_gamma : float
    max_angle_deg : float
        Parameters of projection angle-based weighting, see `calc_angle_weight`.
    occ_thres : float
        Scan points below the occluding surface deeper than this are occluded.
    pad : int
        Images are cropped to the bounding box of the ROI points with this padding before feature extraction.
    samples_n : int
        Number of scan points to keep for further optimization.

    Returns
    -------
    scan : Scan
    """
    'Load rec and occ' >> logger.debug
    rec = helper.load_rec()
    rec.compute_vertex_normals()
    verts = torch.from_numpy(np.asarray(rec.vertices)).to(device, dtype)
    normals = torch.from_numpy(np.asarray(rec.vertex_normals)).to(device, dtype); del rec

    occ = helper.load_occ()
    occ_renderer = MeshRenderer(occ); del occ

    'Init scan' >> logger.debug
    scan = Scan(verts, normals, occ_renderer, occ_thres); del normals, occ_renderer
    scan.init_feats_from_cams(cams, imgs, fe, angle_gamma, max_angle_deg, pad); del cams, imgs, fe
    scan.remove_occluded_pts()

    'Downsample scan points' >> logger.debug
    scan.resample(samples_n)
    scan.normalize_feats()

    return scan, verts


if __name__ == '__main__':
    main()
