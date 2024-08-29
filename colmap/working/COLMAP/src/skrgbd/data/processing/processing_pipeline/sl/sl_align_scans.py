from argparse import ArgumentParser
from pathlib import Path

from PIL import Image
import numpy as np
import open3d as o3d
from pathos.pools import ProcessPool as Pool
from torch.utils.tensorboard import SummaryWriter
import torch

from skrgbd.data.processing.alignment.scan_alignment import Aligner, Scan
from skrgbd.calibration.camera_models import load_from_pt
from skrgbd.utils.logging import logger, tqdm
from skrgbd.data.io.rv_scan import RVScan
from skrgbd.data.io.ply import save_ply
from skrgbd.data.dataset.scene_paths import ScenePaths

from s2dnet.s2dnet import S2DNet


def main():
    description = 'Aligns reference partial SL scans'
    parser = ArgumentParser(description=description)
    parser.add_argument('--aux-dir', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--raw-dir', type=str, required=True)
    parser.add_argument('--raw-calib-dir', type=str, required=True)
    parser.add_argument('--fe-weights-pth', type=str, required=True)
    parser.add_argument('--scene-name', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--workers-n', type=int)
    args = parser.parse_args()

    f'SL align {args.scene_name}' >> logger.debug
    scene_paths = ScenePaths(args.scene_name, data_dir=args.data_dir,
                             raw_dir=args.raw_dir, aux_dir=args.aux_dir, raw_calib_dir=args.raw_calib_dir)
    align_scans(scene_paths, args.fe_weights_pth,
                f'{args.log_dir}/{args.scene_name}', device=args.device, workers_n=args.workers_n)


def align_scans(
        scene_paths, fe_weights_pth, log_dir, device='cpu', dtype=torch.float, precomp_device='cuda',
        scan_ids=None, perturbs_n_per_scan=10, seed=481953483, workers_n=None,
):
    r"""Aligns reference partial SL scans.

    Parameters
    ----------
    scene_paths : ScenePaths
    fe_weights_pth : str
    log_dir : str
    device : torch.device
        The device used for optimization.
        The closest points are searched on CPU,
        the feature maps are stored in RAM since they do not fit on GPU for many scans,
        so optimization on GPU is usually slower.
    dtype : torch.dtype
        The dtype used for optimization.
    precomp_device : torch.device
        The device used for the initial feature extraction and optimization of camera projection model.
    scan_ids : iterable of int
    perturbs_n_per_scan : int
    seed : int
    workers_n : int
    """
    cam_name = 'stl_right'  # use the intrinsics and images from the right SL camera
    if scan_ids is None:
        scan_ids = range(27)

    f'Random seed {seed}' >> logger.debug
    np.random.seed(seed)
    torch.manual_seed(seed)

    'Load scan data' >> logger.debug
    def load_scan_data(scan_i):
        # 'Load scan' >> logger.debug
        scan = RVScan(scene_paths.sl_raw(), scan_i)
        scan = scan.to_board().to_meters()
        scan = scan.mesh
        verts = np.asarray(scan.vertices)
        tris = np.asarray(scan.triangles)

        # 'Load image' >> logger.debug
        img = scene_paths.img(cam_name, 'partial', scan_i, 'maxwhite_00_0000', 'raw')
        img = Image.open(img)
        img = np.asarray(img).copy()
        img = torch.from_numpy(img)
        img = torch.atleast_3d(img).permute(2, 0, 1).to(dtype).contiguous().div(255)
        return scan_i, verts, tris, img

    scan_data = dict()
    with Pool(workers_n) as pool:
        for scan_i, verts, tris, img in tqdm(pool.uimap(load_scan_data, scan_ids), total=len(scan_ids)):
            scan_data[scan_i] = (verts, tris, img)

    'Load camera model' >> logger.debug
    cam_model = scene_paths.cam_model(cam_name, 'rgb', 'pt')
    cam_model = load_from_pt(cam_model).to(device, dtype)

    'Init feature extractor' >> logger.debug
    fe = S2DNet(precomp_device, checkpoint_path=fe_weights_pth)
    fe = fe.eval().requires_grad_(False).to(precomp_device, dtype)

    scans = []
    'Extract features and setup scans' >> logger.debug
    for scan_i in tqdm(scan_ids):
        verts, tris, img = scan_data[scan_i]

        # 'Extract features' >> logger.debug
        feat_maps = fe(img.to(precomp_device).expand(3, -1, -1).unsqueeze(0)); del img
        feat_maps = [feat_map.squeeze(0).to(device) for feat_map in feat_maps]

        # 'Setup scan data' >> logger.debug
        scan = Scan(verts, tris, cam_model, cam_model.board_to_camera, feat_maps); del verts, tris, feat_maps
        scans.append(scan); del scan
    del scan_data, fe, cam_model
    torch.cuda.empty_cache()

    'Load initial scan aligment' >> logger.debug
    scan0_to_world = scene_paths.raw_calib.scan0_to_world
    scan0_to_world = torch.load(scan0_to_world)
    sl_right_poses = scene_paths.raw_calib.sl_right_poses
    sl_right_poses = torch.load(sl_right_poses)
    scan_to_w_init = sl_right_poses @ scan0_to_world; del scan0_to_world, sl_right_poses
    scan_to_w_init = scan_to_w_init[scan_ids]
    w_to_scan_init = scan_to_w_init.inverse().to(device, dtype)

    'Init aligner' >> logger.debug
    aligner = Aligner(scans); del scans
    aligner.set_transforms(w_to_scan_init); del w_to_scan_init

    'Pick alignment parameters' >> logger.debug
    aligner.samples_n = 2 ** 9
    aligner.max_dist_mean = 1e-2
    aligner.set_optimal_params(t_mag=1e-5, perturbs_n_per_scan=perturbs_n_per_scan)

    'Align' >> logger.debug
    start_iter_i = 0
    for stage_i, params in enumerate([
        dict(optim='adam',  lr=1e-4, iters_n=50,  samples_n=2**9,  max_dists=dict(sdf=3e-3, feats_0=3e-3, feats_1=3e-3), target_term_vals=dict(sdf=1/2, feats_0=0, feats_1=1/2)),
        dict(optim='adam',  lr=1e-4, iters_n=150, samples_n=2**9,  max_dists=dict(sdf=1e-3, feats_0=1e-3, feats_1=1e-3), target_term_vals=dict(sdf=1/2, feats_0=1/2, feats_1=0)),
        dict(optim='lbfgs', lr=1,    iters_n=3,   samples_n=2**14, max_dists=dict(sdf=1e-3, feats_0=1e-3, feats_1=1e-3), target_term_vals=dict(sdf=1/2, feats_0=1/2, feats_1=0)),
    ]):
        log_writer = SummaryWriter(f'{log_dir}/stage_{stage_i}', flush_secs=5)
        aligner.samples_n = params['samples_n']
        aligner.max_dists = params['max_dists']
        aligner.target_weighted_term_vals = params['target_term_vals']
        aligner.align(log_writer, optim=params['optim'],
                      lr=params['lr'], iters_n=params['iters_n'], start_iter_i=start_iter_i)
        log_writer.close()
        start_iter_i += params['iters_n']

    'Transform scans back to original global space' >> logger.debug
    ref_scan_i = 0
    w_to_scan_refined = aligner.get_transforms('cpu', torch.double)
    w_to_w_init = scan_to_w_init[ref_scan_i].double() @ w_to_scan_refined[ref_scan_i]; del scan_to_w_init
    scan_to_w_refined = [w_to_w_init @ w2s.inverse() for w2s in w_to_scan_refined]
    del w_to_w_init, w_to_scan_refined

    'Save refined transforms' >> logger.debug
    scan_to_w_refined_pth = scene_paths.sl_board_to_w_refined()
    Path(scan_to_w_refined_pth).parent.mkdir(parents=True, exist_ok=True)
    torch.save(scan_to_w_refined, scan_to_w_refined_pth)

    'Save aligned scans' >> logger.debug
    for scan_i, scan, scan_to_w in tqdm(list(zip(scan_ids, aligner.scans, scan_to_w_refined))):
        scan = o3d.geometry.TriangleMesh(scan.scan)
        scan.transform(scan_to_w.cpu().numpy())
        verts = np.asarray(scan.vertices)
        tris = np.asarray(scan.triangles)
        scan_ply = scene_paths.sl_part(scan_i)
        Path(scan_ply).parent.mkdir(parents=True, exist_ok=True)
        save_ply(scan_ply, verts, tris)
    'Finished' >> logger.debug

    # Return aligner for debugging
    return aligner


if __name__ == '__main__':
    main()
