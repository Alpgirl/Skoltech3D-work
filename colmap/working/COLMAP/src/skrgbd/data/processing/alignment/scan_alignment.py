import numpy as np
import open3d as o3d
import roma
from torch.utils.tensorboard import SummaryWriter
import torch

from skrgbd.calibration.camera_models.camera_model import CameraModel
from skrgbd.utils.logging import tqdm


class Aligner:
    r"""Implements multi-way alignment of colored scans.

    Parameters
    ----------
    scans : list of Scan

    Notes
    -----
    Implements the following algorithm.
    `make_samples`
    1. Sample points from scan vertices.
    2. Refine their coordinates by averaging their projections onto the scans which are close to the samples.

    `collect_sample_feats`
    3. For each sample, calculate the SDF w.r.t to each scan, and the CNN features from the respective photo.

    `calc_loss_terms`
    4. For each sample, calculate the variances of the collected SDF and features,
       using only the scans close to the sample.

    `align`
    5. Calculate the loss value as the weighed sum of the variances, and optimize it iteratively.
    """

    term_labels = ['sdf', 'feats_0', 'feats_1']

    def __init__(self, scans):
        self.scans = scans

        self.fixed_transform_ids = None
        self.w_to_scan_t = None
        self.w_to_scan_rotq = None
        self.params = None
        self.samples = None
        self.sample_normals = None

        self.samples_n = 2 ** 14  # used in `make_samples`
        self.max_dist_mean = 1e-2  # used in `make_samples`
        self.max_dists = dict(sdf=3e-3, feats_0=3e-3, feats_1=3e-3)  # used in `calc_loss_terms`
        self.term_consts = dict(sdf=1, feats_0=1, feats_1=1)
        self.target_weighted_term_vals = dict(sdf=1/3, feats_0=1/3, feats_1=1/3)

    # Transforms
    # ----------
    def set_transforms(self, w_to_scan, fixed_tansform_ids=None):
        r"""Sets initial transformation matrices for the scans.

        Parameters
        ----------
        w_to_scan: iterable of torch.Tensor
            of shape [scans_n, 4, 4].
        fixed_tansform_ids : set of int
            Indices of transforms that will not be optimized.
        """
        if fixed_tansform_ids is None:
            fixed_tansform_ids = set()
        self.fixed_transform_ids = fixed_tansform_ids
        self.w_to_scan_t = []
        self.w_to_scan_rotq = []
        self.params = []

        for i, w_to_scan_i in enumerate(w_to_scan):
            w_to_scan_i_t = w_to_scan_i[:3, 3].detach().clone()
            self.w_to_scan_t.append(w_to_scan_i_t)

            w_to_scan_i_rotm = w_to_scan_i[:3, :3].detach().clone()
            w_to_scan_i_rotq = roma.rotmat_to_unitquat(w_to_scan_i_rotm); del w_to_scan_i_rotm
            self.w_to_scan_rotq.append(w_to_scan_i_rotq)

            if i not in fixed_tansform_ids:
                w_to_scan_i_t.requires_grad_()
                w_to_scan_i_rotq.requires_grad_()
                self.params.extend([w_to_scan_i_t, w_to_scan_i_rotq])

    def get_transforms(self, device=None, dtype=None):
        r"""Returns transformation matrices for the scans.

        Returns
        -------
        w_to_scan: list of torch.Tensor
            of shape [scans_n, 4, 4]
        """
        if device is None:
            device = self.w_to_scan_t[0].device
        if dtype is None:
            dtype = self.w_to_scan_t[0].dtype
        w_to_scan = []
        for w_to_scan_i_t, w_to_scan_i_rotq in zip(self.w_to_scan_t, self.w_to_scan_rotq):
            w_to_scan_i = w_to_scan_i_t.new_zeros(4, 4, device=device, dtype=dtype)
            w_to_scan_i[3, 3] = 1
            w_to_scan_i[:3, 3].copy_(w_to_scan_i_t.detach())
            w_to_scan_i[:3, :3].copy_(roma.unitquat_to_rotmat(w_to_scan_i_rotq.detach()))
            w_to_scan.append(w_to_scan_i)
        return w_to_scan

    def perturb_translations(self, t_mag):
        r"""Adds a random displacement of a fixed length to each non-fixed scan transform.

        Parameters
        ----------
        t_mag : float
            The length of the displacement.
        """
        scans_n = len(self.w_to_scan_t)
        device = self.w_to_scan_t[0].device
        dtype = self.w_to_scan_t[0].dtype

        t_dirs = torch.randn(scans_n, 3, device=device, dtype=dtype)
        t_dirs = torch.nn.functional.normalize(t_dirs, dim=1)
        rand_t = t_dirs * t_mag; del t_dirs

        for i, w2s_t in enumerate(self.w_to_scan_t):
            if i in self.fixed_transform_ids:
                continue
            w2s_t.data.add_(rand_t[i])

    def normalize_rotq(self):
        r"""Normalizes the rotation quaternions."""
        for rotq in self.w_to_scan_rotq:
            rotq.data.copy_(torch.nn.functional.normalize(rotq.data, dim=0))

    # Alignment
    # ---------
    def align(self, log_writer, optim='adam', lr=1e-4, iters_n=100, start_iter_i=0):
        r"""Runs the alignment iterations.

        Parameters
        ----------
        log_writer : SummaryWriter
            Object to log loss terms into. Must have add_scalar(tag: str, value: float, step: int) method.
        optim : {'adam', 'lbfgs'}
            Optimizer to use: faster 1st order Adam or slower 2nd order L-BFGS.
        lr : float
            Learning rate.
        iters_n : int
            Number of iterations of alignment.
        start_iter_i : int
            The starting step value in logs.

        Intrinsic parameters
        --------------------
        term_weights : dict
            sdf: float
            feats_0: float
            feats_1: float
                Weights of the loss terms.
        """
        if optim == 'adam':
            optim = torch.optim.Adam(self.params, lr=lr)
        elif optim == 'lbfgs':
            optim = torch.optim.LBFGS(self.params, lr=lr, line_search_fn='strong_wolfe',
                                      tolerance_change=1e-15, tolerance_grad=1e-15)
        prog_bar = tqdm(range(start_iter_i, start_iter_i + iters_n))
        iter_i = None
        log_scalars = [True]

        def closure():
            self.normalize_rotq()
            self.make_samples()
            terms = self.calc_loss_terms()
            weighted_terms = {label: term * self.term_weights[label] for (label, term) in terms.items()}
            loss = sum(weighted_terms.values())
            optim.zero_grad()
            loss.backward()

            prog_bar.set_description(f'Loss {loss:.2e}')
            if log_scalars[0]:
                log_writer.add_scalar('loss', loss.item(), iter_i)
                for label in terms.keys():
                    log_writer.add_scalar(label, terms[label].item(), iter_i)
                    log_writer.add_scalar(f'w_{label}', weighted_terms[label].item(), iter_i)
                log_scalars[0] = False
            return loss

        for iter_i in prog_bar:
            log_scalars = [True]
            optim.step(closure)
        self.normalize_rotq()

    @torch.no_grad()
    def make_samples(self, make_normals=False):
        r"""Samples points near scan surfaces, which are used for calculation of loss terms.

        Intrinsic parameters
        --------------------
        samples_n : int
            Number of points to sample.
        max_dist_mean : float
            We start with samples from vertices of all scans and then recalculate them
            as their mean projection to all scans, closer to the initial sample than this.

        Parameters
        ----------
        make_normals : bool
        """
        scans_n = len(self.scans)

        # Sample points
        samples_per_scan_n = self.samples_n // scans_n
        pts = []
        for scan, w2s_rotq, w2s_t in zip(self.scans, self.w_to_scan_rotq, self.w_to_scan_t):
            s2w_rotm = roma.unitquat_to_rotmat(roma.quat_conjugation(w2s_rotq)); del w2s_rotq
            samples_on_scan = scan.sample_pts_uniformly(samples_per_scan_n); del scan
            samples_on_scan = samples_on_scan.to(s2w_rotm)
            samples_on_scan = (samples_on_scan - w2s_t) @ s2w_rotm.T; del w2s_t, s2w_rotm
            pts.append(samples_on_scan); del samples_on_scan
        pts = torch.cat(pts, 0)

        pts_n = len(pts)
        pt_sup = pts.new_zeros(pts_n)
        refined_pts = torch.zeros_like(pts)
        if make_normals:
            refined_normals = torch.zeros_like(pts)

        for scan, w2s_rotq, w2s_t in zip(self.scans, self.w_to_scan_rotq, self.w_to_scan_t):
            # Transform pts to scan space
            w2s_rotm = roma.unitquat_to_rotmat(w2s_rotq); del w2s_rotq
            pts_s = pts @ w2s_rotm.T + w2s_t

            # Find closest points
            closest_pts_s, closest_normals_s = scan.find_closest(pts_s.to(scan.normals)); del scan, pts_s
            closest_pts_s = closest_pts_s.to(pts)
            closest_pts = (closest_pts_s - w2s_t) @ w2s_rotm; del closest_pts_s, w2s_t
            if make_normals:
                closest_normals_s = closest_normals_s.to(pts)
                closest_normals = closest_normals_s @ w2s_rotm
            del closest_normals_s, w2s_rotm

            # Keep only the close points
            dists = (pts - closest_pts).norm(dim=1)
            is_close = dists <= self.max_dist_mean; del dists
            pt_sup[is_close] += 1
            refined_pts[is_close] += closest_pts[is_close]; del closest_pts
            if make_normals:
                refined_normals[is_close] += closest_normals[is_close]; del closest_normals
            del is_close
        assert (pt_sup > 0).all()  # each sample is a vertex of some scan, so it has to be close to at least one scan
        del pts

        refined_pts /= pt_sup.unsqueeze(1)
        self.samples = refined_pts
        if make_normals:
            refined_normals /= pt_sup.unsqueeze(1); del pt_sup
            refined_normals = torch.nn.functional.normalize(refined_normals, dim=1)
            self.sample_normals = refined_normals

    def calc_loss_terms(self):
        r"""Calculates the loss terms, as squared_resid_norm.max_clamp(max_squared_norm) / (scans_n * samples_n).
        For each scan, uses only the samples close to the scan.

        Intrinsic parameters
        --------------------
        max_dists : dict
            sdf: float
            feats_0: float
            feats_1: float
                Max displacement at which the term should 'start working'.
        max_norms2 : dict
            sdf: float
            feats_0: float
            feats_1: float
                Max squared norm of each feature.

        Returns
        -------
        terms : dict
            sdf: float
            feats_0: float
            feats_1: float
        """
        feats_per_term = self.collect_sample_feats()
        terms = dict()
        for label in self.term_labels:
            is_close = feats_per_term['dist'].squeeze(1) <= self.max_dists[label]
            feats = feats_per_term[label]
            feats = feats.where(is_close.unsqueeze(1), feats.new_tensor(0))
            mean_feats = feats.sum(0) / is_close.sum(0)
            resids = feats - mean_feats; del feats, mean_feats
            sq_resids = resids.pow(2).sum(1).clamp(max=self.max_norms2[label]); del resids
            sq_resids = sq_resids.where(is_close, sq_resids.new_tensor(0))
            terms[label] = (sq_resids.sum(1) / is_close.sum(1)).mean(); del sq_resids, is_close
        del feats_per_term
        return terms

    def collect_sample_feats(self, div_eps=1e-12):
        r"""Collects sample features from scans.

        Returns
        -------
        feats_per_term: dict
            dists: torch.Tensor
                of shape [scans_n, 1, samples_n].
            sdf: torch.Tensor
                of shape [scans_n, 1, samples_n].
            feats_0: torch.Tensor
                of shape [scans_n, 128, samples_n].
            feats_1: torch.Tensor
                of shape [scans_n, 128, samples_n].
        """
        scans_n = len(self.scans)
        pts = self.samples

        # Init stats
        feats_per_term = dict()
        pts_n = len(pts)
        for label, channels_n in zip(['dist'] + self.term_labels,
                                     [1, 1] + [len(feat_map) for feat_map in self.scans[0].feat_maps]):
            feats_per_term[label] = pts.new_zeros(scans_n, channels_n, pts_n)

        # Calculate stats
        for scan_i, scan, w2s_rotq, w2s_t in zip(range(scans_n), self.scans, self.w_to_scan_rotq, self.w_to_scan_t):
            # Transform pts to scan space
            w2s_rotq = torch.nn.functional.normalize(w2s_rotq, dim=0)
            w2s_rotm = roma.unitquat_to_rotmat(w2s_rotq); del w2s_rotq
            pts_s = pts @ w2s_rotm.T + w2s_t; del w2s_rotm, w2s_t

            # Find closest points and normals there
            with torch.no_grad():
                closest_pts, closest_normals = scan.find_closest(pts_s.detach().to(scan.normals))
                closest_pts = closest_pts.to(pts_s)
                closest_normals = closest_normals.to(pts_s)

            # Calculate distances
            to_pts = pts_s - closest_pts; del closest_pts

            with torch.no_grad():
                # Accumulate distances
                to_pt_dists = to_pts.norm(dim=1)
                feats_per_term['dist'][scan_i, 0] = to_pt_dists

                # Substitute normals
                to_pt_dirs = to_pts / (to_pt_dists + div_eps).unsqueeze(1); del to_pt_dists
                angle_cos = (to_pt_dirs.unsqueeze(1) @ closest_normals.unsqueeze(2)).squeeze(2).squeeze(1)
                closest_normals = to_pt_dirs * angle_cos.sign().unsqueeze(1); del to_pt_dirs, angle_cos

            # Calculate SDF
            sdfs = (to_pts.unsqueeze(1) @ closest_normals.unsqueeze(2)).squeeze(2).squeeze(1)
            del to_pts, closest_normals
            feats_per_term['sdf'][scan_i, 0] = sdfs; del sdfs

            # Get feats
            feats_at_scales = scan.get_feats(pts_s.T); del scan, pts_s
            for label, feats in feats_at_scales.items():
                feats_per_term[label][scan_i] = feats; del feats
            del feats_at_scales
        del pts
        return feats_per_term

    @torch.no_grad()
    def set_optimal_params(self, t_mag=1e-5, perturbs_n_per_scan=1, scan_ids=None, replicas_n=None):
        r"""Calculates the optimal parameters of the algorithm.

        Parameters
        ----------
        t_mag : float
            The value of the random displacement.
        perturbs_n_per_scan : int
            Number of random displacement samples per scan.
        scan_ids : iterable of int
            Ids of the scans to use for estimation.
            Default: all scans.
        replicas_n : int
            Number of replicas per scan.
            Default: the number of scans.

        Notes
        -----
        Uses the following parameter estimation algorithm.
        1. Keep only one scan and replicate it several times.
        2. Apply a small random displacement to each replica,
           and calculate the values of loss terms for this known displacement,
        3. Estimate the zero-neighborhood dependency of loss terms w.r.t the displacement value.
           For a better estimate, average the values over multiple random displacements and different kept scans.
        4. Based on the expected values of the loss terms at the displacement of `max_dists`,
           calculate the inlier-outlier thresholds `max_norms2`,
           and the loss term weights.
        """
        if scan_ids is None:
            scan_ids = range(len(self.scans))
        if replicas_n is None:
            replicas_n = len(self.scans)

        # Store original scans, transforms, and parameters
        orig_scans = self.scans
        orig_w_to_scan = self.get_transforms()
        orig_fixed_transform_ids = self.fixed_transform_ids
        max_dist_mean, max_dists = self.max_dist_mean, self.max_dists

        # Relax all thresholds
        self.max_dist_mean = 1e10
        self.max_dists = {label: 1e10 for label in self.term_labels}
        self.term_consts = {label: 1 for label in self.term_labels}

        # Each term is expected to depend on the misalignment value as C * disp^2.
        # Estimate the constants C by replicating each scan multiple times,
        # perturbing transforms for the replicas with a known displacement, and calculating the term values
        terms = {label: [] for label in self.term_labels}
        for scan_i in scan_ids:
            self.scans = [orig_scans[scan_i]] * replicas_n
            for iter_i in range(perturbs_n_per_scan):
                self.set_transforms([orig_w_to_scan[scan_i]] * replicas_n)
                self.perturb_translations(t_mag)
                self.make_samples()
                terms_sample = self.calc_loss_terms()
                for k, v in terms_sample.items():
                    terms[k].append(v.item())
                del terms_sample
        terms = {k: np.mean(v) for (k, v) in terms.items()}

        # Return the original scans, transforms, and thresholds
        self.scans = orig_scans
        self.set_transforms(orig_w_to_scan, orig_fixed_transform_ids)
        self.max_dist_mean = max_dist_mean
        self.max_dists = max_dists

        # Set term consts
        self.term_consts = {label: term / (t_mag ** 2) for (label, term) in terms.items()}

    @property
    def max_norms2(self):
        return {label: term_const * self.max_dists[label] ** 2 for (label, term_const) in self.term_consts.items()}

    @property
    def term_weights(self):
        return {label: self.target_weighted_term_vals[label] / (term_const * self.max_dists['sdf'] ** 2)
                for (label, term_const) in self.term_consts.items()}


class Scan:
    r"""A helper scan class.

    Parameters
    ----------
    verts : np.ndarray
        of shape [verts_n, 3].
    tris : np.ndarray
        of shape [tris_n, 3].
    cam_model : CameraModel
    scan_to_cam : torch.Tensor
        of shape [4, 4].
    feat_maps : iterable of torch.Tensor
        of shape [maps_n, feat_channels_n, h, w].
    """
    def __init__(self, verts, tris, cam_model, scan_to_cam, feat_maps):
        verts = o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64))
        tris = o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32))
        scan = o3d.geometry.TriangleMesh(verts, tris); del verts, tris
        scan.compute_vertex_normals()
        self.scan = scan

        mesh = o3d.t.geometry.TriangleMesh.from_legacy(scan)
        self.raycasting = o3d.t.geometry.RaycastingScene()
        self.raycasting.add_triangles(mesh)

        verts = np.asarray(scan.vertices)
        self.verts = torch.from_numpy(verts)

        normals = mesh.triangle['normals'].numpy()
        self.normals = torch.from_numpy(normals)

        self.cam_model = cam_model
        self.scan_to_cam = scan_to_cam
        self.feat_maps = feat_maps

    def sample_pts_uniformly(self, pts_n):
        r"""Samples scan vertices quazi-randomly.

        Parameters
        ----------
        pts_n : int
            Number of samples.

        Returns
        -------
        pts : torch.DoubleTensor
            of shape [pts_n, 3].
        """
        verts_n = len(self.verts)
        device = self.verts.device

        interval_bounds = torch.linspace(0, verts_n, pts_n + 1, device=device).round().long()
        inter_starts = interval_bounds[:-1]
        inter_sizes = interval_bounds[1:] - inter_starts; del interval_bounds
        rand_hi = inter_sizes.max()
        ids = torch.randint(0, rand_hi, [pts_n], device=device).remainder(inter_sizes) + inter_starts
        del inter_starts, inter_sizes
        pts = self.verts[ids]
        return pts

    def find_closest(self, pts):
        r"""Finds the closest points on the surface and the normals at these points.

        Parameters
        ----------
        pts : torch.FloatTensor
            of shape [pts_n, 3]. Points in the scan space.

        Returns
        -------
        closest_pts : torch.FloatTensor
            of shape [pts_n, 3]. Inf if closest point is not found.
        normals : torch.FloatTensor
            of shape [pts_n, 3]. Undefined if closest point is not found.
        """
        result = self.raycasting.compute_closest_points(pts.numpy())
        closest_pts = result['points']
        tri_ids = result['primitive_ids']; del result

        closest_pts = torch.from_numpy(closest_pts.numpy())
        tri_ids = torch.from_numpy(tri_ids.numpy().astype(np.int64))
        normals = self.normals[tri_ids.clamp_(max=len(self.normals) - 1)]; del tri_ids
        return closest_pts, normals

    def get_feats(self, pts):
        r"""Retrieves CNN features at the points.

        Parameters
        ----------
        pts : torch.Tensor
            of shape [3, pts_n]. Points in the scan space.

        Returns
        -------
        feats : dict
            'feats_0': torch.Tensor
            'feats_1': torch.Tensor
                of shape [feat_channels_n, pts_n].
        """
        pts_cam = self.scan_to_cam[:3, :3] @ pts + self.scan_to_cam[:3, 3:4]; del pts
        uv = self.cam_model.project_fine(pts_cam); del pts_cam
        uv = (uv.T / (self.cam_model.size_wh.to(uv) / 2)).sub_(1)
        uv = uv.where(uv.isfinite(), uv.new_tensor(-2))

        all_feats = dict()
        for label, feat_map in zip(['feats_0', 'feats_1'], self.feat_maps):
            feat_map = feat_map.to(uv)
            feats = torch.nn.functional.grid_sample(feat_map.unsqueeze(0), uv.unsqueeze(0).unsqueeze(1), mode='bilinear', align_corners=False)
            feats = feats.squeeze(2).squeeze(0); del feat_map
            all_feats[label] = feats; del feats
        del uv
        return all_feats
