import numpy as np
import open3d as o3d
import roma
import torch

from skrgbd.utils.logging import logger, tqdm
from skrgbd.utils.math import mat2trotq
from skrgbd.data.processing.depth_utils.mesh_rendering import MeshRenderer
from skrgbd.data.processing.depth_utils.occluded_mesh_rendering import MeshRenderer as OccMeshRenderer

from s2dnet.s2dnet import S2DNet


class Camera(torch.nn.Module):
    r"""Represents a camera with intrinsic camera model, position, and feature maps.

    Parameters
    ----------
    cam_model : CameraModel
    w2c: torch.Tensor
        of shape [4, 4], world-to-camera transformation matrix.
    feat_maps : dict
        label: torch.Tensor
            of shape [feat_channels_n, h, w].
    requires_grad : bool

    Attributes
    ----------
    w2c_t : torch.Tensor
        of shape [3], world-to-camera translation vector.
    w2c_rotq : torch.Tensor
        of shape [4], world-to-camera rotation quaternion, in xyzw notation.
    w2c_rotm : torch.Tensor
        of shape [3, 3], world-to-camera rotation matrix.
    w2c : torch.Tensor
        of shape [4, 4], world-to-camera transformation matrix.
    c2w_t : torch.Tensor
        of shape [3], camera-to-world translation vector.
    c2w_rotm : torch.Tensor
        of shape [3, 3], camera-to-world rotation matrix.
    """
    def __init__(self, cam_model, w2c, feat_maps=None, requires_grad=False):
        super().__init__()
        self.cam_model = cam_model

        w2c_t, w2c_rotq = mat2trotq(w2c.detach())
        self.w2c_t = torch.nn.Parameter(w2c_t.clone(), requires_grad)
        self.w2c_rotq = torch.nn.Parameter(w2c_rotq, requires_grad)

        self.feat_maps = feat_maps if (feat_maps is not None) else dict()

    @property
    def w2c_rotm(self):
        return roma.unitquat_to_rotmat(self.w2c_rotq)

    @property
    def w2c(self):
        w2c = torch.eye(4, device=self.w2c_t.device, dtype=self.w2c_t.dtype)
        w2c[:3, :3].copy_(self.w2c_rotm)
        w2c[:3, 3].copy_(self.w2c_t)
        return w2c

    @property
    def c2w_rotm(self):
        return self.w2c_rotm.T

    @property
    def c2w_t(self):
        return self.c2w_rotm @ (-self.w2c_t)

    def get_feats(self, uv, label, mode='bilinear'):
        r"""Samples features from the feature map.

        Parameters
        ----------
        uv : torch.Tensor
            of shape [n, 2]. The origin of the image space is in the top-left corner of the top-left pixel:
            the coordinates of the top-left corner are (-1,-1), the coordinates of the bottom-right corner are (1,1).
        label : str
        mode : str

        Returns
        -------
        feats : torch.Tensor
            of shape [feat_channels_n, n].
        """
        align_corners = False if (mode != 'nearest') else None
        feat_map = self.feat_maps[label].to(uv)
        feats = torch.nn.functional.grid_sample(feat_map.unsqueeze(0), uv.unsqueeze(0).unsqueeze(1),
                                                mode=mode, align_corners=align_corners).squeeze(2).squeeze(0)
        del feat_map
        return feats


class Aligner:
    r"""Implements alignment of multiple images to a textured surface via maximization of photometric consistency
    between each image and the surface, and between different images themselves.

    Parameters
    ----------
    scan : Scan
    angle_gamma : float
    max_angle_deg : float
        Parameters of projection angle-based weighting, see `calc_angle_weight`.

    Attributes
    ----------
    term_weights : dict
        label: float
            Set this attribute manually. If scan.feats and cams[i].feat_maps contain feature label 'feat_x',
            then `term_weights` must contain weights for 'feat_x' and 'ref_feat_x'.
    """
    def __init__(self, scan, angle_gamma=4, max_angle_deg=70):
        self.scan = scan
        self.angle_gamma = angle_gamma
        self.max_angle_deg = max_angle_deg

        self.cams = None
        self.params = None
        self.term_weights = None

    def set_cams(self, cams):
        r"""
        Parameters
        ----------
        cams : list of Camera
        """
        self.cams = cams
        self.params = []
        for cam in cams:
            self.params.extend([cam.w2c_t, cam.w2c_rotq])

    def reset_cams(self):
        self.cams = None
        self.params = None

    def normalize_rotq(self):
        r"""Normalizes the rotation quaternions."""
        for cam in self.cams:
            rotq = cam.w2c_rotq
            rotq.data.copy_(torch.nn.functional.normalize(rotq.data, dim=0))

    # Alignment
    # ---------
    def align(self, log_writer, optim='adam', lr=1e-4, iters_n=100, start_iter_i=0):
        r"""Runs the alignment using Adam optimizer.

        Parameters
        ----------
        log_writer : SummaryWriter
            Logging interface, must have add_scalar(tag: str, value: float, step: int) method.
        optim : {'adam', 'lbfgs'}
        lr : float
            Learning rate.
        iters_n : int
            Number of iterations of alignment.
        start_iter_i : int
            The starting step value in logs.
        """
        if optim == 'adam':
            optim = torch.optim.Adam(self.params, lr=lr)
        elif optim == 'lbfgs':
            optim = torch.optim.LBFGS(self.params, lr=lr, line_search_fn='strong_wolfe',
                                      tolerance_change=1e-15, tolerance_grad=1e-15)
        prog_bar = tqdm(range(start_iter_i, start_iter_i + iters_n))
        iter_i = None
        log_scalars = [False]

        def closure():
            self.normalize_rotq()
            loss, terms, weighted_terms = self.calc_loss()
            optim.zero_grad()
            loss.backward()

            prog_bar.set_description(f'Loss {loss:.2e}')
            if log_scalars[0]:
                log_writer.add_scalar('loss', float(loss), iter_i)
                for label in terms.keys():
                    log_writer.add_scalar(label, float(terms[label]), iter_i)
                    log_writer.add_scalar(f'w_{label}', float(weighted_terms[label]), iter_i)
                log_scalars[0] = False
            return loss

        for iter_i in prog_bar:
            log_scalars[0] = True
            optim.step(closure)
        self.normalize_rotq()

    def calc_loss(self):
        r"""Calculates the loss value as weighted sum of loss terms.

        Returns
        -------
        loss : torch.Tensor
        terms : dict
        weighed_terms : dict
        """
        terms = self.calc_loss_terms()
        weighted_terms = {label: term * self.term_weights[label] for (label, term) in terms.items()}
        loss = sum(weighted_terms.values())
        return loss, terms, weighted_terms

    def calc_loss_terms(self, div_eps=1e-8):
        r"""Calculates the loss terms as 1 - (weighted_cosine_similarity / samples_n) / cams_n.

        Returns
        -------
        terms : dict
            feat_0: torch.Tensor
            feat_1: torch.Tensor
                Cosine distance from the average features across cameras.
            ref_feat_0: torch.Tensor
            ref_feat_1: torch.Tensor
                Cosine distance from the reference features.
        """
        feats_per_term = self.collect_sample_feats()
        terms = dict()
        for label in feats_per_term:
            feats = feats_per_term[label]  # cams_n, channels_n, samples_n
            norms = feats.norm(dim=1)  # cams_n, samples_n
            is_valid = norms != 0
            mean_feats = feats.sum(0).div_(norms.sum(0).add_(div_eps))  # channels_n, samples_n
            mean_feats = torch.nn.functional.normalize(mean_feats, dim=0)
            weight_sums = norms.sum(1).add_(div_eps); del norms
            for tgt_feats, term_label in [(mean_feats, label), (self.scan.feats[label], f'ref_{label}')]:
                weighted_cos = torch.einsum('ijk,jk->ik', feats, tgt_feats)  # cams_n, samples_n
                weighted_means = weighted_cos.where(is_valid, weighted_cos.new_tensor(0)).sum(1).div_(weight_sums)
                del weighted_cos
                terms[term_label] = 1 - weighted_means.mean(); del weighted_means
            del feats, is_valid, mean_feats, weight_sums
        del feats_per_term
        return terms

    def collect_sample_feats(self):
        r"""Collects sample features from images.

        Returns
        -------
        feats: dict
            label: torch.Tensor
                of shape [cams_n, channels_n, samples_n].
        """
        pts = self.scan.pts

        # 'Init stats' >> logger.debug
        feats = dict()
        for label, vert_feats in self.scan.feats.items():
            feats[label] = pts.new_zeros(len(self.cams), vert_feats.shape[0], len(pts)); del vert_feats

        # 'Calculate stats' >> logger.debug
        for cam_i, cam in enumerate(self.cams):
            # 'Trace visible verts from cam' >> logger.debug
            with torch.no_grad():
                vis_pts, vis_ids = self.scan.get_vis_pts(pts, cam)

            # 'Transform pts to cam space' >> logger.debug
            w2c_rotq = torch.nn.functional.normalize(cam.w2c_rotq, dim=0)
            w2c_rotm = roma.unitquat_to_rotmat(w2c_rotq); del w2c_rotq
            pts_cam = w2c_rotm @ vis_pts.T + cam.w2c_t.unsqueeze(1); del vis_pts

            # 'Project pts to cam' >> logger.debug
            uvs = cam.cam_model.project_fine(pts_cam); del pts_cam
            with torch.no_grad():
                is_in_bounds = cam.cam_model.uv_is_in_bounds(uvs)
            uvs = uvs[:, is_in_bounds]
            vis_ids = vis_ids[is_in_bounds]; del is_in_bounds

            # 'Calc weights' >> logger.debug
            with torch.no_grad():
                vis_pts = pts[vis_ids]
                vis_normals = self.scan.normals[vis_ids]
                cos = calc_cos(vis_pts, vis_normals, cam); del vis_pts, vis_normals
                weights = calc_angle_weight(cos, self.angle_gamma, self.max_angle_deg); del cos

            # 'Get feats' >> logger.debug
            uvs = cam.cam_model.uv_to_torch(uvs.T)
            for label in feats:
                feats[label][cam_i, :, vis_ids] = cam.get_feats(uvs, label) * weights
            del uvs, vis_ids, weights
        return feats


class Scan:
    r"""Represents a set of points with reference features, and the occluding surface.

    Parameters
    ----------
    pts : torch.Tensor
        of shape [pts_n, 3].
    normals : torch.Tensor
        of shape [pts_n, 3].
    occ_renderer : MeshRenderer
    occ_thres : float
        Scan points below the occluding surface deeper than this are occluded.

    Attributes
    ----------
    feats : dict
        label: torch.Tensor
            of shape [feat_channels_n, pts_n].
    """

    def __init__(self, pts, normals, occ_renderer, occ_thres):
        self.pts = pts
        self.normals = normals
        self.occ_renderer = occ_renderer
        self.occ_thres = occ_thres
        self.feats = None

    def to(self, device=None, dtype=None):
        self.pts = self.pts.to(device, dtype)
        self.normals = self.normals.to(device, dtype)
        for k, v in self.feats.items():
            self.feats[k] = v.to(device, dtype)
        return self

    def init_feats_from_cams(self, cams, imgs, fe, angle_gamma=4, max_angle_deg=70, pad=50):
        r"""Initializes reference point features from posed images.

        Parameters
        ----------
        cams : iterable of Camera
        imgs : iterable of torch.Tensor
            of shape [cams_n, 3, height, width].
        fe : FeatureExtractor
        angle_gamma : float
        max_angle_deg : float
            Parameters of projection angle-based weighting, see `calc_angle_weight`.
        pad : int
            Images are cropped to the bounding box of the ROI points with this padding before feature extraction.
        """
        self.feats = dict()
        tot_feats = dict()
        tot_weights = dict()

        for cam, img in tqdm(list(zip(cams, imgs))):
            # 'Crop to ROI' >> logger.debug
            img, cam_model = crop_to_roi(cam, img.to(self.pts), self.pts.T, pad)
            cam = Camera(cam_model, cam.w2c); del cam_model

            # 'Extract feat maps' >> logger.debug
            feat_maps = fe(img.unsqueeze(0).to(fe.device, fe.dtype)); del img
            feat_labels = []
            for label, feat_map in feat_maps.items():
                cam.feat_maps[label] = feat_map.squeeze(0).to(self.pts); del feat_map
                feat_labels.append(label)
            del feat_maps

            # 'Project points and calculate visibility' >> logger.debug
            uvs, vis_ids = self.project_pts(self.pts, cam)

            # 'Collect point feats' >> logger.debug
            feats = dict()
            uvs = cam.cam_model.uv_to_torch(uvs.T)
            for label in feat_labels:
                feats[label] = cam.get_feats(uvs, label); del cam.feat_maps[label]
            del uvs

            # 'Calculate cam weights' >> logger.debug
            pts = self.pts[vis_ids]
            normals = self.normals[vis_ids]
            cos = calc_cos(pts, normals, cam); del pts, normals, cam
            angle_weights = calc_angle_weight(cos, angle_gamma, max_angle_deg); del cos

            # 'Aggregate point feats' >> logger.debug
            if len(tot_feats) == 0:
                pts_n = len(self.pts)
                for label in feat_labels:
                    channels_n = feats[label].shape[0]
                    tot_feats[label] = feats[label].new_zeros(channels_n, pts_n)
                    tot_weights[label] = feats[label].new_zeros(pts_n)
            for label, feats_l in feats.items():
                feats_l = feats_l.mul_(angle_weights)
                weights = feats_l.norm(dim=0)
                tot_feats[label][:, vis_ids] += feats_l; del feats_l
                tot_weights[label][vis_ids] += weights; del weights
            del feats, angle_weights, vis_ids

        # 'Calc the average' >> logger.debug
        for label, feats in tot_feats.items():
            feats = feats.div_(tot_weights[label])
            self.feats[label] = feats; del feats
        del tot_feats, tot_weights

    def project_pts(self, pts, cam):
        r"""Projects points to camera.

        Parameters
        ----------
        pts : torch.Tensor
            of shape [pts_n, 3].
        cam : Camera

        Returns
        -------
        uvs : torch.Tensor
            of shape [2, vis_pts_n].
        vis_ids : torch.Tensor
            of shape [vis_pts_n], ids of the visible points.
        """
        # 'Trace verts from the camera' >> logger.debug
        pts, vis_ids = self.get_vis_pts(pts, cam)

        # 'Project verts to the camera' >> logger.debug
        pts_cam = cam.w2c_rotm @ pts.T + cam.w2c_t.unsqueeze(1); del pts
        uvs = cam.cam_model.project_fine(pts_cam); del pts_cam
        is_in_bounds = cam.cam_model.uv_is_in_bounds(uvs)
        uvs = uvs[:, is_in_bounds]
        vis_ids = vis_ids[is_in_bounds]; del is_in_bounds
        return uvs, vis_ids

    def get_vis_pts(self, pts, cam):
        r"""Filters out points hidden with the occluding surface.

        Parameters
        ----------
        pts : torch.Tensor
            of shape [pts_n, 3].
        cam : Camera

        Returns
        -------
        vis_pts : torch.Tensor
            of shape [vis_pts_n, 3].
        vis_ids : torch.Tensor
            of shape [vis_pts_n], ids of the visible points.
        """
        is_vis = self.occ_renderer.calc_pts_visibility(pts, cam.c2w_t, self.occ_thres)
        vis_ids = is_vis.nonzero(as_tuple=True)[0]; del is_vis
        pts = pts[vis_ids]
        return pts, vis_ids

    def remove_occluded_pts(self):
        r"""Removes points without features."""
        pt_has_feats = self.pts.new_ones(len(self.pts), dtype=torch.bool)
        for feats in self.feats.values():
            pt_has_feats = pt_has_feats.logical_and_(feats.isfinite().all(0))
        for label, feats in self.feats.items():
            self.feats[label] = feats[:, pt_has_feats].contiguous(); del feats
        self.pts = self.pts[pt_has_feats].contiguous()
        self.normals = self.normals[pt_has_feats].contiguous(); del pt_has_feats

    def resample(self, samples_n):
        r"""Downsamples scan points to an exact number, keeping only the farthest points from the already kept,
        effectively doing quazi-uniform resampling.

        Parameters
        ----------
        samples_n : int
            Number of samples to keep.
        """
        pc = o3d.geometry.PointCloud()
        pts = self.pts.to('cpu', torch.double).numpy()
        pc.points = o3d.utility.Vector3dVector(pts)

        ids = np.arange(len(pts), dtype=np.int64)
        fake_cols = np.zeros([len(pts), 3], dtype=np.float64); del pts
        fake_cols[:, 0] = ids.view(np.float64); del ids
        pc.colors = o3d.utility.Vector3dVector(fake_cols); del fake_cols

        pc = pc.farthest_point_down_sample(samples_n)
        ids = np.asarray(pc.colors)[:, 0].view(np.int64); del pc
        ids = torch.from_numpy(ids).to(self.pts.device)

        self.pts = self.pts[ids].contiguous()
        self.normals = self.normals[ids].contiguous()
        for label, feats in self.feats.items():
            self.feats[label] = feats[:, ids].contiguous()
        del ids

    def normalize_feats(self):
        r"""Normalizes reference feature vectors."""
        for label, feats in self.feats.items():
            self.feats[label] = torch.nn.functional.normalize(feats, dim=0)


class FeatureExtractor(S2DNet):
    r"""Wraps feature extractor for alignment. See `forward` for details.

    Parameters
    ----------
    device : torch.device
    checkpoint_pt : str
    """
    def __init__(self, device, checkpoint_pt):
        super().__init__(device, checkpoint_path=checkpoint_pt)
        self.eval().requires_grad_(False)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def forward(self, img):
        r"""Extracts features.

        Parameters
        ----------
        img : torch.Tensor
            of shape [batch_size, 3, height, width], in range [0, 1].

        Returns
        -------
        feat_maps : dict
            feat_0: torch.Tensor
                of shape [batch_size, 128, height, width], fine-scale features.
            feat_1: torch.Tensor
                of shape [batch_size, 128, height // 4, width // 4], coarse-scale features.
        """
        feat_maps = super().forward(img)
        feat_maps = {f'feat_{i}': feat_map for (i, feat_map) in enumerate(feat_maps)}
        return feat_maps


class Reprojector:
    r"""Reprojects a set of source images to the destination camera.

    Parameters
    ----------
    renderer : OccMeshRenderer
    occ_renderer : MeshRenderer
    occ_thres : float
        Scan points below the occluding surface deeper than this are occluded.

    Attributes
    ----------
    vert_src_uvs_per_cam : dict
        cam_id: torch.Tensor
            of shape [verts_n, 2].
    cam_weights : dict
        cam_id: torch.Tensor
            of shape [tris_n].
    """
    def __init__(self, renderer, occ_renderer, occ_thres=1e-3):
        self.renderer = renderer
        self.occ_renderer = occ_renderer
        self.occ_thres = occ_thres

        self.src_cams = dict()
        self.vert_src_uvs_per_cam = dict()
        self.cam_weights = dict()

    def calc_src_uvs(self, src_cams, angle_gamma=4, max_angle_deg=70):
        r"""Pre-calculates UV coordinates of the scan vertices in the image space of each source camera.

        Parameters
        ----------
        src_cams : dict
            cam_id: Camera
                A camera with a feat_map 'img'.
        angle_gamma : float
        max_angle_deg : float
            Parameters of projection angle-based weighting, see `calc_angle_weight`.
        """
        verts = torch.from_numpy(self.renderer.mesh.vertex['positions'].numpy())
        tri_vert_ids = torch.from_numpy(self.renderer.mesh.triangle['indices'].numpy())
        tri_centers = verts[tri_vert_ids.view(-1)].view(-1, 3, 3).mean(1); del tri_vert_ids
        tri_normals = torch.from_numpy(self.renderer.mesh.triangle['normals'].numpy())
        for cam_id, cam in tqdm(src_cams.items()):
            # 'Trace verts from the source camera' >> logger.debug
            is_vis = self.occ_renderer.calc_pts_visibility(verts, cam.c2w_t, self.occ_thres)
            vis_ids = is_vis.nonzero(as_tuple=True)[0]; del is_vis
            vis_verts = verts[vis_ids]

            # 'Project verts to the source camera' >> logger.debug
            xyz_cam = cam.w2c_rotm @ vis_verts.T.to(cam.w2c_t) + cam.w2c_t.unsqueeze(1); del vis_verts
            uvs = cam.cam_model.project_fine(xyz_cam); del xyz_cam
            is_in_bounds = cam.cam_model.uv_is_in_bounds(uvs)
            uvs = uvs[:, is_in_bounds]
            vis_ids = vis_ids[is_in_bounds]; del is_in_bounds

            uvs = cam.cam_model.uv_to_torch(uvs.T)
            vert_uvs = uvs.new_full([len(verts), 2], float('nan'))
            vert_uvs.index_copy_(0, vis_ids, uvs); del vis_ids, uvs
            self.vert_src_uvs_per_cam[cam_id] = vert_uvs; del vert_uvs

            # 'Calc cam weights' >> logger.debug
            cos = calc_cos(tri_centers, tri_normals, cam)
            angle_weights = calc_angle_weight(cos, angle_gamma, max_angle_deg); del cos
            self.cam_weights[cam_id] = angle_weights; del angle_weights
            self.src_cams[cam_id] = cam

    def reproject(self, dst_cam, src_cam_ids):
        r"""Reprojects the selected source images to the destination camera.

        Parameters
        ----------
        dst_cam : Camera
        src_cam_ids : iterable of str

        Returns
        -------
        repr_img : torch.Tensor
            of shape [channels_n, height, width], with nan in pixels without any projection.
        vis_pix_ids : torch.Tensor
            of shape [pixs_n], ids of the pixels in which the surface is visible.
        """
        'Trace pixel rays from the destination camera' >> logger.debug
        self.renderer.set_rays_from_camera(dst_cam.cam_model)
        casted_rays = self.renderer.make_rays_from_cam(dst_cam.c2w_t, dst_cam.c2w_rotm)
        render = self.renderer.render_rays(casted_rays, cull_back_faces=True, get_tri_ids=True, get_bar_uvs=True)
        hit_depths, tri_ids, bar_uvs = render['ray_hit_depth'], render['tri_ids'], render['bar_uvs']
        del casted_rays, render

        'Reject occluded points' >> logger.debug
        w, h = dst_cam.cam_model.size_wh.tolist()
        if self.renderer.valid_ray_ids is not None:
            vis_dst_ids = self.renderer.valid_ray_ids
        else:
            vis_dst_ids = torch.arange(w * h, dtype=torch.long)
        is_vis = hit_depths.isfinite(); del hit_depths
        vis_dst_ids = vis_dst_ids[is_vis]
        tri_ids = tri_ids[is_vis]
        bar_uvs = bar_uvs[is_vis]; del is_vis

        'Prepare barycentric weights' >> logger.debug
        tri_vert_ids = torch.from_numpy(self.renderer.mesh.triangle['indices'].numpy())
        vert_ids = tri_vert_ids[tri_ids]; del tri_vert_ids
        vert_weights = bar_uvs.new_empty(len(bar_uvs), 3)
        vert_weights[:, 1:3] = bar_uvs
        vert_weights[:, 0] = 1 - bar_uvs.sum(1); del bar_uvs

        'Collect image data' >> logger.debug
        src_cam = self.src_cams[next(iter(src_cam_ids))]
        img = src_cam.feat_maps['img']; del src_cam
        channels_n = img.shape[0]
        total_img = img.new_zeros(channels_n, h * w)
        total_weights = img.new_zeros(h * w); del img

        for cam_id in tqdm(src_cam_ids):
            src_cam = self.src_cams[cam_id]
            vert_uvs = self.vert_src_uvs_per_cam[cam_id]
            uvs = vert_uvs[vert_ids.view(-1)].view(-1, 3, 2); del vert_uvs

            # 'Reject invisible hits' >> logger.debug
            is_vis_src = uvs.view(-1, 6).isfinite().all(1)
            vis_src_ids = is_vis_src.nonzero(as_tuple=True)[0]; del is_vis_src
            uvs = (vert_weights[vis_src_ids].unsqueeze(1) @ uvs[vis_src_ids]).squeeze(1)

            # 'Sample source image' >> logger.debug
            src_img_vals = src_cam.get_feats(uvs, 'img'); del src_cam, uvs

            # 'Collect cam weights' >> logger.debug
            vis_tri_ids = tri_ids[vis_src_ids]
            cam_weights = self.cam_weights[cam_id][vis_tri_ids]; del vis_tri_ids

            # 'Collect image values' >> logger.debug
            both_ids = vis_dst_ids[vis_src_ids]; del vis_src_ids
            total_img[:, both_ids] += src_img_vals.mul_(cam_weights); del src_img_vals
            total_weights[both_ids] += cam_weights; del both_ids, cam_weights
        del tri_ids, vert_ids, vert_weights

        'Calculate weighted average' >> logger.debug
        total_img /= total_weights; del total_weights
        repr_img = total_img.view(channels_n, h, w); del total_img
        return repr_img, vis_dst_ids


def calc_cos(pts, normals, cam):
    r"""Calculates cosine of the angle between the direction from a point to the camera and the normal at the point.

    Parameters
    ----------
    pts : torch.Tensor
        of shape [pts_n, 3].
    normals : torch.Tensor
        of shape [pts_n, 3].
    cam : Camera

    Returns
    -------
    cos : torch.Tensor
        of shape [pts_n].
    """
    pts_to_cam = cam.c2w_t - pts; del pts
    pts_to_cam = torch.nn.functional.normalize(pts_to_cam, dim=1)
    cos = (pts_to_cam.unsqueeze(1) @ normals.unsqueeze(2)).squeeze(2).squeeze(1); del pts_to_cam, normals
    return cos


def calc_angle_weight(cos, angle_gamma, max_angle_deg):
    r"""Calculates the weight related to the projection angle as
        10^-[(cos - 1) / (cos max_angle_deg - 1)]^angle_gamma.

    Parameters
    ----------
    cos : torch.Tensor
        of shape [pts_n].
    angle_gamma : float
    max_angle_deg : float

    Returns
    -------
    weights : torch.Tensor
        of shape [pts_n].
    """
    weights = cos.sub(1).div_(np.cos(np.deg2rad(max_angle_deg)) - 1); del cos
    weights = torch.pow(10, weights.pow_(angle_gamma).neg_())
    return weights


def init_cams(init_cams, imgs, fe, roi_pts, pad=50):
    r"""Initializes cameras for alignment.

    Parameters
    ----------
    init_cams : iterable of Camera
    imgs : iterable of torch.Tensor
        of shape [cams_n, 3, height, width].
    fe : FeatureExtractor
    roi_pts : torch.Tensor
        of shape [3, pts_n].
    pad : int
        Images are cropped to the bounding box of the ROI points with this padding before feature extraction.

    Returns
    -------
    cams : iterable of Camera
    """
    cams = []
    for cam, img in tqdm(list(zip(init_cams, imgs))):
        # 'Crop to ROI' >> logger.debug
        img, cam_model = crop_to_roi(cam, img, roi_pts, pad)

        # 'Extract feat maps' >> logger.debug
        feat_maps = fe(img.unsqueeze(0).to(fe.device, fe.dtype)); del img
        feat_maps = {label: feat_map.squeeze(0).to(cam.w2c_t) for (label, feat_map) in feat_maps.items()}

        # 'Init Camera' >> logger.debug
        cam = Camera(cam_model, cam.w2c, feat_maps, True); del cam_model, feat_maps
        cams.append(cam); del cam
    return cams


def crop_to_roi(cam, img, roi_pts, pad=50):
    r"""Crops an image and a camera model to the bounding box of the ROI points.

    Parameters
    ----------
    cam : Camera
    img : torch.Tensor
        of shape [3, height, width].
    roi_pts : torch.Tensor
        of shape [3, pts_n].
    pad : int
        Image is cropped to the bounding box of the ROI points with this padding.

    Returns
    -------
    cropped_img : torch.Tensor
        of shape [3, height, width].
    cropped_cam_model : CameraModel
    """
    # 'Find ROI' >> logger.debug
    pts_cam = cam.w2c_rotm @ roi_pts + cam.w2c_t.unsqueeze(1)
    uvs = cam.cam_model.project_fine(pts_cam); del pts_cam
    crop_left_top = (uvs.min(1)[0] - pad).floor().maximum(uvs.new_tensor(0)).int()
    new_wh = (uvs.max(1)[0] + pad).ceil().minimum(cam.cam_model.size_wh).int(); del uvs
    new_wh = new_wh - crop_left_top

    # 'Crop to ROI' >> logger.debug
    cropped_img = img[:, crop_left_top[1]: crop_left_top[1] + new_wh[1], crop_left_top[0]: crop_left_top[0] + new_wh[0]]
    cropped_cam_model = cam.cam_model.clone().crop_(crop_left_top, new_wh); del crop_left_top, new_wh
    return cropped_img, cropped_cam_model
