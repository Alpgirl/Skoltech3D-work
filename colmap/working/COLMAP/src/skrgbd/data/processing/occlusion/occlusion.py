import numpy as np
import open3d as o3d
import scipy.spatial
import torch
import trimesh

from skrgbd.data.processing.mesh_utils.triangulation import clean_triangles
from skrgbd.data.processing.depth_utils.mesh_rendering import MeshRenderer


class OcclusionHelper:
    r"""See the description and a usage example in
        skrgbd/data/processing/processing_pipeline/sl_06_make_occlusion_surface.py"""

    def __init__(self, device='cpu', dtype=torch.float32):
        self.device = device
        self.dtype = dtype

        self.rec = None
        self.rec_renderer = None
        self.env = None
        self.env_renderer = None

        self.pix_rays = None
        self.pix_tris = None
        self.cam_to_world = None
        self.partial_scans = None

        self.carving_rays = None
        self.grid_wh = None
        self.fxy = None
        self.cxy = None

        self.free_depths = None

        self.samples = None
        self.normals = None

    def set_reconstruction(self, rec):
        self.rec = rec
        self.rec_renderer = MeshRenderer(rec)

    def set_envelope(self, dilation=3e-3, subdivs_n=1):
        verts = np.asarray(self.rec.vertices)
        if dilation is not None:
            aug_mesh = trimesh.creation.icosphere(subdivisions=subdivs_n, radius=dilation)
            aug_verts = np.asarray(aug_mesh.vertices); del aug_mesh
            verts = verts[:, None, :] + aug_verts; del aug_verts
            verts = verts.reshape(-1, 3)
        env = compute_hull(verts); del verts
        self.env = env
        self.env_renderer = MeshRenderer(env)

    def init_partial_scans(self):
        r"""Must set pix_rays, pix_tris, cam_to_world, and partial_scans."""
        raise NotImplementedError

    def init_carving_rays(self, grid_wh):
        grid_wh = torch.tensor(grid_wh, dtype=torch.long)

        uvn_min = []
        uvn_max = []
        for pix_rays in self.pix_rays.values():
            uvn = pix_rays[:, :2] / pix_rays[:, 2:3]; del pix_rays
            uvn_min.append(uvn.min(0)[0])
            uvn_max.append(uvn.max(0)[0]); del uvn
        uvn_min = torch.stack(uvn_min).min(0)[0]
        uvn_max = torch.stack(uvn_max).max(0)[0]

        fxy = (grid_wh - 1) / (uvn_max - uvn_min); del uvn_max
        cxy = .5 - uvn_min * fxy; del uvn_min

        w, h = grid_wh.tolist()
        v, u = torch.meshgrid([torch.linspace(.5, h - .5, h), torch.linspace(.5, w - .5, w)])
        uv = torch.stack([u, v], -1).view(-1, 2); del u, v
        uvn = (uv - cxy).div_(fxy)
        pix_rays = torch.cat([uvn, torch.ones_like(uvn[:, :1])], 1); del uvn
        pix_rays = torch.nn.functional.normalize(pix_rays, dim=1)
        pix_rays = pix_rays.mul_(-1)

        pix_rays = pix_rays.to(self.rec_renderer.device, self.rec_renderer.dtype)
        grid_wh = grid_wh.to(self.device)
        fxy = fxy.to(self.device, self.dtype)
        cxy = cxy.to(self.device, self.dtype)

        self.carving_rays = pix_rays
        self.grid_wh = grid_wh
        self.fxy = fxy
        self.cxy = cxy

    def init_partial_carving_depths(self, rec_fidelity_threshold=1e-3, free_space_depth=1e-4, show_progres=None):
        show_progres = show_progres or (lambda x: x)

        self.free_depths = dict()

        pix_rays = self.carving_rays
        casted_rays = torch.empty([len(pix_rays), 6], device=self.rec_renderer.device, dtype=self.rec_renderer.dtype)

        for scan_i, scan in show_progres(self.partial_scans.items()):
            scan = o3d.io.read_triangle_mesh(scan)
            part_renderer = MeshRenderer(scan); del scan
            cam_to_world = self.cam_to_world[scan_i].to(casted_rays)
            cam_center = cam_to_world[:3, 3]
            casted_rays[:, :3] = cam_center; del cam_center
            torch.mm(pix_rays, cam_to_world[:3, :3].T, out=casted_rays[:, 3:6]); del cam_to_world
            rec_depth, part_depth = [renderer.render_rays(casted_rays, cull_back_faces=True)['ray_hit_depth']
                                     for renderer in [self.rec_renderer, part_renderer]]

            depth_is_defined = part_depth.sub_(rec_depth).abs_() <= rec_fidelity_threshold; del part_depth
            free_depth = rec_depth.add_(free_space_depth).where(depth_is_defined, rec_depth.new_tensor(0))
            del rec_depth, depth_is_defined

            self.free_depths[scan_i] = free_depth.to(self.device, self.dtype); del free_depth
            del part_renderer

    def sample_occlusion_surface(
            self, rec_fidelity_threshold=1e-3, sample_size=1e-4, batch_size=10_240_000,
            resample_rec=True, show_progres=None
    ):
        show_progres = show_progres or (lambda x: x)

        self.samples = []
        self.normals = []

        # Sample boundary surfaces
        for scan_i, scan in show_progres(self.partial_scans.items()):
            scan = o3d.io.read_triangle_mesh(scan)
            renderer = MeshRenderer(scan); del scan
            pix_rays, pix_tris = self.pix_rays[scan_i], self.pix_tris[scan_i]
            cam_to_world = self.cam_to_world[scan_i].to(pix_rays)
            verts, tris = self.get_boundary_surface(renderer, cam_to_world, pix_rays, pix_tris, rec_fidelity_threshold)
            del renderer, cam_to_world, pix_rays, pix_tris

            verts = verts.double().numpy()
            tris = tris.int().numpy()
            samples, normals = sample_surface_rand(verts, tris, sample_size); del verts, tris

            samples = torch.from_numpy(samples)
            normals = torch.from_numpy(normals)
            kept_ids = self.carve_samples(samples, batch_size, scan_i)
            self.samples.append(samples[kept_ids]); del samples
            self.normals.append(normals[kept_ids]); del normals, kept_ids

        # Sample envelope
        verts = np.asarray(self.env.vertices)
        tris = np.asarray(self.env.triangles)
        samples, normals = sample_surface_rand(verts, tris, sample_size); del verts, tris

        samples = torch.from_numpy(samples)
        normals = torch.from_numpy(normals)
        kept_ids = self.carve_samples(samples, batch_size)
        self.samples.append(samples[kept_ids]); del samples
        self.normals.append(normals[kept_ids]); del normals, kept_ids

        # Sample reconstruction
        # Resampling with the same density as the other parts
        # only required if the new density is higher than the original one
        if resample_rec:
            verts = np.asarray(self.rec.vertices)
            tris = np.asarray(self.rec.triangles)
            samples, normals = sample_surface_rand(verts, tris, sample_size); del verts, tris
        else:
            rec_clone = o3d.geometry.TriangleMesh(self.rec)
            rec_clone.compute_vertex_normals()
            samples = np.asarray(rec_clone.vertices)
            normals = np.asarray(rec_clone.vertex_normals); del rec_clone
        self.samples.append(samples); del samples
        self.normals.append(normals); del normals

    def get_boundary_surface(self, part_renderer, cam_to_world, pix_rays, pix_tris, rec_fidelity_threshold):
        r"""
        Parameters
        ----------
        part_renderer
        cam_to_world : torch.Tensor
            of shape [4, 4]
        pix_rays : torch.Tensor
            of shape [rays_n, 3]
        pix_tris : torch.Tensor
            of shape [pix_tris_n, 3]
        rec_fidelity_threshold : float

        Returns
        -------
        verts : torch.Tensor
            of shape [verts_n, 3]
        tris : torch.Tensor
            of shape [tris_n, 3]
        """
        cam_center = cam_to_world[:3, 3]
        rays_world = pix_rays @ cam_to_world[:3, :3].T; del cam_to_world, pix_rays

        casted_rays = torch.empty([len(rays_world), 6], device=part_renderer.device, dtype=part_renderer.dtype)
        casted_rays[:, :3] = cam_center
        casted_rays[:, 3:6] = rays_world
        rec_depth, part_depth, env_depth = [renderer.render_rays(casted_rays, cull_back_faces=True)['ray_hit_depth']
                                            for renderer in [self.rec_renderer, part_renderer, self.env_renderer]]
        del casted_rays, part_renderer
        depth_is_defined = part_depth.sub_(rec_depth).abs_() <= rec_fidelity_threshold; del part_depth
        free_depth = rec_depth.where(depth_is_defined, env_depth); del rec_depth, env_depth

        tri_depth_is_defined = depth_is_defined.view(-1)[pix_tris.view(-1)].view_as(pix_tris); del depth_is_defined
        tri_has_defined_verts = tri_depth_is_defined.any(1)
        tri_has_undef_verts = tri_depth_is_defined.all(1).logical_not_(); del tri_depth_is_defined
        tri_is_boundary = tri_has_undef_verts.logical_and_(tri_has_defined_verts)
        del tri_has_undef_verts, tri_has_defined_verts
        tris = pix_tris[tri_is_boundary]; del tri_is_boundary, pix_tris
        ray_ids, tris = clean_triangles(tris)
        rays_world = rays_world[ray_ids]
        free_depth = free_depth[ray_ids]; del ray_ids

        free_depth = free_depth.to(rays_world)
        verts = (rays_world * free_depth.unsqueeze(1)).add_(cam_center); del rays_world, free_depth, cam_center
        return verts, tris

    def carve_samples(self, samples, batch_size=None, skip_i=None):
        r"""
        Parameters
        ----------
        samples : torch.Tensor
            of shape [samples_n, 3]
        batch_size : int
        skip_i : int

        Returns
        -------
        kept_sample_ids : list of torch.Tensor
            of shape [kept_samples_n]
        """
        if batch_size is None:
            return self._carve_samples(samples, skip_i)
        else:
            kept_sample_ids = []

            for batch_start in range(0, len(samples), batch_size):
                batch_end = min(batch_start + batch_size, len(samples))
                samples_batch = samples[batch_start: batch_end]
                kept_sample_ids_batch = self._carve_samples(samples_batch, skip_i); del samples_batch

                kept_sample_ids_batch = kept_sample_ids_batch.add_(batch_start).cpu()
                kept_sample_ids.append(kept_sample_ids_batch); del kept_sample_ids_batch
            kept_sample_ids = torch.cat(kept_sample_ids)
            return kept_sample_ids

    def _carve_samples(self, samples, skip_i=None):
        r"""
        Parameters
        ----------
        samples : torch.Tensor
            of shape [samples_n, 3]
        skip_i : int

        Returns
        -------
        kept_sample_ids : torch.Tensor
            of shape [kept_samples_n]
        """
        kept_samples = samples.to(self.device, self.dtype); del samples
        keep_sample = kept_samples.new_ones(len(kept_samples), dtype=torch.bool)
        kept_sample_ids = torch.arange(len(kept_samples), device=kept_samples.device)

        for depth_i, free_depth in self.free_depths.items():
            if depth_i == skip_i:
                continue
            cam_to_world = self.cam_to_world[depth_i]
            cam_center = cam_to_world[:3, 3]
            samples_cam = (kept_samples - cam_center) @ cam_to_world[:3, :3]; del cam_to_world, cam_center
            sample_depths = samples_cam.norm(dim=1)
            uvn = samples_cam[:, :2] / samples_cam[:, 2:3]; del samples_cam
            uv = uvn.mul_(self.fxy).add_(self.cxy); del uvn
            ji = uv.floor_().long(); del uv

            in_bounds = (ji >= 0).all(1).logical_and_((ji < self.grid_wh).all(1))
            sample_depths = sample_depths[in_bounds]
            ji = ji[in_bounds]

            pix_id = (ji[:, 1] * self.grid_wh[0]).add_(ji[:, 0]); del ji
            free_depth = free_depth[pix_id]; del pix_id
            is_hidden = sample_depths >= free_depth; del sample_depths, free_depth

            keep_sample.masked_scatter_(in_bounds, is_hidden); del in_bounds, is_hidden
            kept_samples = kept_samples[keep_sample]
            kept_sample_ids = kept_sample_ids[keep_sample]
            keep_sample = keep_sample[keep_sample]
            if len(kept_samples) == 0:
                break
        return kept_sample_ids


def compute_hull(verts, clean=True, orient=True):
    r"""

    Parameters
    ----------
    verts : np.ndarray
        of shape [verts_n, 3]
    clean : bool
    orient : bool

    Returns
    -------
    hull : o3d.geometry.TriangleMesh
    """
    hull = scipy.spatial.ConvexHull(verts)
    verts = hull.points
    tris = hull.simplices; del hull
    hull = o3d.geometry.TriangleMesh()
    hull.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64)); del verts
    hull.triangles = o3d.utility.Vector3iVector(tris.astype(np.int32)); del tris

    if clean or orient:
        hull = hull.remove_duplicated_vertices()
        hull = hull.remove_degenerate_triangles()
        hull = hull.remove_duplicated_triangles()
        hull = hull.remove_unreferenced_vertices()

    if orient:
        hull.orient_triangles()
        hull.compute_vertex_normals()
        verts = torch.from_numpy(np.asarray(hull.vertices))
        normals = torch.from_numpy(np.asarray(hull.vertex_normals))
        hull.vertex_normals.clear()

        center = verts.mean(0)
        normals_point_inwards = (center - verts).mul_(normals).sum(1).mean() > 0; del normals
        if normals_point_inwards:
            tris = np.asarray(hull.triangles)
            tris = tris[:, [0, 2, 1]]
            hull.triangles.clear()
            hull.triangles = o3d.utility.Vector3iVector(tris.astype(np.int32))
            hull.compute_triangle_normals()

    return hull


def sample_surface_rand(verts, tris, sample_size):
    r"""
    Parameters
    ----------
    verts : np.ndarray
        of shape [verts_n, 3], float64
    tris : np.ndarray
        of shape [tris_n, 3], int32
    sample_size : float

    Returns
    -------
    samples : np.ndarray
        of shape [samples_n, 3], float64
    normals : np.ndarray
        of shape [samples_n, 3], float64
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts); del verts
    mesh.triangles = o3d.utility.Vector3iVector(tris); del tris

    # Assuming uniform triangular tiling of the surface with sides equal to sample_size,
    # the area per each vertex-sample is the area of two equilateral triangles
    s_per_sample = (sample_size ** 2) * np.sqrt(3) / 2
    samples_n = round(mesh.get_surface_area() / s_per_sample)
    samples = mesh.sample_points_uniformly(samples_n, use_triangle_normal=True)

    normals = np.asarray(samples.normals)
    samples = np.asarray(samples.points)
    return samples, normals
