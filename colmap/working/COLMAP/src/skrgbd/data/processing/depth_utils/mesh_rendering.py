import numpy as np
import open3d as o3d
import torch

from skrgbd.calibration.camera_models.camera_model import CameraModel


class MeshRenderer:
    r"""
    Parameters
    ----------
    scan : o3d.geometry.TriangleMesh
    """
    def __init__(self, scan):
        self.mesh = o3d.t.geometry.TriangleMesh.from_legacy(scan)
        self.raycasting = o3d.t.geometry.RaycastingScene()
        self.raycasting.add_triangles(self.mesh)

        self.device = 'cpu'
        self.dtype = torch.float32

        self.rays = None
        self.valid_ray_ids = None
        self.w = None
        self.h = None

    def set_rays_from_camera(self, camera_model):
        r"""
        Parameters
        ----------
        camera_model : CameraModel
        """
        rays = camera_model.get_pixel_rays()
        rays = rays.permute(1, 2, 0).to(self.device, self.dtype, memory_format=torch.contiguous_format).view(-1, 3)
        valid_ray_ids = rays.isfinite().all(1).nonzero().squeeze(1)
        if len(valid_ray_ids) == len(rays):
            self.valid_ray_ids = None
            self.rays = rays
        else:
            self.valid_ray_ids = valid_ray_ids
            self.rays = rays[valid_ray_ids].contiguous()
        self.w, self.h = camera_model.size_wh.tolist()

    def render_to_camera(self, cam_to_mesh_t, cam_to_mesh_rot, outputs, cull_back_faces=False):
        r"""
        Parameters
        ----------
        cam_to_mesh_t : torch.Tensor
            of shape [3]
        cam_to_mesh_rot : torch.Tensor
            of shape [:3, :3]
        outputs : list of str
            from {'ray_depth', 'z_depth', 'world_normals', 'cam_normals', 'world_ray_dirs'}
        cull_back_faces : bool
            If True, invalidate the results for the rays hitting the back of the face.

        Returns
        -------
        render : dict
            ray_depth, z_depth: torch.Tensor
                of shape [height, width], inf for rays that didn't hit any face.
            world_normals, cam_normals: torch.Tensor
                of shape [height, width, 3], all 0 for rays that didn't hit any face.
            world_xyz : torch.Tensor
                of shape [height, width, 3], inf for rays that didn't hit any face.
        """
        # Make rays
        cam_to_mesh_rot = cam_to_mesh_rot.to(self.rays)
        casted_rays = self.make_rays_from_cam(cam_to_mesh_t, cam_to_mesh_rot)

        # Trace rays
        raw_render = self.render_rays(casted_rays, cull_back_faces=cull_back_faces)

        # Collect outputs
        render = dict()
        for key in outputs:
            if key in {'ray_depth', 'z_depth'}:
                render[key] = self.get_depth(raw_render, key)
            elif key in {'world_normals', 'cam_normals'}:
                render[key] = self.get_normals(raw_render, cam_to_mesh_rot, key)
            elif key in {'world_xyz', 'cam_xyz'}:
                render[key] = self.get_xyz(raw_render, casted_rays, key)
            elif key in {'world_ray_dirs', 'cam_ray_dirs'}:
                render[key] = self.get_ray_dirs(casted_rays[:, 3:6], key)
        return render

    def make_rays_from_cam(self, cam_to_mesh_t, cam_to_mesh_rot, rays=None):
        r"""Calculates the optical rays passing through centers of the pixels of the destination camera in the world space.

        Parameters
        ----------
        cam_to_mesh_t : torch.Tensor
            of shape [3]
        cam_to_mesh_rot : torch.Tensor
            of shape [:3, :3]
        rays : torch.Tensor
            of shape [rays_n, 3]

        Returns
        -------
        casted_rays : torch.Tensor
            of shape [rays_n, 6].
        """
        if rays is None:
            rays = self.rays
        casted_rays = rays.new_empty([len(rays), 6])
        ray_origins, ray_dirs = casted_rays[:, :3], casted_rays[:, 3:6]
        ray_origins.copy_(cam_to_mesh_t); del cam_to_mesh_t
        cam_to_mesh_rot = cam_to_mesh_rot.to(rays)
        torch.mm(rays, cam_to_mesh_rot.T, out=ray_dirs)
        return casted_rays

    def get_depth(self, raw_render, var='ray_depth'):
        r"""
        Parameters
        ----------
        raw_render
            ray_hit_depth : torch.Tensor
                of shape [rays_n].
        var : {'ray_depth', 'z_depth'}

        Returns
        -------
        depth : torch.Tensor
            of shape [height, width], inf for rays that didn't hit any face.
        """
        ray_depth = raw_render['ray_hit_depth']

        # Optionally, transform
        if var == 'ray_depth':
            depth = ray_depth
        elif var == 'z_depth':
            depth = ray_depth * self.rays[:, 2]
        del ray_depth

        # Set depth for invalid hits to inf
        if self.valid_ray_ids is not None:
            depth = self.scatter_ray_data(depth.unsqueeze(1), self.valid_ray_ids, float('inf'))
        depth = depth.view(self.h, self.w)
        return depth

    def get_normals(self, raw_render, cam_to_mesh_rot, var='world_normals'):
        r"""
        Parameters
        ----------
        raw_render : dict
            ray_hit_depth : torch.Tensor
                of shape [rays_n].
            normals : torch.Tensor
                of shape [rays_n, 3].
        cam_to_mesh_rot : torch.Tensor
            of shape [3, 3].
        var : {'world_normals', 'cam_normals'}

        Returns
        -------
        normals : torch.Tensor
            of shape [height, width, 3], 0 for rays that didn't hit any face.
        """
        world_normals = raw_render['normals']

        # Optionally, transform to cam space
        if var == 'world_normals':
            normals = world_normals
        elif var == 'cam_normals':
            normals = world_normals @ cam_to_mesh_rot
        del world_normals, cam_to_mesh_rot

        # Set normals for invalid hits to 0
        valid_ray_ids = raw_render['ray_hit_depth'].isfinite().nonzero(as_tuple=True)[0]
        normals = normals[valid_ray_ids]
        if self.valid_ray_ids is not None:
            valid_ray_ids = self.valid_ray_ids[valid_ray_ids]
        normals = self.scatter_ray_data(normals, valid_ray_ids, 0)
        normals = normals.view(self.h, self.w, 3)
        return normals

    def get_xyz(self, raw_render, casted_rays, var='world_xyz', as_map=True):
        r"""
        Parameters
        ----------
        raw_render : dict
            ray_hit_depth : torch.Tensor
                of shape [rays_n].
        casted_rays : torch.Tensor
            of shape [rays_n, 6].
        var : {'world_xyz'}
        as_map : bool

        Returns
        -------
        xyz : torch.Tensor
            of shape [height, width, 3] if as_map or [rays_n, 3], inf for rays that didn't hit any face.
        """
        ray_depth = raw_render['ray_hit_depth']
        world_xyz = casted_rays[:, :3] + casted_rays[:, 3:6] * ray_depth.unsqueeze(1); del ray_depth

        xyz = world_xyz

        if not as_map:
            return xyz

        # Set xyz for invalid hits to inf
        if self.valid_ray_ids is not None:
            xyz = self.scatter_ray_data(xyz, self.valid_ray_ids, float('inf'))
        xyz = xyz.view(self.h, self.w, 3)
        return xyz

    def get_ray_dirs(self, ray_dirs, var='world_ray_dirs'):
        r"""
        Parameters
        ----------
        ray_dirs : torch.Tensor
            of shape [rays_n, 3].
        var : {'world_ray_dirs'}

        Returns
        -------
        ray_dirs : torch.Tensor
            of shape [height, width, 3], nan for invalid rays.
        """
        world_ray_dirs = ray_dirs; del ray_dirs

        ray_dirs = world_ray_dirs; del world_ray_dirs

        # Set invalid rays to nan
        if self.valid_ray_ids is not None:
            ray_dirs = self.scatter_ray_data(ray_dirs, self.valid_ray_ids, float('nan'))
        ray_dirs = ray_dirs.view(self.h, self.w, 3)
        return ray_dirs

    def scatter_ray_data(self, ray_data, ray_ids, default_val=float('inf')):
        r"""Scatters ray data into image pixels.

        Parameters
        ----------
        ray_data : torch.Tensor
            of shape [valid_rays_n, channels_n].
        ray_ids : torch.LongTensor
            of shape [valid_rays_n].
        default_val : float

        Returns
        -------
        img_data : torch.Tensor
            of shape [height, width, channels_n].
        """
        channels_n = ray_data.shape[1]
        img_data = ray_data.new_full([self.h, self.w, channels_n], default_val)
        img_data.view(-1, channels_n).index_copy_(0, ray_ids, ray_data); del ray_ids, ray_data
        return img_data

    def render_rays(
            self, casted_rays, cull_back_faces=False, backface_val=float('inf'), get_tri_ids=False, get_bar_uvs=False,
    ):
        r"""
        Parameters
        ----------
        casted_rays : torch.Tensor
            of shape [rays_n, 6].
        cull_back_faces : bool
            If True, set the depth value for the rays hitting the back of the face to `backface_val`.
        backface_val : float
        get_tri_ids : bool
        get_bar_uvs : bool

        Returns
        -------
        render : dict
            ray_hit_depth : torch.Tensor
                of shape [rays_n], inf for rays that didn't hit any face.
            normals : torch.Tensor
                of shape [rays_n, 3], all 0 for rays that didn't hit any face.
            tri_ids : torch.Tensor
                of shape [rays_n], o3d.t.geometry.RaycastingScene.INVALID_ID for rays that didn't hit any face.
            bar_uvs : torch.Tensor
                of shape [rays_n, 2],
        """
        casted_rays_t = o3d.core.Tensor.from_numpy(casted_rays.numpy())
        result = self.raycasting.cast_rays(casted_rays_t); del casted_rays_t
        ray_hit_depth = torch.from_numpy(result['t_hit'].numpy())
        normals = torch.from_numpy(result['primitive_normals'].numpy())

        if cull_back_faces:
            ray_dirs = casted_rays[:, 3:6]
            hit_front_facing = (normals.unsqueeze(1) @ ray_dirs.unsqueeze(2)).squeeze(2).squeeze(1) < 0; del ray_dirs
            ray_hit_depth = ray_hit_depth.where(hit_front_facing, ray_hit_depth.new_tensor(backface_val))
            del hit_front_facing
        del casted_rays

        render = dict(ray_hit_depth=ray_hit_depth, normals=normals)
        if get_tri_ids:
            render['tri_ids'] = torch.from_numpy(result['primitive_ids'].numpy().astype(np.int64))
        if get_bar_uvs:
            render['bar_uvs'] = torch.from_numpy(result['primitive_uvs'].numpy())
        return render

    def calc_pts_visibility(self, pts, viewpoint, occ_thres, div_eps=1e-12):
        r"""
        Parameters
        ----------
        pts : torch.Tensor
            of shape [pts_n, 3].
        viewpoint : torch.Tensor
            of shape [3].
        occ_thres : float

        Returns
        -------
        is_visible : torch.BoolTensor
            of shape [pts_n].
        """
        casted_rays = pts.new_empty([len(pts), 6])
        ray_origins, ray_dirs = casted_rays[:, :3], casted_rays[:, 3:6]
        ray_origins.copy_(viewpoint); del ray_origins

        torch.sub(pts, viewpoint.to(pts), out=ray_dirs); del viewpoint
        pt_depths = ray_dirs.norm(dim=1)
        ray_dirs /= pt_depths.unsqueeze(1) + div_eps; del ray_dirs
        return self.calc_ray_visibility(casted_rays, pt_depths, occ_thres)

    def calc_ray_visibility(self, casted_rays, ray_depths, occ_thres):
        r"""
        Parameters
        ----------
        casted_rays : torch.Tensor
            of shape [rays_n, 6].
        ray_depths : torch.Tensor
            of shape [rays_n].
        occ_thres : float

        Returns
        -------
        is_visible : torch.BoolTensor
            of shape [pts_n].
        """
        hit_depths = self.render_rays(casted_rays.to(self.device, self.dtype), cull_back_faces=True)['ray_hit_depth']
        del casted_rays
        hit_depths_shifted = hit_depths.to(ray_depths).add_(occ_thres); del hit_depths
        is_visible = hit_depths_shifted.isfinite().logical_and_(ray_depths <= hit_depths_shifted)
        del hit_depths_shifted, ray_depths
        return is_visible
