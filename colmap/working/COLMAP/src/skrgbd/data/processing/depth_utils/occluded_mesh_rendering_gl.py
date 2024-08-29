import nvdiffrast.torch as dr
import open3d as o3d
import torch

from skrgbd.data.processing.depth_utils.mesh_rendering_gl import MeshRenderer as BaseMeshRenderer


class MeshRenderer(BaseMeshRenderer):
    def __init__(self, scan, occlusion_mesh, occ_thres, device='cuda'):
        super().__init__(scan, device)
        self.occ_mesh = o3d.geometry.TriangleMesh(occlusion_mesh)
        self.occ_tris = None
        self.occ_xyzw = None
        self.occ_ranges = None
        self.occ_thres = occ_thres

    def init_mesh_data(self, tri_normals=False):
        r"""Initializes the intrinsic data required for rendering.

        Parameters
        ----------
        tri_normals : bool
            If True, initialize triangle normals to be able to render normal maps.
        """
        mesh = self.mesh
        self.mesh = self.occ_mesh
        super().init_mesh_data()
        self.occ_tris = self.tris
        self.occ_xyzw = self.xyzw

        self.mesh = mesh
        super().init_mesh_data(tri_normals)

        verts_n = len(self.xyzw)
        tris_n = len(self.tris)
        occ_tris_n = len(self.occ_tris)
        self.occ_xyzw = torch.cat([self.xyzw, self.occ_xyzw], 0)
        self.occ_tris = torch.cat([self.tris, self.occ_tris + verts_n], 0)
        self.occ_ranges = self.occ_tris.new_tensor([[0, tris_n], [tris_n, occ_tris_n]]).cpu()

    def render_to_camera(self, mesh_to_cam):
        r"""Does rasterization.

        Parameters
        ----------
        mesh_to_cam : torch.Tensor
            of shape [4, 4]. World-to-camera extrinsics matrix.
        occ_threshold : float

        Returns
        -------
        rast : torch.Tensor
            of shape [height, width, 4], which contains the rasterizer output in order (u, v, z/w, triangle_id).
            Pixels with missing values have zero in all channels.
        z : torch.Tensor
            of shape [h, w]. Z-coordinate in camera space with z-axis pointing from the camera.
            Pixels with missing values contain the value (2 * self.far * self.near) / (self.far + self.near).
        """
        mat = self.proj_mat @ mesh_to_cam; del mesh_to_cam
        xyzw_clip = self.occ_xyzw @ mat.T; del mat
        rast, _ = dr.rasterize(self.glctx, xyzw_clip, self.occ_tris, resolution=self.resolution, ranges=self.occ_ranges)
        del _, xyzw_clip

        z = self.get_z(rast)
        z, occ_z = z
        rast = rast[0]

        occ_z = occ_z.add_(self.occ_thres)
        not_occluded = z <= occ_z; del occ_z
        z = z.where(not_occluded, z.new_tensor((2 * self.far * self.near) / (self.far + self.near)))
        rast = rast.where(not_occluded.unsqueeze(-1), rast.new_tensor(0)); del not_occluded
        return rast, z
