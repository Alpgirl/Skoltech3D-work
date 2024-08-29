import nvdiffrast.torch as dr
import numpy as np
import open3d as o3d
import torch

from skrgbd.calibration.camera_models.pinhole import PinholeCameraModel


class MeshRenderer:
    r"""Triangle mesh rasterizer based on OpenGL.

    Parameters
    ----------
    scan : o3d.geometry.TriangleMesh
    """
    dtype = torch.float

    def __init__(self, scan, device='cuda'):
        self.device = device
        self.mesh = o3d.geometry.TriangleMesh(scan)
        self.glctx = dr.RasterizeGLContext(output_db=False, device=self.device)

        self.tris = None
        self.tri_normals = None
        self.xyzw = None

        self.far = None
        self.near = None
        self.proj_mat = None
        self.resolution = None

    def init_mesh_data(self, tri_normals=False):
        r"""Initializes the intrinsic data required for rendering.

        Parameters
        ----------
        tri_normals : bool
            If True, initialize triangle normals to be able to render normal maps.
        """
        self.tris = torch.from_numpy(np.asarray(self.mesh.triangles)).to(self.device, torch.int)

        verts = torch.from_numpy(np.asarray(self.mesh.vertices))
        xyzw = torch.cat([verts, torch.ones_like(verts[:, :1])], 1); del verts
        self.xyzw = xyzw.to(self.device, self.dtype)

        if tri_normals:
            self.mesh.compute_triangle_normals()
            self.tri_normals = torch.from_numpy(np.asarray(self.mesh.triangle_normals)).to(self.device, self.dtype)

    def set_cam_model(self, cam_model, near=.5, far=1.5):
        r"""Calculates OpengGL projection matrix, as described here
          http://www.songho.ca/opengl/gl_projectionmatrix.html

        Parameters
        ----------
        cam_model : PinholeCameraModel
        near : float
            Distance to the near plane.
        far : float
            Distance to the far plane.

        """
        self.far = far
        self.near = near

        lb = -cam_model.principal / cam_model.focal * near
        rt = (cam_model.size_wh - cam_model.principal) / cam_model.focal * near

        proj_mat = torch.zeros(4, 4, device=self.device, dtype=self.dtype)
        proj_mat[[0, 1], [0, 1]] = (2 * near / (rt - lb)).to(self.device, self.dtype)
        proj_mat[:2, 2] = ((lb + rt) / (rt - lb)).to(self.device, self.dtype)
        proj_mat[2, 2] = - (far + near) / (far - near)
        proj_mat[2, 3] = - 2 * far * near / (far - near)
        proj_mat[3, 2] = -1
        # Z-axis in camera space in OpenGL points to camera, and in PinholeCameraModel it points from camera,
        # so we flip it.
        proj_mat[:, 2] *= -1
        # Y-axis should also be flipped, since in OpenGL it points up, and in PinholeCameraModel it points down.
        # However, the vertical orientation of the rendered image also differs, so we keep the orientation of the Y-axis
        # to negate the flipping of the image.
        self.proj_mat = proj_mat

    def set_resolution(self, h, w):
        r"""Sets the resolution of render."""
        self.resolution = (h, w)

    def render_to_camera(self, mesh_to_cam):
        r"""Does rasterization.

        Parameters
        ----------
        mesh_to_cam : torch.Tensor
            of shape [4, 4]. World-to-camera extrinsics matrix.

        Returns
        -------
        rast : torch.Tensor
            of shape [height, width, 4], which contains the rasterizer output in order (u, v, z/w, triangle_id).
            Pixels with missing values have zero in all channels.
        """
        mat = self.proj_mat @ mesh_to_cam; del mesh_to_cam
        xyzw_clip = self.xyzw @ mat.T; del mat
        rast, _ = dr.rasterize(self.glctx, xyzw_clip.unsqueeze(0), self.tris, resolution=self.resolution)
        del _, xyzw_clip
        rast = rast.squeeze(0)
        return rast

    def get_z(self, rast):
        r"""Calculates z-depth in camera space from the output of the rasterizer.

        Parameters
        ----------
        rast : torch.Tensor
            of shape [**, 4]. The output of `render_to_camera`.

        Returns
        -------
        z : torch.Tensor
            of shape [**]. Z-coordinate in camera space with z-axis pointing from the camera.
            Pixels with missing values contain the value (2 * self.far * self.near) / (self.far + self.near).
        """
        ndc_z = rast[..., 2]
        z = ndc_z.mul(self.near - self.far).add_(self.far + self.near); del ndc_z
        z = (2 * self.far * self.near) / z
        return z

    def get_normals(self, rast):
        r"""Calculates normal map in world space from the output of the rasterizer.

        Parameters
        ----------
        rast : torch.Tensor
            of shape [height, width, 4]. The output of `render_to_camera`.

        Returns
        -------
        normals : torch.Tensor
            of shape [height, width, 3].
        """
        normals = self.tri_normals[rast.view(-1, 4)[:, 3].long().sub_(1)].view(*rast.shape[:2], 3)
        return normals

    def interpolate(self, attrs, rast):
        r"""Renders vertex attributes.

        Parameters
        ----------
        attrs : torch.Tensor
            of shape [verts_n, attrs_n].
        rast : torch.Tensor
            of shape [height, width, 4]. The output of `render_to_camera`.

        Returns
        -------
        samples : torch.Tensor
            of shape [height, width, attrs_n].
        """
        attrs, _ = dr.interpolate(attrs.unsqueeze(0), rast.unsqueeze(0), self.tris); del _
        attrs = attrs.squeeze(0)
        return attrs

    def render_colors_to_camera(self, colors, mesh_to_cam, bg_color=None, antialias=False):
        r"""Renders mesh color on vertices.

        Parameters
        ----------
        colors : torch.Tensor
            of shape [verts_n, colors_n].
        mesh_to_cam : torch.Tensor
            of shape [4, 4]. World-to-camera extrinsics matrix.
        bg_color : torch.Tensor
            of shape [3].
        antialias : bool

        Returns
        -------
        img : torch.Tensor
            of shape [height, width, colors_n].
        """
        mat = self.proj_mat @ mesh_to_cam; del mesh_to_cam
        xyzw_clip = self.xyzw @ mat.T; del mat
        rast, _ = dr.rasterize(self.glctx, xyzw_clip.unsqueeze(0), self.tris, resolution=self.resolution); del _
        img, _ = dr.interpolate(colors.unsqueeze(0), rast, self.tris); del _, colors
        if bg_color is not None:
            img = img.where(rast[..., 3].ne(0).unsqueeze(3), bg_color)
        if antialias:
            img = dr.antialias(img, rast, xyzw_clip.unsqueeze(0), self.tris)
        img = img.squeeze(0)
        return img
