import open3d as o3d
import torch

from skrgbd.data.processing.depth_utils.mesh_rendering_gl import MeshRenderer


class SplatRenderer(MeshRenderer):
    def __init__(self, pts, device='cuda'):
        super().__init__(o3d.geometry.TriangleMesh(), device)
        self.pts = pts
        self.template_verts = None
        self.template_tris = None
        self.cam_model = None
        self._near = None
        self._far = None

    def set_cam_model(self, cam_model, near=.5, far=1.5):
        self.cam_model = cam_model
        self._near = near
        self._far = far

    def set_template(self, template_verts, template_tris):
        self.template_verts = template_verts
        self.template_tris = template_tris

    def render_colors_to_camera_patches(self, colors, bg_color=None, antialias=False, patch_size=256, half_overlap=4):
        all_colors = colors
        mesh_to_cam = torch.eye(4, device=self.device, dtype=self.dtype)

        cam_size_wh = self.cam_model.size_wh.clone()
        crop_left_top = cam_size_wh.new_tensor([0, 0])
        h = self.cam_model.size_wh[1]

        patches = []
        for crop_top in range(0, h, patch_size - half_overlap * 2):
            crop_left_top[1] = crop_top
            cam_size_wh[1] = min(patch_size, h - crop_top)

            cam_model = self.cam_model.clone()
            cam_model = cam_model.crop_(crop_left_top, cam_size_wh)
            super().set_cam_model(cam_model, self._near, self._far)
            self.set_resolution(cam_size_wh[1], cam_size_wh[0])

            is_in_bounds = cam_model.uv_is_in_bounds(cam_model.project(self.pts.T)); del cam_model
            if is_in_bounds.any():
                pts = self.pts[is_in_bounds]
                colors = all_colors[is_in_bounds]; del is_in_bounds

                verts, tris, colors = make_instances(pts, colors, self.template_verts, self.template_tris)
                self.tris = tris.to(self.device, torch.int); del tris
                self.xyzw = verts.new_ones([len(verts), 4], device=self.device)
                self.xyzw[:, :3].copy_(verts); del verts
                colors = colors.to(self.device, self.dtype)

                img = self.render_colors_to_camera(colors, mesh_to_cam, bg_color, antialias).cpu(); del colors
                self.tris = None
                self.xyzw = None
            else:
                del is_in_bounds
                img = bg_color.cpu().unsqueeze(0).unsqueeze(1).expand(cam_size_wh[1], cam_size_wh[0], -1)
            if len(patches) > 0:
                patches[-1] = patches[-1][:-half_overlap]
                patches.append(img[half_overlap:])
            else:
                patches.append(img)
            del img
            if crop_top + cam_size_wh[1].item() >= h:
                break
        img = torch.cat(patches, 0); del patches
        assert img.shape[0] == h
        return img


def make_instances(pts, attrs, template_verts, template_tris):
    r"""Makes multiple instances of a mesh defined with vertices and triangles,
    positioned at a set of points.

    Parameters
    ----------
    pts : torch.Tensor
        of shape [pts_n, 3]. Centers of the instances.
    attrs : torch.Tensor
        of shape [pts_n, attrs_n]. Instance attributes.
    template_verts : torch.Tensor
        of shape [verts_n, 3]. Vertices of the mesh.
    template_tris : torch.Tensor
        of shape [tris_n, 3]. Triangles of the mesh.

    Returns
    -------
    verts : torch.Tensor
        of shape [verts_n * pts_n, 3]. Vertices of the instances.
    tris : torch.Tensor
        of shape [tris_n * pts_n, 3]. Triangles of the instances.
    attrs : torch.Tensor
        of shape [verts_n * pts_n, attrs_n]. Instance attributes, propagated to each vertex.
    """
    verts = pts.unsqueeze(1).add(template_verts).reshape(-1, 3)
    tris = torch.arange(len(pts), dtype=template_tris.dtype, device=template_tris.device).mul(len(template_verts))
    tris = tris.unsqueeze(1).unsqueeze(2).add(template_tris).view(-1, 3)
    attrs_n = attrs.shape[1]
    attrs = attrs.unsqueeze(1).expand(-1, len(template_verts), -1).reshape(-1, attrs_n)
    return verts, tris, attrs


def make_square_splat(splat_size=5e-4):
    r"""Makes a square splat.

    Parameters
    ----------
    splat_size : float

    Returns
    -------
    verts : torch.FloatTensor
        of shape [4, 3].
    tris : torch.IntTensor
        of shape [2, 3].
    """
    verts = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=torch.float)
    verts = verts.sub_(verts.new_tensor([.5, .5, 0])).mul_(splat_size)
    tris = torch.tensor([[0, 2, 1], [0, 3, 2]], dtype=torch.int)
    return verts, tris
