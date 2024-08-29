import open3d as o3d
import torch


def vis_cam_pose(cam_to_world, ax_len=2e-2, center_size=1e-6):
    r"""Makes a triangular mesh from camera poses for visualizations.
    Each camera is represented with a triplet of edges (actually, very thin triangles):
    the red / green / blue edge corresponds to the X / Y / Z axis in the camera space.

    Parameters
    ----------
    cam_to_world : torch.Tensor
        of shape [cams_n, 4, 4].
    ax_len : float
        Length of axes.
    center_size : float

    Returns
    -------
    mesh : o3d.geometry.TriangleMesh
    """
    cam_center = cam_to_world[..., :3, 3]
    cam_x = cam_to_world[..., :3, :3] @ cam_to_world.new_tensor([1., 0, 0])
    cam_y = cam_to_world[..., :3, :3] @ cam_to_world.new_tensor([0, 1., 0])
    cam_z = cam_to_world[..., :3, :3] @ cam_to_world.new_tensor([0, 0, 1.])

    pts = torch.stack([cam_center, cam_x, cam_y, cam_z, cam_center + center_size], 1)
    pts[:, 1:4] = cam_center.unsqueeze(1) + pts[:, 1:4] * ax_len

    colors = pts.new_tensor([
        [1., 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1., 1, 1],
    ])
    colors = colors.unsqueeze(0).expand(len(pts), -1, -1).contiguous()

    tris = torch.tensor([
        [0, 1, 4],
        [0, 2, 4],
        [0, 3, 4]
    ])
    tris = torch.arange(len(pts)).unsqueeze(1).unsqueeze(2) * 5 + tris

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices.extend(pts.view(-1, 3).double().numpy())
    mesh.vertex_colors.extend(colors.view(-1, 3).double().numpy())
    mesh.triangles.extend(tris.view(-1, 3).long().numpy())
    return mesh
