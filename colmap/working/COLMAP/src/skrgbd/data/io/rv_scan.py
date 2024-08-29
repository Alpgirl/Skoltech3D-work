from pathlib import Path

import numpy as np
import open3d as o3d
import torch


class RVScan:
    r"""
    Parameters
    ----------
    project_dir : str or Path
    scan_i : int
    """

    def __init__(self, project_dir, scan_i, mesh_ext='obj', scans_n=27):
        self.project_dir = Path(project_dir)
        self.scan_i = scan_i

        meshes = list(self.project_dir.glob(f'**/Mesh_*.{mesh_ext}'))
        if len(meshes) != scans_n:
            raise RuntimeError(f'Found {len(meshes)} instead of {scans_n}')
        ids = [int(mesh.name[-len(f'0000.{mesh_ext}'):-len(f'.{mesh_ext}')]) for mesh in meshes]
        if min(ids) == 0:
            mesh_i = scan_i
        elif min(ids) == 1:
            mesh_i = scan_i + 1
        else:
            raise RuntimeError(f'Unexpected minimal mesh_i {min(ids)}')
        meshes = list(filter(lambda p: p.name == f'Mesh_{mesh_i:04}.{mesh_ext}', meshes))
        if len(meshes) != 1:
            raise RuntimeError(f'Found {len(meshes)} meshes: {meshes}')
        self.mesh = meshes[0]
        self.meshdir = self.mesh.parent

        self.mesh = o3d.io.read_triangle_mesh(str(self.mesh))
        self.mesh = self.mesh.remove_duplicated_vertices()
        self.coordinates = 'world'
        self.units = 'mm'

        self.mesh_to_world = self.project_dir / f'scan_res_{scan_i:04}/Raw/M_scenemesh.bin'
        self.mesh_to_world = torch.from_numpy(np.fromfile(self.mesh_to_world, dtype=np.float32).reshape(4, 4)).double()

        self.board_to_mesh = self.project_dir / f'scan_res_{scan_i:04}/vertex_matrix.txt'
        self.board_to_mesh = torch.from_numpy(np.loadtxt(self.board_to_mesh)).double()

    def transform(self, transform):
        self.mesh = self.mesh.transform(transform)

    def to_meters(self):
        if self.units == 'm':
            pass
        elif self.units == 'mm':
            transform = np.identity(4)
            transform[:3] /= 1000
            self.transform(transform)
            self.units = 'm'
        else:
            raise RuntimeError(f'Unknown units {self.units}')
        return self

    def to_board(self):
        if (self.coordinates != 'world') or (self.units != 'mm'):
            raise NotImplementedError

        world_to_board = self.board_to_mesh.inverse() @ self.mesh_to_world.inverse()
        self.transform(world_to_board.numpy())
        self.coordinates = 'board'
        return self

    @property
    def vertices(self):
        return torch.from_numpy(np.asarray(self.mesh.vertices))

    @vertices.setter
    def vertices(self, vertices):
        if (len(vertices.shape) != 2) or (vertices.shape[1] != 3):
            raise ValueError('Vertices have shape [n, 3]')
        self.mesh.vertices = o3d.utility.Vector3dVector(np.ascontiguousarray(vertices, dtype=np.float64))

    def unproject(self, uv, eps=1e-8):
        r"""For points in the texture space calculates the respective 3D points on the mesh.

        Parameters
        ----------
        uv : torch.Tensor
            of shape [n, 2], texture coordinates of the points.
            The coordinates of the bottom-left corner of the texture (not the center of the pixel), are (0,0),
            the coordinates of the top-right corner are (w,h).

        Returns
        -------
        xyz : torch.Tensor
            of shape [n, 3], 3D coordinates of the points on mesh.
            If a point lies outside of the mesh, its coordinates are nan.
        """
        points_n = len(uv)

        triangle_uvs = torch.from_numpy(np.asarray(self.mesh.triangle_uvs))
        triangle_uvs = triangle_uvs.view(-1, 3, 2)

        # Calculate the lengths of the edges of mesh triangles and the respective normals in the texture space
        tri_side_normals = triangle_uvs.roll(-1, 1).sub_(triangle_uvs).roll(1, -1)
        tri_side_normals[..., 0] *= -1
        tri_side_len = tri_side_normals.norm(dim=-1)
        tri_side_normals = tri_side_normals.div_(tri_side_len.unsqueeze(-1) + eps)

        # Calculate the signed distance from the UV points to the triangle edges
        subtri_h = (uv.unsqueeze(1).unsqueeze(2) - triangle_uvs).mul_(tri_side_normals).sum(-1)
        del triangle_uvs, tri_side_normals, uv

        # Find the triangles containing the UV points
        keypoints_on_mesh, keypoint_tri = subtri_h.ge(0).all(-1).nonzero(as_tuple=True)
        unique = keypoints_on_mesh != keypoints_on_mesh.roll(1)
        keypoints_on_mesh = keypoints_on_mesh[unique]
        keypoint_tri = keypoint_tri[unique]; del unique

        # Calculate the barycentric coordinates of the UV points
        subtri_area = subtri_h[keypoints_on_mesh, keypoint_tri].mul_(tri_side_len[keypoint_tri])
        del tri_side_len, subtri_h
        bary_coords = subtri_area.roll(2, -1).div_(subtri_area.sum(-1, keepdim=True)); del subtri_area

        # Calculate the 3D coords of the UV points
        vertices = torch.from_numpy(np.asarray(self.mesh.vertices))
        triangles = torch.from_numpy(np.asarray(self.mesh.triangles)).long()
        keypoint_verts = vertices[triangles[keypoint_tri].view(-1)].view(-1, 3, 3); del vertices, triangles

        xyz = keypoint_verts.new_full([points_n, 3], np.nan)
        xyz[keypoints_on_mesh] = keypoint_verts.mul_(bary_coords.unsqueeze(-1)).sum(1)
        return xyz
