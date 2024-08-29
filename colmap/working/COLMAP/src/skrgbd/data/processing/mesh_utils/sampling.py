import numpy as np
import torch


def sample_on_grid(verts, tris, step, grid_type='square', progress_fn=None):
    device = verts.device
    tri_verts = verts[tris.view(-1)].view(*tris.shape, 3); del tris, verts

    # Order the vertices so that 0-1 are on the shortest edge
    edge_lens = (tri_verts.roll(-1, 1) - tri_verts).norm(dim=2)
    min_edge_i = edge_lens.min(1)[1]; del edge_lens
    perms = torch.stack([min_edge_i, min_edge_i + 1, min_edge_i + 2], 1); del min_edge_i
    perms = perms.remainder_(3)
    tri_verts = tri_verts.gather(1, perms.unsqueeze(2).expand_as(tri_verts)); del perms

    # For each triangle, calculate transformation to 2D "uv" coordinates,
    # with the origin at the center of the triangle, and v-axis pointing from the shortest edge to the opposite vertex
    uv_y = tri_verts[:, 2] - tri_verts[:, :2].mean(1)
    uv_x = tri_verts[:, 1] - tri_verts[:, 0]
    tri_normals = torch.cross(uv_x, uv_y, 1)
    tri_normals = torch.nn.functional.normalize(tri_normals, dim=1)
    uv_x = torch.cross(uv_y, tri_normals, 1)
    uv_y = torch.nn.functional.normalize(uv_y, dim=1)
    uv_x = torch.nn.functional.normalize(uv_x, dim=1)
    if grid_type == 'tri':
        uv_x = (uv_y * np.cos(np.pi / 3)).add_(uv_x * np.sin(np.pi / 3))
        uv_x = torch.nn.functional.normalize(uv_x, dim=1)
    elif grid_type != 'square':
        raise ValueError

    uv_0 = tri_verts.mean(1)
    tri_verts = tri_verts - uv_0.unsqueeze(1)

    if grid_type == 'square':
        p_u = uv_x
        p_v = uv_y
    elif grid_type == 'tri':
        # https://math.stackexchange.com/a/148218
        # https://en.wikipedia.org/wiki/Cramer%27s_rule#Explicit_formulas_for_small_systems
        uv_xy = (uv_x * uv_y).sum(1)
        den = uv_xy.pow(2).sub_(1)
        # Note: for some reason, the code in the end of this function is significantly slower
        # if pow(2) above is done in-place. Why, PyTorch?
        p_u = (uv_y * uv_xy.unsqueeze(1)).sub_(uv_x).div_(den.unsqueeze(1))
        p_v = (uv_x * uv_xy.unsqueeze(1)).sub_(uv_y).div_(den.unsqueeze(1))
        del uv_xy, den
    tri_verts_uv = torch.stack([(tri_verts * p_u.unsqueeze(1)).sum(2), (tri_verts * p_v.unsqueeze(1)).sum(2)], 2)
    del p_u, p_v, tri_verts

    # Calculate the grid range
    uv_min = tri_verts_uv.min(1)[0]
    uv_max = tri_verts_uv.max(1)[0]
    ji_min = uv_min.div_(step).floor_().long(); del uv_min
    ji_max = uv_max.div_(step).ceil_().long(); del uv_max

    def get_samples(tri_verts_uv, ji_min, ji_max, uv_0, uv_x, uv_y):
        r"""Samples points inside triangle from a square grid.
        Can be parallelized for multiple triangles, but is sufficiently fast in serial."""
        j, i = torch.meshgrid([
            torch.arange(ji_min[0], ji_max[0] + 1, device=device),
            torch.arange(ji_min[1], ji_max[1] + 1, device=device)
        ])
        del ji_min, ji_max
        uv = tri_verts_uv.new_empty([j.shape[0] * j.shape[1], 2])
        torch.mul(j.reshape(-1), step, out=uv[:, 0])
        torch.mul(i.reshape(-1), step, out=uv[:, 1]); del j, i

        for i in range(3):
            v1 = uv - tri_verts_uv[i]
            v2 = tri_verts_uv.roll(-1, 0)[i] - tri_verts_uv[i]
            cross = v1[:, 1] * v2[0] - v1[:, 0] * v2[1]; del v1, v2
            in_tri = cross >= 0; del cross
            uv = uv[in_tri]; del in_tri

        xyz = (uv[:, :1] * uv_x).add_(uv[:, 1:2] * uv_y).add_(uv_0)
        return xyz

    samples = []
    normals = []

    tri_ids = range(len(tri_verts_uv))
    if progress_fn is not None:
        tri_ids = progress_fn(tri_ids)
    for tri_i in tri_ids:
        new_samples = get_samples(
            tri_verts_uv[tri_i], ji_min[tri_i], ji_max[tri_i], uv_0[tri_i], uv_x[tri_i], uv_y[tri_i])
        samples.append(new_samples)
        new_normals = tri_normals[tri_i].unsqueeze(0).expand_as(new_samples)
        normals.append(new_normals)
        del new_samples, new_normals
    del tri_verts_uv, uv_0, uv_x, uv_y, ji_min, ji_max

    samples = torch.cat(samples, 0)
    normals = torch.cat(normals, 0)
    return samples, normals

