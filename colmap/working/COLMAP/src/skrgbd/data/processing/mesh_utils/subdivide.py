import open3d as o3d
import torch

from skrgbd.utils.logging import logger


def split_long_edges(verts_xyz, tri_vert_ids, max_edge_len, max_iters_n=1_000_000, dtype=torch.double):
    r"""Subdivides long edges of the mesh.

    Parameters
    ----------
    verts_xyz : torch.Tensor
        of shape [verts_n, 3].
    tri_vert_ids : torch.LongTensor
        of shape [tris_n, 3].
    max_edge_len : float
    max_iters_n : int
    dtype : torch.dtype

    Returns
    -------
    mesh : o3d.geometry.TriangleMesh
    """
    tris = verts_xyz[tri_vert_ids.view(-1)].view(*tri_vert_ids.shape, 3); del verts_xyz, tri_vert_ids
    tris = tris.to(dtype)

    final_tris = []
    final_tris_n = 0

    f'Find edge lens' >> logger.debug
    edge_lens = tris.roll(-1, 1).sub(tris).norm(dim=2)

    for split_i in range(max_iters_n):
        tri_to_subdiv_n = len(tris)
        f'Iter {split_i:02}, final {final_tris_n:>9}, to subdiv {tri_to_subdiv_n:>9}' >> logger.debug

        'Find longest edges' >> logger.debug
        l_edge_lens, l_edge_ids = edge_lens.max(dim=1)

        'Find long edges' >> logger.debug
        tri_is_final = l_edge_lens.le(max_edge_len); del l_edge_lens
        final_tris.append(tris[tri_is_final].clone())
        final_tris_n += len(final_tris[-1])

        tri_ids = tri_is_final.logical_not_().nonzero(as_tuple=True)[0]; del tri_is_final
        if len(tri_ids) == 0:
            del l_edge_ids
            break
        l_edge_ids = l_edge_ids[tri_ids]

        'Make new tris' >> logger.debug
        v1_ids = l_edge_ids; del l_edge_ids
        v1 = tris[tri_ids, v1_ids]

        v2_ids = v1_ids.add(1).remainder_(3)
        v2 = tris[tri_ids, v2_ids]
        e2 = edge_lens[tri_ids, v2_ids]; del v2_ids

        v3_ids = v1_ids.add(2).remainder_(3); del v1_ids
        v3 = tris[tri_ids, v3_ids]; del tris
        e3 = edge_lens[tri_ids, v3_ids]; del v3_ids, tri_ids, edge_lens

        v_new = v1.add(v2).div_(2)
        e_new_1 = v_new.sub(v1).norm(dim=1)
        e_new_2 = v_new.sub(v2).norm(dim=1)
        e_new_3 = v_new.sub(v3).norm(dim=1)

        edge_lens = torch.stack([e_new_1, e_new_3, e3, e_new_3, e_new_2, e2], 1); del e2, e3, e_new_1, e_new_2, e_new_3
        edge_lens = edge_lens.view(-1, 3)

        tris = torch.stack([v1, v_new, v3, v3, v_new, v2], 1); del v1, v2, v3, v_new
        tris = tris.view(-1, 3, 3)

        split_i += 1
    del edge_lens

    'Combine' >> logger.debug
    final_tris.append(tris); del tris
    verts_xyz_sub = torch.cat(final_tris, 0).view(-1, 3); del final_tris
    tri_vert_ids_sub = torch.arange(0, len(verts_xyz_sub), dtype=torch.int, device=verts_xyz_sub.device).view(-1, 3)

    f'Build mesh: V# {len(verts_xyz_sub)}, F# {len(tri_vert_ids_sub)}' >> logger.debug
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_xyz_sub.double().numpy()); del verts_xyz_sub
    mesh.triangles = o3d.utility.Vector3iVector(tri_vert_ids_sub.int().numpy()); del tri_vert_ids_sub

    'Remove duplicates' >> logger.debug
    mesh = mesh.remove_duplicated_vertices()
    f'Final mesh: V# {len(mesh.vertices)}' >> logger.debug

    return mesh
