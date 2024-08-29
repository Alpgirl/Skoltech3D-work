import numpy as np


def save_ply(file, verts, tris=None, *, vert_normals=None, vert_colors=None, vert_quality=None):
    r"""Saves mesh data to binary PLY.
    
    Parameters
    ----------
    file : str
    verts : np.ndarray
        of shape [verts_n, 3].
    tris : np.ndarray
        of shape [tris_n, 3].
    vert_normals : np.ndarray
        of shape [verts_n, 3].
    vert_colors : np.ndarray
        of shape [verts_n, 3].
    vert_quality : np.ndarray
        of shape [verts_n].
    """
    header = ['ply', 'format binary_little_endian 1.0']

    # Verts
    verts_n = len(verts)
    verts_header = [f'element vertex {verts_n}', 'property float x', 'property float y', 'property float z']
    verts_dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
    if vert_normals is not None:
        verts_header.extend(['property float nx', 'property float ny', 'property float nz'])
        verts_dtype.extend([('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4')])
    if vert_colors is not None:
        verts_header.extend(['property uchar red', 'property uchar green', 'property uchar blue'])
        verts_dtype.extend([('red', '<u1'), ('green', '<u1'), ('blue', '<u1')])
    if vert_quality is not None:
        verts_header.append('property float quality')
        verts_dtype.append(('q', '<f4'))
    header += verts_header

    verts_data = np.empty([verts_n], dtype=verts_dtype)
    for prop, data in zip(['x', 'y', 'z'], verts.T):
        verts_data[prop] = data
    if vert_normals is not None:
        for prop, data in zip(['nx', 'ny', 'nz'], vert_normals.T):
            verts_data[prop] = data
    if vert_colors is not None:
        for prop, data in zip(['red', 'green', 'blue'], vert_colors.T):
            verts_data[prop] = data
    if vert_quality is not None:
        verts_data['q'] = vert_quality

    # Tris
    if tris is not None:
        tris_n = len(tris)
        faces_header = [f'element face {tris_n}', 'property list uchar uint vertex_indices']
        faces_dtype = [('vn', '<u1'), ('v1', '<u4'), ('v2', '<u4'), ('v3', '<u4')]
        header += faces_header

        faces_data = np.empty([tris_n], dtype=faces_dtype)
        faces_data['vn'] = 3
        for prop, data in zip(['v1', 'v2', 'v3'], tris.T):
            faces_data[prop] = data

    header += ['end_header']

    with open(file, 'w') as file:
        header = '\n'.join(header) + '\n'
        file.write(header)
        verts_data.tofile(file)
        if tris is not None:
            faces_data.tofile(file)
