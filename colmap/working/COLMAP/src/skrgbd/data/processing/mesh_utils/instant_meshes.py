import subprocess
import tempfile

import open3d as o3d


def remesh(mesh, instant_meshes_bin, edge_len=None, tri_n=None, verts_n=None, tmpdir='/tmp'):
    r"""Does isotropic triangular remeshing of a mesh using the method of
        Jakob et al (2015). Instant Field-Aligned Meshes. SIGGRAPH Asia.

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
    instant_meshes_bin : str
    edge_len : float
        Desired edge length.
    tri_n : int
        Desired face count.
    vert_n : int
        Desired vertex count.
    tmpdir : str

    Returns
    -------
    out_mesh : o3d.geometry.TriangleMesh
    """
    with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdir:
        in_mesh = f'{tmpdir}/in.ply'
        o3d.io.write_triangle_mesh(in_mesh, mesh)

        out_mesh = f'{tmpdir}/out.ply'
        command = f'{instant_meshes_bin} -o {out_mesh} -r 6 -p 6'
        if edge_len is not None:
            command = f'{command} -s {edge_len}'
        if tri_n is not None:
            command = f'{command} -f {tri_n}'
        if verts_n is not None:
            command = f'{command} -v {verts_n}'
        command = f'{command} {in_mesh}'

        subprocess.run(command.split(), check=True)
        out_mesh = o3d.io.read_triangle_mesh(out_mesh)
    return out_mesh
