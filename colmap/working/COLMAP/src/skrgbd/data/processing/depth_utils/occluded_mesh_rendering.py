import open3d as o3d

from skrgbd.data.processing.depth_utils.mesh_rendering import MeshRenderer as BaseMeshRenderer


class MeshRenderer(BaseMeshRenderer):
    def __init__(self, scan, occlusion_mesh, occ_thres):
        super().__init__(scan)

        self.occ_mesh = o3d.t.geometry.TriangleMesh.from_legacy(occlusion_mesh)
        self.occ_raycasting = o3d.t.geometry.RaycastingScene()
        self.occ_raycasting.add_triangles(self.occ_mesh)
        self.occ_thres = occ_thres

    def render_rays(self, casted_rays, **kwargs):
        # See the docstring from the base class
        scan_renderer = self.raycasting
        occ_renderer = self.occ_raycasting

        self.raycasting = scan_renderer
        scan_render = super().render_rays(casted_rays, **kwargs)
        self.raycasting = occ_renderer
        # For a watertight occlusion mesh hitting the back of a face means hitting the inside, which may happen due to
        # bugs in raycasting. So we unconditionally cull here to set the depth of these buggy hits to inf,
        # and reject the respective scan hits later.
        occ_render = super().render_rays(casted_rays, cull_back_faces=True, backface_val=float('nan'))
        self.raycasting = scan_renderer

        scan_depth = scan_render['ray_hit_depth']
        occ_depth = occ_render['ray_hit_depth'].add_(self.occ_thres); del occ_render
        not_occluded = scan_depth <= occ_depth; del occ_depth
        scan_depth = scan_depth.where(not_occluded, scan_depth.new_tensor(float('inf')))

        scan_render['ray_hit_depth'] = scan_depth
        return scan_render
