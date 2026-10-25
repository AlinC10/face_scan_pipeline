"""
Metric 3D Triangulation & Screened Poisson Surface Reconstruction (Etapa 4)
===========================================================================
Triangulates metric 3D point cloud from calibrated disparity maps (Z = f * B / d),
computes surface normal vectors, reconstructs watertight manifold geometry using
Screened Poisson (Octree Depth 9-10), and performs Density Trimming to remove
the spherical bloated geometry behind the open facial point cloud.
"""

import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import cv2
import numpy as np
import open3d as o3d
import trimesh

log = logging.getLogger("Photogrammetry.Poisson")

DEFAULT_OCTREE_DEPTH = 9
DEFAULT_DENSITY_TRIM_PERCENTILE = 6.0  # Trims bottom 6% low-density extrapolated triangles


@dataclass
class PoissonReconstructionResult:
    """Contains the reconstructed master high-poly facial mesh and metric statistics."""
    mesh: trimesh.Trimesh
    num_vertices: int
    num_faces: int
    point_cloud_size: int
    density_trim_percentile: float
    octree_depth: int
    bbox_dimensions_mm: Tuple[float, float, float]  # (dx, dy, dz)
    execution_time_s: float
    centroid_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def save_mesh(self, output_path: Union[str, Path]) -> Path:
        """Exports the master high-poly mesh as OBJ."""
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        self.mesh.export(str(out_p))
        log.info("Saved Master High-Poly Mesh (%d faces) to: %s", self.num_faces, out_p)
        return out_p


class MetricPoissonReconstructor:
    """
    Executes metric 3D triangulation and density-trimmed Screened Poisson reconstruction.
    """

    def __init__(
        self,
        octree_depth: int = DEFAULT_OCTREE_DEPTH,
        density_trim_percentile: float = DEFAULT_DENSITY_TRIM_PERCENTILE,
        min_depth_mm: float = 120.0,
        max_depth_mm: float = 750.0,
    ):
        self.octree_depth = octree_depth
        self.density_trim_percentile = density_trim_percentile
        self.min_depth_mm = min_depth_mm
        self.max_depth_mm = max_depth_mm

    def triangulate_and_reconstruct(
        self,
        disparity_2k: np.ndarray,
        rect_left_2k: np.ndarray,
        focal_px_2k: float,
        baseline_mm: float,
        mask_2k: Optional[np.ndarray] = None,
        d0_shift: int = 0,
    ) -> PoissonReconstructionResult:
        """
        Converts disparity map into metric 3D point cloud and executes Screened Poisson surface reconstruction.
        """
        t_start = time.time()
        h, w = disparity_2k.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        # 1. Mask valid pixels
        valid_mask = disparity_2k > (float(d0_shift) + 2.0)
        if mask_2k is not None:
            valid_mask = valid_mask & (mask_2k > 127)

        y_idx, x_idx = np.where(valid_mask)
        if len(y_idx) < 1000:
            raise ValueError(f"Insufficient valid 3D points ({len(y_idx)}) for Poisson reconstruction.")

        disp_vals = disparity_2k[valid_mask]

        # 2. Metric Triangulation: Z = f * B / d (in millimeters)
        z_mm = (focal_px_2k * baseline_mm) / disp_vals

        # Filter depths outside plausible human head range
        depth_filter = (z_mm >= self.min_depth_mm) & (z_mm <= self.max_depth_mm)
        z_mm = z_mm[depth_filter]
        x_idx = x_idx[depth_filter]
        y_idx = y_idx[depth_filter]

        if len(z_mm) < 1000:
            raise ValueError(f"No points remaining in plausible depth range [{self.min_depth_mm}, {self.max_depth_mm}] mm.")

        # Metric X, Y in mm:
        x_mm = (x_idx - cx) * (z_mm / focal_px_2k)
        y_mm = -(y_idx - cy) * (z_mm / focal_px_2k)  # Invert so +Y points UP (towards forehead)

        # Center point cloud coordinates around facial centroid
        x_med = float(np.median(x_mm))
        y_med = float(np.median(y_mm))
        z_med = float(np.median(z_mm))
        x_c = x_mm - x_med
        y_c = y_mm - y_med
        z_c = -(z_mm - z_med)  # Invert so +Z points outward from face towards camera

        points_3d = np.stack([x_c, y_c, z_c], axis=1)
        colors = rect_left_2k[y_idx, x_idx, ::-1].astype(np.float64) / 255.0  # RGB in [0, 1]

        log.info("Triangulated %d metric 3D surface points. Estimating surface normals...", len(points_3d))

        # 3. Open3D Point Cloud & Normal Estimation
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Statistical outlier removal
        pcd, inlier_indices = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
        log.info("Filtered point cloud: %d clean inliers retained.", len(pcd.points))

        # Estimate surface normals oriented outwards towards the camera
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30)
        )
        pcd.orient_normals_towards_camera_location(camera_location=[0.0, 0.0, 500.0])

        # 4. Screened Poisson Surface Reconstruction (Depth 9-10)
        log.info("Executing Screened Poisson Reconstruction (Octree Depth: %d)...", self.octree_depth)
        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=self.octree_depth,
            width=0,
            scale=1.1,
            linear_fit=False
        )

        densities_np = np.asarray(densities)
        raw_faces = len(mesh_o3d.triangles)
        log.info("Poisson raw mesh generated: %d vertices, %d faces.", len(mesh_o3d.vertices), raw_faces)

        # 5. Density Trimming: eliminate spherical bloated geometry behind the head
        if self.density_trim_percentile > 0.0:
            density_thresh = float(np.percentile(densities_np, self.density_trim_percentile))
            low_density_mask = densities_np < density_thresh
            mesh_o3d.remove_vertices_by_mask(low_density_mask)
            mesh_o3d.remove_degenerate_triangles()
            mesh_o3d.remove_non_manifold_edges()
            mesh_o3d.remove_unreferenced_vertices()
            log.info(
                "Density Trimming (Threshold: %.2f, percentile: %.1f%%): Trimmed to %d vertices, %d faces.",
                density_thresh, self.density_trim_percentile, len(mesh_o3d.vertices), len(mesh_o3d.triangles)
            )

        # 6. Convert to Trimesh for downstream processing
        vertices = np.asarray(mesh_o3d.vertices)
        triangles = np.asarray(mesh_o3d.triangles)
        v_colors = np.asarray(mesh_o3d.vertex_colors) if mesh_o3d.has_vertex_colors() else None

        tri_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=triangles,
            vertex_colors=(v_colors * 255.0).astype(np.uint8) if v_colors is not None and len(v_colors) > 0 else None,
            process=True
        )

        bbox = tri_mesh.bounding_box.extents
        bbox_dims = (float(bbox[0]), float(bbox[1]), float(bbox[2]))
        dt = time.time() - t_start

        log.info(
            "Etapa 4 Completed in %.2fs: Master High-Poly Mesh = %d faces, BBox: %.1fx%.1fx%.1f mm",
            dt, len(tri_mesh.faces), bbox_dims[0], bbox_dims[1], bbox_dims[2]
        )

        return PoissonReconstructionResult(
            mesh=tri_mesh,
            num_vertices=len(tri_mesh.vertices),
            num_faces=len(tri_mesh.faces),
            point_cloud_size=len(points_3d),
            density_trim_percentile=self.density_trim_percentile,
            octree_depth=self.octree_depth,
            bbox_dimensions_mm=bbox_dims,
            execution_time_s=dt,
            centroid_mm=(x_med, y_med, z_med),
        )
