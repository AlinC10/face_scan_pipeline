"""
Mesh Decimation (QEM) & xatlas UV Unwrapping Module (Etapa 5)
=============================================================
Performs Quadric Error Metric (QEM) decimation on the master high-poly mesh
to produce a clean, production-ready Low-Poly manifold (15,000 – 30,000 faces).
Unwraps UV charts with xatlas directly on the decimated mesh for optimal UV packing,
ready for texture projection (Albedo) and tangent-space normal map baking.
"""

import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import open3d as o3d
import trimesh
import xatlas

log = logging.getLogger("Photogrammetry.DecimationUV")

DEFAULT_TARGET_FACES = 20000


@dataclass
class DecimationUVResult:
    """Contains the UV-unwrapped low-poly mesh and decimation statistics."""
    low_mesh: trimesh.Trimesh
    high_face_count: int
    low_face_count: int
    low_vertex_count: int
    target_faces: int
    uv_coordinates: np.ndarray          # (V, 2) float32 in [0, 1]
    execution_time_s: float

    def save_mesh(self, output_path: Union[str, Path]) -> Path:
        """Exports the UV-unwrapped low-poly mesh as OBJ."""
        out_p = Path(output_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        self.low_mesh.export(str(out_p))
        log.info("Saved UV-unwrapped Low-Poly Mesh (%d faces) to: %s", self.low_face_count, out_p)
        return out_p


class MeshDecimatorUV:
    """
    Executes Quadric Error Decimation and xatlas UV atlas parameterization.
    """

    def __init__(self, target_faces: int = DEFAULT_TARGET_FACES):
        self.target_faces = target_faces

    def decimate_and_unwrap(
        self,
        high_mesh: trimesh.Trimesh,
        target_faces: Optional[int] = None,
    ) -> DecimationUVResult:
        """
        Simplifies the high-poly mesh using Open3D QEM decimation, then unwraps UV coordinates with xatlas.
        """
        t_start = time.time()
        n_target = target_faces or self.target_faces
        high_faces = len(high_mesh.faces)

        log.info("Starting QEM Decimation: High-Poly (%d faces) -> Target Low-Poly (%d faces)...", high_faces, n_target)

        # 1. Open3D Quadric Decimation
        o3d_high = o3d.geometry.TriangleMesh()
        o3d_high.vertices = o3d.utility.Vector3dVector(high_mesh.vertices.astype(np.float64))
        o3d_high.triangles = o3d.utility.Vector3iVector(high_mesh.faces.astype(np.int32))

        o3d_decimated = o3d_high.simplify_quadric_decimation(target_number_of_triangles=n_target)
        o3d_decimated.remove_degenerate_triangles()
        o3d_decimated.remove_non_manifold_edges()
        o3d_decimated.remove_unreferenced_vertices()

        dec_verts = np.asarray(o3d_decimated.vertices, dtype=np.float32)
        dec_faces = np.asarray(o3d_decimated.triangles, dtype=np.int32)
        low_faces = len(dec_faces)
        log.info("QEM Decimation complete: %d vertices, %d faces.", len(dec_verts), low_faces)

        # 2. xatlas UV Parameterization on Decimated Mesh
        log.info("Executing xatlas UV unwrapping on %d faces...", low_faces)
        t_uv = time.time()
        vmapping, indices, uvs = xatlas.parametrize(dec_verts, dec_faces)
        dt_uv = time.time() - t_uv

        unwrapped_verts = dec_verts[vmapping]
        unwrapped_faces = indices.astype(np.int64)

        # 3. Construct Textured Trimesh
        low_mesh = trimesh.Trimesh(
            vertices=unwrapped_verts,
            faces=unwrapped_faces,
            visual=trimesh.visual.TextureVisuals(uv=uvs.astype(np.float32)),
            process=True
        )

        dt = time.time() - t_start
        log.info(
            "Etapa 5 Completed in %.2fs (xatlas took %.2fs): Low-Poly Mesh = %d faces, %d vertices with UV atlas.",
            dt, dt_uv, len(low_mesh.faces), len(low_mesh.vertices)
        )

        return DecimationUVResult(
            low_mesh=low_mesh,
            high_face_count=high_faces,
            low_face_count=len(low_mesh.faces),
            low_vertex_count=len(low_mesh.vertices),
            target_faces=n_target,
            uv_coordinates=uvs,
            execution_time_s=dt,
        )
