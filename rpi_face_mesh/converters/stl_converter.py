"""
STL Converter: Stereolithography Binary Exporter
================================================
Exports 3D mesh to STL format for slicers (Bambu Studio, Cura, PrusaSlicer).
"""

from pathlib import Path
from typing import Optional, Union

import trimesh


def _apply_export_scale(mesh: trimesh.Trimesh, scale: Optional[float]) -> None:
    """Explicit factor when given; otherwise the legacy guess from the mesh size (metres / viewport -> mm)."""
    if scale is not None:
        if scale != 1.0:
            mesh.apply_scale(float(scale))
        return
    max_d = float(mesh.extents.max())
    if max_d < 0.5:
        mesh.apply_scale(1000.0)
    elif max_d < 5.0:
        mesh.apply_scale(100.0)


def convert_to_stl(
    mesh: trimesh.Trimesh,
    output_stl_path: Union[str, Path],
    enforce_watertight: bool = True,
    scale: Optional[float] = None,
) -> Path:
    """
    Exports 3D mesh to STL binary format for 3D slicing software.
    Optionally repairs open boundary seams to ensure manifold geometry.

    Args:
        mesh: trimesh.Trimesh object to export.
        output_stl_path: Destination path for the .stl file.
        enforce_watertight: Whether to run boundary hole filling.
        scale: Factor applied to the coordinates. None keeps the legacy guess from the mesh size;
            the face pipeline always passes its export scale explicitly (same units in every format).

    Returns:
        Path to the saved .stl file.
    """
    out_path = Path(output_stl_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_mesh = mesh.copy()
    _apply_export_scale(export_mesh, scale)

    if enforce_watertight and not export_mesh.is_watertight:
        trimesh.repair.fill_holes(export_mesh)

    export_mesh.export(str(out_path))
    return out_path
