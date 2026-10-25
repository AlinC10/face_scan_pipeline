"""
3MF Converter: 3D Manufacturing Format Exporter
==============================================
Exports 3D mesh to modern .3MF format, preserving unit metadata and coordinate systems.
"""

from pathlib import Path
from typing import Union
import trimesh


def convert_to_3mf(
    mesh: trimesh.Trimesh,
    output_3mf_path: Union[str, Path],
) -> Path:
    """
    Exports 3D mesh to modern .3MF format preserving geometric integrity.

    Args:
        mesh: trimesh.Trimesh object to export.
        output_3mf_path: Destination path for the .3mf file.

    Returns:
        Path to the saved .3mf file.
    """
    out_path = Path(output_3mf_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_mesh = mesh.copy()
    # Convert viewport scale or meters to millimeters (~150 mm) for 3MF manufacturing.
    max_d = float(export_mesh.extents.max())
    if max_d < 0.5:
        export_mesh.apply_scale(1000.0)
    elif max_d < 5.0:
        export_mesh.apply_scale(100.0)

    export_mesh.export(str(out_path))
    return out_path
