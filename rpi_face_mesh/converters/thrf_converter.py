"""
3MF Converter: 3D Manufacturing Format Exporter
==============================================
Exports 3D mesh to modern .3MF format, preserving unit metadata and coordinate systems.
"""

from pathlib import Path
from typing import Optional, Union

import trimesh

from .stl_converter import _apply_export_scale


def convert_to_3mf(
    mesh: trimesh.Trimesh,
    output_3mf_path: Union[str, Path],
    scale: Optional[float] = None,
) -> Path:
    """
    Exports 3D mesh to modern .3MF format preserving geometric integrity.

    Args:
        mesh: trimesh.Trimesh object to export.
        output_3mf_path: Destination path for the .3mf file.
        scale: Factor applied to the coordinates. None keeps the legacy guess from the mesh size;
            the face pipeline always passes its export scale explicitly (same units in every format).

    Returns:
        Path to the saved .3mf file.
    """
    out_path = Path(output_3mf_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_mesh = mesh.copy()
    _apply_export_scale(export_mesh, scale)

    export_mesh.export(str(out_path))
    return out_path
