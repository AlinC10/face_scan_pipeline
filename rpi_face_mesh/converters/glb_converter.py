"""
GLB Converter: Standard glTF 2.0 Binary Export
=============================================
Exports a 3D trimesh to standard binary .GLB format.
"""

from pathlib import Path
from typing import Union
import trimesh


def convert_to_glb(
    mesh: trimesh.Trimesh,
    output_glb_path: Union[str, Path],
) -> Path:
    """
    Exports 3D mesh to standard binary .GLB format with embedded materials and textures.

    Args:
        mesh: trimesh.Trimesh object to export.
        output_glb_path: Destination path for the .glb file.

    Returns:
        Path to the saved .glb file.
    """
    out_path = Path(output_glb_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mesh.export(str(out_path))
    return out_path
