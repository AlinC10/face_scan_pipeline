"""
3D Model Format Converters & Draco Compressor Package
=====================================================
Modular converters for OBJ, GLB, STL, and 3MF formats,
featuring decoupled GLB generation and Draco compression.
"""

from pathlib import Path
from typing import Union
import trimesh

from .glb_converter import convert_to_glb
from .draco_compressor import apply_draco_compression
from .stl_converter import convert_to_stl
from .thrf_converter import convert_to_3mf


def convert_to_glb_draco(
    mesh: trimesh.Trimesh,
    output_glb_path: Union[str, Path],
    draco_compression_level: int = 7,
    draco: bool = True,
) -> Path:
    """
    Combined pipeline helper:
      1. Converts mesh to standard .GLB
      2. Applies Draco compression hook to that .GLB

    Maintains backward compatibility with earlier code while keeping
    the two operations decoupled internally.
    """
    out_path = Path(output_glb_path).resolve()
    # 1. Generate standard GLB
    convert_to_glb(mesh, out_path)
    # 2. Draco compression of the geometry (kept uncompressed if DracoPy is missing or encoding fails)
    if draco:
        apply_draco_compression(out_path, out_path, compression_level=draco_compression_level)
    return out_path


__all__ = [
    "convert_to_glb",
    "apply_draco_compression",
    "convert_to_stl",
    "convert_to_3mf",
    "convert_to_glb_draco",
]
