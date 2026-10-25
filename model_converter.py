"""
Model Converter: Backward Compatibility Facade
==============================================
Re-exports conversion functions from the new modular 'converters' package.
"""

from converters import (
    convert_to_glb,
    apply_draco_compression,
    convert_to_stl,
    convert_to_3mf,
    convert_to_glb_draco,
)

__all__ = [
    "convert_to_glb",
    "apply_draco_compression",
    "convert_to_stl",
    "convert_to_3mf",
    "convert_to_glb_draco",
]
