"""
3D Reconstruction & Topology Package
====================================
"""

from .ear_generator import (
    attach_ears_to_canonical_mesh,
    deform_mesh_with_ears,
    compute_ear_coordinate_frame,
    compute_2d_ear_pixel_points,
)
from .relief_generator import (
    generate_facial_hair_height_map,
    generate_tangent_normal_map,
    apply_hair_displacement_to_mesh,
)

__all__ = [
    "attach_ears_to_canonical_mesh",
    "deform_mesh_with_ears",
    "compute_ear_coordinate_frame",
    "compute_2d_ear_pixel_points",
    "generate_facial_hair_height_map",
    "generate_tangent_normal_map",
    "apply_hair_displacement_to_mesh",
]
