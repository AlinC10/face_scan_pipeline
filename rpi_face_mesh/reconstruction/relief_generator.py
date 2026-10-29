"""
Relief Generator: Facial Hair Micro-Displacement & Normal Map Baking
=====================================================================
Extracts semantic height maps for eyebrows, mustache, and beard from facial textures,
applies geometric normal displacement at Subdivision Level 1, and bakes PBR normal maps.
"""

from typing import Tuple, Optional
from pathlib import Path
import cv2
import numpy as np
import trimesh
from PIL import Image


def generate_facial_hair_height_map(
    texture_bgr: np.ndarray,
    hair_regions: bool = True
) -> np.ndarray:
    """
    Computes a continuous, normalized [0.0, 1.0] height displacement map
    from facial texture luminance and micro-gradients.

    Semantic UV Zones:
      - Right Eyebrow: U in [0.18, 0.48], V in [0.65, 0.82]
      - Left Eyebrow:  U in [0.52, 0.82], V in [0.65, 0.82]
      - Mustache:      U in [0.36, 0.64], V in [0.30, 0.44]
      - Beard / Jaw:   U in [0.22, 0.78], V in [0.02, 0.30]

    Args:
        texture_bgr: Fused face texture in BGR format (H, W, 3).
        hair_regions: Whether to confine displacement to facial hair regions.

    Returns:
        Floating-point Height Map of shape (H, W) with values in [0.0, 1.0].
    """
    h_tex, w_tex = texture_bgr.shape[:2]

    # Helper function: converts normalized UV (0..1) to image coordinates (x, y)
    def to_img(u: float, v: float) -> Tuple[int, int]:
        return int(np.clip(u, 0.0, 1.0) * w_tex), int(np.clip(1.0 - v, 0.0, 1.0) * h_tex)

    # 1. Build smooth semantic mask for facial hair zones with strict anatomical exclusions
    mask = np.zeros((h_tex, w_tex), dtype=np.float32)
    exclusion = np.zeros((h_tex, w_tex), dtype=np.float32)

    if hair_regions:
        # --- POSITIVE REGIONS (Hair Candidates) ---
        # Right Eyebrow (Strict anatomical boundary, does not bleed onto forehead)
        cv2.ellipse(mask, to_img(0.34, 0.71), (int(0.12 * w_tex), int(0.04 * h_tex)), 0, 0, 360, 1.0, -1)

        # Left Eyebrow (Strict anatomical boundary, does not bleed onto forehead)
        cv2.ellipse(mask, to_img(0.66, 0.71), (int(0.12 * w_tex), int(0.04 * h_tex)), 0, 0, 360, 1.0, -1)

        # Mustache candidate (philtrum and nasolabial wings)
        cv2.ellipse(mask, to_img(0.50, 0.375), (int(0.17 * w_tex), int(0.045 * h_tex)), 0, 0, 360, 1.0, -1)

        # Chin and Jawline / Beard
        cv2.ellipse(mask, to_img(0.50, 0.12), (int(0.28 * w_tex), int(0.11 * h_tex)), 0, 0, 360, 1.0, -1)
        # Lateral beard wings / sideburns
        pts_jaw_r = np.array([to_img(0.20, 0.04), to_img(0.20, 0.28), to_img(0.35, 0.22), to_img(0.35, 0.04)], dtype=np.int32)
        pts_jaw_l = np.array([to_img(0.80, 0.04), to_img(0.80, 0.28), to_img(0.65, 0.22), to_img(0.65, 0.04)], dtype=np.int32)
        cv2.fillPoly(mask, [pts_jaw_r, pts_jaw_l], 1.0)

        # --- STRICT ANATOMICAL EXCLUSION ZONES (Zero Hair Allowed) ---
        # 1. LIPS (Vermilion border & oral aperture): V in [0.23, 0.355], U in [0.37, 0.63]
        cv2.ellipse(exclusion, to_img(0.50, 0.295), (int(0.13 * w_tex), int(0.058 * h_tex)), 0, 0, 360, 1.0, -1)

        # 2. NOSE (Nose bridge, dorsum, tip, columella, and nostrils): V in [0.395, 0.65]
        # Main nose body & bridge
        cv2.ellipse(exclusion, to_img(0.50, 0.52), (int(0.09 * w_tex), int(0.12 * h_tex)), 0, 0, 360, 1.0, -1)
        # Nostrils & columella base
        cv2.ellipse(exclusion, to_img(0.50, 0.425), (int(0.085 * w_tex), int(0.030 * h_tex)), 0, 0, 360, 1.0, -1)

        # 3. EYES (Orbit & eyelids)
        cv2.ellipse(exclusion, to_img(0.35, 0.62), (int(0.09 * w_tex), int(0.045 * h_tex)), 0, 0, 360, 1.0, -1)
        cv2.ellipse(exclusion, to_img(0.65, 0.62), (int(0.09 * w_tex), int(0.045 * h_tex)), 0, 0, 360, 1.0, -1)

        # Subtract exclusions from candidate mask
        mask = np.clip(mask - exclusion, 0.0, 1.0)

        # Feather mask edges with blur
        ksize = max(9, (min(h_tex, w_tex) // 100) | 1)
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        # Hard-enforce zero on all exclusion zones (lips, nose, eyes) to prevent Gaussian bleed
        mask[exclusion > 0.01] = 0.0
    else:
        mask.fill(1.0)

    # 2. Extract luminance and hair micro-contrast
    gray = cv2.cvtColor(texture_bgr, cv2.COLOR_BGR2GRAY)

    # Baseline skin tone via large median blur
    k_median = max(15, (min(h_tex, w_tex) // 60) | 1)
    skin_ref = cv2.medianBlur(gray, k_median)

    # Hair contrast: darker pixels relative to local skin reference
    darkness = np.clip((skin_ref.astype(np.float32) - gray.astype(np.float32)) / 45.0, 0.0, 1.0)

    # Hair strand edges via Sobel gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_norm = np.clip(grad_mag / 40.0, 0.0, 1.0)

    # 3. Combine signals and filter noise
    combined_signal = (darkness * 0.65 + grad_norm * 0.35) * mask
    k_smooth = max(7, (min(h_tex, w_tex) // 180) | 1)
    height_map = cv2.GaussianBlur(combined_signal, (k_smooth, k_smooth), 0)

    # Ensure hard-zero exclusion on height map as well
    if hair_regions:
        height_map[exclusion > 0.01] = 0.0

    max_val = float(np.max(height_map))
    if max_val > 1e-6:
        height_map = np.clip(height_map / max_val, 0.0, 1.0)
    else:
        height_map.fill(0.0)

    return height_map


def generate_tangent_normal_map(
    height_map: np.ndarray,
    strength: float = 3.5
) -> np.ndarray:
    """
    Computes a Tangent-Space Normal Map (BGR 8-bit image) from a height map.

    In standard tangent space:
      - X (Red)   = -dH/dx * strength
      - Y (Green) = -dH/dy * strength
      - Z (Blue)  = 1.0 (pointing outward)
      - Flat neutral normal = RGB(128, 128, 255) / BGR(255, 128, 128)
    """
    dx = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3)

    nx = -dx * strength
    ny = -dy * strength
    nz = np.ones_like(height_map, dtype=np.float32)

    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= norm
    ny /= norm
    nz /= norm

    normal_bgr = np.zeros((*height_map.shape, 3), dtype=np.uint8)
    normal_bgr[..., 0] = np.clip((nz * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)  # Blue = Z
    normal_bgr[..., 1] = np.clip((ny * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)  # Green = Y
    normal_bgr[..., 2] = np.clip((nx * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)  # Red = X

    return normal_bgr


def apply_hair_displacement_to_mesh(
    mesh_low: trimesh.Trimesh,
    height_map: Optional[np.ndarray] = None,
    max_displacement_cm: float = 0.0,
    subdivide_level: int = 1,
    enable_displacement: bool = False,
) -> trimesh.Trimesh:
    """
    Subdivides the canonical face mesh to Level 1 or Level 2 preserving clean anatomical curvature.
    Geometric vertex displacement is DEACTIVATED by default to guarantee a completely smooth,
    anatomically pure surface with zero artificial ripples or waves over nose, lips, or skin.

    Args:
        mesh_low: Input mesh with ears (490 vertices, 934 faces).
        height_map: Optional height map. Unused when displacement is disabled.
        max_displacement_cm: Displacement magnitude (0.0 = completely smooth).
        subdivide_level: Subdivision iterations (1 = 1,913 verts, 2 = 7,561 verts).
        enable_displacement: False by default to prevent 2D image pixel noise from warping geometry.

    Returns:
        Clean, smoothly subdivided face mesh without geometric ripples.
    """
    # 1. Subdivide mesh to create smooth, dense anatomical curvature
    subdivided = mesh_low.copy()
    for _ in range(subdivide_level):
        subdivided = subdivided.subdivide()

    # If displacement is deactivated, return the clean, perfectly smooth subdivided mesh
    if not enable_displacement or height_map is None or max_displacement_cm <= 0.0:
        return subdivided

    uvs = subdivided.visual.uv
    h_tex, w_tex = height_map.shape[:2]

    # Map UV [0..1] to texture image indices
    px = np.clip((uvs[:, 0] * (w_tex - 1)).astype(np.int32), 0, w_tex - 1)
    py = np.clip(((1.0 - uvs[:, 1]) * (h_tex - 1)).astype(np.int32), 0, h_tex - 1)

    # Sample height values
    h_samples = height_map[py, px]  # (N_verts,)

    # 2. Displace vertices along outward normal vectors (adapts to mesh viewport scale)
    normals = subdivided.vertex_normals
    max_dim = float(subdivided.extents.max())
    if max_dim < 0.5:
        actual_displacement = max_displacement_cm * 0.01  # meters
    elif max_dim < 5.0:
        actual_displacement = max_displacement_cm * 0.10  # viewport decimeters
    else:
        actual_displacement = max_displacement_cm  # centimeters
    displacement_vectors = normals * (h_samples[:, np.newaxis] * actual_displacement)

    displaced_mesh = subdivided.copy()
    displaced_mesh.vertices += displacement_vectors

    return displaced_mesh
