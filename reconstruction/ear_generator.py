"""
Ear Generator: Anatomical Ear Topology & Stitching for Canonical Face Mesh
==========================================================================
Generates realistic 3D human ears (helix, antihelix, concha, lobule)
stitched directly onto the MediaPipe canonical face boundary.

Anchors:
  - Right Ear (Subject's right, seen by Camera 0):
      Anchored to vertices [127 (top), 234 (tragus), 93 (concha), 132 (lobule)]
  - Left Ear (Subject's left, seen by Camera 1):
      Anchored to vertices [356 (top), 454 (tragus), 323 (concha), 361 (lobule)]
"""

from typing import Tuple, List, Dict, Any
import numpy as np
import trimesh

# Local ear offsets in (Outward Normal, Up Vector, Forward Vector)
# normalized as fractions of ear height (~5.3 units in canonical scale)
EAR_OFFSETS: List[Tuple[float, float, float]] = [
    # Helix outer rim (H0..H5) - Anatomical human auriculocephalic angle (~18-20 deg to skull)
    (0.15,  0.48, -0.05), # 0: H0 (Helix apex)
    (0.26,  0.30, -0.18), # 1: H1 (Upper posterior helix)
    (0.32,  0.05, -0.28), # 2: H2 (Mid posterior helix)
    (0.28, -0.18, -0.25), # 3: H3 (Lower posterior helix)
    (0.18, -0.38, -0.15), # 4: H4 (Lobule lower)
    (0.08, -0.48, -0.05), # 5: H5 (Lobule bottom tip, hugs jawline)
    # Antihelix / Scapha rim (I0..I2)
    (0.14,  0.25, -0.12), # 6: I0 (Upper antihelix)
    (0.18,  0.05, -0.18), # 7: I1 (Mid antihelix)
    (0.15, -0.14, -0.15), # 8: I2 (Lower antihelix)
    # Concha depth (C0..C1)
    (0.06,  0.10, -0.08), # 9: C0 (Concha bowl upper)
    (0.06, -0.08, -0.08), # 10: C1 (Concha bowl lower)
]

# Texture UV coordinates for right ear (mapped near U=0, seen by Camera 0)
EAR_UVS_RIGHT: List[Tuple[float, float]] = [
    (0.045, 0.64), # H0
    (0.065, 0.58), # H1
    (0.080, 0.50), # H2
    (0.075, 0.42), # H3
    (0.055, 0.36), # H4
    (0.035, 0.33), # H5
    (0.035, 0.56), # I0
    (0.045, 0.50), # I1
    (0.038, 0.42), # I2
    (0.022, 0.53), # C0
    (0.022, 0.45), # C1
]


def compute_ear_coordinate_frame(anchors: np.ndarray, is_right: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Computes an orthonormal coordinate frame for the ear based on 4 facial anchor landmarks:
      anchors[0]: Top insertion (127 or 356)
      anchors[1]: Tragus (234 or 454)
      anchors[2]: Concha base (93 or 323)
      anchors[3]: Lobule insertion (132 or 361)

    Returns:
      (center, up_vector, normal_out_vector, forward_vector, ear_height)
    """
    c = anchors.mean(axis=0)
    u = anchors[0] - anchors[3]
    h = float(np.linalg.norm(u))
    u /= max(1e-6, h)

    f = anchors[1] - anchors[2]
    f /= max(1e-6, np.linalg.norm(f))

    n = np.cross(u, f)
    n /= max(1e-6, np.linalg.norm(n))
    if is_right and n[0] > 0:
        n = -n
    elif (not is_right) and n[0] < 0:
        n = -n
    f = np.cross(n, u)
    return c, u, n, f, h


def attach_ears_to_canonical_mesh(canonical_mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    """
    Extends the canonical MediaPipe 468-vertex mesh by stitching realistic 3D ears.
    Returns:
      (mesh_with_ears, ear_metadata)
    """
    v_base = canonical_mesh.vertices.copy()
    f_base = canonical_mesh.faces.copy()
    uv_base = canonical_mesh.visual.uv.copy() if hasattr(canonical_mesh.visual, "uv") and canonical_mesh.visual.uv is not None else np.zeros((len(v_base), 2), dtype=np.float32)

    num_base = len(v_base) # 468

    # 1. Right Ear (anchors [127, 234, 93, 132])
    c_r, u_r, n_r, f_r, h_r = compute_ear_coordinate_frame(v_base[[127, 234, 93, 132]], is_right=True)
    v_r = np.array([c_r + h_r * (off[0] * n_r + off[1] * u_r + off[2] * f_r) for off in EAR_OFFSETS], dtype=np.float32)

    # 2. Left Ear (anchors [356, 454, 323, 361])
    c_l, u_l, n_l, f_l, h_l = compute_ear_coordinate_frame(v_base[[356, 454, 323, 361]], is_right=False)
    v_l = np.array([c_l + h_l * (off[0] * n_l + off[1] * u_l + off[2] * f_l) for off in EAR_OFFSETS], dtype=np.float32)

    # 3. Triangles Right Ear
    A0_r, A1_r, A2_r, A3_r = 127, 234, 93, 132
    H0_r, H1_r, H2_r, H3_r, H4_r, H5_r = range(num_base, num_base + 6)
    I0_r, I1_r, I2_r = range(num_base + 6, num_base + 9)
    C0_r, C1_r = range(num_base + 9, num_base + 11)

    tris_r = [
        [A0_r, H0_r, I0_r],
        [A0_r, I0_r, C0_r],
        [A0_r, C0_r, A1_r],
        [A1_r, C0_r, C1_r],
        [A1_r, C1_r, A2_r],
        [A2_r, C1_r, I2_r],
        [A2_r, I2_r, A3_r],
        [A3_r, I2_r, H5_r],
        [I0_r, H0_r, H1_r],
        [I0_r, H1_r, I1_r],
        [I1_r, H1_r, H2_r],
        [I1_r, H2_r, I2_r],
        [I2_r, H2_r, H3_r],
        [I2_r, H3_r, H4_r],
        [I2_r, H4_r, H5_r],
        [C0_r, I0_r, I1_r],
        [C0_r, I1_r, C1_r],
        [C1_r, I1_r, I2_r],
        # Back-shell closure triangles (solid volume from behind)
        [A0_r, H1_r, H0_r],
        [A0_r, H2_r, H1_r],
        [A1_r, H2_r, A0_r],
        [A1_r, H3_r, H2_r],
        [A2_r, H3_r, A1_r],
        [A2_r, H4_r, H3_r],
        [A3_r, H4_r, A2_r],
        [A3_r, H5_r, H4_r],
    ]

    # 4. Triangles Left Ear (reversed winding for outward normals)
    offset_l = num_base + len(EAR_OFFSETS)
    A0_l, A1_l, A2_l, A3_l = 356, 454, 323, 361
    H0_l, H1_l, H2_l, H3_l, H4_l, H5_l = range(offset_l, offset_l + 6)
    I0_l, I1_l, I2_l = range(offset_l + 6, offset_l + 9)
    C0_l, C1_l = range(offset_l + 9, offset_l + 11)

    tris_l = [
        [A0_l, I0_l, H0_l],
        [A0_l, C0_l, I0_l],
        [A0_l, A1_l, C0_l],
        [A1_l, C1_l, C0_l],
        [A1_l, A2_l, C1_l],
        [A2_l, I2_l, C1_l],
        [A2_l, A3_l, I2_l],
        [A3_l, H5_l, I2_l],
        [I0_l, H1_l, H0_l],
        [I0_l, I1_l, H1_l],
        [I1_l, H2_l, H1_l],
        [I1_l, I2_l, H2_l],
        [I2_l, H3_l, H2_l],
        [I2_l, H4_l, H3_l],
        [I2_l, H5_l, H4_l],
        [C0_l, I1_l, I0_l],
        [C0_l, C1_l, I1_l],
        [C1_l, I2_l, I1_l],
        # Back-shell closure triangles (solid volume from behind)
        [A0_l, H0_l, H1_l],
        [A0_l, H1_l, H2_l],
        [A1_l, A0_l, H2_l],
        [A1_l, H2_l, H3_l],
        [A2_l, A1_l, H3_l],
        [A2_l, H3_l, H4_l],
        [A3_l, A2_l, H4_l],
        [A3_l, H4_l, H5_l],
    ]

    # 5. Combine Vertices, Faces, and UVs
    all_verts = np.vstack([v_base, v_r, v_l])
    all_faces = np.vstack([f_base, tris_r, tris_l])

    uv_r = np.array(EAR_UVS_RIGHT, dtype=np.float32)
    uv_l = np.array([(1.0 - u, v) for u, v in EAR_UVS_RIGHT], dtype=np.float32)
    all_uvs = np.vstack([uv_base, uv_r, uv_l])

    mesh_with_ears = trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=False)
    mesh_with_ears.visual = trimesh.visual.TextureVisuals(uv=all_uvs)

    ear_meta = {
        "num_base_verts": num_base,
        "ear_offsets": EAR_OFFSETS,
        "anchors_right": [127, 234, 93, 132],
        "anchors_left": [356, 454, 323, 361],
        "right_ear_range": (num_base, num_base + len(EAR_OFFSETS)),
        "left_ear_range": (offset_l, offset_l + len(EAR_OFFSETS)),
    }

    return mesh_with_ears, ear_meta


def deform_mesh_with_ears(
    canonical_mesh_with_ears: trimesh.Trimesh,
    pixel_landmarks: List[Dict[str, float]],
    ear_meta: Dict[str, Any],
    target_width_m: float = 0.155,
    use_biometric_scale: bool = True,
) -> trimesh.Trimesh:
    """
    Deforms facial vertices using MediaPipe landmarks, scaling the mesh to real-world
    human metric dimensions (in meters: ~0.155 m width, ~0.20 m height).
    Uses biometric references: Interpupillary Distance (IPD = 63 mm) or Outer Eye Span (96 mm).
    """
    num_base = ear_meta.get("num_base_verts", 468)
    pts = pixel_landmarks[:num_base]

    xs = np.array([p["x"] for p in pts], dtype=np.float32)
    ys = np.array([p["y"] for p in pts], dtype=np.float32)
    zs = np.array([p["z"] for p in pts], dtype=np.float32)

    cx, cy, cz = float(xs.mean()), float(ys.mean()), float(zs.mean())

    # Biometric scaling to real-world human proportions:
    # Scales to standard 3D viewport dimensions (~1.5 units wide, perfectly sized in Blender)
    scale = None
    if use_biometric_scale and len(pixel_landmarks) >= 474:
        # 1. Interpupillary Distance (IPD: Landmark 468 Right Iris, 473 Left Iris)
        p_r = np.array([pixel_landmarks[468]["x"], pixel_landmarks[468]["y"]], dtype=np.float32)
        p_l = np.array([pixel_landmarks[473]["x"], pixel_landmarks[473]["y"]], dtype=np.float32)
        ipd_px = float(np.linalg.norm(p_r - p_l))
        if ipd_px > 30.0:
            scale = 0.63 / ipd_px  # Standardized viewport scale (IPD ~0.63 units, face ~1.3-1.5 units)

    if scale is None and len(pixel_landmarks) >= 264:
        # 2. Outer canthal distance (Outer eye corners: Landmark 33 Right, 263 Left)
        p_r = np.array([pixel_landmarks[33]["x"], pixel_landmarks[33]["y"]], dtype=np.float32)
        p_l = np.array([pixel_landmarks[263]["x"], pixel_landmarks[263]["y"]], dtype=np.float32)
        eye_span_px = float(np.linalg.norm(p_r - p_l))
        if eye_span_px > 40.0:
            scale = 0.96 / eye_span_px  # Outer eye span ~0.96 units

    if scale is None:
        # 3. Fallback to bounding face width
        face_width_px = max(1.0, float(xs.max() - xs.min()))
        scale = 1.55 / face_width_px

    # 1. Deform base face vertices (Y Up, Z Forward) in meters
    deformed_face = np.zeros((num_base, 3), dtype=np.float32)
    deformed_face[:, 0] = (xs - cx) * scale
    deformed_face[:, 1] = -(ys - cy) * scale  # Invert Y so up is +Y
    deformed_face[:, 2] = -(zs - cz) * scale  # Invert Z so face points forward

    # 2. Deform Right Ear using deformed anchor coordinates
    idx_r = ear_meta["anchors_right"]
    c_r, u_r, n_r, f_r, h_r = compute_ear_coordinate_frame(deformed_face[idx_r], is_right=True)
    deformed_ear_r = np.array(
        [c_r + h_r * (off[0] * n_r + off[1] * u_r + off[2] * f_r) for off in ear_meta["ear_offsets"]],
        dtype=np.float32
    )

    # 3. Deform Left Ear using deformed anchor coordinates
    idx_l = ear_meta["anchors_left"]
    c_l, u_l, n_l, f_l, h_l = compute_ear_coordinate_frame(deformed_face[idx_l], is_right=False)
    deformed_ear_l = np.array(
        [c_l + h_l * (off[0] * n_l + off[1] * u_l + off[2] * f_l) for off in ear_meta["ear_offsets"]],
        dtype=np.float32
    )

    # Combine all vertices
    all_deformed = np.vstack([deformed_face, deformed_ear_r, deformed_ear_l])

    mesh_out = canonical_mesh_with_ears.copy()
    mesh_out.vertices = all_deformed
    return mesh_out


def compute_2d_ear_pixel_points(
    pixel_landmarks: List[Dict[str, float]],
    ear_meta: Dict[str, Any],
) -> np.ndarray:
    """
    Computes 2D image pixel coordinates for the stitched ear vertices based on facial landmark anchors.
    Returns:
      Array of shape (num_all_vertices, 2) with (x, y) coordinates for all face + ear vertices.
    """
    num_base = ear_meta.get("num_base_verts", 468)
    base_pts = np.array([[pt["x"], pt["y"]] for pt in pixel_landmarks[:num_base]], dtype=np.float32)

    # 1. Right Ear 2D Frame (Camera 0 views subject's right cheek and right ear)
    idx_r = ear_meta["anchors_right"]
    anchors_r = base_pts[idx_r]
    c_r = anchors_r.mean(axis=0)
    u_r = anchors_r[0] - anchors_r[3]  # top - lobule
    h_r = float(np.linalg.norm(u_r))
    u_r /= max(1e-6, h_r)
    # Perpendicular outward pointing left (-X)
    n_r = np.array([u_r[1], -u_r[0]], dtype=np.float32)
    if n_r[0] > 0:
        n_r = -n_r

    pts_ear_r = np.array(
        [c_r + h_r * (off[0] * n_r + off[1] * u_r) for off in ear_meta["ear_offsets"]],
        dtype=np.float32
    )

    # 2. Left Ear 2D Frame (Camera 1 views subject's left cheek and left ear)
    idx_l = ear_meta["anchors_left"]
    anchors_l = base_pts[idx_l]
    c_l = anchors_l.mean(axis=0)
    u_l = anchors_l[0] - anchors_l[3]
    h_l = float(np.linalg.norm(u_l))
    u_l /= max(1e-6, h_l)
    # Perpendicular outward pointing right (+X)
    n_l = np.array([-u_l[1], u_l[0]], dtype=np.float32)
    if n_l[0] < 0:
        n_l = -n_l

    pts_ear_l = np.array(
        [c_l + h_l * (off[0] * n_l + off[1] * u_l) for off in ear_meta["ear_offsets"]],
        dtype=np.float32
    )

    return np.vstack([base_pts, pts_ear_r, pts_ear_l])
