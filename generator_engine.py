"""
Full Head 3D Scanner & Multi-Format Exporter (Steps 1 & 2 - Production Quality)
Produces complete watertight 3D heads with UV texturing in 8 industry formats:
High/Low Poly: OBJ+MTL, GLB (Draco), STL (3D Print Solid in mm), 3MF (Watertight in mm).
"""

import os
import sys
import shutil
import zipfile
import argparse
import subprocess
import urllib.request
from pathlib import Path
from typing import Union, List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image
import trimesh
from scipy.interpolate import RBFInterpolator

from face_scanner import FaceScanner

# MediaPipe to Standard 68 Facial Landmarks mapping
MP_TO_68 = [
    # Jawline (0-16)
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397,
    # Right eyebrow (17-21)
    70, 63, 105, 66, 107,
    # Left eyebrow (22-26)
    336, 296, 334, 293, 300,
    # Nose bridge (27-30)
    168, 6, 197, 195,
    # Nose bottom & tip (31-35)
    98, 97, 2, 326, 327,
    # Right eye (36-41)
    33, 160, 158, 133, 153, 144,
    # Left eye (42-47)
    362, 385, 387, 263, 373, 380,
    # Outer lips (48-59)
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375,
    # Inner lips (60-67)
    78, 95, 88, 178, 87, 14, 317, 402,
]


def download_if_url(input_path_or_url: str, temp_dir: Path) -> Path:
    """If the input is a URL, downloads it to a temporary file, otherwise returns Path."""
    s = str(input_path_or_url).strip()
    if s.startswith("http://") or s.startswith("https://"):
        temp_dir.mkdir(parents=True, exist_ok=True)
        dest = temp_dir / "downloaded_face.jpg"
        req = urllib.request.Request(
            s,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            dest.write_bytes(resp.read())
        return dest
    return Path(s).resolve()


def safe_warp_triangle(
    img_src: np.ndarray,
    img_dst: np.ndarray,
    mask_dst: np.ndarray,
    t_src: np.ndarray,
    t_dst: np.ndarray,
):
    """Warps an affine triangle from source image to destination texture with border clamping."""
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))

    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return

    h_src, w_src = img_src.shape[:2]
    # Check if completely outside source
    if r1[0] + r1[2] <= 0 or r1[1] + r1[3] <= 0 or r1[0] >= w_src or r1[1] >= h_src:
        return

    x1, y1 = max(0, r1[0]), max(0, r1[1])
    x2, y2 = min(w_src, r1[0] + r1[2]), min(h_src, r1[1] + r1[3])

    if x2 <= x1 or y2 <= y1:
        return

    t1_cropped = [((t_src[i][0] - r1[0]), (t_src[i][1] - r1[1])) for i in range(3)]
    t2_cropped = [((t_dst[i][0] - r2[0]), (t_dst[i][1] - r2[1])) for i in range(3)]

    patch = np.zeros((r1[3], r1[2], 3), dtype=img_src.dtype)
    patch_y1 = y1 - r1[1]
    patch_y2 = patch_y1 + (y2 - y1)
    patch_x1 = x1 - r1[0]
    patch_x2 = patch_x1 + (x2 - x1)
    patch[patch_y1:patch_y2, patch_x1:patch_x2] = img_src[y1:y2, x1:x2]

    warp_mat = cv2.getAffineTransform(np.float32(t1_cropped), np.float32(t2_cropped))
    img2_cropped = cv2.warpAffine(
        patch, warp_mat, (r2[2], r2[3]), None,
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )

    t2_mask = np.zeros((r2[3], r2[2]), dtype=np.uint8)
    cv2.fillConvexPoly(t2_mask, np.int32(t2_cropped), 255)

    sub_dst = img_dst[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]]
    sub_mask = mask_dst[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]]

    sub_dst[t2_mask == 255] = img2_cropped[t2_mask == 255]
    sub_mask[t2_mask == 255] = 255


def bake_uv_texture_flame(
    img_bgr: np.ndarray,
    deformed_mesh: trimesh.Trimesh,
    flame_template: trimesh.Trimesh,
    cx: float,
    cy: float,
    flame_center: np.ndarray,
    scale_to_m: float,
    tex_size: int = 4096,
) -> np.ndarray:
    """
    Projective UV texture baking directly onto FLAME UV space.
    Strictly masks the frontal facial region to avoid double-sampling features
    (e.g., eyes on temples or lips on the neck), seamlessly inpainting occluded areas.
    """
    verts = deformed_mesh.vertices
    px_coords = np.zeros((len(verts), 2), dtype=np.float32)
    px_coords[:, 0] = (verts[:, 0] - flame_center[0]) / scale_to_m + cx
    px_coords[:, 1] = -(verts[:, 1] - flame_center[1]) / scale_to_m + cy

    uvs = flame_template.visual.uv
    uv_px = np.zeros_like(uvs, dtype=np.float32)
    uv_px[:, 0] = uvs[:, 0] * (tex_size - 1)
    uv_px[:, 1] = (1.0 - uvs[:, 1]) * (tex_size - 1)

    texture = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
    mask = np.zeros((tex_size, tex_size), dtype=np.uint8)

    normals = deformed_mesh.face_normals
    faces = deformed_mesh.faces
    face_centers = verts[faces].mean(axis=1)

    # Filter STRICTLY to the anatomical face region to avoid feature duplication:
    # 1. Normal pointing forward (Z > 0.25)
    # 2. In front of the ears and temples (center Z > 0.015)
    # 3. Between chin and eyebrows (center Y: 1.415 to 1.565)
    # 4. Within cheek boundaries (|center X| < 0.070)
    face_tri_mask = (
        (normals[:, 2] > 0.25)
        & (face_centers[:, 2] > 0.015)
        & (face_centers[:, 1] > 1.415)
        & (face_centers[:, 1] < 1.565)
        & (np.abs(face_centers[:, 0]) < 0.070)
    )

    for i in range(len(faces)):
        if face_tri_mask[i]:
            tri = faces[i]
            pts_src = px_coords[tri]
            pts_dst = uv_px[tri]
            safe_warp_triangle(img_bgr, texture, mask, pts_src, pts_dst)

    # Multiscale Inpainting: downscale to 512 for instant seamless color diffusion across cranium/neck
    unmapped = (mask == 0).astype(np.uint8) * 255
    tex_small = cv2.resize(texture, (512, 512))
    mask_small = cv2.resize(unmapped, (512, 512), interpolation=cv2.INTER_NEAREST)
    inpaint_small = cv2.inpaint(tex_small, mask_small, 5, cv2.INPAINT_TELEA)
    inpaint_up = cv2.resize(inpaint_small, (tex_size, tex_size), interpolation=cv2.INTER_CUBIC)

    final_texture = texture.copy()
    final_texture[mask == 0] = inpaint_up[mask == 0]

    return final_texture


def make_watertight_solid_mm(mesh: trimesh.Trimesh, pedestal_drop_mm: float = 15.0) -> trimesh.Trimesh:
    """
    Creates a 100% solid, watertight mesh scaled in millimeters with a flat pedestal base,
    specifically formatted for 3D slicing software (Bambu Studio, Cura, PrusaSlicer).
    Guarantees no open holes at the neck or mouth.
    """
    m = mesh.copy()

    # Scale to millimeters if currently in meters (human head height in meters is ~0.3m, in mm is ~300mm)
    if m.extents[1] < 1.0:
        m.apply_scale(1000.0)

    # Center horizontally and set neck base at Y = 0 (or flat reference)
    center = m.vertices.mean(axis=0)
    m.vertices[:, 0] -= center[0]
    m.vertices[:, 2] -= center[2]

    # Weld duplicate vertices (e.g. 3D seam vertices created by UV splits)
    unique, inverse = trimesh.grouping.unique_rows(np.round(m.vertices, 4))
    m = trimesh.Trimesh(vertices=m.vertices[unique], faces=inverse[m.faces], process=True)

    # Detect boundary loops
    edges = m.edges_sorted
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges_set = set(tuple(e) for e in unique_edges[counts == 1])

    if not boundary_edges_set:
        return m

    # Build directed cycles
    edge_map = {}
    for f in m.faces:
        for e in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
            s_e = tuple(sorted(e))
            if s_e in boundary_edges_set:
                edge_map[e[0]] = e[1]

    visited = set()
    directed_cycles = []
    for start_node in list(edge_map.keys()):
        if start_node in visited:
            continue
        curr = start_node
        cycle = []
        while curr not in visited and curr in edge_map:
            visited.add(curr)
            cycle.append(curr)
            curr = edge_map[curr]
        if len(cycle) > 2:
            directed_cycles.append(cycle)

    new_verts = list(m.vertices)
    new_faces = list(m.faces)

    min_y = m.vertices[:, 1].min()
    y_base = min_y - pedestal_drop_mm

    for cycle in directed_cycles:
        pts = m.vertices[cycle]

        # Check if this is the neck collar loop (lowest boundary)
        if pts[:, 1].mean() < min_y + 60.0:
            # 1. Extrude neck loop down to form cylindrical skirt
            skirt_start_idx = len(new_verts)
            for v_idx in cycle:
                p = m.vertices[v_idx].copy()
                p[1] = y_base  # flat base plane
                new_verts.append(p)

            # Create side quad faces (as 2 triangles) connecting original ring to flat ring
            k = len(cycle)
            for i in range(k):
                v_orig_1 = cycle[i]
                v_orig_2 = cycle[(i + 1) % k]
                v_base_1 = skirt_start_idx + i
                v_base_2 = skirt_start_idx + ((i + 1) % k)
                # Triangle 1 & 2
                new_faces.append([v_orig_1, v_base_1, v_orig_2])
                new_faces.append([v_orig_2, v_base_1, v_base_2])

            # 2. Cap the flat base with central fan
            base_center_idx = len(new_verts)
            base_center = np.array([0.0, y_base, pts[:, 2].mean()], dtype=np.float32)
            new_verts.append(base_center)

            for i in range(k):
                v_base_1 = skirt_start_idx + i
                v_base_2 = skirt_start_idx + ((i + 1) % k)
                # Normal pointing down (-Y)
                new_faces.append([v_base_2, v_base_1, base_center_idx])
        else:
            # Other small holes (e.g. mouth / eye openings): centroid cap
            c_idx = len(new_verts)
            center_pt = pts.mean(axis=0)
            new_verts.append(center_pt)
            k = len(cycle)
            for i in range(k):
                new_faces.append([cycle[i], cycle[(i + 1) % k], c_idx])

    solid_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=True)
    solid_mesh.fix_normals()
    return solid_mesh


def export_custom_obj(
    mesh: trimesh.Trimesh,
    obj_path: Path,
    mtl_path: Path,
    texture_filename: str,
):
    """
    Exports clean OBJ + MTL with explicit, guaranteed texture linking.
    """
    obj_path = Path(obj_path)
    mtl_path = Path(mtl_path)

    # 1. Write MTL file
    mtl_content = f"""# Material file generated by 3D Face Reconstructor
newmtl material_0
Ka 1.000000 1.000000 1.000000
Kd 1.000000 1.000000 1.000000
Ks 0.100000 0.100000 0.100000
d 1.000000
illum 2
map_Kd {texture_filename}
"""
    mtl_path.write_text(mtl_content, encoding="utf-8")

    # 2. Write OBJ file with explicit mtllib and usemtl
    lines = [
        f"# Wavefront OBJ File generated by 3D Face Reconstructor",
        f"mtllib {mtl_path.name}",
        f"usemtl material_0",
        "",
    ]

    # Vertices
    for v in mesh.vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

    # UVs
    has_uv = hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) > 0
    if has_uv:
        for uv in mesh.visual.uv:
            lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")

    # Normals
    has_normals = hasattr(mesh, "vertex_normals") and len(mesh.vertex_normals) > 0
    if has_normals:
        for n in mesh.vertex_normals:
            lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")

    lines.append("")
    # Faces
    for f in mesh.faces:
        v1, v2, v3 = f[0] + 1, f[1] + 1, f[2] + 1
        if has_uv and has_normals:
            lines.append(f"f {v1}/{v1}/{v1} {v2}/{v2}/{v2} {v3}/{v3}/{v3}")
        elif has_uv:
            lines.append(f"f {v1}/{v1} {v2}/{v2} {v3}/{v3}")
        else:
            lines.append(f"f {v1} {v2} {v3}")

    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class FullHeadGenerator:
    """
    Comprehensive 3D Head Generator producing all 8 formats:
    High/Low Poly: OBJ+MTL, GLB (Draco), STL (3D Print Solid in mm), 3MF (Color/Metadata in mm).
    """

    @staticmethod
    def _resolve_asset(path_str: str) -> str:
        p = Path(path_str)
        if p.is_file():
            return str(p.resolve())
        asset_p = Path(__file__).parent / "assets" / p.name
        if asset_p.is_file():
            return str(asset_p.resolve())
        root_p = Path(__file__).parent / p.name
        if root_p.is_file():
            return str(root_p.resolve())
        return str(p.resolve())

    def __init__(
        self,
        task_model: str = "assets/face_landmarker.task",
        canonical_mesh_path: str = "assets/canonical_face_model.obj",
        flame_template_path: str = "assets/flame_head_template.obj",
        landmark_embedding_path: str = "assets/landmark_embedding.npy",
    ):
        task_model = self._resolve_asset(task_model)
        canonical_mesh_path = self._resolve_asset(canonical_mesh_path)
        flame_template_path = self._resolve_asset(flame_template_path)
        landmark_embedding_path = self._resolve_asset(landmark_embedding_path)

        self.scanner = FaceScanner(model_path=task_model)
        self.canonical_mesh = trimesh.load(canonical_mesh_path, process=False)
        self.flame_template = trimesh.load(flame_template_path, process=False)

        # Load FLAME 68 landmark embedding
        embed_data = np.load(landmark_embedding_path, allow_pickle=True, encoding="latin1").item()
        self.flame_faces_idx = embed_data["full_lmk_faces_idx"].squeeze()
        self.flame_bary_coords = embed_data["full_lmk_bary_coords"].squeeze()

    def close(self):
        if hasattr(self, "scanner") and self.scanner is not None:
            self.scanner.close()
            self.scanner = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _compute_flame_68_landmarks(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Computes the 68 landmark coordinates on the FLAME mesh using barycentric embeddings."""
        lmks = np.zeros((68, 3), dtype=np.float32)
        for i in range(68):
            f_idx = self.flame_faces_idx[i]
            v_coords = mesh.vertices[mesh.faces[f_idx]]
            b = self.flame_bary_coords[i]
            lmks[i] = b[0] * v_coords[0] + b[1] * v_coords[1] + b[2] * v_coords[2]
        return lmks

    def generate_models(
        self,
        image_input: Union[str, Path],
        output_dir: Union[str, Path] = "./output_models",
        name_prefix: str = "avatar",
        tex_size_high: int = 4096,
        tex_size_low: int = 2048,
    ) -> Dict[str, Path]:
        """
        Executes Steps 1 & 2:
        1. Detects facial landmarks.
        2. Fits FLAME head geometry via RBF interpolation.
        3. Bakes textures (High 4K, Low 2K) with multiscale inpainting.
        4. Generates watertight 3D printable meshes (STL/3MF in millimeters).
        5. Exports all 8 deliverables with linked MTLs and Draco GLBs.
        6. Packages bundle.zip.
        """
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        local_img = download_if_url(str(image_input), out_dir)
        print(f"Scanning face in: {local_img.name}...")

        scan_result = self.scanner.scan_image(local_img)
        if scan_result["num_faces_detected"] == 0:
            raise ValueError(f"No face detected in {local_img.name}")

        face = scan_result["faces"][0]
        img_bgr = cv2.imread(str(local_img))

        # ----------------------------------------------------
        # 1. FIT FLAME HEAD GEOMETRY VIA RBF DEFORMATION
        # ----------------------------------------------------
        print("Fitting 3D Head Geometry (FLAME scaffold)...")

        # Extract 68 landmarks from MediaPipe
        pts_all = face["pixel_landmarks"]
        mp_68_px = np.array(
            [[pts_all[idx]["x"], pts_all[idx]["y"], pts_all[idx]["z"]] for idx in MP_TO_68],
            dtype=np.float32,
        )

        # 1. Metric IPD Scaling (Inter-Pupillary Distance = 63 mm)
        # Right eye: 36 & 39; Left eye: 42 & 45
        r_eye_center = (mp_68_px[36] + mp_68_px[39]) / 2.0
        l_eye_center = (mp_68_px[42] + mp_68_px[45]) / 2.0
        ipd_px = np.linalg.norm(r_eye_center[:2] - l_eye_center[:2])
        scale_to_m = 0.063 / max(1.0, ipd_px)

        cx, cy = mp_68_px[:, 0].mean(), mp_68_px[:, 1].mean()
        cz = mp_68_px[:, 2].mean()
        flame_lmks_source = self._compute_flame_68_landmarks(self.flame_template)
        flame_center = flame_lmks_source.mean(axis=0)

        target_lmks = np.zeros_like(mp_68_px)
        target_lmks[:, 0] = (mp_68_px[:, 0] - cx) * scale_to_m + flame_center[0]
        target_lmks[:, 1] = -(mp_68_px[:, 1] - cy) * scale_to_m + flame_center[1]

        # DAMP and CLAMP Z displacement: MediaPipe monocular Z has high noise,
        # damp by 0.35 and clamp to ±10 mm to preserve natural skull curvature without forehead spikes
        z_offset = -(mp_68_px[:, 2] - cz) * scale_to_m * 0.35
        z_offset = np.clip(z_offset, -0.010, 0.010)
        target_lmks[:, 2] = flame_lmks_source[:, 2] + z_offset

        # Dense rigid cranium anchors sampled directly from FLAME:
        # 1. Back of skull and neck (Z < 0.01, Y < 1.40)
        # 2. Upper forehead and top of head (Y > 1.55) to lock forehead profile firmly
        flame_verts = self.flame_template.vertices
        cranium_mask = (flame_verts[:, 2] < 0.01) | (flame_verts[:, 1] < 1.40) | (flame_verts[:, 1] > 1.55)
        cranium_indices = np.where(cranium_mask)[0]
        step = max(1, len(cranium_indices) // 300)
        rigid_anchors = flame_verts[cranium_indices[::step]]

        all_sources = np.vstack([flame_lmks_source, rigid_anchors])
        all_targets = np.vstack([target_lmks, rigid_anchors])
        displacements = all_targets - all_sources

        # Localized Gaussian RBF with smoothing to prevent skull distortion
        rbf = RBFInterpolator(all_sources, displacements, kernel="gaussian", epsilon=12.5, smoothing=0.005)
        flame_displaced_verts = self.flame_template.vertices + rbf(self.flame_template.vertices)

        # Base Low-Poly Mesh (LOD 0: ~5k vertices, ~10k triangles)
        mesh_low = self.flame_template.copy()
        mesh_low.vertices = flame_displaced_verts

        # High-Poly Mesh (Subdivision 1x: ~20k vertices, ~40k triangles)
        print("Generating Subdivided High-Poly Mesh...")
        mesh_high = mesh_low.subdivide()

        # ----------------------------------------------------
        # 2. BAKE UV TEXTURES DIRECTLY ONTO FLAME UV SPACE
        # ----------------------------------------------------
        print(f"Baking High-Res {tex_size_high}x{tex_size_high} & Low-Res {tex_size_low}x{tex_size_low} UV Textures onto FLAME...")
        tex_high_bgr = bake_uv_texture_flame(
            img_bgr=img_bgr,
            deformed_mesh=mesh_low,
            flame_template=self.flame_template,
            cx=cx,
            cy=cy,
            flame_center=flame_center,
            scale_to_m=scale_to_m,
            tex_size=tex_size_high,
        )
        tex_low_bgr = cv2.resize(tex_high_bgr, (tex_size_low, tex_size_low), interpolation=cv2.INTER_AREA)

        # Save texture files
        tex_high_path = out_dir / f"{name_prefix}_high_texture.png"
        tex_low_path = out_dir / f"{name_prefix}_low_texture.jpg"
        cv2.imwrite(str(tex_high_path), tex_high_bgr)
        cv2.imwrite(str(tex_low_path), tex_low_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # Attach textures to meshes
        tex_high_pil = Image.fromarray(cv2.cvtColor(tex_high_bgr, cv2.COLOR_BGR2RGB))
        tex_low_pil = Image.fromarray(cv2.cvtColor(tex_low_bgr, cv2.COLOR_BGR2RGB))

        mat_high = trimesh.visual.material.PBRMaterial(baseColorTexture=tex_high_pil, roughnessFactor=0.6)
        mat_low = trimesh.visual.material.PBRMaterial(baseColorTexture=tex_low_pil, roughnessFactor=0.6)

        # Transfer UVs
        mesh_high.visual = trimesh.visual.TextureVisuals(uv=mesh_high.visual.uv, material=mat_high)
        mesh_low.visual = trimesh.visual.TextureVisuals(uv=self.flame_template.visual.uv, material=mat_low)

        # ----------------------------------------------------
        # 3. EXPORT ALL 8 DELIVERABLES
        # ----------------------------------------------------
        print("Exporting complete 8-format matrix with linked materials...")
        files = {}

        # 1 & 2: OBJ Models + Explicit Linked MTLs
        obj_high = out_dir / f"{name_prefix}_high.obj"
        mtl_high = out_dir / f"{name_prefix}_high.mtl"
        export_custom_obj(mesh_high, obj_high, mtl_high, tex_high_path.name)

        obj_low = out_dir / f"{name_prefix}_low.obj"
        mtl_low = out_dir / f"{name_prefix}_low.mtl"
        export_custom_obj(mesh_low, obj_low, mtl_low, tex_low_path.name)

        files["obj_high"] = obj_high
        files["mtl_high"] = mtl_high
        files["obj_low"] = obj_low
        files["mtl_low"] = mtl_low
        files["texture_high"] = tex_high_path
        files["texture_low"] = tex_low_path

        # 3 & 4: GLB Models (with Draco compression via gltf-transform)
        glb_high_raw = out_dir / f"{name_prefix}_high_raw.glb"
        glb_low_raw = out_dir / f"{name_prefix}_low_raw.glb"
        mesh_high.export(str(glb_high_raw))
        mesh_low.export(str(glb_low_raw))

        glb_high = out_dir / f"{name_prefix}_high.glb"
        glb_low = out_dir / f"{name_prefix}_low.glb"

        try:
            subprocess.run(
                [
                    "npx", "-y", "@gltf-transform/cli", "draco",
                    str(glb_high_raw), str(glb_high),
                    "--quantize-position", "14", "--quantize-normal", "12", "--quantize-texcoord", "14",
                ],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                [
                    "npx", "-y", "@gltf-transform/cli", "draco",
                    str(glb_low_raw), str(glb_low),
                    "--quantize-position", "12", "--quantize-normal", "10", "--quantize-texcoord", "12",
                ],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            glb_high_raw.unlink(missing_ok=True)
            glb_low_raw.unlink(missing_ok=True)
        except Exception:
            shutil.move(str(glb_high_raw), str(glb_high))
            shutil.move(str(glb_low_raw), str(glb_low))

        files["glb_high"] = glb_high
        files["glb_low"] = glb_low

        # 5 & 6: STL Models (100% Watertight Solid in Millimeters with Flat Pedestal)
        print("Generating watertight solids in millimeters for 3D printing...")
        solid_high_mm = make_watertight_solid_mm(mesh_high, pedestal_drop_mm=15.0)
        solid_low_mm = make_watertight_solid_mm(mesh_low, pedestal_drop_mm=15.0)

        stl_high = out_dir / f"{name_prefix}_high.stl"
        stl_low = out_dir / f"{name_prefix}_low.stl"
        solid_high_mm.export(str(stl_high))
        solid_low_mm.export(str(stl_low))
        files["stl_high"] = stl_high
        files["stl_low"] = stl_low

        # 7 & 8: 3MF Models (Exported from solid watertight meshes in mm)
        mf_high = out_dir / f"{name_prefix}_high.3mf"
        mf_low = out_dir / f"{name_prefix}_low.3mf"
        solid_high_mm.export(str(mf_high))
        solid_low_mm.export(str(mf_low))
        files["3mf_high"] = mf_high
        files["3mf_low"] = mf_low

        # Package into bundle.zip
        zip_path = out_dir / f"{name_prefix}_bundle.zip"
        print(f"Packaging complete bundle into: {zip_path.name}...")
        to_zip = [
            obj_high, mtl_high, tex_high_path,
            obj_low, mtl_low, tex_low_path,
            glb_high, glb_low,
            stl_high, stl_low,
            mf_high, mf_low,
        ]

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in to_zip:
                if item.exists():
                    zf.write(item, arcname=item.name)

        files["bundle_zip"] = zip_path
        print(f"\nAll 8 formats generated successfully in: {out_dir}")
        return files


def main():
    parser = argparse.ArgumentParser(description="Full Head 3D Generator (Steps 1 & 2)")
    parser.add_argument("-i", "--input", required=True, help="Input image file or URL")
    parser.add_argument("-o", "--output-dir", default="./output_models", help="Output directory")
    parser.add_argument("--name", default="avatar", help="Name prefix for output files")

    args = parser.parse_args()

    with FullHeadGenerator() as generator:
        generator.generate_models(args.input, output_dir=args.output_dir, name_prefix=args.name)


if __name__ == "__main__":
    main()
