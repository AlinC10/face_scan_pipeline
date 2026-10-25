"""
3D Face Reconstructor: Single Image (URL / Local File) or Multi-Image Folder to 3D Head
Supports Windows, Linux, and Cloud Docker (RunPod).
Exports GLB, OBJ+MTL+PNG/JPG, STL, and 3MF in High and Low poly hierarchies.
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path
from typing import Union, List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image, ExifTags
import trimesh

from face_scanner import FaceScanner
from converters import (
    convert_to_glb,
    apply_draco_compression,
    convert_to_stl,
    convert_to_3mf,
    convert_to_glb_draco,
)
from reconstruction import (
    attach_ears_to_canonical_mesh,
    deform_mesh_with_ears,
    compute_2d_ear_pixel_points,
    generate_facial_hair_height_map,
    generate_tangent_normal_map,
    apply_hair_displacement_to_mesh,
)


def extract_image_exif(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extracts key camera EXIF metadata (UniqueID, Camera Serial, Model, Timestamp, Resolution).
    """
    path_obj = Path(image_path).resolve()
    info = {
        "filename": path_obj.name,
        "path": str(path_obj),
        "unique_id": None,
        "serial": None,
        "model": None,
        "timestamp": None,
        "dimensions": None,
    }
    try:
        with Image.open(path_obj) as img:
            info["dimensions"] = f"{img.width}x{img.height}"
            exif = img.getexif()
            if exif:
                for tag_id, val in exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if tag_name == "Model":
                        info["model"] = str(val).strip()
                    elif tag_name in ("DateTime", "DateTimeOriginal"):
                        info["timestamp"] = str(val).strip()
                    elif tag_name in ("BodySerialNumber", "CameraSerialNumber", "SerialNumber") or tag_id in (0xA431, 0x0211):
                        info["serial"] = str(val).strip()
                    elif tag_name == "ImageUniqueID" or tag_id == 0xA420:
                        info["unique_id"] = str(val).strip()

                # Search Exif sub-IFD
                for ifd_id in (getattr(ExifTags.IFD, "Exif", 0x8769), getattr(ExifTags.IFD, "Makernote", 0x927C)):
                    try:
                        sub_ifd = exif.get_ifd(ifd_id)
                        for sub_tag_id, sub_val in sub_ifd.items():
                            sub_tag_name = ExifTags.TAGS.get(sub_tag_id, str(sub_tag_id))
                            if "serial" in sub_tag_name.lower() or sub_tag_id in (0xA431, 0x0211):
                                if not info["serial"]:
                                    info["serial"] = str(sub_val).strip()
                            elif "unique" in sub_tag_name.lower() or sub_tag_id == 0xA420:
                                if not info["unique_id"]:
                                    info["unique_id"] = str(sub_val).strip()
                            elif sub_tag_name in ("DateTimeOriginal", "DateTimeDigitized") and not info["timestamp"]:
                                info["timestamp"] = str(sub_val).strip()
                    except Exception:
                        pass
    except Exception:
        pass
    return info


def download_image_from_url(url: str, save_path: Path) -> Path:
    """Downloads an image from a URL with browser-like headers."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        },
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        data = response.read()
    with open(save_path, "wb") as f:
        f.write(data)
    return save_path


def warp_triangle(
    img_src: np.ndarray,
    img_dst: np.ndarray,
    mask_dst: np.ndarray,
    t_src: np.ndarray,
    t_dst: np.ndarray,
):
    """Warps an affine triangular patch from source image to destination texture."""
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))

    if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
        return

    t1_cropped = []
    t2_cropped = []
    for i in range(3):
        t1_cropped.append(((t_src[i][0] - r1[0]), (t_src[i][1] - r1[1])))
        t2_cropped.append(((t_dst[i][0] - r2[0]), (t_dst[i][1] - r2[1])))

    img1_cropped = img_src[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]

    warp_mat = cv2.getAffineTransform(np.float32(t1_cropped), np.float32(t2_cropped))
    img2_cropped = cv2.warpAffine(
        img1_cropped,
        warp_mat,
        (r2[2], r2[3]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    t2_mask = np.zeros((r2[3], r2[2]), dtype=np.uint8)
    cv2.fillConvexPoly(t2_mask, np.int32(t2_cropped), 255)

    sub_dst = img_dst[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]]
    sub_mask = mask_dst[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]]

    sub_dst[t2_mask == 255] = img2_cropped[t2_mask == 255]
    sub_mask[t2_mask == 255] = 255


def bake_uv_texture(
    img_bgr: np.ndarray,
    pts_pixel: np.ndarray,
    canonical_mesh: trimesh.Trimesh,
    tex_size: int = 2048,
) -> np.ndarray:
    """
    Bakes the face image onto the canonical UV map and applies fast multiscale
    inpainting to seamlessly fill unobserved regions.
    """
    uvs = canonical_mesh.visual.uv
    faces = canonical_mesh.faces

    uv_px = np.zeros_like(uvs, dtype=np.float32)
    uv_px[:, 0] = uvs[:, 0] * (tex_size - 1)
    uv_px[:, 1] = (1.0 - uvs[:, 1]) * (tex_size - 1)

    texture = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
    mask = np.zeros((tex_size, tex_size), dtype=np.uint8)

    # Warp each triangle
    for tri in faces:
        warp_triangle(img_bgr, texture, mask, pts_pixel[tri], uv_px[tri])

    # Multiscale Inpainting: downscale to 512 for instant seamless color propagation
    unmapped = (mask == 0).astype(np.uint8) * 255
    tex_small = cv2.resize(texture, (512, 512))
    mask_small = cv2.resize(unmapped, (512, 512), interpolation=cv2.INTER_NEAREST)
    inpaint_small = cv2.inpaint(tex_small, mask_small, 3, cv2.INPAINT_TELEA)
    inpaint_up = cv2.resize(inpaint_small, (tex_size, tex_size), interpolation=cv2.INTER_CUBIC)

    # Composite: original sharp face pixels + smooth background skin tone
    final_texture = texture.copy()
    final_texture[mask == 0] = inpaint_up[mask == 0]

    # Anti-Shadow Bottom-Edge Extrusion:
    # Replaces neck shadows and dark clothing along the lower jawline with the clean beard/skin tone
    y_cutoff = int(tex_size * 0.78)
    y_ref = int(tex_size * 0.74)
    ref_row = final_texture[y_ref, :].copy()
    for y in range(y_cutoff, tex_size):
        dark_pixels = np.mean(final_texture[y], axis=-1) < 80
        final_texture[y, dark_pixels] = ref_row[dark_pixels]

    return final_texture


def bake_stereo_uv_texture(
    img_left: np.ndarray,
    pts_left: np.ndarray,
    img_right: np.ndarray,
    pts_right: np.ndarray,
    canonical_mesh: trimesh.Trimesh,
    tex_size: int = 2048,
    blend_width: float = 0.12,
) -> np.ndarray:
    """
    Fuses UV textures from both Left and Right stereo cameras:
      - Left camera (27° convergence inward) sees subject's right cheek (U in [0.0, 0.50])
      - Right camera (27° convergence inward) sees subject's left cheek (U in [0.50, 1.0])
      - Seamless alpha ramp along the U axis at the center (U in [0.44, 0.56]).
    """
    tex_left = bake_uv_texture(img_left, pts_left, canonical_mesh, tex_size=tex_size)
    tex_right = bake_uv_texture(img_right, pts_right, canonical_mesh, tex_size=tex_size)

    # Linear horizontal gradient along U [0..1]
    u_coords = np.linspace(0.0, 1.0, tex_size, dtype=np.float32)
    u_min = 0.5 - blend_width / 2.0
    u_max = 0.5 + blend_width / 2.0
    alpha_left = np.clip((u_max - u_coords) / (u_max - u_min), 0.0, 1.0)

    # Broadcast to (H, W, 3)
    alpha_2d = np.tile(alpha_left[np.newaxis, :, np.newaxis], (tex_size, 1, 3))
    blended = (tex_left.astype(np.float32) * alpha_2d + tex_right.astype(np.float32) * (1.0 - alpha_2d)).astype(np.uint8)
    return blended


def deform_canonical_mesh(
    canonical_mesh: trimesh.Trimesh,
    pixel_landmarks: List[Dict[str, float]],
    target_width_cm: float = 15.5,
) -> trimesh.Trimesh:
    """
    Deforms the canonical 3D mesh vertices to match the subject's detected facial landmarks,
    properly orienting the axes for standard 3D graphics (Y Up, Z Forward).
    """
    pts = pixel_landmarks[: len(canonical_mesh.vertices)]

    xs = np.array([p["x"] for p in pts], dtype=np.float32)
    ys = np.array([p["y"] for p in pts], dtype=np.float32)
    zs = np.array([p["z"] for p in pts], dtype=np.float32)

    cx, cy, cz = xs.mean(), ys.mean(), zs.mean()

    # Normalization scale to human face proportions (centimeters)
    face_width_px = max(1.0, xs.max() - xs.min())
    scale = target_width_cm / face_width_px

    deformed_verts = np.zeros((len(pts), 3), dtype=np.float32)
    deformed_verts[:, 0] = (xs - cx) * scale
    deformed_verts[:, 1] = -(ys - cy) * scale  # Invert Y so up is +Y
    deformed_verts[:, 2] = -(zs - cz) * scale  # Invert Z so face points forward

    mesh_out = canonical_mesh.copy()
    mesh_out.vertices = deformed_verts
    return mesh_out


# =============================================================================
# MULTI-FORMAT EXPORTERS (Using model_converter.py)
# =============================================================================


def export_multi_format_hierarchy(
    mesh_low: trimesh.Trimesh,
    tex_bgr: np.ndarray,
    output_dir: Path,
    name_prefix: str = "model",
) -> Dict[str, Any]:
    """
    Organizează și exportă modelele 3D conform structurii ierarhice solicitate:
      output_dir/
      ├── obj/
      │   ├── high/ (high_model.obj, high_model_texture.png, high_model_normal.png, material.mtl)
      │   └── low/  (low_model.obj, low_model_texture.jpg, low_model_normal.png, material.mtl)
      ├── glb/      (high_model.glb, low_model.glb cu loc de completare Draco)
      ├── stl/      (high_model.stl, low_model.stl)
      └── 3mf/      (high_model.3mf, low_model.3mf)
    """
    output_dir = Path(output_dir).resolve()

    dir_obj_high = output_dir / "obj" / "high"
    dir_obj_low = output_dir / "obj" / "low"
    dir_glb = output_dir / "glb"
    dir_stl = output_dir / "stl"
    dir_3mf = output_dir / "3mf"

    for d in (dir_obj_high, dir_obj_low, dir_glb, dir_stl, dir_3mf):
        d.mkdir(parents=True, exist_ok=True)

    # Determinare convenție de nume: high_{base}.* și low_{base}.*
    base = name_prefix or "model"
    for pfx in ("high_", "low_"):
        if base.startswith(pfx):
            base = base[len(pfx):]
    for sfx in ("_high", "_low"):
        if base.endswith(sfx):
            base = base[:-len(sfx)]
    if not base or base.startswith("f_"):
        base = "model"

    high_name = f"high_{base}"
    low_name = f"low_{base}"

    # 1. Texturi:
    # High: PNG necomprimat
    tex_high_png_path = dir_obj_high / f"{high_name}_texture.png"
    cv2.imwrite(str(tex_high_png_path), tex_bgr)

    # Low: JPG comprimat
    h, w = tex_bgr.shape[:2]
    tex_low_size = min(2048, w)
    tex_bgr_low = cv2.resize(tex_bgr, (tex_low_size, tex_low_size), interpolation=cv2.INTER_AREA)
    tex_low_jpg_path = dir_obj_low / f"{low_name}_texture.jpg"
    cv2.imwrite(str(tex_low_jpg_path), tex_bgr_low, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

    # 2. Harta de Normale PBR:
    print("Preparing clean PBR textures & neutral normal maps (geometric vertex relief deactivated)...")
    height_map = np.zeros(tex_bgr.shape[:2], dtype=np.float32)
    normal_bgr = generate_tangent_normal_map(height_map, strength=0.0)

    normal_low_png_path = dir_obj_low / f"{low_name}_normal.png"
    cv2.imwrite(str(normal_low_png_path), normal_bgr)
    normal_high_png_path = dir_obj_high / f"{high_name}_normal.png"
    cv2.imwrite(str(normal_high_png_path), normal_bgr)

    # 3. Subdiviziune Nivel 1 (Low: 1.9k verts) & Nivel 2 (High: 7.5k verts) - Netede, curate, fara valuri:
    print("Applying Subdivision Level 1 to Low-Poly model (1.9k verts, perfectly smooth)...")
    mesh_low_displaced = apply_hair_displacement_to_mesh(
        mesh_low, height_map=None, max_displacement_cm=0.0, subdivide_level=1, enable_displacement=False
    )
    print("Applying Subdivision Level 2 to High-Poly model (7.5k verts, dense smooth curvature)...")
    mesh_high = apply_hair_displacement_to_mesh(
        mesh_low, height_map=None, max_displacement_cm=0.0, subdivide_level=2, enable_displacement=False
    )

    # 4. Atasare materiale PBR cu harta de normale:
    tex_low_pil = Image.fromarray(cv2.cvtColor(tex_bgr_low, cv2.COLOR_BGR2RGB))
    normal_low_pil = Image.fromarray(cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2RGB))
    mat_low = trimesh.visual.material.PBRMaterial(
        baseColorTexture=tex_low_pil,
        normalTexture=normal_low_pil,
        roughnessFactor=0.6,
        metallicFactor=0.1
    )
    mesh_low_displaced.visual = trimesh.visual.TextureVisuals(uv=mesh_low_displaced.visual.uv, material=mat_low)

    tex_high_pil = Image.fromarray(cv2.cvtColor(tex_bgr, cv2.COLOR_BGR2RGB))
    normal_high_pil = Image.fromarray(cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2RGB))
    mat_high = trimesh.visual.material.PBRMaterial(
        baseColorTexture=tex_high_pil,
        normalTexture=normal_high_pil,
        roughnessFactor=0.6,
        metallicFactor=0.1
    )
    mesh_high.visual = trimesh.visual.TextureVisuals(uv=mesh_high.visual.uv, material=mat_high)

    # 5. Export OBJ:
    obj_high_path = dir_obj_high / f"{high_name}.obj"
    mesh_high.export(str(obj_high_path))
    mtl_high = dir_obj_high / "material.mtl"
    with open(mtl_high, "w") as f:
        f.write(
            f"# Material for {high_name}.obj\n"
            f"newmtl material_0\n"
            f"Ka 1.000000 1.000000 1.000000\n"
            f"Kd 1.000000 1.000000 1.000000\n"
            f"Ks 0.100000 0.100000 0.100000\n"
            f"Ns 10.000000\n"
            f"map_Kd {tex_high_png_path.name}\n"
            f"norm {normal_high_png_path.name}\n"
            f"map_Bump -bm 1.0 {normal_high_png_path.name}\n"
        )
    # Curățare textură duplicat generată implicit de trimesh
    mat0_high = dir_obj_high / "material_0.png"
    if mat0_high.exists():
        mat0_high.unlink()
    print(f"  -> Saved OBJ High: {obj_high_path.relative_to(output_dir)}")

    obj_low_path = dir_obj_low / f"{low_name}.obj"
    mesh_low_displaced.export(str(obj_low_path))
    mtl_low = dir_obj_low / "material.mtl"
    with open(mtl_low, "w") as f:
        f.write(
            f"# Material for {low_name}.obj\n"
            f"newmtl material_0\n"
            f"Ka 1.000000 1.000000 1.000000\n"
            f"Kd 1.000000 1.000000 1.000000\n"
            f"Ks 0.100000 0.100000 0.100000\n"
            f"Ns 10.000000\n"
            f"map_Kd {tex_low_jpg_path.name}\n"
            f"norm {normal_low_png_path.name}\n"
            f"map_Bump -bm 1.0 {normal_low_png_path.name}\n"
        )
    mat0_low = dir_obj_low / "material_0.png"
    if mat0_low.exists():
        mat0_low.unlink()
    print(f"  -> Saved OBJ Low (Subdivided Level 1 with 3D Relief): {obj_low_path.relative_to(output_dir)}")

    # 6. Export GLB (cu loc de completare Draco):
    glb_high_path = dir_glb / f"{high_name}.glb"
    convert_to_glb_draco(mesh_high, glb_high_path)
    print(f"  -> Saved GLB High: {glb_high_path.relative_to(output_dir)}")

    glb_low_path = dir_glb / f"{low_name}.glb"
    convert_to_glb_draco(mesh_low_displaced, glb_low_path)
    print(f"  -> Saved GLB Low (with PBR Normal Map & 3D Relief): {glb_low_path.relative_to(output_dir)}")

    # 7. Export STL:
    stl_high_path = dir_stl / f"{high_name}.stl"
    convert_to_stl(mesh_high, stl_high_path)
    print(f"  -> Saved STL High: {stl_high_path.relative_to(output_dir)}")

    stl_low_path = dir_stl / f"{low_name}.stl"
    convert_to_stl(mesh_low_displaced, stl_low_path)
    print(f"  -> Saved STL Low (Physical 3D Relief for printing): {stl_low_path.relative_to(output_dir)}")

    # 8. Export 3MF:
    thrf_high_path = dir_3mf / f"{high_name}.3mf"
    convert_to_3mf(mesh_high, thrf_high_path)
    print(f"  -> Saved 3MF High: {thrf_high_path.relative_to(output_dir)}")

    thrf_low_path = dir_3mf / f"{low_name}.3mf"
    convert_to_3mf(mesh_low_displaced, thrf_low_path)
    print(f"  -> Saved 3MF Low (3D Manufacturing Format): {thrf_low_path.relative_to(output_dir)}")

    return {
        "obj": {"high": obj_high_path, "low": obj_low_path},
        "textures": {"high_png": tex_high_png_path, "low_jpg": tex_low_jpg_path, "normal_png": normal_low_png_path},
        "glb": {"high": glb_high_path, "low": glb_low_path},
        "stl": {"high": stl_high_path, "low": stl_low_path},
        "3mf": {"high": thrf_high_path, "low": thrf_low_path},
    }


class Face3DReconstructor:
    def __init__(
        self,
        model_task_path: str = "assets/face_landmarker.task",
        canonical_mesh_path: str = "assets/canonical_face_model.obj",
    ):
        p_task = Path(model_task_path)
        if not p_task.is_file():
            if (Path(__file__).parent / "assets" / p_task.name).is_file():
                p_task = Path(__file__).parent / "assets" / p_task.name
            elif (Path(__file__).parent / p_task.name).is_file():
                p_task = Path(__file__).parent / p_task.name
        self.model_task_path = p_task.resolve()

        p_mesh = Path(canonical_mesh_path)
        if not p_mesh.is_file():
            if (Path(__file__).parent / "assets" / p_mesh.name).is_file():
                p_mesh = Path(__file__).parent / "assets" / p_mesh.name
            elif (Path(__file__).parent / p_mesh.name).is_file():
                p_mesh = Path(__file__).parent / p_mesh.name
        self.canonical_mesh_path = p_mesh.resolve()

        if not self.canonical_mesh_path.exists():
            raise FileNotFoundError(f"Canonical mesh not found at {self.canonical_mesh_path}")

        self.canonical_mesh_raw = trimesh.load(str(self.canonical_mesh_path), process=False)
        # Stitch anatomical ears to canonical mesh
        self.canonical_mesh, self.ear_meta = attach_ears_to_canonical_mesh(self.canonical_mesh_raw)
        self.scanner = FaceScanner(model_path=str(self.model_task_path))

    def close(self):
        if hasattr(self, "scanner") and self.scanner is not None:
            self.scanner.close()
            self.scanner = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def reconstruct_from_image(
        self,
        image_input: Union[str, Path],
        output_dir: Union[str, Path] = "./output_3d",
        name_prefix: str = "model",
        tex_size: int = 2048,
    ) -> Dict[str, Any]:
        """
        Processes a single image (local file path or http/https URL), extracts 3D geometry,
        bakes texture, and exports in the multi-format hierarchy (obj/high, obj/low, glb, stl, 3mf).
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        input_str = str(image_input).strip()

        # Handle URL download
        if input_str.startswith("http://") or input_str.startswith("https://"):
            print(f"Downloading online image from: {input_str}...")
            temp_img_path = output_dir / "downloaded_input.jpg"
            download_image_from_url(input_str, temp_img_path)
            local_img_path = temp_img_path
        else:
            local_img_path = Path(input_str).resolve()
            if not local_img_path.exists():
                raise FileNotFoundError(f"Image file not found: {local_img_path}")

        print(f"Scanning face in: {local_img_path.name}...")
        scan_result = self.scanner.scan_image(local_img_path)

        if scan_result["num_faces_detected"] == 0:
            raise ValueError(f"No human face detected in {local_img_path.name}")

        face = scan_result["faces"][0]
        img_bgr = cv2.imread(str(local_img_path))

        # 1. Deform geometry to match detected landmarks (including stitched ears)
        print("Deforming 3D facial geometry (including stitched ears)...")
        mesh_3d = deform_mesh_with_ears(self.canonical_mesh, face["pixel_landmarks"], self.ear_meta)

        # 2. Bake UV texture with complete face + ear pixel points
        print(f"Baking {tex_size}x{tex_size} UV texture with seamless inpainting...")
        pts_pixel = compute_2d_ear_pixel_points(face["pixel_landmarks"], self.ear_meta)
        baked_bgr = bake_uv_texture(img_bgr, pts_pixel, self.canonical_mesh, tex_size=tex_size)

        # 3. Export multi-format hierarchy
        outputs = export_multi_format_hierarchy(mesh_3d, baked_bgr, output_dir, name_prefix)
        return outputs

    def reconstruct_from_stereo_pair(
        self,
        left_image_input: Union[str, Path],
        right_image_input: Union[str, Path],
        output_dir: Union[str, Path] = "./output_3d",
        name_prefix: str = "face_mesh",
        tex_size: int = 2048,
    ) -> Dict[str, Any]:
        """
        Processes a calibrated stereo image pair (Left & Right cameras at 27° convergence),
        fusing 3D landmark geometry and baking seamless dual-camera UV textures,
        exporting all models in the multi-format hierarchy.
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        left_p = Path(left_image_input).resolve()
        right_p = Path(right_image_input).resolve()

        print(f"Scanning stereo face pair: {left_p.name} and {right_p.name}...")
        scan_left = self.scanner.scan_image(left_p)
        scan_right = self.scanner.scan_image(right_p)

        if scan_left["num_faces_detected"] == 0 and scan_right["num_faces_detected"] == 0:
            raise ValueError("No human face detected in either camera frame.")

        img_left = cv2.imread(str(left_p))
        img_right = cv2.imread(str(right_p))

        # Determine primary face geometry
        if scan_left["num_faces_detected"] > 0:
            face_primary = scan_left["faces"][0]
        else:
            face_primary = scan_right["faces"][0]

        print("Deforming 3D facial geometry from stereo landmarks (including stitched ears)...")
        mesh_3d = deform_mesh_with_ears(self.canonical_mesh, face_primary["pixel_landmarks"], self.ear_meta)

        # Bake fused stereo UV texture if both faces are detected
        if scan_left["num_faces_detected"] > 0 and scan_right["num_faces_detected"] > 0:
            print(f"Baking fused dual-camera stereo UV texture ({tex_size}x{tex_size})...")
            face_l = scan_left["faces"][0]
            face_r = scan_right["faces"][0]
            pts_l = compute_2d_ear_pixel_points(face_l["pixel_landmarks"], self.ear_meta)
            pts_r = compute_2d_ear_pixel_points(face_r["pixel_landmarks"], self.ear_meta)
            baked_bgr = bake_stereo_uv_texture(img_left, pts_l, img_right, pts_r, self.canonical_mesh, tex_size=tex_size)
        else:
            # Fallback to single available camera
            print("Notice: Single camera available for texture baking. Using fallback inpainting...")
            pts_pixel = compute_2d_ear_pixel_points(face_primary["pixel_landmarks"], self.ear_meta)
            img_active = img_left if scan_left["num_faces_detected"] > 0 else img_right
            baked_bgr = bake_uv_texture(img_active, pts_pixel, self.canonical_mesh, tex_size=tex_size)

        # Export multi-format hierarchy
        return export_multi_format_hierarchy(mesh_3d, baked_bgr, output_dir, name_prefix)

    def reconstruct_from_rig_directory(
        self,
        rig_dir: Union[str, Path],
        output_dir: Union[str, Path] = "./output_scan",
        tex_size: int = 2048,
        name_prefix: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Processes stereo photos organized in camera subdirectories (e.g. rig/0 and rig/1).
        Pairs them by matching filenames, EXIF UniqueIDs, or 1-to-1 matching,
        logs camera serials and EXIF metadata, and generates the complete 3D model hierarchy.
        """
        rig_path = Path(rig_dir).resolve()
        if not rig_path.is_dir():
            raise FileNotFoundError(f"Rig directory not found: {rig_path}")

        # Search for 0 and 1 subdirectories
        dir_0 = rig_path / "0"
        dir_1 = rig_path / "1"

        if not dir_0.is_dir() or not dir_1.is_dir():
            for c0, c1 in [("cam0", "cam1"), ("camera0", "camera1"), ("left", "right")]:
                if (rig_path / c0).is_dir() and (rig_path / c1).is_dir():
                    dir_0 = rig_path / c0
                    dir_1 = rig_path / c1
                    break

        if not dir_0.is_dir() or not dir_1.is_dir():
            raise ValueError(
                f"Rig directory '{rig_path}' must contain '0' (Camera 0 / Left) and '1' (Camera 1 / Right) subdirectories."
            )

        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        images_0 = sorted([p for p in dir_0.iterdir() if p.suffix.lower() in valid_exts])
        images_1 = sorted([p for p in dir_1.iterdir() if p.suffix.lower() in valid_exts])

        if not images_0:
            raise FileNotFoundError(f"No valid images found in Camera 0 directory: {dir_0}")
        if not images_1:
            raise FileNotFoundError(f"No valid images found in Camera 1 directory: {dir_1}")

        # Pair images across 0 and 1
        pairs: List[Tuple[Path, Path]] = []
        map_1 = {p.name: p for p in images_1}

        # 1. Exact filename match
        unmatched_0 = []
        for p0 in images_0:
            if p0.name in map_1:
                pairs.append((p0, map_1[p0.name]))
            else:
                unmatched_0.append(p0)

        # 2. Single pair fallback
        if not pairs and len(images_0) == 1 and len(images_1) == 1:
            pairs.append((images_0[0], images_1[0]))
        elif unmatched_0:
            # 3. Match by EXIF UniqueID
            map_1_remaining = [p for p in images_1 if p not in [pr[1] for pr in pairs]]
            for p0 in unmatched_0:
                exif_0 = extract_image_exif(p0)
                matched = False
                if exif_0["unique_id"]:
                    for p1 in map_1_remaining:
                        exif_1 = extract_image_exif(p1)
                        if exif_1["unique_id"] and exif_1["unique_id"] == exif_0["unique_id"]:
                            pairs.append((p0, p1))
                            map_1_remaining.remove(p1)
                            matched = True
                            break
                if not matched and len(unmatched_0) == 1 and len(map_1_remaining) == 1:
                    pairs.append((p0, map_1_remaining[0]))

        if not pairs:
            raise ValueError(
                f"Could not pair any stereo images between {dir_0} ({len(images_0)} files) and {dir_1} ({len(images_1)} files)."
            )

        print("\n" + "=" * 60)
        print(f"FOUND {len(pairs)} STEREO RIG PAIR(S) IN: {rig_path.name}")
        print("=" * 60)

        all_results = []
        out_base = Path(output_dir).resolve()

        for idx, (p_left, p_right) in enumerate(pairs, start=1):
            exif_l = extract_image_exif(p_left)
            exif_r = extract_image_exif(p_right)

            print(f"\n[Pair #{idx}]")
            print(f"  • Camera 0 (Left):  {p_left.name}")
            print(f"    └─ File: {p_left}")
            if exif_l['serial'] or exif_l['unique_id'] or exif_l['dimensions']:
                print(f"    └─ EXIF: Serial={exif_l['serial'] or 'N/A'} | UniqueID={exif_l['unique_id'] or 'N/A'} | Size={exif_l['dimensions'] or 'N/A'}")
            
            print(f"  • Camera 1 (Right): {p_right.name}")
            print(f"    └─ File: {p_right}")
            if exif_r['serial'] or exif_r['unique_id'] or exif_r['dimensions']:
                print(f"    └─ EXIF: Serial={exif_r['serial'] or 'N/A'} | UniqueID={exif_r['unique_id'] or 'N/A'} | Size={exif_r['dimensions'] or 'N/A'}")

            pair_prefix = name_prefix or ("model" if len(pairs) == 1 else p_left.stem)
            target_out = out_base if len(pairs) == 1 else out_base / pair_prefix

            res = self.reconstruct_from_stereo_pair(
                left_image_input=p_left,
                right_image_input=p_right,
                output_dir=target_out,
                name_prefix=pair_prefix,
                tex_size=tex_size,
            )
            all_results.append(res)
            print(f"  -> Generated 3D models in: {target_out}")

        return all_results

    def reconstruct_from_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path] = "./output_3d",
        tex_size: int = 2048,
    ) -> List[Dict[str, Path]]:
        """Processes all images in a directory."""
        input_dir = Path(input_dir).resolve()
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        images = [p for p in input_dir.iterdir() if p.suffix.lower() in valid_exts]
        images.sort()

        if not images:
            print(f"No valid images found in directory: {input_dir}")
            return []

        print(f"Found {len(images)} images in {input_dir.name}. Processing...")
        all_results = []
        for img_p in images:
            prefix = img_p.stem
            res = self.reconstruct_from_image(
                img_p, output_dir=output_dir, name_prefix=prefix, tex_size=tex_size
            )
            all_results.append(res)
        return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Unified 3D Face Reconstructor (Stereo Rig 2-Photos / Single Image / Folder to 3D Head)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  1. Process stereo rig directory (reads 0/ and 1/ subdirectories):
     ./run_reconstruct.sh --rig-dir input_images/rig -o output_scan/
     ./run_reconstruct.sh -i input_images/rig -o output_scan/

  2. Process explicit 2 stereo photos:
     ./run_reconstruct.sh -l input_images/rig/0/face.jpg -r input_images/rig/1/face.jpg -o output_scan/

  3. Process single image:
     ./run_reconstruct.sh -i photo.jpg -o output_scan/
        """
    )
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        help="Path to an image file, a directory of images, a web URL, or a rig directory (with 0/ and 1/)",
    )
    parser.add_argument(
        "-l",
        "--left",
        default=None,
        help="Path to Left camera photo (Camera 0 in rig)",
    )
    parser.add_argument(
        "-r",
        "--right",
        default=None,
        help="Path to Right camera photo (Camera 1 in rig)",
    )
    parser.add_argument(
        "--rig-dir",
        default=None,
        help="Path to stereo rig directory containing '0' and '1' subdirectories (e.g. input_images/rig)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="./output_scan",
        help="Directory to save the resulting 3D models (default: ./output_scan)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Prefix name for the output files (default: face_mesh or input filename)",
    )
    parser.add_argument(
        "--tex-size",
        type=int,
        default=2048,
        help="Texture resolution (e.g. 1024, 2048, 4096)",
    )

    args = parser.parse_args()

    # Determine execution mode
    with Face3DReconstructor() as reconstructor:
        # Mode 1: Explicit stereo pair via --left and --right
        if args.left and args.right:
            prefix = args.name or Path(args.left).stem
            print(f"\nRunning 2-photo stereo reconstruction:")
            print(f"  • Left  (Cam 0): {args.left}")
            print(f"  • Right (Cam 1): {args.right}")
            res = reconstructor.reconstruct_from_stereo_pair(
                left_image_input=args.left,
                right_image_input=args.right,
                output_dir=args.output_dir,
                name_prefix=prefix,
                tex_size=args.tex_size,
            )
            print(f"\nSuccess! Stereo 3D models created in: {Path(args.output_dir).resolve()}")
            sys.exit(0)

        # Mode 2: Rig directory via --rig-dir
        rig_path_candidate = args.rig_dir
        if not rig_path_candidate and args.input:
            inp_p = Path(args.input)
            if inp_p.is_dir() and ((inp_p / "0").is_dir() and (inp_p / "1").is_dir()):
                rig_path_candidate = str(inp_p)

        if rig_path_candidate:
            print(f"\nRunning rig stereo reconstruction from: {rig_path_candidate}")
            results = reconstructor.reconstruct_from_rig_directory(
                rig_dir=rig_path_candidate,
                output_dir=args.output_dir,
                name_prefix=args.name,
                tex_size=args.tex_size,
            )
            print(f"\nSuccess! All rig pairs processed in: {Path(args.output_dir).resolve()}")
            sys.exit(0)

        # Mode 3: Fallback check for input_images/rig if no input given
        if not args.input:
            default_rig = Path("input_images/rig")
            if default_rig.is_dir() and (default_rig / "0").is_dir() and (default_rig / "1").is_dir():
                print(f"\nAuto-detected stereo rig directory: {default_rig.resolve()}")
                results = reconstructor.reconstruct_from_rig_directory(
                    rig_dir=default_rig,
                    output_dir=args.output_dir,
                    name_prefix=args.name,
                    tex_size=args.tex_size,
                )
                print(f"\nSuccess! All rig pairs processed in: {Path(args.output_dir).resolve()}")
                sys.exit(0)
            else:
                parser.print_help()
                print("\nError: Please provide --rig-dir, --left and --right, or --input.")
                sys.exit(1)

        # Mode 4: Standard input (URL, image file, or directory)
        inp = args.input.strip()
        if inp.startswith("http://") or inp.startswith("https://") or Path(inp).is_file():
            prefix = args.name or (Path(inp).stem if Path(inp).is_file() else "model")
            results = reconstructor.reconstruct_from_image(
                inp, output_dir=args.output_dir, name_prefix=prefix, tex_size=args.tex_size
            )
            print(f"\nSuccess! 3D Model created in: {Path(args.output_dir).resolve()}")
        elif Path(inp).is_dir():
            reconstructor.reconstruct_from_directory(
                inp, output_dir=args.output_dir, tex_size=args.tex_size
            )
            print(f"\nSuccess! All models processed in: {Path(args.output_dir).resolve()}")
        else:
            print(f"Error: Invalid input '{inp}'. Must be a file, directory, or URL.")
            sys.exit(1)


if __name__ == "__main__":
    main()
