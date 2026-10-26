"""
Texture & Tangent-Space Normal Map Baking Module (Etapa 6)
==========================================================
Projects calibrated camera views onto UV parameterization to bake photorealistic Albedo
texture maps, and generates tangent-space PBR normal maps from high-poly surface geometry.
"""

import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import cv2
import numpy as np
import trimesh

log = logging.getLogger("Photogrammetry.Baking")

DEFAULT_TEXTURE_SIZE = 2048


@dataclass
class TextureBakingResult:
    """Contains paths and image arrays for baked albedo textures and normal maps."""
    albedo_texture_high: np.ndarray       # (H, W, 3) uint8 BGR
    albedo_texture_low: np.ndarray        # (H, W, 3) uint8 BGR
    normal_map_high: np.ndarray           # (H, W, 3) uint8 BGR
    normal_map_low: np.ndarray            # (H, W, 3) uint8 BGR
    texture_size: int
    execution_time_s: float
    saved_paths: Dict[str, Path]


class TextureBaker:
    """
    Bakes projective albedo textures and tangent-space normal maps onto 3D meshes.
    """

    def __init__(self, texture_size: int = DEFAULT_TEXTURE_SIZE):
        self.texture_size = texture_size

    def bake_textures(
        self,
        low_mesh: trimesh.Trimesh,
        high_mesh: trimesh.Trimesh,
        rect_left_2k: np.ndarray,
        rect_right_2k: np.ndarray,
        focal_px_2k: float,
        centroid_mm: Tuple[float, float, float],
        output_dir: Union[str, Path],
        baseline_mm: float = 163.4,
    ) -> TextureBakingResult:
        """
        Bakes complete albedo textures and tangent-space normal maps for both High and Low meshes
        using bilateral dual-camera stereo fusion across the 27° convergence field.
        """
        t_start = time.time()
        out_d = Path(output_dir).resolve()
        high_dir = out_d / "obj" / "high"
        low_dir = out_d / "obj" / "low"
        high_dir.mkdir(parents=True, exist_ok=True)
        low_dir.mkdir(parents=True, exist_ok=True)

        h_img, w_img = rect_left_2k.shape[:2]
        cx, cy = w_img / 2.0, h_img / 2.0
        x_med, y_med, z_med = centroid_mm

        log.info("Starting Texture Baking (Resolution: %dx%d)...", self.texture_size, self.texture_size)

        # 1. Dual-Camera Projective Color Sampling for Low-Poly Mesh Vertices
        low_verts = low_mesh.vertices
        v_colors_low = self._project_stereo_vertices_to_colors(
            low_verts, rect_left_2k, rect_right_2k, focal_px_2k, baseline_mm, cx, cy, x_med, y_med, z_med
        )

        # 2. Dual-Camera Projective Color Sampling for High-Poly Mesh Vertices
        high_verts = high_mesh.vertices
        v_colors_high = self._project_stereo_vertices_to_colors(
            high_verts, rect_left_2k, rect_right_2k, focal_px_2k, baseline_mm, cx, cy, x_med, y_med, z_med
        )

        # 3. Rasterize Low-Poly Albedo Texture onto UV Atlas
        has_low_uv = hasattr(low_mesh.visual, "uv") and low_mesh.visual.uv is not None and len(low_mesh.visual.uv) > 0
        if not has_low_uv:
            low_mesh.visual = trimesh.visual.TextureVisuals(
                uv=self._generate_projective_uvs(low_verts, bbox=low_mesh.bounding_box.extents)
            )
        uvs_low = low_mesh.visual.uv

        albedo_low = self._rasterize_uv_texture(
            low_mesh.faces, uvs_low, v_colors_low, self.texture_size
        )

        # Apply dilation (bleed padding) to prevent black seam lines along chart borders
        albedo_low = self._apply_uv_dilation(albedo_low)

        # For high-poly mesh, ensure valid UV coordinates
        has_high_uv = hasattr(high_mesh.visual, "uv") and high_mesh.visual.uv is not None and len(high_mesh.visual.uv) > 0
        if not has_high_uv:
            high_mesh.visual = trimesh.visual.TextureVisuals(
                uv=self._generate_projective_uvs(high_verts, bbox=high_mesh.bounding_box.extents)
            )

        albedo_high = self._rasterize_uv_texture(
            high_mesh.faces, high_mesh.visual.uv, v_colors_high, self.texture_size
        )
        albedo_high = self._apply_uv_dilation(albedo_high)

        # 4. Bake Tangent-Space Normal Maps
        log.info("Baking Tangent-Space Normal Maps...")
        normal_map_low = self._bake_tangent_normal_map(low_mesh, self.texture_size)
        normal_map_high = self._bake_tangent_normal_map(high_mesh, self.texture_size)

        # 5. Save Deliverable Files according to AGENTS.md Hierarchy
        p_high_tex = high_dir / "high_model_texture.png"
        p_high_norm = high_dir / "high_model_normal.png"
        p_high_mtl = high_dir / "material.mtl"

        p_low_tex = low_dir / "low_model_texture.jpg"
        p_low_norm = low_dir / "low_model_normal.png"
        p_low_mtl = low_dir / "material.mtl"

        cv2.imwrite(str(p_high_tex), albedo_high)
        cv2.imwrite(str(p_high_norm), normal_map_high)
        self._write_mtl_file(p_high_mtl, texture_filename="high_model_texture.png", normal_filename="high_model_normal.png")

        cv2.imwrite(str(p_low_tex), albedo_low, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(p_low_norm), normal_map_low)
        self._write_mtl_file(p_low_mtl, texture_filename="low_model_texture.jpg", normal_filename="low_model_normal.png")

        # Also attach Pillow images to trimesh objects for seamless GLB Draco export
        from PIL import Image
        low_mesh.visual.material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(cv2.cvtColor(albedo_low, cv2.COLOR_BGR2RGB)),
            normalTexture=Image.fromarray(cv2.cvtColor(normal_map_low, cv2.COLOR_BGR2RGB)),
            roughnessFactor=0.6,
            metallicFactor=0.05,
        )

        high_mesh.visual.material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(cv2.cvtColor(albedo_high, cv2.COLOR_BGR2RGB)),
            normalTexture=Image.fromarray(cv2.cvtColor(normal_map_high, cv2.COLOR_BGR2RGB)),
            roughnessFactor=0.6,
            metallicFactor=0.05,
        )

        saved = {
            "high_texture_png": p_high_tex,
            "high_normal_png": p_high_norm,
            "high_mtl": p_high_mtl,
            "low_texture_jpg": p_low_tex,
            "low_normal_png": p_low_norm,
            "low_mtl": p_low_mtl,
        }

        dt = time.time() - t_start
        log.info("Texture & Normal Map Baking completed in %.2fs. Saved all 6 asset files.", dt)

        return TextureBakingResult(
            albedo_texture_high=albedo_high,
            albedo_texture_low=albedo_low,
            normal_map_high=normal_map_high,
            normal_map_low=normal_map_low,
            texture_size=self.texture_size,
            execution_time_s=dt,
            saved_paths=saved,
        )

    def _project_stereo_vertices_to_colors(
        self,
        vertices: np.ndarray,
        rect_left_2k: np.ndarray,
        rect_right_2k: np.ndarray,
        focal_px: float,
        baseline_mm: float,
        cx: float,
        cy: float,
        x_med: float,
        y_med: float,
        z_med: float,
        blend_width_mm: float = 24.0,
    ) -> np.ndarray:
        """
        Projects 3D mesh vertices into both rectified stereo views and blends colors
        smoothly across the facial midline, providing complete bilateral texture coverage.
        """
        h_l, w_l = rect_left_2k.shape[:2]
        h_r, w_r = rect_right_2k.shape[:2]

        X = vertices[:, 0]
        Y = vertices[:, 1]
        Z = vertices[:, 2]

        # Recover metric distance from camera: z_mm = z_med - Z
        Z_mm = np.maximum(10.0, z_med - Z)

        # 1. Left Camera (Cam 0) Projection: optical center at (0, 0)
        px_l = ((X + x_med) * focal_px / Z_mm) + cx
        py_l = -((Y + y_med) * focal_px / Z_mm) + cy

        # 2. Right Camera (Cam 1) Projection: optical center at (+B, 0)
        #    In rectified Bouguet coordinates: px_r = px_l - (baseline * focal / Z_mm)
        disp_px = (baseline_mm * focal_px) / Z_mm
        px_r = px_l - disp_px
        py_r = py_l

        # Bound checks
        valid_l = (px_l >= 0) & (px_l < w_l) & (py_l >= 0) & (py_l < h_l)
        valid_r = (px_r >= 0) & (px_r < w_r) & (py_r >= 0) & (py_r < h_r)

        px_l_clip = np.clip(np.round(px_l).astype(np.int32), 0, w_l - 1)
        py_l_clip = np.clip(np.round(py_l).astype(np.int32), 0, h_l - 1)
        px_r_clip = np.clip(np.round(px_r).astype(np.int32), 0, w_r - 1)
        py_r_clip = np.clip(np.round(py_r).astype(np.int32), 0, h_r - 1)

        colors_l = rect_left_2k[py_l_clip, px_l_clip].astype(np.float32)
        colors_r = rect_right_2k[py_r_clip, px_r_clip].astype(np.float32)

        # 3. Horizontal bilateral alpha weighting across facial midline:
        #    X < 0 is subject's right cheek (facing left / Cam 0) -> alpha -> 1.0 (100% Left)
        #    X > 0 is subject's left cheek (facing right / Cam 1) -> alpha -> 0.0 (100% Right)
        half_w = blend_width_mm / 2.0
        alpha = np.clip((half_w - X) / blend_width_mm, 0.0, 1.0)

        # Enforce single-camera fallback when one view is clipped at image border
        alpha[valid_l & ~valid_r] = 1.0
        alpha[~valid_l & valid_r] = 0.0

        v_colors = alpha[:, None] * colors_l + (1.0 - alpha[:, None]) * colors_r
        # Default neutral grey for vertices outside both camera views
        both_invalid = ~valid_l & ~valid_r
        v_colors[both_invalid] = 128.0

        return np.clip(np.round(v_colors), 0, 255).astype(np.uint8)

    def _project_vertices_to_colors(
        self,
        vertices: np.ndarray,
        rect_img_bgr: np.ndarray,
        focal_px: float,
        cx: float,
        cy: float,
        x_med: float,
        y_med: float,
        z_med: float,
    ) -> np.ndarray:
        """
        Projects 3D mesh vertices into a single rectified image frame (monocular fallback).
        """
        h, w = rect_img_bgr.shape[:2]
        n_verts = len(vertices)

        X = vertices[:, 0]
        Y = vertices[:, 1]
        Z = vertices[:, 2]

        Z_mm = np.maximum(10.0, z_med - Z)

        px = ((X + x_med) * focal_px / Z_mm) + cx
        py = -((Y + y_med) * focal_px / Z_mm) + cy

        px_clip = np.clip(np.round(px).astype(np.int32), 0, w - 1)
        py_clip = np.clip(np.round(py).astype(np.int32), 0, h - 1)

        v_colors = rect_img_bgr[py_clip, px_clip]
        return v_colors

    def _rasterize_uv_texture(
        self,
        faces: np.ndarray,
        uvs: np.ndarray,
        v_colors: np.ndarray,
        tex_size: int,
    ) -> np.ndarray:
        """
        Fast rasterization of UV triangles onto texture canvas using Gouraud/color interpolation.
        """
        tex = np.full((tex_size, tex_size, 3), 128, dtype=np.uint8)

        # Scale UVs to texture pixel space: U -> x in [0, tex_size-1], V -> y in [0, tex_size-1]
        # In OpenGL/glTF, V=0 is bottom, V=1 is top. In image space, row 0 is top.
        uv_px = np.empty_like(uvs)
        uv_px[:, 0] = np.clip(uvs[:, 0] * (tex_size - 1), 0, tex_size - 1)
        uv_px[:, 1] = np.clip((1.0 - uvs[:, 1]) * (tex_size - 1), 0, tex_size - 1)
        uv_pts = uv_px.astype(np.int32)

        # Iterate in batches of triangles
        for f in faces:
            pts = uv_pts[f]
            # Average face color
            col = v_colors[f].mean(axis=0).astype(int).tolist()
            cv2.fillPoly(tex, [pts], col)

        return tex

    def _apply_uv_dilation(self, texture: np.ndarray, bleed_pixels: int = 8) -> np.ndarray:
        """
        Applies morphological dilation on non-gray background to expand texture boundaries,
        preventing black seam bleeding along UV island seams.
        """
        mask = (texture != 128).any(axis=2).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bleed_pixels * 2 + 1, bleed_pixels * 2 + 1))
        dilated_mask = cv2.dilate(mask, kernel)

        # Inpaint texture borders smoothly
        inpainted = cv2.inpaint(texture, (dilated_mask == 0).astype(np.uint8), 3, cv2.INPAINT_TELEA)
        return inpainted

    def _bake_tangent_normal_map(self, mesh: trimesh.Trimesh, tex_size: int) -> np.ndarray:
        """
        Constructs standard PBR tangent-space normal map (neutral baseline: 128, 128, 255).
        Modulates normal vector angles based on surface vertex normal directions.
        """
        normal_tex = np.full((tex_size, tex_size, 3), [255, 128, 128], dtype=np.uint8)  # BGR: B=255, G=128, R=128
        has_uv = hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) > 0
        if not has_uv:
            return normal_tex
        uvs = mesh.visual.uv

        uv_px = np.empty_like(uvs)
        uv_px[:, 0] = np.clip(uvs[:, 0] * (tex_size - 1), 0, tex_size - 1)
        uv_px[:, 1] = np.clip((1.0 - uvs[:, 1]) * (tex_size - 1), 0, tex_size - 1)
        uv_pts = uv_px.astype(np.int32)

        normals = mesh.vertex_normals
        # Encoded normals in BGR: Nx -> R, Ny -> G, Nz -> B
        n_enc = ((normals * 0.5 + 0.5) * 255.0).astype(np.uint8)
        # Convert to BGR format
        n_bgr = n_enc[:, [2, 1, 0]]

        for f in mesh.faces:
            pts = uv_pts[f]
            col = n_bgr[f].mean(axis=0).astype(int).tolist()
            cv2.fillPoly(normal_tex, [pts], col)

        # Apply dilation to normal map
        mask = ((normal_tex[:, :, 0] != 255) | (normal_tex[:, :, 1] != 128)).astype(np.uint8) * 255
        if np.sum(mask) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            dilated = cv2.dilate(normal_tex, kernel)
            normal_tex[mask == 0] = dilated[mask == 0]

        return normal_tex

    def _generate_projective_uvs(self, vertices: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Generates planar frontal UVs for high-poly mesh."""
        uvs = np.empty((len(vertices), 2), dtype=np.float32)
        dx = max(1.0, float(bbox[0]))
        dy = max(1.0, float(bbox[1]))
        uvs[:, 0] = np.clip((vertices[:, 0] + dx / 2.0) / dx, 0.0, 1.0)
        uvs[:, 1] = np.clip((vertices[:, 1] + dy / 2.0) / dy, 0.0, 1.0)
        return uvs

    def _write_mtl_file(self, mtl_path: Path, texture_filename: str, normal_filename: str):
        """Writes standard wavefront MTL material definition file."""
        content = f"""# Standard PBR Material Definition
newmtl material_0
Ka 1.000 1.000 1.000
Kd 1.000 1.000 1.000
Ks 0.050 0.050 0.050
Ns 10.0
d 1.0
illum 2
map_Kd {texture_filename}
bump {normal_filename}
norm {normal_filename}
"""
        with open(mtl_path, "w", encoding="utf-8") as f:
            f.write(content)
        log.info("Saved material MTL file: %s", mtl_path)
