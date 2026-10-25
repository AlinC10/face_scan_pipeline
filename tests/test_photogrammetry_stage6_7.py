"""
Unit Tests for Photogrammetry Etapa 6 & Etapa 7
==============================================
Tests:
  1. Texture & Tangent-Space Normal Map Baking (Etapa 6).
  2. Standard 8 Deliverables Exporter: OBJ, GLB Draco, STL, 3MF (Etapa 7).
  3. End-to-end validation of full 7-stage pipeline.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import trimesh

from photogrammetry.texture_baker import (
    TextureBaker,
    TextureBakingResult,
)
from photogrammetry.export_manager import (
    StandardDeliverablesExporter,
    ExportResult,
)
from run_photogrammetry_runpod import run_photogrammetry_pipeline


def test_texture_baker_synthetic():
    """Verifies Albedo texture and Tangent Normal Map baking on synthetic mesh."""
    # Create simple low-poly mesh with UV coordinates
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=40.0)
    # Add simple planar UV coordinates
    uvs = np.zeros((len(mesh.vertices), 2), dtype=np.float32)
    uvs[:, 0] = np.clip((mesh.vertices[:, 0] + 40.0) / 80.0, 0.0, 1.0)
    uvs[:, 1] = np.clip((mesh.vertices[:, 1] + 40.0) / 80.0, 0.0, 1.0)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uvs)

    high_mesh = trimesh.creation.icosphere(subdivisions=3, radius=40.0)

    # Dummy images (1536 x 2048)
    img_l = np.full((1536, 2048, 3), (120, 150, 190), dtype=np.uint8)
    img_r = np.full((1536, 2048, 3), (115, 145, 185), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as td:
        baker = TextureBaker(texture_size=512)
        res = baker.bake_textures(
            low_mesh=mesh,
            high_mesh=high_mesh,
            rect_left_2k=img_l,
            rect_right_2k=img_r,
            focal_px_2k=1271.3,
            centroid_mm=(0.0, 0.0, 350.0),
            output_dir=td,
        )

        assert res.albedo_texture_low.shape == (512, 512, 3)
        assert res.normal_map_low.shape == (512, 512, 3)
        assert res.saved_paths["high_texture_png"].is_file()
        assert res.saved_paths["high_normal_png"].is_file()
        assert res.saved_paths["high_mtl"].is_file()
        assert res.saved_paths["low_texture_jpg"].is_file()
        assert res.saved_paths["low_normal_png"].is_file()
        assert res.saved_paths["low_mtl"].is_file()


def test_standard_deliverables_export():
    """Verifies generation of all 8 standard deliverable files."""
    high_mesh = trimesh.creation.icosphere(subdivisions=3, radius=40.0)
    low_mesh = trimesh.creation.icosphere(subdivisions=2, radius=40.0)

    with tempfile.TemporaryDirectory() as td:
        exporter = StandardDeliverablesExporter(output_dir=td)
        res = exporter.export_all_deliverables(high_mesh, low_mesh)

        assert res.is_complete is True
        assert res.total_files_count == 8

        # Check required files
        expected_keys = [
            "obj_high", "obj_low",
            "glb_high", "glb_low",
            "stl_high", "stl_low",
            "3mf_high", "3mf_low"
        ]
        for key in expected_keys:
            assert key in res.deliverables
            p = res.deliverables[key]
            assert p.is_file()
            assert p.stat().st_size > 0


def test_stage6_7_end_to_end_on_real_sample():
    """Executes full 7-stage photogrammetry pipeline on repository rig sample f_150."""
    p_l = Path("input_images/rig/0/f_150.jpg")
    p_r = Path("input_images/rig/1/f_150.jpg")
    if not p_l.is_file() or not p_r.is_file():
        print("Skipping real sample test: files not found.")
        return

    with tempfile.TemporaryDirectory() as td:
        summary = run_photogrammetry_pipeline(
            left_path=p_l,
            right_path=p_r,
            output_dir=td,
            check_apriltags=False,
            stop_after_stage=7,
        )

        assert summary["status"] == "success"
        assert len(summary["stages_completed"]) == 7

        out_path = Path(td)
        assert (out_path / "obj" / "high" / "high_model.obj").is_file()
        assert (out_path / "obj" / "high" / "high_model_texture.png").is_file()
        assert (out_path / "obj" / "high" / "high_model_normal.png").is_file()
        assert (out_path / "obj" / "high" / "material.mtl").is_file()

        assert (out_path / "obj" / "low" / "low_model.obj").is_file()
        assert (out_path / "obj" / "low" / "low_model_texture.jpg").is_file()
        assert (out_path / "obj" / "low" / "low_model_normal.png").is_file()
        assert (out_path / "obj" / "low" / "material.mtl").is_file()

        assert (out_path / "glb" / "high_model.glb").is_file()
        assert (out_path / "glb" / "low_model.glb").is_file()

        assert (out_path / "stl" / "high_model.stl").is_file()
        assert (out_path / "stl" / "low_model.stl").is_file()

        assert (out_path / "3mf" / "high_model.3mf").is_file()
        assert (out_path / "3mf" / "low_model.3mf").is_file()

        assert (out_path / "photogrammetry_manifest.json").is_file()


if __name__ == "__main__":
    print("Testing test_texture_baker_synthetic...")
    test_texture_baker_synthetic()
    print("  [PASS] test_texture_baker_synthetic")

    print("Testing test_standard_deliverables_export...")
    test_standard_deliverables_export()
    print("  [PASS] test_standard_deliverables_export")

    print("Testing test_stage6_7_end_to_end_on_real_sample...")
    test_stage6_7_end_to_end_on_real_sample()
    print("  [PASS] test_stage6_7_end_to_end_on_real_sample")

    print("\n>>> ALL STAGE 6 & 7 TESTS PASSED SUCCESSFULLY! <<<")
