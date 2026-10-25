"""
Unit Tests for Photogrammetry Etapa 4 & Etapa 5
==============================================
Tests:
  1. Metric 3D Triangulation & Screened Poisson Reconstruction (Depth 9, Density Trimming).
  2. Quadric Error Decimation (QEM) to 20,000 triangles.
  3. xatlas UV atlas parameterization on decimated mesh.
  4. End-to-end execution on real sample f_150.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import trimesh

from photogrammetry.ingestion import StereoRigIngestor
from photogrammetry.rectification import StereoRectifier
from photogrammetry.face_mask import FacialMaskExtractor
from photogrammetry.dense_stereo import DenseStereoMatcher
from photogrammetry.triangulation_poisson import (
    MetricPoissonReconstructor,
    PoissonReconstructionResult,
)
from photogrammetry.decimation_uv import (
    MeshDecimatorUV,
    DecimationUVResult,
)


def test_poisson_reconstruction_synthetic():
    """Verifies Screened Poisson reconstruction and density trimming on synthetic surface."""
    H, W = 400, 400
    focal_px = 1000.0
    baseline_mm = 163.4

    # Create synthetic curved dome (sphere patch) with depth Z ~ 350 mm
    y_grid, x_grid = np.mgrid[:H, :W]
    cx, cy = W / 2.0, H / 2.0
    r_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
    r_max_sq = 120 ** 2

    valid_mask = r_sq < r_max_sq
    # Z in [330, 370] mm
    z_mm = np.full((H, W), 350.0, dtype=np.float32)
    z_mm[valid_mask] = 350.0 - np.sqrt(np.maximum(0.0, r_max_sq - r_sq[valid_mask])) * 0.2

    # Disparity d = f * B / Z
    disparity = np.zeros((H, W), dtype=np.float32)
    disparity[valid_mask] = (focal_px * baseline_mm) / z_mm[valid_mask]

    mask = (valid_mask * 255).astype(np.uint8)
    rect_img = np.full((H, W, 3), 180, dtype=np.uint8)

    reconstructor = MetricPoissonReconstructor(octree_depth=7, density_trim_percentile=5.0)
    res = reconstructor.triangulate_and_reconstruct(
        disparity_2k=disparity,
        rect_left_2k=rect_img,
        focal_px_2k=focal_px,
        baseline_mm=baseline_mm,
        mask_2k=mask,
        d0_shift=0
    )

    assert res.num_vertices > 100
    assert res.num_faces > 100
    assert isinstance(res.mesh, trimesh.Trimesh)
    assert res.mesh.is_empty is False


def test_decimation_uv_synthetic():
    """Verifies QEM decimation and xatlas UV unwrapping on a high-poly sphere."""
    # Create high-poly icosphere (approx 5,120 faces)
    high_mesh = trimesh.creation.icosphere(subdivisions=4, radius=50.0)
    assert len(high_mesh.faces) > 2000

    target_faces = 800
    decimator = MeshDecimatorUV(target_faces=target_faces)
    res = decimator.decimate_and_unwrap(high_mesh, target_faces=target_faces)

    assert abs(res.low_face_count - target_faces) < target_faces * 0.15
    assert len(res.uv_coordinates) == res.low_vertex_count
    # Check that UV coordinates are bounded in [0, 1]
    assert np.all(res.uv_coordinates >= -0.01)
    assert np.all(res.uv_coordinates <= 1.01)

    # Test export
    with tempfile.TemporaryDirectory() as td:
        p_obj = Path(td) / "low_model.obj"
        res.save_mesh(p_obj)
        assert p_obj.is_file()


def test_stage4_5_end_to_end_on_real_sample():
    """Executes full pipeline Etapele 1-5 on f_150 real photos."""
    p_l = Path("input_images/rig/0/f_150.jpg")
    p_r = Path("input_images/rig/1/f_150.jpg")
    if not p_l.is_file() or not p_r.is_file():
        print("Skipping real sample test: files not found.")
        return

    # Etapele 1-3
    ingestor = StereoRigIngestor(check_apriltags=False)
    ingestion_res = ingestor.ingest_pair(p_l, p_r, pair_id="f_150")

    rectifier = StereoRectifier()
    rect_res = rectifier.rectify_pair(p_l, p_r, ingestion_result=ingestion_res, refine_with_features=True)

    mask_extractor = FacialMaskExtractor()
    mask_res = mask_extractor.process_rectified_pair(rect_res.rect_left_2k, rect_res.rect_right_2k)

    matcher = DenseStereoMatcher(num_residual_disparities=256)
    disp_res = matcher.compute_disparity(
        rect_res.rect_left_2k,
        rect_res.rect_right_2k,
        d0_shift=mask_res.d0_shift,
        mask_2k=mask_res.mask_2k,
        pair_id="f_150"
    )

    # Etapa 4: Triangulation & Screened Poisson
    reconstructor = MetricPoissonReconstructor(octree_depth=9, density_trim_percentile=6.0)
    poisson_res = reconstructor.triangulate_and_reconstruct(
        disparity_2k=disp_res.disparity_2k,
        rect_left_2k=rect_res.rect_left_2k,
        focal_px_2k=rect_res.effective_focal_2k,
        baseline_mm=rect_res.baseline_mm,
        mask_2k=mask_res.mask_2k,
        d0_shift=mask_res.d0_shift
    )

    assert poisson_res.num_faces > 10000
    assert poisson_res.mesh.is_empty is False

    # Etapa 5: Decimation & xatlas UV
    decimator = MeshDecimatorUV(target_faces=20000)
    uv_res = decimator.decimate_and_unwrap(poisson_res.mesh, target_faces=20000)

    assert abs(uv_res.low_face_count - 20000) < 4000
    assert len(uv_res.uv_coordinates) > 0

    with tempfile.TemporaryDirectory() as td:
        p_high = Path(td) / "high_model.obj"
        p_low = Path(td) / "low_model.obj"
        poisson_res.save_mesh(p_high)
        uv_res.save_mesh(p_low)

        assert p_high.is_file()
        assert p_low.is_file()


if __name__ == "__main__":
    print("Testing test_poisson_reconstruction_synthetic...")
    test_poisson_reconstruction_synthetic()
    print("  [PASS] test_poisson_reconstruction_synthetic")

    print("Testing test_decimation_uv_synthetic...")
    test_decimation_uv_synthetic()
    print("  [PASS] test_decimation_uv_synthetic")

    print("Testing test_stage4_5_end_to_end_on_real_sample...")
    test_stage4_5_end_to_end_on_real_sample()
    print("  [PASS] test_stage4_5_end_to_end_on_real_sample")

    print("\n>>> ALL STAGE 4 & 5 TESTS PASSED SUCCESSFULLY! <<<")
