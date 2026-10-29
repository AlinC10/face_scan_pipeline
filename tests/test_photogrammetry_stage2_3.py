"""
Unit Tests for Photogrammetry Etapa 2 & Etapa 3
==============================================
Tests:
  1. Bouguet Stereo Rectifier (Nominal & AprilTag Extrinsics, Q matrix, Epipolar Inspection).
  2. Facial Mask Extractor (Face Oval, Dilation, d0 Pre-Shift calculation).
  3. Dense Stereo Matcher (Pre-Shift d0 right shift, SGBM+WLS, 16-bit and color exports).
  4. End-to-end execution on real rig sample pair f_150.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np

from photogrammetry.ingestion import StereoRigIngestor
from photogrammetry.rectification import (
    StereoRectifier,
    StereoRectificationResult,
    NOMINAL_BASELINE_MM,
    NOMINAL_CONVERGENCE_DEG,
)
from photogrammetry.face_mask import (
    FacialMaskExtractor,
    FacialMaskResult,
)
from photogrammetry.dense_stereo import (
    DenseStereoMatcher,
    DenseDisparityResult,
)


def test_stereo_rectifier_nominal():
    """Verifies that nominal rig extrinsics are correctly computed and Q matrix generated."""
    rectifier = StereoRectifier()
    assert abs(rectifier.nominal_baseline_mm - 163.4) < 1e-4
    assert abs(rectifier.nominal_convergence_deg - 27.0) < 1e-4

    # Dummy images (2048 x 1536)
    dummy_l = np.full((1536, 2048, 3), 120, dtype=np.uint8)
    dummy_r = np.full((1536, 2048, 3), 120, dtype=np.uint8)

    res = rectifier.rectify_pair(dummy_l, dummy_r, refine_with_features=False)
    assert res.rect_left_2k.shape == (1536, 2048, 3)
    assert res.rect_right_2k.shape == (1536, 2048, 3)
    assert res.Q_2k.shape == (4, 4)
    assert res.P_rect0.shape == (3, 4)
    assert res.P_rect1.shape == (3, 4)

    # In rectified pair, focal lengths match exactly
    assert abs(res.P_rect0[0, 0] - res.P_rect1[0, 0]) < 1e-3
    assert abs(res.P_rect0[1, 1] - res.P_rect1[1, 1]) < 1e-3
    # P_rect1 has non-zero baseline translation along X
    assert abs(res.P_rect1[0, 3]) > 0.0


def test_epipolar_inspection_composite():
    """Verifies the side-by-side composite visualization with horizontal scanlines."""
    rectifier = StereoRectifier()
    dummy_l = np.full((1536, 2048, 3), 50, dtype=np.uint8)
    dummy_r = np.full((1536, 2048, 3), 70, dtype=np.uint8)
    res = rectifier.rectify_pair(dummy_l, dummy_r, refine_with_features=False)

    comp = res.generate_epipolar_inspection_image(num_scanlines=10)
    assert comp.shape == (1536, 4096, 3)


def test_facial_mask_fallback():
    """Verifies fallback elliptical mask and nominal d0 when no face is present."""
    extractor = FacialMaskExtractor()
    blank_l = np.zeros((1536, 2048, 3), dtype=np.uint8)
    blank_r = np.zeros((1536, 2048, 3), dtype=np.uint8)

    res = extractor.process_rectified_pair(blank_l, blank_r)
    assert res.face_detected_cam0 is False
    assert res.mask_2k.shape == (1536, 2048)
    assert np.sum(res.mask_2k == 255) > 0  # Fallback ellipse contains valid pixels
    # Pre-shift d0 is non-negative
    assert res.d0_shift >= 0


def test_dense_stereo_preshift():
    """Verifies dense disparity matching with pre-shift d0."""
    matcher = DenseStereoMatcher(num_residual_disparities=256, block_size=5)

    H, W = 1536, 2048
    # Create textured background
    np.random.seed(42)
    left_img = np.random.randint(50, 200, (H, W, 3), dtype=np.uint8)

    # Shift right image by true disparity = 500 px
    true_disp = 500
    right_img = np.zeros_like(left_img)
    right_img[:, :W - true_disp] = left_img[:, true_disp:]

    # Mask over center region
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[400:1100, 500:1500] = 255

    # If median disparity is 500, d0 = max(0, 500 - 128) = 372
    d0_shift = 372
    res = matcher.compute_disparity(left_img, right_img, d0_shift=d0_shift, mask_2k=mask)

    assert res.disparity_2k.shape == (H, W)
    assert res.residual_disp.shape == (H, W)
    assert res.d0_shift == 372
    # Verify pixels outside mask are 0
    assert np.all(res.disparity_2k[mask == 0] == 0.0)

    # Test file saving
    with tempfile.TemporaryDirectory() as td:
        tmp_d = Path(td)
        p_raw = tmp_d / "disp_raw.png"
        p_col = tmp_d / "disp_col.jpg"
        res.save_disparity_16bit(p_raw)
        res.save_colored_visualization(p_col)

        assert p_raw.is_file()
        assert p_col.is_file()
        loaded_raw = cv2.imread(str(p_raw), cv2.IMREAD_UNCHANGED)
        assert loaded_raw.dtype == np.uint16


def test_stage2_3_end_to_end_on_real_sample():
    """
    Executes Etapa 1 -> Etapa 2 -> Etapa 3 on the actual repository pair:
    input_images/rig/0/f_150.jpg and input_images/rig/1/f_150.jpg.
    """
    p_l = Path("input_images/rig/0/f_150.jpg")
    p_r = Path("input_images/rig/1/f_150.jpg")
    if not p_l.is_file() or not p_r.is_file():
        print("Skipping real sample test: files not found.")
        return

    # Etapa 1: Ingestion
    ingestor = StereoRigIngestor(check_apriltags=True)
    ingestion_res = ingestor.ingest_pair(p_l, p_r, pair_id="f_150")
    assert ingestion_res.z_rig_mm == 150.0

    # Etapa 2: Rectification & Facial Masking
    rectifier = StereoRectifier()
    rect_res = rectifier.rectify_pair(p_l, p_r, ingestion_result=ingestion_res, refine_with_features=True)
    assert rect_res.rect_left_2k.shape == (1536, 2048, 3)
    assert rect_res.rect_right_2k.shape == (1536, 2048, 3)

    mask_extractor = FacialMaskExtractor()
    mask_res = mask_extractor.process_rectified_pair(rect_res.rect_left_2k, rect_res.rect_right_2k)
    assert mask_res.face_detected_cam0 is True
    assert mask_res.mask_2k.shape == (1536, 2048)
    assert mask_res.d0_shift >= 0

    # Etapa 3: Dense Stereo Disparity
    matcher = DenseStereoMatcher(num_residual_disparities=256)
    disp_res = matcher.compute_disparity(
        rect_res.rect_left_2k,
        rect_res.rect_right_2k,
        d0_shift=mask_res.d0_shift,
        mask_2k=mask_res.mask_2k,
        pair_id="f_150"
    )

    assert disp_res.valid_pixels_count > 1000
    assert disp_res.disparity_2k.shape == (1536, 2048)

    # Validate output saving
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        rect_paths = rect_res.save_rectified_pair(tmp_dir)
        mask_paths = mask_res.save_mask(tmp_dir / "face_mask.png")
        p_raw = disp_res.save_disparity_16bit(tmp_dir / "disparity_16bit.png")
        p_col = disp_res.save_colored_visualization(tmp_dir / "disparity_turbo.jpg")

        assert rect_paths["rect_left_2k"].is_file()
        assert mask_paths["mask"].is_file()
        assert p_raw.is_file()
        assert p_col.is_file()


if __name__ == "__main__":
    print("Testing test_stereo_rectifier_nominal...")
    test_stereo_rectifier_nominal()
    print("  [PASS] test_stereo_rectifier_nominal")

    print("Testing test_epipolar_inspection_composite...")
    test_epipolar_inspection_composite()
    print("  [PASS] test_epipolar_inspection_composite")

    print("Testing test_facial_mask_fallback...")
    test_facial_mask_fallback()
    print("  [PASS] test_facial_mask_fallback")

    print("Testing test_dense_stereo_preshift...")
    test_dense_stereo_preshift()
    print("  [PASS] test_dense_stereo_preshift")

    print("Testing test_stage2_3_end_to_end_on_real_sample...")
    test_stage2_3_end_to_end_on_real_sample()
    print("  [PASS] test_stage2_3_end_to_end_on_real_sample")

    print("\n>>> ALL STAGE 2 & 3 TESTS PASSED SUCCESSFULLY! <<<")


from tests._functional import function_tests  # noqa: E402

load_tests = function_tests(globals())
