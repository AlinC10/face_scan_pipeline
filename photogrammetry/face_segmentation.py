"""
Face Region from Segmentation (MediaPipe SelfieMulticlass)
==========================================================
The MediaPipe face mesh stops 2-3 cm below the hairline and well before the ears, so
the reconstructed face was cut. The multiclass selfie segmenter labels every pixel as
background, hair, body skin, face skin, clothes or others (accessories); its face-skin
class reaches the hairline, the lateral cheeks and the ears, and the hair is the border.

  * face region: face skin (+ accessories inside it, i.e. spectacle frames), holes filled
    (eyes, mouth), the component under the nose tip, edges refined on the full-resolution
    photograph with a guided filter (the network runs at 256 x 256);
  * spectacle frames: the accessories class next to the face (used to inpaint the
    frames out of the photographs before texture baking, plan P1.9).

Model: assets/selfie_multiclass_256x256.tflite (Google, MediaPipe Image Segmenter, 16 MB).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

log = logging.getLogger("Photogrammetry.FaceSegmentation")

SEGMENTER_PATH = Path(__file__).resolve().parent.parent / "assets" / "selfie_multiclass_256x256.tflite"
BACKGROUND, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, OTHERS = range(6)


@dataclass
class FaceSegmentation:
    face: np.ndarray          # bool, full resolution: face skin + ears, hair excluded, holes filled
    frames: np.ndarray        # bool, full resolution: spectacle frames (accessories next to the face)
    hair: np.ndarray          # bool, full resolution
    face_fraction: float      # of the image


def _largest_component_at(mask: np.ndarray, seed: Optional[np.ndarray]) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    if seed is not None:
        x, y = int(np.clip(seed[0], 0, mask.shape[1] - 1)), int(np.clip(seed[1], 0, mask.shape[0] - 1))
        if labels[y, x] > 0:
            return labels == labels[y, x]
    return labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(out, contours, -1, 1, thickness=cv2.FILLED)
    return out.astype(bool)


def segment_faces(images_bgr: List[np.ndarray], nose_px: Optional[List[np.ndarray]] = None,
                  refine_scale: float = 0.5) -> List[Optional[FaceSegmentation]]:
    """Segments each photograph; `nose_px` (landmark 1 per image) selects the face component."""
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    if not SEGMENTER_PATH.is_file():
        log.warning("Segmentation model missing (%s): the face region comes from the MediaPipe mesh.", SEGMENTER_PATH)
        return [None] * len(images_bgr)
    opts = vision.ImageSegmenterOptions(base_options=mpt.BaseOptions(model_asset_path=str(SEGMENTER_PATH)),
                                        output_category_mask=True, output_confidence_masks=True)
    out: List[Optional[FaceSegmentation]] = []
    with vision.ImageSegmenter.create_from_options(opts) as seg:
        for i, img in enumerate(images_bgr):
            res = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            cat = res.category_mask.numpy_view()[..., 0]
            conf_face = res.confidence_masks[FACE_SKIN].numpy_view()[..., 0].astype(np.float32)
            conf_other = res.confidence_masks[OTHERS].numpy_view()[..., 0].astype(np.float32)
            h, w = cat.shape
            # Guided filter on the photograph: the 256 x 256 network output follows the real edges (hairline)
            small = cv2.resize(img, None, fx=refine_scale, fy=refine_scale, interpolation=cv2.INTER_AREA)
            p = cv2.resize(conf_face + conf_other, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
            try:
                refined = cv2.ximgproc.guidedFilter(small, p, radius=8, eps=1e-3)
            except AttributeError:           # opencv without contrib
                refined = p
            face = cv2.resize(refined, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
            seed = nose_px[i] if nose_px is not None else None
            face = _fill_holes(_largest_component_at(face, seed))
            near_face = cv2.dilate(face.astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool)
            frames = (cat == OTHERS) & near_face
            # The segmenter misses thin parts of the frames: add the thin dark structures (black-hat) that touch them
            frames = _complete_frames(img, frames, face)
            frames = cv2.dilate(frames.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))).astype(bool)
            out.append(FaceSegmentation(face=face, frames=frames, hair=cat == HAIR, face_fraction=float(face.mean())))
            log.info("Segmentation view %d: face region %.1f%% of the image, spectacle frames %.2f%%",
                     i, 100 * face.mean(), 100 * frames.mean())
    return out


def _complete_frames(img: np.ndarray, frames: np.ndarray, face: np.ndarray, scale: float = 0.25) -> np.ndarray:
    """
    Segmented frames + the black-hat components (long, thin, dark) that touch them, inside the face region.
    Eyebrows, lashes and the lip line are thin and dark too, but they do not touch the frames, so they stay.
    """
    if not frames.any():
        return frames
    g = cv2.cvtColor(cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    small = lambda m: cv2.resize(m.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    face_s = cv2.dilate(small(face).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    frames_s = small(frames)
    thin = (bh > 14) & face_s
    n, labels = cv2.connectedComponents((thin | frames_s).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(labels[frames_s])] = True
    keep[0] = False
    grown = keep[labels]
    return frames | (cv2.resize(grown.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0)


def inpaint_frames(image_bgr: np.ndarray, frames: np.ndarray, radius: int = 7) -> np.ndarray:
    """The photograph with the spectacle frames painted over from the skin around them (Telea), at full resolution."""
    if not frames.any():
        return image_bgr
    ys, xs = np.nonzero(frames)
    pad = 3 * radius
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, image_bgr.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, image_bgr.shape[1])
    out = image_bgr.copy()
    out[y0:y1, x0:x1] = cv2.inpaint(image_bgr[y0:y1, x0:x1], frames[y0:y1, x0:x1].astype(np.uint8), radius,
                                    cv2.INPAINT_TELEA)
    return out
