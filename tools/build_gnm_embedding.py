#!/usr/bin/env python3
"""
build_gnm_embedding.py — MediaPipe (478 landmarks) -> GNM Head correspondence, built once (plan P2.2).

GNM is fitted (identity + expression, ICP) to the MediaPipe canonical face (assets/canonical_face_model.obj,
the mean face of the landmark model), started on the inner iBUG-68 landmarks (GNM HEAD_SPARSE_68 vs the usual
MediaPipe -> 68 mapping). Each of the 468 mesh landmarks then takes the closest point of the fitted GNM skin
(3 vertex ids + barycentric weights); the 10 iris landmarks take the eyeball centres' surface points of the
nearest skin. Written to assets/gnm_mediapipe_embedding.npz with the residual of every landmark.

  python tools/build_gnm_embedding.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from photogrammetry.gnm_fit import MP_TO_IBUG68, GNMFitter  # noqa: E402

OUT = ROOT / "assets" / "gnm_mediapipe_embedding.npz"


def main() -> int:
    fitter = GNMFitter()
    canon = np.asarray(trimesh.load(ROOT / "assets" / "canonical_face_model.obj", process=False).vertices)[:468] * 10.0
    res = fitter.fit(canon, landmarks=fitter.landmarks68_correspondence(canon[MP_TO_IBUG68]), iterations=12,
                     lambda_identity=1.0, lambda_expression=3.0)
    dist, vid, bary = fitter.surface_correspondence(res.vertices, canon)
    # residual of the barycentric points (closest dense sample) to the canonical landmarks
    resid = np.linalg.norm(np.einsum("lk,lkc->lc", bary, res.vertices[vid]) - canon, axis=1)
    np.savez_compressed(OUT, vid=vid.astype(np.int32), bary=bary.astype(np.float32), residual_mm=resid.astype(np.float32))
    report = {"fit": res.stats, "embedding_residual_median_mm": round(float(np.median(resid)), 3),
              "embedding_residual_p95_mm": round(float(np.percentile(resid, 95)), 3),
              "embedding_residual_max_mm": round(float(resid.max()), 3)}
    print(json.dumps(report, indent=2))
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
