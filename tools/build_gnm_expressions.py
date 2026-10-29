#!/usr/bin/env python3
"""
build_gnm_expressions.py — Semantic expression vectors for the GLB morph targets (plan P2.7), built once.

GNM's ExpressionSampler decodes (latent 64 + one-hot class 20) -> 383 expression coefficients with a Keras MLP
(gnm/shape/data/semantic_sampler/expression_decoder_model.h5, Apache 2.0). The pipeline must not depend on
TensorFlow, so the decoder is evaluated here with NumPy (Dense + ReLU, weights read with h5py) at the latent mean
(z = 0): the prototype expression of each class. Output: assets/gnm_semantic_expressions.npz (names, (20, 383)).

  python tools/build_gnm_expressions.py path/to/expression_decoder_model.h5
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "gnm_semantic_expressions.npz"
CLASSES = ["SURPRISE", "DISGUST", "SUCK", "COMPRESS_FACE", "STRETCH_FACE", "HAPPY", "SQUINT", "PLATYSMA", "BLOW",
           "FUNNELER", "SMILE_WIDE", "CORNERS_DOWN", "PUCKER", "WINK_LEFT", "WINK_RIGHT", "MOUTH_LEFT",
           "MOUTH_RIGHT", "LIPS_ROLL_IN", "SNARL", "TONGUE_CENTER"]
LAYERS = ["dense_13", "dense_14", "dense_15", "dense_16", "dense_17"]


def load_dense(path: str):
    import h5py
    with h5py.File(path, "r") as f:
        mw = f["model_weights"]
        return [(np.asarray(mw[n][n]["kernel:0"]), np.asarray(mw[n][n]["bias:0"])) for n in LAYERS]


def decode(layers, z: np.ndarray, onehot: np.ndarray) -> np.ndarray:
    x = np.concatenate([z, onehot], axis=-1)              # Concatenate([latent_input, decoder_label_input])
    for i, (W, b) in enumerate(layers):
        x = x @ W + b
        if i < len(layers) - 1:
            x = np.maximum(x, 0.0)                         # relu; the last Dense is linear
    return x


def main(argv=None) -> int:
    path = (argv or sys.argv[1:])[0]
    layers = load_dense(path)
    onehot = np.eye(len(CLASSES))
    vectors = decode(layers, np.zeros((len(CLASSES), 64)), onehot)
    np.savez_compressed(OUT, names=np.array(CLASSES), expressions=vectors.astype(np.float32))
    for n, v in zip(CLASSES, vectors):
        print(f"{n:<14} |coef| rms {np.sqrt(np.mean(v ** 2)):.3f}")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
