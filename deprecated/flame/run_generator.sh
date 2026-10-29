#!/usr/bin/env bash
# Faza 3 (FLAME, pe pauză). Rulează din rădăcina repo-ului: assets/face_landmarker.task și
# assets/canonical_face_model.obj vin de acolo, iar șabloanele FLAME din deprecated/flame/assets/.
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$ROOT/.venv/bin/python" "$DIR/generator_engine.py" "$@"
