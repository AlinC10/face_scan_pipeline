#!/bin/bash
set -e

# ==============================================================================
# Docker Entrypoint for 3D Facial Scanner & Photogrammetry Pipeline
# ==============================================================================

# RunPod pod (JOB_ID set by cloud/job.py): download the job from R2, run, upload the result
if [ -n "$JOB_ID" ]; then
    exec python3 runpod_job.py
fi

# Ensure output directory exists and is writable
mkdir -p /app/output 2>/dev/null || true

# Auto-detect input rig directory layout
if [ -d "/app/input_images/rig/0" ] && [ -d "/app/input_images/rig/1" ]; then
    RIG_DIR="/app/input_images/rig"
elif [ -d "/app/input_images/0" ] && [ -d "/app/input_images/1" ]; then
    RIG_DIR="/app/input_images"
else
    RIG_DIR="/app/input_images/rig"
fi

OUT_DIR="/app/output"

cleanup_permissions() {
    # If running as root, make generated output accessible to host users
    if [ "$(id -u)" = "0" ] && [ -d "/app/output" ]; then
        chmod -R a+rw /app/output 2>/dev/null || true
    fi
}
trap cleanup_permissions EXIT

# Route commands based on first argument
case "$1" in
    photogrammetry)
        shift
        if [ $# -eq 0 ]; then
            echo "[Docker Entrypoint] Running Dense Stereo Photogrammetry Pipeline (Stages 1-7)..."
            echo "  • Rig Input: $RIG_DIR"
            echo "  • Output Dir: $OUT_DIR"
            exec python3 run_photogrammetry_runpod.py --rig-dir "$RIG_DIR" -o "$OUT_DIR"
        else
            exec python3 run_photogrammetry_runpod.py "$@"
        fi
        ;;

    reconstruct)
        shift
        if [ $# -eq 0 ]; then
            echo "[Docker Entrypoint] Running Canonical Stereo 3D Reconstructor..."
            echo "  • Rig Input: $RIG_DIR"
            echo "  • Output Dir: $OUT_DIR"
            exec python3 reconstruct_3d.py -i "$RIG_DIR" -o "$OUT_DIR"
        else
            exec python3 reconstruct_3d.py "$@"
        fi
        ;;

    bash|sh|python|python3|pytest)
        exec "$@"
        ;;

    -*)
        # User passed CLI flags directly (e.g. --rig-dir, -l, -r, -i, -o)
        exec python3 run_photogrammetry_runpod.py "$@"
        ;;

    "")
        # Default: Dense Stereo Photogrammetry Pipeline on mounted volumes
        echo "========================================================================"
        echo " 3D Facial Photogrammetry & Surface Reconstruction Pipeline"
        echo "========================================================================"
        echo "  • Ingesting stereo pair from: $RIG_DIR"
        echo "  • Exporting deliverables to:  $OUT_DIR"
        echo "========================================================================"
        exec python3 run_photogrammetry_runpod.py --rig-dir "$RIG_DIR" -o "$OUT_DIR"
        ;;

    *)
        exec "$@"
        ;;
esac
