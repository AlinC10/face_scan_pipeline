#!/usr/bin/env bash
# docker_scan.sh — One face scan with the Docker image on this machine; photos and results on the local disk.
# Nothing goes through Cloudflare R2: without JOB_ID the container runs the pipeline on the mounted folders.
#
#   tools/docker_scan.sh alin                                    # input_images/ -> teste/alin/
#   tools/docker_scan.sh alin ~/scans/alin                       # another input folder (rig/ + optional calib/)
#   IMAGE=alinc10/face-scan:v1 tools/docker_scan.sh alin "" --no-spectacles --texture-size 2048
#
# Results: <repo>/teste/<name>/ (OBJ, GLB, STL, 3MF, object.json, stats.json, photogrammetry_manifest.json).
# Docker Desktop: the default 4096 px atlas needs ~10 GB of RAM in the Docker VM (Settings -> Resources);
# with less, the RAM check stops in a second -- use --texture-size 2048 (~4.4 GB on a half-size pair).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:?usage: tools/docker_scan.sh <name> [input_dir] [pipeline options...]}"
INPUT_DIR="${2:-$ROOT/input_images}"
[ -z "$INPUT_DIR" ] && INPUT_DIR="$ROOT/input_images"
shift $(( $# >= 2 ? 2 : 1 ))
IMAGE="${IMAGE:-face-scan:local}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/teste/$NAME}"
INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"
[ -d "$INPUT_DIR/rig/0" ] && [ -d "$INPUT_DIR/rig/1" ] || { echo "No rig/0 and rig/1 in $INPUT_DIR" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"
echo "Scan $NAME: $INPUT_DIR -> $OUTPUT_DIR ($IMAGE) $*"
t0=$(date +%s)
set +e
docker run --rm \
    -v "$INPUT_DIR:/app/input_images:ro" \
    -v "$OUTPUT_DIR:/app/output" \
    "$IMAGE" --rig-dir /app/input_images/rig -o /app/output --name model "$@"
code=$?
set -e
echo "Done in $(( $(date +%s) - t0 )) s, exit code $code. Results: $OUTPUT_DIR"
exit $code
