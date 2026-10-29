# ==============================================================================
# Dockerfile: 3D Facial Scanner (dense two-view face reconstruction, CPU only)
# ==============================================================================
# Two stages, as Turntable Studio:
#   builder : Python 3.12 + compilers, installs the Python packages into /opt/venv
#   runtime : Python 3.12 slim + only the shared libraries OpenCV / MediaPipe / Open3D load,
#             the virtual environment and the code (no compilers, no ffmpeg)
# Python 3.12: GNM Head (plan P2) requires >= 3.12; Open3D 0.20 needs glibc >= 2.35 (bookworm: 2.36).
#
#   docker build -t face-scan:local .
#   docker run --rm -v $(pwd)/input_images:/app/input_images -v $(pwd)/output:/app/output face-scan:local
# With JOB_ID set (RunPod pod started by cloud/job.py) the container runs runpod_job.py.
# ==============================================================================

FROM python:3.12-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements-runtime.txt requirements-open3d.txt /tmp/
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /tmp/requirements-runtime.txt \
    && pip install --no-deps -r /tmp/requirements-open3d.txt \
    && find /opt/venv -name "__pycache__" -type d -prune -exec rm -rf {} + \
    && find /opt/venv -name "*.pyc" -delete


FROM python:3.12-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PATH=/opt/venv/bin:$PATH \
    MPLCONFIGDIR=/tmp/matplotlib

# Shared libraries loaded by OpenCV (GL, glib), MediaPipe (EGL / GLES) and Open3D (OpenMP, libusb)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 libegl1 libgles2 libsm6 libxext6 libxrender1 libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY . /app/

# Unix line endings and the execute bit for the entrypoint (Windows checkouts)
RUN sed -i 's/\r$//' /app/docker-entrypoint.sh \
    && chmod 755 /app/docker-entrypoint.sh \
    && mkdir -p /app/input_images /app/output \
    && chmod 777 /app/output /app/input_images

# GNM Head v3 (53 MB, gitignored): reused from the build context when present, otherwise downloaded; SHA-256 checked
RUN /opt/venv/bin/python -c "from photogrammetry.gnm_head import ensure_model; ensure_model()"

VOLUME ["/app/input_images", "/app/output"]
ENTRYPOINT ["/bin/bash", "/app/docker-entrypoint.sh"]
# Default: the pair in input_images/rig (+ calibration from input_images/calib) -> output/
CMD []
