# ==============================================================================
# Dockerfile: 3D Facial Scanner & Multi-Pipeline 3D Reconstructor
# ==============================================================================
# Suport complet pentru:
#   1. Dense Stereo Photogrammetry (Etapele 1-7: Bouguet, MediaPipe Pre-shift d0,
#      CREStereo / SGBM WLS, Screened Poisson, Decimare QEM 20k, xatlas UV, 8 Livrabile)
#   2. Canonical 3D Stereo Reconstructor (reconstruct_3d.py cu urechi sudate)
# Compatibil cu:
#   - CPU (standard)
#   - GPU NVIDIA (cu flag-ul --gpus all)
# ==============================================================================

FROM python:3.11-slim-bookworm

# Prevenire prompt-uri interactive și configurare runtime Python
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    MPLCONFIGDIR=/tmp/matplotlib \
    TORCH_HOME=/tmp/torch \
    NUMBA_CACHE_DIR=/tmp/numba

# Instalare dependențe de sistem (OpenCV, Open3D, MediaPipe, OpenGL/Mesa)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libegl1 \
    libgles2 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalare pachete Python (separat pentru a profita de Docker layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copiere cod sursă și active (conform regulilor din .dockerignore)
COPY . /app/

# Creare directoare pentru mount-uri locale și asigurare permisiuni de scriere
RUN mkdir -p /app/input_images /app/output && \
    chmod 777 /app/output /app/input_images && \
    chmod +x /app/docker-entrypoint.sh

# Declarare volume pentru persistență date
VOLUME ["/app/input_images", "/app/output"]

# Punct de intrare inteligent
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Comandă implicită: procesează perechile stereo din input_images/rig și exportă în output/
CMD []
