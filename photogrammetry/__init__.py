"""
Dense Stereo Photogrammetry Package
===================================
Pipeline for High-Fidelity 3D Facial Reconstruction from Dual-Camera Stereo Rigs.
"""

from .apriltag_tracker import AprilTagTracker, AprilTagPoseResult
from .ingestion import StereoRigIngestor, StereoIngestionResult, load_stereo_rig_pair
from .rectification import StereoRectifier, StereoRectificationResult
from .face_mask import FacialMaskExtractor, FacialMaskResult
from .dense_stereo import DenseStereoMatcher, DenseDisparityResult
from .triangulation_poisson import MetricPoissonReconstructor, PoissonReconstructionResult
from .decimation_uv import MeshDecimatorUV, DecimationUVResult
from .texture_baker import TextureBaker, TextureBakingResult
from .export_manager import StandardDeliverablesExporter, ExportResult

__all__ = [
    # Etapa 1: Ingestion & Metric Calibration
    "AprilTagTracker",
    "AprilTagPoseResult",
    "StereoRigIngestor",
    "StereoIngestionResult",
    "load_stereo_rig_pair",
    # Etapa 2: Rectification & Facial Masking
    "StereoRectifier",
    "StereoRectificationResult",
    "FacialMaskExtractor",
    "FacialMaskResult",
    # Etapa 3: Dense Disparity
    "DenseStereoMatcher",
    "DenseDisparityResult",
    # Etapa 4: Triangulation & Screened Poisson
    "MetricPoissonReconstructor",
    "PoissonReconstructionResult",
    # Etapa 5: Decimation & UV Unwrap
    "MeshDecimatorUV",
    "DecimationUVResult",
    # Etapa 6: Texture & Normal Map Baking
    "TextureBaker",
    "TextureBakingResult",
    # Etapa 7: Standard Deliverables Export
    "StandardDeliverablesExporter",
    "ExportResult",
]
