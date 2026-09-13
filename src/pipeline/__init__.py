"""Pipeline package (Part 10): orchestrator + configuration factory."""

from src.pipeline.factory import build_detector, build_pipeline
from src.pipeline.pipeline import (
    NavigationPipeline,
    PipelineFrameResult,
)

__all__ = [
    "NavigationPipeline",
    "PipelineFrameResult",
    "build_detector",
    "build_pipeline",
]
