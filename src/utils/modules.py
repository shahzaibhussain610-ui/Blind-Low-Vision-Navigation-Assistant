"""Shared application constants: pipeline phases and module registry."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PipelineModule:
    """One stage of the eventual navigation pipeline."""

    label: str
    part: str
    implemented: bool


#: Ordered list of pipeline modules; only the foundation is implemented.
PIPELINE_MODULES: Tuple[PipelineModule, ...] = (
    PipelineModule("Project Foundation", "Part 1", True),
    PipelineModule("Video Input", "Part 2", True),
    PipelineModule("YOLO11 Object Detection", "Part 3", True),
    PipelineModule("Relative Depth Estimation", "Part 4", True),
    PipelineModule("Object Tracking & Movement", "Part 5", True),
    PipelineModule("Spatial & Walkable-Area Analysis", "Part 6", True),
    PipelineModule("Risk Assessment Engine", "Part 7", True),
    PipelineModule("Navigation Decision Engine", "Part 8", True),
    PipelineModule("Virtual Haptic Feedback", "Part 9", True),
    PipelineModule("Complete Dashboard, Testing & Evaluation", "Part 10", True),
)

#: Human-readable roadmap used in the sidebar.
SIDEBAR_ROADMAP: Tuple[str, ...] = (
    "Part 1 — Project Foundation",
    "Part 2 — Video Input",
    "Part 3 — YOLO11",
    "Part 4 — Depth",
    "Part 5 — Tracking",
    "Part 6 — Spatial Analysis",
    "Part 7 — Risk Assessment",
    "Part 8 — Navigation",
    "Part 9 — Virtual Haptic",
    "Part 10 — Testing & Evaluation",
)

CURRENT_PHASE: str = "Part 10 — Complete Dashboard, Testing & Evaluation"
