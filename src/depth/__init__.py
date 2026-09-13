"""Depth package (Part 4): relative monocular depth estimation.

Public API::

    from src.depth import RelativeDepthEstimator, ObjectDepth

Depth is RELATIVE (0-1, higher = closer), never a metric distance.
"""

from src.depth.depth_estimator import DepthError, ObjectDepth, RelativeDepthEstimator

__all__ = ["DepthError", "ObjectDepth", "RelativeDepthEstimator"]
