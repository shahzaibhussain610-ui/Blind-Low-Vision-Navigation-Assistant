"""Navigation-relevant class filtering for YOLO detections.

Class IDs are never hard-coded: the filter works with class *names* and
resolves them against the actual class mapping of the loaded model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Iterable, Optional, Set

from src.logging.logger import get_logger

logger = get_logger("class_filter")


#: Classes from the pretrained COCO model that are relevant for walking
#: navigation. Requests for classes outside the model's mapping are
#: reported and ignored (documented limitation; a custom-trained model
#: would be required for classes such as stairs).
NAVIGATION_CLASSES: FrozenSet[str] = frozenset(
    {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "traffic light",
        "stop sign",
        "bench",
        "chair",
        "couch",
        "bed",
        "dining table",
        "toilet",
        "backpack",
        "handbag",
        "suitcase",
        "dog",
        "cat",
        "potted plant",
        "tv",
        "refrigerator",
    }
)


@dataclass
class ClassFilter:
    """Configurable detection class filter.

    Modes:
        * ``all`` — keep every class the model outputs.
        * ``navigation`` — keep only classes in :data:`NAVIGATION_CLASSES`
          that exist in the model's actual class mapping.
        * ``custom`` — keep only the explicitly configured names that
          exist in the model's mapping.
    """

    mode: str = "navigation"
    custom_classes: Set[str] = field(default_factory=set)
    model_class_names: Optional[dict] = None  # {id: name} of loaded model

    def __post_init__(self) -> None:
        if self.mode not in ("all", "navigation", "custom"):
            logger.warning("Unknown class filter mode '%s'; using 'navigation'", self.mode)
            self.mode = "navigation"

    # ------------------------------------------------------------------ #
    def set_model_classes(self, class_names: dict) -> list:
        """Provide the loaded model's class mapping and resolve it.

        Args:
            class_names: Mapping of class id -> class name.

        Returns:
            Names requested by the current mode that are NOT available
            in the model (useful for UI warnings / documentation).
        """
        self.model_class_names = dict(class_names)
        available = set(class_names.values())
        missing = sorted(self.requested_names() - available)
        if missing:
            logger.warning(
                "Requested class(es) not available in the pretrained model: %s. "
                "A custom-trained detector would be required for these.",
                ", ".join(missing),
            )
        return missing

    def requested_names(self) -> Set[str]:
        """Class names the current mode ideally keeps."""
        if self.mode == "all":
            return set(self.model_class_names.values()) if self.model_class_names else set()
        if self.mode == "navigation":
            return set(NAVIGATION_CLASSES)
        return set(self.custom_classes)

    def allowed_ids(self) -> Optional[FrozenSet[int]]:
        """Class IDs to keep, or None to keep everything.

        Returns None only when the model classes are unknown or the mode
        is 'all'; otherwise returns the intersection of requested names
        with the model's actual mapping.
        """
        if self.model_class_names is None:
            return None  # filter not yet bound to a model: keep all
        if self.mode == "all":
            return None
        allowed = {
            class_id
            for class_id, name in self.model_class_names.items()
            if name in self.requested_names()
        }
        return frozenset(allowed)

    def allows(self, class_id: int) -> bool:
        """True when a class id passes the current filter."""
        allowed = self.allowed_ids()
        return True if allowed is None else class_id in allowed

    def filter_ids(self, class_ids: Iterable[int]) -> list:
        """Filter an iterable of class ids, preserving order."""
        allowed = self.allowed_ids()
        if allowed is None:
            return list(class_ids)
        return [cid for cid in class_ids if cid in allowed]
