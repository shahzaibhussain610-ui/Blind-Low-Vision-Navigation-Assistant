"""Tests for the navigation-relevant class filter."""

import pytest

from src.detection import NAVIGATION_CLASSES, ClassFilter


#: Minimal COCO-like mapping matching the real pretrained model.
COCO_MAPPING = {0: "person", 1: "bicycle", 2: "car", 5: "bus",
                7: "truck", 9: "traffic light", 56: "chair", 58: "dog"}


class TestModes:
    def test_default_mode_is_navigation(self):
        assert ClassFilter().mode == "navigation"

    def test_invalid_mode_falls_back(self):
        assert ClassFilter(mode="bogus").mode == "navigation"

    def test_all_mode_returns_none(self):
        f = ClassFilter(mode="all")
        f.set_model_classes(COCO_MAPPING)
        assert f.allowed_ids() is None
        assert f.allows(3) is True  # any class passes

    def test_unknown_model_keeps_all(self):
        f = ClassFilter(mode="navigation")
        assert f.allowed_ids() is None  # not yet bound to a model


class TestNavigationFiltering:
    def test_filters_to_model_intersection(self):
        f = ClassFilter(mode="navigation")
        f.set_model_classes(COCO_MAPPING)
        allowed = f.allowed_ids()
        assert allowed is not None
        # person(0), bicycle(1), car(2), bus(5), truck(7), traffic light(9) in
        assert 0 in allowed and 9 in allowed
        # couch/bed etc. not in this toy mapping
        assert all(cid in COCO_MAPPING for cid in allowed)

    def test_allows_method(self):
        f = ClassFilter(mode="navigation")
        f.set_model_classes(COCO_MAPPING)
        assert f.allows(0) is True        # person
        assert f.allows(58) is True       # dog
        assert f.allows(62) is False      # 'chair-like' class not relevant? (id not in mapping)

    def test_filter_ids_preserves_order(self):
        f = ClassFilter(mode="navigation")
        f.set_model_classes(COCO_MAPPING)
        ids = f.filter_ids([2, 999, 0, 5])
        assert ids == [2, 0, 5]

    def test_no_fabricated_classes(self):
        """Filter must never create IDs absent from the model mapping."""
        f = ClassFilter(mode="navigation")
        f.set_model_classes(COCO_MAPPING)
        for cid in f.allowed_ids():
            assert cid in COCO_MAPPING


class TestCustomFiltering:
    def test_custom_mode(self):
        f = ClassFilter(mode="custom", custom_classes={"person", "dog"})
        missing = f.set_model_classes(COCO_MAPPING)
        allowed = f.allowed_ids()
        assert allowed == frozenset({0, 58})
        assert missing == []

    def test_missing_class_reported(self):
        """Requested classes absent from the model must be reported, not faked."""
        f = ClassFilter(mode="custom", custom_classes={"person", "stairs"})
        missing = f.set_model_classes(COCO_MAPPING)
        assert "stairs" in missing
        assert "person" not in missing

    def test_navigation_mode_missing_report(self):
        f = ClassFilter(mode="navigation")
        missing = f.set_model_classes({0: "person"})  # tiny model
        assert "car" in missing  # requested but unavailable

    def test_navigation_subset_of_coco(self):
        f = ClassFilter(mode="navigation")
        f.set_model_classes(COCO_MAPPING)
        for cid in f.allowed_ids():
            assert COCO_MAPPING[cid] in NAVIGATION_CLASSES
