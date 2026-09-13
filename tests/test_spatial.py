"""Tests for spatial & walkable-area analysis (Part 6)."""

import pytest

from src.detection.detection_result import Detection, ImagePosition
from src.spatial import SpatialAnalyzer, SpatialResult, ZoneState
from src.tracking.tracker import MovementStatus, TrackedObject


def make_tracked(track_id, x1, x2, y1=100, y2=400, depth=0.8,
                 movement=MovementStatus.STATIC, class_name="person"):
    detection = Detection(
        class_id=0, class_name=class_name, confidence=0.9,
        x1=x1, y1=y1, x2=x2, y2=y2, frame_number=0, timestamp=0.0,
        frame_width=640, frame_height=480, track_id=track_id,
    )
    return TrackedObject(
        detection=detection, relative_depth=depth, depth_label="NEAR",
        movement=movement,
    )


@pytest.fixture
def analyzer():
    return SpatialAnalyzer(path_width_ratio=0.40, path_min_depth=0.30)


class TestWalkingPath:
    def test_corridor_geometry(self, analyzer):
        result = analyzer.analyze(640, 480, [], frame_number=0)
        # 40% of 640 = 256 px wide corridor centered at 320.
        assert result.path_x1 == 192
        assert result.path_x2 == 448

    def test_object_in_path(self, analyzer):
        result = analyzer.analyze(640, 480, [make_tracked(1, 250, 400)])
        assert len(result.path_obstacles) == 1
        assert result.path_obstacles[0].in_walking_path is True

    def test_distant_object_not_in_path(self, analyzer):
        result = analyzer.analyze(640, 480, [make_tracked(1, 250, 400, depth=0.10)])
        assert result.path_obstacles == []

    def test_outside_path(self, analyzer):
        result = analyzer.analyze(640, 480, [make_tracked(1, 0, 100, depth=0.8)])
        assert result.path_obstacles == []

    def test_overlap_ratio(self, analyzer):
        tracked = make_tracked(1, 250, 450)  # overlaps 198 px of 256 corridor
        result = analyzer.analyze(640, 480, [tracked])
        obj = result.path_obstacles[0]
        assert obj.overlap_ratio == pytest.approx(198 / 256, abs=0.02)


class TestZones:
    def test_zones_clear_when_empty(self, analyzer):
        result = analyzer.analyze(640, 480, [])
        assert result.zone_states == {"LEFT": "CLEAR", "CENTER": "CLEAR",
                                      "RIGHT": "CLEAR"}
        assert result.clear_zones == ["LEFT", "CENTER", "RIGHT"]

    def test_center_blocked_by_large_object(self, analyzer):
        result = analyzer.analyze(640, 480, [make_tracked(1, 200, 440)])
        assert result.zone_states["CENTER"] == ZoneState.BLOCKED.value
        assert result.path_blocked is True

    def test_side_object_only_blocks_its_zone(self, analyzer):
        result = analyzer.analyze(640, 480, [make_tracked(1, 0, 150)])
        assert result.zone_states["LEFT"] == ZoneState.PARTIALLY_BLOCKED.value
        assert result.zone_states["CENTER"] == ZoneState.CLEAR.value
        assert result.zone_states["RIGHT"] == ZoneState.CLEAR.value

    def test_approaching_object_blocks_its_zone(self, analyzer):
        tracked = make_tracked(1, 0, 150, movement=MovementStatus.APPROACHING)
        result = analyzer.analyze(640, 480, [tracked])
        assert result.zone_states["LEFT"] == ZoneState.BLOCKED.value


class TestSpatialObject:
    def test_position_from_bbox(self, analyzer):
        tracked = make_tracked(1, 250, 400)
        result = analyzer.analyze(640, 480, [tracked])
        assert result.objects[0].position is ImagePosition.CENTER

    def test_to_dict_roundtrip(self, analyzer):
        result = analyzer.analyze(640, 480, [make_tracked(1, 250, 400)])
        data = result.to_dict()
        assert data["path_blocked"] is True
        assert data["objects"][0]["in_walking_path"] is True
        assert set(data["zones"].keys()) == {"LEFT", "CENTER", "RIGHT"}
