"""Tests for the risk assessment engine (Part 7)."""

import pytest

from src.risk import ObjectRisk, RiskAssessment, RiskAssessor, RiskLevel
from src.spatial.spatial_analyzer import SpatialObject, SpatialResult
from src.tracking.tracker import MovementStatus


def make_spatial_object(track_id=1, depth=0.8, in_path=True, overlap=0.8,
                        movement=MovementStatus.APPROACHING, class_name="person",
                        position="CENTER"):
    from src.detection.detection_result import ImagePosition

    return SpatialObject(
        track_id=track_id, class_name=class_name,
        position=ImagePosition(position), in_walking_path=in_path,
        relative_depth=depth, depth_label="NEAR", overlap_ratio=overlap,
        movement=movement,
    )


def make_result(objects):
    return SpatialResult(
        frame_number=0, frame_width=640, frame_height=480,
        path_x1=192, path_x2=448, objects=objects,
        zone_states={"LEFT": "CLEAR", "CENTER": "BLOCKED", "RIGHT": "CLEAR"},
    )


@pytest.fixture
def assessor():
    return RiskAssessor()


class TestRiskLevels:
    def test_critical_case(self, assessor):
        obj = make_spatial_object()  # close + approaching + in path + person
        assessment = assessor.assess(make_result([obj]))
        assert assessment.highest >= RiskLevel.HIGH
        assert assessment.highest in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    def test_safe_when_far_and_outside_path(self, assessor):
        obj = make_spatial_object(depth=0.05, in_path=False, overlap=0.0,
                                  movement=MovementStatus.MOVING_AWAY,
                                  class_name="chair")
        assessment = assessor.assess(make_result([obj]))
        assert assessment.highest is RiskLevel.SAFE

    def test_ordering(self):
        assert RiskLevel.SAFE < RiskLevel.LOW < RiskLevel.MEDIUM
        assert RiskLevel.MEDIUM < RiskLevel.HIGH < RiskLevel.CRITICAL

    def test_empty_assessment_is_safe(self, assessor):
        assessment = assessor.assess(make_result([]))
        assert assessment.highest is RiskLevel.SAFE
        assert assessment.highest_object is None

    def test_uncertain_movement_is_conservative(self, assessor):
        static = make_spatial_object(movement=MovementStatus.STATIC)
        uncertain = make_spatial_object(track_id=2, movement=MovementStatus.UNCERTAIN)
        a_static = assessor.assess(make_result([static])).objects[0]
        a_uncertain = assessor.assess(make_result([uncertain])).objects[0]
        assert a_uncertain.score > a_static.score


class TestRiskFactors:
    def test_closer_is_riskier(self, assessor):
        near = assessor.assess(make_result([make_spatial_object(depth=0.8)])).objects[0]
        far = assessor.assess(make_result([make_spatial_object(depth=0.05)])).objects[0]
        assert near.score > far.score

    def test_path_overlap_increases_risk(self, assessor):
        low = make_spatial_object(overlap=0.1)
        high = make_spatial_object(track_id=2, overlap=0.9)
        a_low = assessor.assess(make_result([low])).objects[0]
        a_high = assessor.assess(make_result([high])).objects[0]
        assert a_high.score > a_low.score

    def test_hazard_class_boost(self, assessor):
        chair = make_spatial_object(class_name="chair")
        person = make_spatial_object(track_id=2, class_name="person")
        a_chair = assessor.assess(make_result([chair])).objects[0]
        a_person = assessor.assess(make_result([person])).objects[0]
        assert a_person.score > a_chair.score

    def test_reasons_attached(self, assessor):
        obj = make_spatial_object()
        risk_obj = assessor.assess(make_result([obj])).objects[0]
        assert risk_obj.reasons  # explanation must exist
        assert any("path" in r for r in risk_obj.reasons)

    def test_highest_object(self, assessor):
        low = make_spatial_object(track_id=1, depth=0.1, in_path=False, overlap=0.0,
                                  movement=MovementStatus.MOVING_AWAY)
        high = make_spatial_object(track_id=2, depth=0.8)
        assessment = assessor.assess(make_result([low, high]))
        assert assessment.highest_object.track_id == 2

    def test_thresholds_configurable(self):
        obj = make_spatial_object(depth=0.5)
        strict = RiskAssessor(critical_depth=0.10)   # 0.5 counts as "very close"
        lenient = RiskAssessor(critical_depth=0.99)  # 0.5 counts as far
        strict_score = strict.assess(make_result([obj])).objects[0].score
        lenient_score = lenient.assess(make_result([obj])).objects[0].score
        assert strict_score > lenient_score
