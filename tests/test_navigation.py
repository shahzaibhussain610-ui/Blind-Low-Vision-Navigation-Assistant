"""Tests for the navigation decision engine (Part 8)."""

import pytest

from src.navigation import NavigationCommand, NavigationDecision, NavigationEngine
from src.risk import RiskAssessor, RiskLevel
from src.spatial.spatial_analyzer import SpatialAnalyzer, SpatialResult
from src.tracking.tracker import MovementStatus
from src.detection.detection_result import Detection
from src.tracking.tracker import TrackedObject


def make_tracked(track_id, x1, x2, depth=0.8, movement=MovementStatus.APPROACHING,
                 class_name="person"):
    detection = Detection(
        class_id=0, class_name=class_name, confidence=0.9,
        x1=x1, y1=100, x2=x2, y2=400, frame_number=0, timestamp=0.0,
        frame_width=640, frame_height=480, track_id=track_id,
    )
    return TrackedObject(detection=detection, relative_depth=depth,
                         depth_label="NEAR", movement=movement)


@pytest.fixture
def engine():
    return NavigationEngine(min_command_interval=0, hysteresis_frames=1)


@pytest.fixture
def analyzer():
    return SpatialAnalyzer(path_width_ratio=0.40, path_min_depth=0.30)


def run_pipeline_frame(engine, analyzer, tracked_objects, frame_number):
    spatial = analyzer.analyze(640, 480, tracked_objects, frame_number)
    risk = RiskAssessor().assess(spatial, frame_number)
    return engine.decide(spatial, risk, frame_number)


class TestCommands:
    def test_proceed_when_clear(self, engine, analyzer):
        decision = run_pipeline_frame(engine, analyzer, [], 0)
        assert decision.command is NavigationCommand.PROCEED
        assert "clear" in decision.reason

    def test_stop_on_critical(self, engine, analyzer):
        # Very close + approaching + fully blocking path => critical.
        decision = run_pipeline_frame(
            engine, analyzer, [make_tracked(1, 150, 490, depth=0.95)], 0
        )
        assert decision.command is NavigationCommand.STOP

    def test_stop_on_uncertain_in_path(self, engine, analyzer):
        decision = run_pipeline_frame(
            engine, analyzer,
            [make_tracked(1, 250, 400, movement=MovementStatus.UNCERTAIN)], 0
        )
        assert decision.command is NavigationCommand.STOP

    def test_move_left_when_right_blocked(self, engine, analyzer):
        tracked = make_tracked(1, 380, 640, depth=0.5,
                               movement=MovementStatus.STATIC)
        decision = run_pipeline_frame(engine, analyzer, [tracked], 0)
        assert decision.command is NavigationCommand.MOVE_LEFT

    def test_move_right_when_left_blocked(self, engine, analyzer):
        tracked = make_tracked(1, 0, 260, depth=0.5, movement=MovementStatus.STATIC)
        decision = run_pipeline_frame(engine, analyzer, [tracked], 0)
        assert decision.command is NavigationCommand.MOVE_RIGHT

    def test_stop_when_both_sides_blocked(self, engine, analyzer):
        objects = [
            make_tracked(1, 0, 260, depth=0.5, movement=MovementStatus.STATIC),
            make_tracked(2, 380, 640, depth=0.5, movement=MovementStatus.STATIC),
        ]
        decision = run_pipeline_frame(engine, analyzer, objects, 0)
        assert decision.command is NavigationCommand.STOP


class TestHysteresis:
    def test_direction_changes_debounced(self):
        engine = NavigationEngine(min_command_interval=0, hysteresis_frames=3)
        analyzer = SpatialAnalyzer()
        objects = [make_tracked(1, 380, 640, depth=0.5,
                                movement=MovementStatus.STATIC)]
        # While the candidate command persists, it is applied after
        # hysteresis_frames and then remains stable.
        commands = []
        for n in range(5):
            commands.append(run_pipeline_frame(engine, analyzer, objects, n).command)
        assert commands[-1] is NavigationCommand.MOVE_LEFT
        assert commands[-1] == commands[-2]  # stable once adopted

    def test_stop_overrides_everything(self, engine, analyzer):
        run_pipeline_frame(engine, analyzer, [], 0)  # PROCEED first
        decision = run_pipeline_frame(
            engine, analyzer, [make_tracked(1, 150, 490, depth=0.95)], 1
        )
        assert decision.command is NavigationCommand.STOP

    def test_reset(self, engine, analyzer):
        run_pipeline_frame(engine, analyzer, [], 0)
        engine.reset()
        assert engine.current is NavigationCommand.PROCEED


class TestDecisionStructure:
    def test_decision_fields(self, engine, analyzer):
        decision = run_pipeline_frame(engine, analyzer, [], 0)
        assert isinstance(decision, NavigationDecision)
        assert decision.frame_number == 0
        assert isinstance(decision.reason, str) and decision.reason
        assert "command" in decision.to_dict()

    def test_all_commands_representable(self):
        for name in ("PROCEED", "MOVE_LEFT", "MOVE_RIGHT", "SLOW_DOWN",
                     "CAUTION", "STOP"):
            assert NavigationCommand(name)
