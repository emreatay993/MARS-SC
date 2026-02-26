"""Tests for staged progress coordination."""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ui.handlers.progress_coordinator import ProgressCoordinator, StageSpec


def test_progress_coordinator_is_monotonic_across_stage_boundaries():
    events = []

    def _sink(current, total, message):
        events.append((current, total, message))

    coordinator = ProgressCoordinator(
        sink_callback=_sink,
        stages=[
            StageSpec(key="stress", label="Stress", weight=1.0),
            StageSpec(key="forces", label="Nodal Forces", weight=1.0),
            StageSpec(key="deformation", label="Deformation", weight=1.0),
        ],
    )

    stress_cb = coordinator.stage_callback("stress")
    forces_cb = coordinator.stage_callback("forces")
    deformation_cb = coordinator.stage_callback("deformation")

    stress_cb(0, 100, "start")
    stress_cb(100, 100, "done")
    forces_cb(0, 100, "start")
    forces_cb(50, 100, "mid")
    deformation_cb(100, 100, "done")

    percents = [int((current / total) * 100) for current, total, _ in events if total > 0]
    assert percents == sorted(percents)
    assert percents[-1] == 100
    assert events[0][2].startswith("Stress:")
    assert any(msg.startswith("Nodal Forces:") for _, _, msg in events)
    assert any(msg.startswith("Deformation:") for _, _, msg in events)


def test_progress_coordinator_preserves_last_progress_for_indeterminate_updates():
    events = []

    def _sink(current, total, message):
        events.append((current, total, message))

    coordinator = ProgressCoordinator(
        sink_callback=_sink,
        stages=[StageSpec(key="stress", label="Stress", weight=1.0)],
        total_units=1000,
    )

    stress_cb = coordinator.stage_callback("stress")
    stress_cb(50, 100, "halfway")
    stress_cb(0, 0, "waiting on external step")

    assert events[0][0] == 500
    assert events[0][1] == 1000
    assert events[1][0] == 500
    assert events[1][1] == 0
    assert events[1][2] == "Stress: waiting on external step"

