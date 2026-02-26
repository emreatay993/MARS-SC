"""Helpers for stable, stage-aware solve progress reporting."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple


@dataclass(frozen=True)
class StageSpec:
    """Description of one solve stage contributing to global progress."""

    key: str
    label: str
    weight: float = 1.0


class ProgressCoordinator:
    """
    Map stage-local progress updates to a single monotonic global timeline.

    The sink callback keeps the existing ``(current, total, message)`` shape.
    """

    def __init__(
        self,
        sink_callback: Callable[[int, int, str], None],
        stages: List[StageSpec],
        total_units: int = 1000,
    ) -> None:
        if not stages:
            raise ValueError("ProgressCoordinator requires at least one stage.")
        if total_units <= 0:
            raise ValueError("total_units must be positive.")

        total_weight = float(sum(stage.weight for stage in stages))
        if total_weight <= 0.0:
            raise ValueError("At least one stage must have positive weight.")

        self._sink_callback = sink_callback
        self._total_units = int(total_units)
        self._ranges: Dict[str, Tuple[float, float, str]] = {}
        self._last_units = 0

        cursor = 0.0
        for idx, stage in enumerate(stages):
            if stage.weight < 0.0:
                raise ValueError(f"Stage '{stage.key}' has negative weight.")
            stage_fraction = stage.weight / total_weight
            end = cursor + stage_fraction
            if idx == len(stages) - 1:
                end = 1.0
            self._ranges[stage.key] = (cursor, end, stage.label)
            cursor = end

    def stage_callback(self, stage_key: str) -> Callable[[int, int, str], None]:
        """Return a callback for one stage in the solve pipeline."""
        if stage_key not in self._ranges:
            raise KeyError(f"Unknown stage key: {stage_key}")

        def _callback(current: int, total: int, message: str) -> None:
            self.update(stage_key, current, total, message)

        return _callback

    def update(self, stage_key: str, current: int, total: int, message: str) -> None:
        """Emit one stage progress update through the global sink callback."""
        if stage_key not in self._ranges:
            raise KeyError(f"Unknown stage key: {stage_key}")

        start, end, label = self._ranges[stage_key]
        full_message = self._format_message(label, message)

        if total <= 0:
            self._sink_callback(self._last_units, 0, full_message)
            return

        ratio = float(current) / float(total)
        ratio = min(max(ratio, 0.0), 1.0)
        mapped_ratio = start + ((end - start) * ratio)
        mapped_units = int(round(mapped_ratio * self._total_units))
        mapped_units = min(self._total_units, max(self._last_units, mapped_units))
        self._last_units = mapped_units
        self._sink_callback(mapped_units, self._total_units, full_message)

    @staticmethod
    def _format_message(label: str, message: str) -> str:
        message = (message or "").strip()
        if not message:
            return label
        if message.startswith(f"{label}:"):
            return message
        return f"{label}: {message}"

