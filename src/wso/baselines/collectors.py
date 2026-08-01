"""Baseline collection scaffolding."""

from __future__ import annotations

from wso.telemetry.schemas import TransientLoRATelemetry


class TelemetryCollector:
    """In-memory collector for prototype task-conditioned telemetry baselines."""

    def __init__(self) -> None:
        self.records: list[TransientLoRATelemetry] = []

    def add(self, record: TransientLoRATelemetry) -> None:
        self.records.append(record)

    def by_task_family(self, task_family: str) -> list[TransientLoRATelemetry]:
        return [r for r in self.records if r.task_family == task_family]
