"""Typed, allowlisted defensive action executors."""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from wso.brain.features import MAX_SINGULAR_VALUES
from wso.brain.types import ActionReceipt, BrainAction
from wso.telemetry.schemas import TransientLoRATelemetry


class BrainActionExecutor(Protocol):
    name: str

    def supports(self, action: BrainAction) -> bool: ...

    def execute(
        self,
        action: BrainAction,
        record: TransientLoRATelemetry,
    ) -> ActionReceipt: ...


class LocalDefensiveExecutor:
    """Default read-only executor.

    It can observe, emit review events, and run a registered derived spectral scan.
    It intentionally cannot quarantine or mutate an external runtime.
    """

    name = "local_read_only_v1"
    _supported: frozenset[BrainAction] = frozenset(
        {"allow", "observe", "deep_scan", "human_review"}
    )

    def supports(self, action: BrainAction) -> bool:
        return action in self._supported

    def execute(
        self,
        action: BrainAction,
        record: TransientLoRATelemetry,
    ) -> ActionReceipt:
        if not self.supports(action):
            return ActionReceipt(
                requested_action=action,
                executed_action="human_review",
                executor=self.name,
                success=False,
                summary=f"{action} requires an explicitly registered production executor.",
            )
        if action == "deep_scan":
            evidence = _registered_spectral_scan(record)
            return ActionReceipt(
                requested_action=action,
                executed_action=action,
                executor=self.name,
                success=True,
                summary="Completed registered read-only spectral concentration scan.",
                evidence=evidence,
            )
        summary = {
            "allow": "Recorded an allow decision; no runtime mutation was requested.",
            "observe": "Recorded the event for continued observation.",
            "human_review": "Emitted a human-review escalation receipt.",
        }[action]
        return ActionReceipt(
            requested_action=action,
            executed_action=action,
            executor=self.name,
            success=True,
            summary=summary,
        )


def _registered_spectral_scan(
    record: TransientLoRATelemetry,
) -> dict[str, float | int | str | bool | None]:
    if len(record.singular_values) > MAX_SINGULAR_VALUES:
        raise ValueError(f"spectral scan accepts at most {MAX_SINGULAR_VALUES} values")
    singular = np.asarray(record.singular_values, dtype=np.float64)
    singular = np.abs(singular[np.isfinite(singular)])
    singular.sort()
    singular = singular[::-1]
    maximum = float(singular[0]) if singular.size else 0.0
    scaled = singular / maximum if maximum > 0 else np.zeros_like(singular)
    energy = np.square(scaled)
    total = float(energy.sum())
    if total > 0:
        probabilities = energy / total
        positive = probabilities[probabilities > 0]
        entropy = -float(np.sum(positive * np.log(positive)))
        normalized_entropy = entropy / math.log(len(positive)) if len(positive) > 1 else 0.0
        concentration = float(probabilities[0])
    else:
        normalized_entropy = 0.0
        concentration = 0.0
    return {
        "scan": "registered_spectral_concentration_v1",
        "singular_value_count": int(singular.size),
        "leading_energy_concentration": concentration,
        "normalized_spectral_entropy": normalized_entropy,
        "raw_tensor_persisted": False,
        "prompt_payload_used": False,
    }
