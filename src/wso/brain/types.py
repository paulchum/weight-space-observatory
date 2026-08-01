"""Schemas for the autonomous weight-space trust gateway.

The learned components are intentionally separated from the hard safety envelope:
models may estimate risk and utility, but they cannot expand the set of actions the
operator has authorized.
"""

from __future__ import annotations

import math
import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BrainAction = Literal["allow", "observe", "deep_scan", "human_review", "quarantine"]
BrainMode = Literal["shadow", "enforce"]

ACTION_SEVERITY: dict[str, int] = {
    "allow": 0,
    "observe": 1,
    "deep_scan": 2,
    "human_review": 3,
    "quarantine": 4,
}

MAX_SELF_PLAY_SEEDS_PER_CLASS = 256
SELF_PLAY_TRAIN_FRACTION = 0.6


class BrainConfig(BaseModel):
    """Operator-owned policy and learning configuration.

    Thresholds are safety constraints, not trainable parameters. The default ceiling
    permits observation and a registered read-only deep scan, while quarantine remains
    a recommendation requiring review.
    """

    model_config = ConfigDict(frozen=True)

    schema_version: str = "2.0"
    feature_dimension: int = Field(default=64, ge=32)
    numeric_feature_count: int = Field(default=30, ge=1)
    min_supervised_examples: int = Field(default=20, ge=2)
    min_positive_examples: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=0.08, gt=0.0)
    l2: float = Field(default=0.002, ge=0.0)
    classifier_epochs: int = Field(default=24, ge=1)
    target_false_positive_rate: float = 0.05
    review_risk_threshold: float = 0.55
    high_risk_threshold: float = 0.82
    critical_risk_threshold: float = 0.94
    max_uncertainty_for_quarantine: float = 0.18
    max_autonomous_action: BrainAction = "deep_scan"
    memory_limit: int = Field(default=512, ge=1)
    pending_decision_limit: int = Field(default=4096, ge=1)
    calibration_limit: int = Field(default=4096, ge=1)
    bandit_exploration: float = 0.18
    promotion_confidence: float = 0.95
    max_self_play_evasion_rate: float = 0.20
    random_seed: int = Field(default=20260801, ge=0)

    @field_validator(
        "learning_rate",
        "l2",
        "target_false_positive_rate",
        "review_risk_threshold",
        "high_risk_threshold",
        "critical_risk_threshold",
        "max_uncertainty_for_quarantine",
        "bandit_exploration",
        "promotion_confidence",
        "max_self_play_evasion_rate",
    )
    @classmethod
    def finite_probability_or_rate(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("brain rates and thresholds must be finite and non-negative")
        return value

    @model_validator(mode="after")
    def valid_threshold_order(self) -> BrainConfig:
        if self.feature_dimension < 32:
            raise ValueError("feature_dimension must be at least 32")
        if not 0 < self.numeric_feature_count < self.feature_dimension:
            raise ValueError("numeric_feature_count must be inside the feature vector")
        if not (
            0
            < self.review_risk_threshold
            < self.high_risk_threshold
            < self.critical_risk_threshold
            <= 1
        ):
            raise ValueError("risk thresholds must be strictly ordered inside (0, 1]")
        if self.high_risk_threshold - self.review_risk_threshold < 0.05:
            raise ValueError(
                "high_risk_threshold must be at least 0.05 above review_risk_threshold"
            )
        if not 0 < self.target_false_positive_rate < 0.5:
            raise ValueError("target_false_positive_rate must be inside (0, 0.5)")
        if not 0 <= self.max_uncertainty_for_quarantine <= 1:
            raise ValueError("max_uncertainty_for_quarantine must be inside [0, 1]")
        if not 0.5 < self.promotion_confidence < 1:
            raise ValueError("promotion_confidence must be inside (0.5, 1)")
        if not 0 < self.max_self_play_evasion_rate < 0.5:
            raise ValueError("max_self_play_evasion_rate must be inside (0, 0.5)")
        if not _zero_event_requirement_fits(
            self.target_false_positive_rate,
            self.promotion_confidence,
            MAX_SELF_PLAY_SEEDS_PER_CLASS,
        ):
            raise ValueError(
                "target_false_positive_rate and promotion_confidence require more "
                f"than {MAX_SELF_PLAY_SEEDS_PER_CLASS} benign self-play records"
            )
        maximum_heldout_incidents = MAX_SELF_PLAY_SEEDS_PER_CLASS - round(
            MAX_SELF_PLAY_SEEDS_PER_CLASS * SELF_PLAY_TRAIN_FRACTION
        )
        if not _zero_event_requirement_fits(
            self.max_self_play_evasion_rate,
            self.promotion_confidence,
            maximum_heldout_incidents,
        ):
            raise ValueError(
                "max_self_play_evasion_rate and promotion_confidence require more "
                f"than {maximum_heldout_incidents} held-out self-play incident seeds"
            )
        return self


def _zero_event_requirement_fits(limit: float, confidence: float, capacity: int) -> bool:
    """Compare the zero-event requirement without an overflow-prone division."""

    return math.log1p(-confidence) >= capacity * math.log1p(-limit)


class RetrievedMemory(BaseModel):
    memory_id: str
    title: str
    similarity: float
    incident_rate: float
    preferred_action: BrainAction
    lesson: str


class BrainDecision(BaseModel):
    """Auditable public result for one telemetry observation."""

    schema_version: str = "1.0"
    decision_id: str = Field(default_factory=lambda: f"decision-{uuid.uuid4().hex}")
    timestamp: float = Field(default_factory=time.time)
    run_id: str
    deployment_id: str | None = None
    module_id: str
    task_family: str | None = None
    risk_probability: float
    novelty_score: float
    drift_score: float
    uncertainty: float
    confidence: float
    recommended_action: BrainAction
    executed_action: BrainAction
    mode: BrainMode
    autonomous: bool
    action_receipt_id: str | None = None
    model_version: int
    feature_schema: str = "weight-space-v2"
    provenance_verified: bool = False
    reasons: list[str] = Field(default_factory=list)
    policy_scores: dict[str, float] = Field(default_factory=dict)
    retrieved_memories: list[RetrievedMemory] = Field(default_factory=list)
    research_basis: list[str] = Field(
        default_factory=lambda: [
            "LoRA weight-space backdoor detection (2026-02)",
            "ReasoningBank experience learning (2025-09/ICLR 2026)",
            "TFlow transient weight communication (2026-05)",
            "AI Control runtime monitoring (2026-06)",
            "GPT-Red self-play hardening (2026-07)",
        ]
    )

    @field_validator(
        "timestamp",
        "risk_probability",
        "novelty_score",
        "drift_score",
        "uncertainty",
        "confidence",
    )
    @classmethod
    def finite_decision_number(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("brain decision values must be finite")
        return value


class ActionReceipt(BaseModel):
    """Immutable receipt from an allowlisted action executor."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0"
    receipt_id: str = Field(default_factory=lambda: f"receipt-{uuid.uuid4().hex}")
    timestamp: float = Field(default_factory=time.time)
    requested_action: BrainAction
    executed_action: BrainAction
    executor: str
    success: bool
    reversible: bool = True
    summary: str
    evidence: dict[str, float | int | str | bool | None] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def finite_receipt_timestamp(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("action receipt timestamp must be finite")
        return value

    @field_validator("evidence")
    @classmethod
    def finite_receipt_evidence(
        cls, values: dict[str, float | int | str | bool | None]
    ) -> dict[str, float | int | str | bool | None]:
        for value in values.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("action receipt numeric evidence must be finite")
        return values


class DecisionEnvelope(BaseModel):
    """Private learning record stored in the local trajectory ledger."""

    decision: BrainDecision
    features: list[float]
    numeric_features: list[float]
    telemetry: dict[str, object]

    @field_validator("features", "numeric_features")
    @classmethod
    def finite_vectors(cls, values: list[float]) -> list[float]:
        if any(not math.isfinite(value) for value in values):
            raise ValueError("brain feature vectors must be finite")
        return values


class BrainFeedback(BaseModel):
    """Verified label and, when supplied, realized action outcome."""

    schema_version: str = "1.0"
    feedback_id: str = Field(default_factory=lambda: f"feedback-{uuid.uuid4().hex}")
    timestamp: float = Field(default_factory=time.time)
    decision_id: str
    incident: bool
    verified: bool = True
    impact: float = Field(default=1.0, ge=0.0, le=10.0)
    actual_action: BrainAction | None = None
    action_succeeded: bool | None = None
    containment_succeeded: bool | None = None
    realized_cost: float = Field(default=0.0, ge=0.0, le=10.0)
    disruption: float = Field(default=0.0, ge=0.0, le=10.0)
    latency_ms: float = Field(default=0.0, ge=0.0, le=86_400_000.0)
    note: str | None = None
    source: Literal["operator", "incident_system", "offline_evaluation"] = "operator"

    @field_validator("timestamp", "impact", "realized_cost", "disruption", "latency_ms")
    @classmethod
    def finite_feedback_number(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("feedback values must be finite")
        return value

    @model_validator(mode="after")
    def valid_outcome(self) -> BrainFeedback:
        if not self.incident and self.containment_succeeded is not None:
            raise ValueError("containment_succeeded is only meaningful for an incident")
        return self

    @property
    def has_action_outcome(self) -> bool:
        if self.incident:
            return self.containment_succeeded is not None
        return self.action_succeeded is not None


class StrategyMemory(BaseModel):
    """A consolidated success/failure lesson inspired by ReasoningBank."""

    schema_version: str = "1.0"
    memory_id: str = Field(default_factory=lambda: f"memory-{uuid.uuid4().hex}")
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    title: str
    lesson: str
    centroid: list[float]
    preferred_action: BrainAction
    incident_count: int = Field(default=0, ge=0)
    benign_count: int = Field(default=0, ge=0)
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)
    mean_reward: float = 0.0
    example_decision_ids: list[str] = Field(default_factory=list)

    @field_validator("centroid")
    @classmethod
    def finite_centroid(cls, values: list[float]) -> list[float]:
        if any(not math.isfinite(value) for value in values):
            raise ValueError("strategy memory centroids must be finite")
        return values

    @field_validator("created_at", "updated_at", "mean_reward")
    @classmethod
    def finite_memory_number(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("strategy memory values must be finite")
        return value

    @property
    def observations(self) -> int:
        return self.incident_count + self.benign_count

    @property
    def incident_rate(self) -> float:
        return (self.incident_count + 1.0) / (self.observations + 2.0)


class SelfPlayReport(BaseModel):
    schema_version: str = "2.1"
    seed_incidents: int
    generated_candidates: int
    hard_examples: int
    mean_risk_before: float
    mean_risk_after: float
    evasion_rate_before: float
    evasion_rate_after: float
    benign_false_positive_rate_before: float
    benign_false_positive_rate_after: float
    benign_false_positive_upper_bound: float
    operating_threshold: float
    benign_holdout_examples: int
    incident_evaluation_examples: int
    incident_evasion_upper_bound: float
    hard_variant_seed_evasion_rate_before: float
    hard_variant_seed_evasion_rate_after: float
    hard_variant_seed_evasion_upper_bound: float
    classifier_hash_before: str
    classifier_hash_after: str
    incident_dataset_hash: str
    benign_holdout_dataset_hash: str
    promoted: bool
    promotion_reason: str
    model_version_before: int
    model_version_after: int
    safety_boundary: str = (
        "Feature-space defensive hardening only. No payload, trigger, tensor, adapter, "
        "or monitor-bypass artifact is emitted."
    )


class BrainStatus(BaseModel):
    schema_version: str = "1.0"
    model_version: int
    baseline_examples: int
    supervised_examples: int
    positive_examples: int
    negative_examples: int
    feedback_examples: int
    calibration_examples: int
    memory_count: int
    pending_decisions: int
    ready_for_learned_risk: bool
    ready_for_quarantine_recommendation: bool
    operating_review_threshold: float
    max_autonomous_action: BrainAction
    last_updated_at: float
