"""Small, auditable online learners used by the autonomous brain.

The implementation deliberately avoids opaque pickle artifacts. Every learned number is
stored as JSON, can be inspected, and is covered by the same immutable policy envelope.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt

from wso.brain.types import (
    BrainAction,
    BrainConfig,
    BrainDecision,
    RetrievedMemory,
    StrategyMemory,
)

ACTIONS: tuple[BrainAction, ...] = (
    "allow",
    "observe",
    "deep_scan",
    "human_review",
    "quarantine",
)

FloatArray = npt.NDArray[np.float64]


class OnlineRiskClassifier:
    """Diagonal-Laplace logistic learner with an AdaGrad update."""

    def __init__(self, dimension: int, *, learning_rate: float, l2: float) -> None:
        self.dimension = dimension
        self.learning_rate = learning_rate
        self.l2 = l2
        self.weights: FloatArray = np.zeros(dimension + 1, dtype=np.float64)
        self.gradient_square: FloatArray = np.full(dimension + 1, 1e-4, dtype=np.float64)
        self.precision: FloatArray = np.ones(dimension + 1, dtype=np.float64)
        self.examples = 0
        self.positive_examples = 0
        self.negative_examples = 0

    def predict(self, features: np.ndarray) -> tuple[float, float]:
        augmented = self._augment(features)
        logit = float(np.dot(self.weights, augmented))
        posterior_variance = float(np.sum(np.square(augmented) / self.precision))
        corrected_logit = logit / math.sqrt(1.0 + math.pi * posterior_variance / 8.0)
        probability = _sigmoid(corrected_logit)
        uncertainty = math.sqrt(max(0.0, posterior_variance))
        uncertainty = uncertainty / (1.0 + uncertainty)
        return probability, float(np.clip(uncertainty, 0.0, 1.0))

    def update(
        self,
        features: np.ndarray,
        label: bool,
        *,
        weight: float | None = None,
        count_observation: bool = True,
        update_precision: bool = True,
    ) -> None:
        augmented = self._augment(features)
        probability = _sigmoid(float(np.dot(self.weights, augmented)))
        if weight is None:
            weight = self._class_weight(label)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("classifier example weight must be finite and positive")
        error = probability - float(label)
        gradient = weight * error * augmented + self.l2 * self.weights
        hessian = weight * probability * (1.0 - probability) * np.square(augmented) + self.l2
        self.gradient_square += np.square(gradient)
        if update_precision:
            self.precision += hessian
        step = self.learning_rate * gradient / np.sqrt(self.gradient_square)
        self.weights -= np.clip(step, -0.25, 0.25)
        if count_observation:
            self.examples += 1
            if label:
                self.positive_examples += 1
            else:
                self.negative_examples += 1

    def fit(
        self,
        examples: list[tuple[np.ndarray, bool]],
        *,
        epochs: int,
        seed: int,
    ) -> None:
        if not examples:
            return
        rng = np.random.default_rng(seed)
        order = np.arange(len(examples))
        for epoch in range(epochs):
            rng.shuffle(order)
            for index in order:
                features, label = examples[int(index)]
                self.update(
                    features,
                    label,
                    count_observation=epoch == 0,
                    update_precision=epoch == epochs - 1,
                )

    def ready(self, config: BrainConfig) -> bool:
        return (
            self.examples >= config.min_supervised_examples
            and self.positive_examples >= config.min_positive_examples
            and self.negative_examples >= config.min_positive_examples
        )

    def clone(self) -> OnlineRiskClassifier:
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "weights": self.weights.tolist(),
            "gradient_square": self.gradient_square.tolist(),
            "precision": self.precision.tolist(),
            "examples": self.examples,
            "positive_examples": self.positive_examples,
            "negative_examples": self.negative_examples,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OnlineRiskClassifier:
        dimension = int(payload["dimension"])
        learning_rate = float(payload["learning_rate"])
        l2 = float(payload["l2"])
        if dimension < 1:
            raise ValueError("persisted classifier dimension must be positive")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("persisted classifier learning rate must be finite and positive")
        if not math.isfinite(l2) or l2 < 0:
            raise ValueError("persisted classifier regularization must be finite and non-negative")
        model = cls(
            dimension,
            learning_rate=learning_rate,
            l2=l2,
        )
        model.weights = np.asarray(payload["weights"], dtype=np.float64)
        model.gradient_square = np.asarray(payload["gradient_square"], dtype=np.float64)
        model.precision = np.asarray(payload["precision"], dtype=np.float64)
        expected = (model.dimension + 1,)
        if not (
            model.weights.shape == expected
            and model.gradient_square.shape == expected
            and model.precision.shape == expected
        ):
            raise ValueError("persisted classifier arrays have invalid dimensions")
        if not (
            np.isfinite(model.weights).all()
            and np.isfinite(model.gradient_square).all()
            and np.isfinite(model.precision).all()
        ):
            raise ValueError("persisted classifier contains non-finite values")
        if np.any(model.gradient_square <= 0) or np.any(model.precision <= 0):
            raise ValueError("persisted classifier precision must be positive")
        model.examples = int(payload.get("examples", 0))
        model.positive_examples = int(payload.get("positive_examples", 0))
        model.negative_examples = int(payload.get("negative_examples", 0))
        if (
            model.examples < 0
            or model.positive_examples < 0
            or model.negative_examples < 0
            or model.examples != model.positive_examples + model.negative_examples
        ):
            raise ValueError("persisted classifier example counts are inconsistent")
        return model

    def _augment(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.shape != (self.dimension,):
            raise ValueError(f"expected {self.dimension} risk features, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("risk features must be finite")
        return np.concatenate((np.ones(1, dtype=np.float64), values))

    def _class_weight(self, label: bool) -> float:
        total = self.positive_examples + self.negative_examples + 2.0
        count = (self.positive_examples if label else self.negative_examples) + 1.0
        return float(np.clip(total / (2.0 * count), 0.5, 5.0))


class MultiTimescaleNovelty:
    """Slow trusted baseline plus fast untrusted drift memory.

    This is a pragmatic continuum-memory design: all observations update the fast
    state, while only verified benign outcomes may update the slow state.
    """

    def __init__(self, dimension: int, *, calibration_limit: int) -> None:
        self.dimension = dimension
        self.calibration_limit = calibration_limit
        self.count = 0
        self.mean: FloatArray = np.zeros(dimension, dtype=np.float64)
        self.m2: FloatArray = np.zeros(dimension, dtype=np.float64)
        self.fast_mean: FloatArray = np.zeros(dimension, dtype=np.float64)
        self.fast_initialized = False
        self.calibration: list[float] = []

    def fit_baseline(self, examples: list[np.ndarray]) -> None:
        if not examples:
            raise ValueError("at least one trusted baseline example is required")
        self.count = 0
        self.mean.fill(0.0)
        self.m2.fill(0.0)
        for values in examples:
            self._update_slow(self._validated(values))
        self.fast_mean = self.mean.copy()
        self.fast_initialized = True
        self.calibration = sorted(self.raw_surprise(values) for values in examples)
        self.calibration = self.calibration[-self.calibration_limit :]

    def calibrate(self, examples: list[np.ndarray]) -> None:
        """Replace operating calibration with a disjoint trusted holdout."""

        if not examples:
            raise ValueError("novelty calibration requires at least one example")
        self.calibration = sorted(self.raw_surprise(self._validated(values)) for values in examples)
        self.calibration = self.calibration[-self.calibration_limit :]

    def scores(self, values: np.ndarray) -> tuple[float, float, list[tuple[int, float]]]:
        values = self._validated(values)
        z = self.z_scores(values)
        ranked = sorted(enumerate(z.tolist()), key=lambda item: item[1], reverse=True)
        raw = self._aggregate_z(z)
        novelty = self._calibrated_risk(raw)
        drift = self._drift_score()
        return novelty, drift, ranked[:5]

    def raw_surprise(self, values: np.ndarray) -> float:
        return self._aggregate_z(self.z_scores(self._validated(values)))

    def z_scores(self, values: np.ndarray) -> np.ndarray:
        if self.count < 2:
            return np.zeros(self.dimension, dtype=np.float64)
        variance = self.m2 / max(1.0, float(self.count - 1))
        floor = np.square(0.025 + 0.08 * np.abs(self.mean))
        scale = np.sqrt(np.maximum(variance, floor))
        return np.abs(values - self.mean) / scale

    def observe(self, values: np.ndarray) -> None:
        values = self._validated(values)
        if not self.fast_initialized:
            self.fast_mean = values.copy()
            self.fast_initialized = True
            return
        self.fast_mean = 0.92 * self.fast_mean + 0.08 * values

    def learn_benign(self, values: np.ndarray) -> None:
        values = self._validated(values)
        raw = self.raw_surprise(values)
        self._update_slow(values)
        self.calibration.append(raw)
        if len(self.calibration) > self.calibration_limit:
            self.calibration = self.calibration[-self.calibration_limit :]

    def _update_slow(self, values: np.ndarray) -> None:
        self.count += 1
        delta = values - self.mean
        self.mean += delta / float(self.count)
        delta2 = values - self.mean
        self.m2 += delta * delta2

    def _aggregate_z(self, z: np.ndarray) -> float:
        if not z.size:
            return 0.0
        width = min(5, z.size)
        top = np.partition(z, -width)[-width:]
        return float(np.mean(top))

    def _calibrated_risk(self, raw: float) -> float:
        if not self.calibration:
            return float(np.clip(1.0 - math.exp(-max(0.0, raw - 1.0)), 0.0, 1.0))
        less = sum(value < raw for value in self.calibration)
        equal = sum(value == raw for value in self.calibration)
        quantile = (less + 0.5 * equal + 0.5) / (len(self.calibration) + 1.0)
        exponent = math.log(0.5) / math.log(1.0 - 0.05)
        return float(np.clip(quantile**exponent, 0.0, 1.0))

    def _drift_score(self) -> float:
        if not self.fast_initialized or self.count < 2:
            return 0.0
        variance = self.m2 / max(1.0, float(self.count - 1))
        floor = np.square(0.025 + 0.08 * np.abs(self.mean))
        scale = np.sqrt(np.maximum(variance, floor))
        drift_z = np.abs(self.fast_mean - self.mean) / scale
        top = np.sort(drift_z)[-min(5, drift_z.size) :]
        return float(np.clip(1.0 - math.exp(-float(np.mean(top)) / 2.0), 0.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "calibration_limit": self.calibration_limit,
            "count": self.count,
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
            "fast_mean": self.fast_mean.tolist(),
            "fast_initialized": self.fast_initialized,
            "calibration": self.calibration,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MultiTimescaleNovelty:
        dimension = int(payload["dimension"])
        calibration_limit = int(payload["calibration_limit"])
        if dimension < 1 or calibration_limit < 1:
            raise ValueError("persisted novelty dimensions and limits must be positive")
        model = cls(
            dimension,
            calibration_limit=calibration_limit,
        )
        model.count = int(payload.get("count", 0))
        model.mean = np.asarray(payload["mean"], dtype=np.float64)
        model.m2 = np.asarray(payload["m2"], dtype=np.float64)
        model.fast_mean = np.asarray(payload["fast_mean"], dtype=np.float64)
        model.fast_initialized = bool(payload.get("fast_initialized", False))
        model.calibration = [float(value) for value in payload.get("calibration", [])]
        expected = (model.dimension,)
        if not (
            model.mean.shape == expected
            and model.m2.shape == expected
            and model.fast_mean.shape == expected
        ):
            raise ValueError("persisted novelty arrays have invalid dimensions")
        if not (
            np.isfinite(model.mean).all()
            and np.isfinite(model.m2).all()
            and np.isfinite(model.fast_mean).all()
            and all(math.isfinite(value) for value in model.calibration)
        ):
            raise ValueError("persisted novelty model contains non-finite values")
        if model.count < 0 or np.any(model.m2 < -1e-9):
            raise ValueError("persisted novelty model has invalid counts or variance")
        if len(model.calibration) > model.calibration_limit:
            raise ValueError("persisted novelty calibration exceeds its limit")
        return model

    def _validated(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self.dimension,):
            raise ValueError(f"expected {self.dimension} novelty features, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("novelty features must be finite")
        return array


class ConstrainedContextualBandit:
    """LinUCB policy inside a non-learned action safety envelope."""

    context_dimension = 7

    def __init__(self, *, exploration: float) -> None:
        self.exploration = exploration
        self.a: dict[BrainAction, FloatArray] = {
            action: np.eye(self.context_dimension) * 2.0 for action in ACTIONS
        }
        self.b: dict[BrainAction, FloatArray] = {
            action: np.zeros(self.context_dimension) for action in ACTIONS
        }
        self.counts = {action: 0 for action in ACTIONS}

    def context(
        self,
        *,
        risk: float,
        novelty: float,
        drift: float,
        uncertainty: float,
        memory_risk: float,
    ) -> np.ndarray:
        values = np.asarray(
            [1.0, risk, novelty, drift, uncertainty, risk * (1.0 - uncertainty), memory_risk],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("policy context must be finite")
        return values

    def select(
        self,
        candidates: Iterable[BrainAction],
        context: np.ndarray,
    ) -> tuple[BrainAction, dict[str, float]]:
        scores: dict[str, float] = {}
        for action in candidates:
            inverse = np.linalg.inv(self.a[action])
            theta = inverse @ self.b[action]
            expected = float(theta @ context)
            bonus = self.exploration * math.sqrt(max(0.0, float(context @ inverse @ context)))
            score = expected + bonus + self._prior(action, context)
            scores[action] = score
        if not scores:
            raise ValueError("policy received no admissible actions")
        chosen = max(scores, key=lambda action: (scores[action], -ACTIONS.index(action)))
        return chosen, scores  # type: ignore[return-value]

    def update(self, action: BrainAction, context: np.ndarray, reward: float) -> None:
        self.a[action] += np.outer(context, context)
        self.b[action] += float(np.clip(reward, -10.0, 10.0)) * context
        self.counts[action] += 1

    def _prior(self, action: BrainAction, context: np.ndarray) -> float:
        _, risk, novelty, drift, uncertainty, _, memory_risk = context
        combined = 0.65 * risk + 0.2 * novelty + 0.15 * memory_risk
        if action == "allow":
            return 0.9 - 1.8 * combined - 0.4 * drift
        if action == "observe":
            return 0.35 - abs(combined - 0.28) - 0.15 * drift
        if action == "deep_scan":
            return 0.55 - abs(combined - 0.58) + 0.35 * uncertainty + 0.2 * drift
        if action == "human_review":
            return 0.25 + 0.65 * uncertainty + 0.55 * combined
        return 1.4 * combined - 0.65 * uncertainty - 0.55

    def to_dict(self) -> dict[str, Any]:
        return {
            "exploration": self.exploration,
            "a": {action: matrix.tolist() for action, matrix in self.a.items()},
            "b": {action: vector.tolist() for action, vector in self.b.items()},
            "counts": self.counts,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConstrainedContextualBandit:
        exploration = float(payload["exploration"])
        if not math.isfinite(exploration) or exploration < 0:
            raise ValueError("persisted policy exploration must be finite and non-negative")
        model = cls(exploration=exploration)
        model.a = {action: np.asarray(payload["a"][action], dtype=np.float64) for action in ACTIONS}
        model.b = {action: np.asarray(payload["b"][action], dtype=np.float64) for action in ACTIONS}
        model.counts = {action: int(payload["counts"].get(action, 0)) for action in ACTIONS}
        for action in ACTIONS:
            if model.a[action].shape != (model.context_dimension, model.context_dimension):
                raise ValueError("persisted policy matrix has invalid dimensions")
            if model.b[action].shape != (model.context_dimension,):
                raise ValueError("persisted policy vector has invalid dimensions")
            if not np.isfinite(model.a[action]).all() or not np.isfinite(model.b[action]).all():
                raise ValueError("persisted policy contains non-finite values")
            if not np.allclose(model.a[action], model.a[action].T, rtol=1e-10, atol=1e-12):
                raise ValueError("persisted policy matrix must be symmetric")
            if float(np.min(np.linalg.eigvalsh(model.a[action]))) <= 0:
                raise ValueError("persisted policy matrix must be positive definite")
            if model.counts[action] < 0:
                raise ValueError("persisted policy counts must be non-negative")
        return model


class StrategyBank:
    """Consolidated retrieval memory learned only from verified outcomes."""

    def __init__(self, *, limit: int) -> None:
        self.limit = limit
        self.memories: list[StrategyMemory] = []

    def retrieve(self, features: np.ndarray, *, limit: int = 3) -> list[RetrievedMemory]:
        ranked: list[tuple[float, StrategyMemory]] = []
        for memory in self.memories:
            similarity = _cosine(features, np.asarray(memory.centroid, dtype=np.float64))
            if similarity >= 0.35:
                ranked.append((similarity, memory))
        ranked.sort(key=lambda item: (item[0], item[1].observations), reverse=True)
        return [
            RetrievedMemory(
                memory_id=memory.memory_id,
                title=memory.title,
                similarity=similarity,
                incident_rate=memory.incident_rate,
                preferred_action=memory.preferred_action,
                lesson=memory.lesson,
            )
            for similarity, memory in ranked[:limit]
        ]

    def consolidate(
        self,
        *,
        decision: BrainDecision,
        features: np.ndarray,
        incident: bool,
        actual_action: BrainAction,
        reward: float,
    ) -> StrategyMemory:
        nearest: StrategyMemory | None = None
        nearest_similarity = -1.0
        for memory in self.memories:
            if memory.preferred_action != actual_action:
                continue
            similarity = _cosine(features, np.asarray(memory.centroid, dtype=np.float64))
            if similarity > nearest_similarity:
                nearest = memory
                nearest_similarity = similarity
        if nearest is None or nearest_similarity < 0.86:
            nearest = StrategyMemory(
                title=_memory_title(decision, incident, actual_action),
                lesson=_memory_lesson(decision, incident, actual_action),
                centroid=features.tolist(),
                preferred_action=actual_action,
            )
            self.memories.append(nearest)

        previous = nearest.observations
        centroid = np.asarray(nearest.centroid, dtype=np.float64)
        centroid += (features - centroid) / float(previous + 1)
        nearest.centroid = centroid.tolist()
        nearest.updated_at = time.time()
        if incident:
            nearest.incident_count += 1
        else:
            nearest.benign_count += 1
        if reward >= 0:
            nearest.success_count += 1
        else:
            nearest.failure_count += 1
        nearest.mean_reward += (reward - nearest.mean_reward) / float(previous + 1)
        nearest.example_decision_ids.append(decision.decision_id)
        nearest.example_decision_ids = nearest.example_decision_ids[-8:]
        self._trim()
        return nearest

    def _trim(self) -> None:
        if len(self.memories) <= self.limit:
            return
        self.memories.sort(
            key=lambda memory: (memory.observations, memory.updated_at), reverse=True
        )
        self.memories = self.memories[: self.limit]

    def to_dict(self) -> dict[str, Any]:
        return {"limit": self.limit, "memories": [memory.model_dump() for memory in self.memories]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StrategyBank:
        limit = int(payload["limit"])
        if limit < 1:
            raise ValueError("persisted strategy-bank limit must be positive")
        bank = cls(limit=limit)
        bank.memories = [
            StrategyMemory.model_validate(item) for item in payload.get("memories", [])
        ]
        return bank


def outcome_reward(
    *,
    incident: bool,
    impact: float,
    action_succeeded: bool | None,
    containment_succeeded: bool | None,
    realized_cost: float,
    disruption: float,
    latency_ms: float,
) -> float:
    """Reward an observed result, never a counterfactual action assumption."""

    if action_succeeded is None and containment_succeeded is None:
        raise ValueError("an observed action outcome is required for policy learning")
    operational_success = 0.0 if action_succeeded is None else (1.0 if action_succeeded else -1.0)
    if incident:
        contained = 1.0 if containment_succeeded is True else -1.0
        benefit = impact * contained + 0.15 * operational_success
    else:
        benefit = 0.25 * operational_success
    latency_penalty = min(latency_ms / 3_600_000.0, 1.0) * 0.1
    return float(
        np.clip(benefit - 0.12 * realized_cost - 0.18 * disruption - latency_penalty, -10, 10)
    )


def memory_incident_risk(memories: list[RetrievedMemory]) -> float:
    if not memories:
        return 0.5
    weights = np.asarray([memory.similarity for memory in memories], dtype=np.float64)
    rates = np.asarray([memory.incident_rate for memory in memories], dtype=np.float64)
    total = float(weights.sum())
    return float(weights @ rates / total) if total > 0 else 0.5


def _sigmoid(value: float) -> float:
    if value >= 0:
        factor = math.exp(-min(value, 60.0))
        return 1.0 / (1.0 + factor)
    factor = math.exp(max(value, -60.0))
    return factor / (1.0 + factor)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _memory_title(decision: BrainDecision, incident: bool, action: BrainAction) -> str:
    outcome = "incident" if incident else "benign shift"
    return f"{action.replace('_', ' ').title()} for {outcome} in {decision.module_id}"


def _memory_lesson(decision: BrainDecision, incident: bool, action: BrainAction) -> str:
    signal = decision.reasons[0] if decision.reasons else "this weight-space signature"
    if incident:
        return (
            f"A verified incident matched {signal.lower()}; prefer {action.replace('_', ' ')} "
            "for a closely matching future signature."
        )
    return (
        f"A benign outcome matched {signal.lower()}; prefer {action.replace('_', ' ')} and "
        "avoid stronger disruption unless new evidence appears."
    )
