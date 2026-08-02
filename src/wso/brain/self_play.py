"""Sealed feature-space adversarial hardening inspired by GPT-Red self-play.

This module never constructs or emits an adapter, trigger, prompt, payload, or bypass
recipe. It searches only the already-derived numeric feature surface and exposes only
aggregate robustness measurements.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Iterable
from typing import Any, cast

import numpy as np

from wso.brain.learning import OnlineRiskClassifier
from wso.brain.runtime import AutonomousWeightSpaceBrain, telemetry_label
from wso.brain.types import (
    MAX_SELF_PLAY_SEEDS_PER_CLASS,
    SELF_PLAY_TRAIN_FRACTION,
    SelfPlayReport,
)
from wso.telemetry.schemas import TransientLoRATelemetry

MAX_CANDIDATES_PER_INCIDENT = 128


def run_defensive_self_play(
    brain: AutonomousWeightSpaceBrain,
    *,
    incidents: list[TransientLoRATelemetry],
    benign_holdout: list[TransientLoRATelemetry],
    candidates_per_incident: int = 48,
    epsilon: float = 0.18,
    promote: bool = True,
    expected_incident_dataset_hash: str | None = None,
    expected_benign_holdout_hash: str | None = None,
) -> SelfPlayReport:
    """Train and gate a challenger on hard derived-feature variants."""

    version_before = brain.refresh()
    if candidates_per_incident < 8:
        raise ValueError("self-play requires at least 8 candidates per incident")
    if candidates_per_incident > MAX_CANDIDATES_PER_INCIDENT:
        raise ValueError(
            f"self-play accepts at most {MAX_CANDIDATES_PER_INCIDENT} candidates per incident"
        )
    if not 0 < epsilon <= 0.5:
        raise ValueError("self-play epsilon must be inside (0, 0.5]")
    if len(incidents) < 2:
        raise ValueError("self-play requires at least two incident seeds")
    if len(incidents) > MAX_SELF_PLAY_SEEDS_PER_CLASS:
        raise ValueError(
            f"self-play accepts at most {MAX_SELF_PLAY_SEEDS_PER_CLASS} incident seeds"
        )
    if len(benign_holdout) > MAX_SELF_PLAY_SEEDS_PER_CLASS:
        raise ValueError(
            f"self-play accepts at most {MAX_SELF_PLAY_SEEDS_PER_CLASS} benign records"
        )
    if any(telemetry_label(record) is not True for record in incidents):
        raise ValueError("every self-play incident must carry an explicit incident label")
    if not benign_holdout:
        raise ValueError("self-play promotion requires a trusted benign holdout")
    if any(telemetry_label(record) is not False for record in benign_holdout):
        raise ValueError("every benign holdout record must carry an explicit benign label")
    incident_ids = [record.run_id for record in incidents]
    benign_ids = [record.run_id for record in benign_holdout]
    if len(set(incident_ids)) != len(incident_ids):
        raise ValueError("self-play incident run IDs must be unique")
    if len(set(benign_ids)) != len(benign_ids):
        raise ValueError("benign holdout run IDs must be unique")
    if set(incident_ids) & set(benign_ids):
        raise ValueError("incident seeds and benign holdout must be disjoint")
    incident_bundles = [brain.encoder.encode(record) for record in incidents]
    benign_bundles = [brain.encoder.encode(record) for record in benign_holdout]
    incident_features = [_feature_hash(bundle.vector) for bundle in incident_bundles]
    benign_features = [_feature_hash(bundle.vector) for bundle in benign_bundles]
    if len(set(incident_features)) != len(incident_features):
        raise ValueError("self-play incident seeds must be unique in encoded feature space")
    if len(set(benign_features)) != len(benign_features):
        raise ValueError("benign holdout must be unique in encoded feature space")
    if set(incident_features) & set(benign_features):
        raise ValueError("incident seeds and benign holdout overlap in encoded feature space")
    _reject_near_duplicates(
        [bundle.vector for bundle in incident_bundles],
        label="incident seeds",
    )
    _reject_near_duplicates(
        [bundle.vector for bundle in benign_bundles],
        label="benign holdout",
    )
    _reject_cross_near_duplicates(
        [bundle.vector for bundle in incident_bundles],
        [bundle.vector for bundle in benign_bundles],
    )
    incident_dataset_hash = telemetry_dataset_hash(incidents)
    benign_dataset_hash = telemetry_dataset_hash(benign_holdout)
    if promote:
        if expected_incident_dataset_hash is None:
            raise ValueError("promotion requires an expected frozen incident-dataset hash")
        if expected_incident_dataset_hash != incident_dataset_hash:
            raise ValueError("incidents do not match their expected frozen dataset hash")
        if expected_benign_holdout_hash is None:
            raise ValueError("promotion requires an expected frozen benign-holdout hash")
        if expected_benign_holdout_hash != benign_dataset_hash:
            raise ValueError("benign holdout does not match its expected frozen dataset hash")
        minimum_benign = _minimum_zero_event_sample(
            brain.config.target_false_positive_rate,
            brain.config.promotion_confidence,
        )
        if len(benign_holdout) < minimum_benign:
            raise ValueError(
                f"promotion requires at least {minimum_benign} benign holdout records "
                "for the configured confidence and false-positive target"
            )

    rng = np.random.default_rng(brain.config.random_seed + brain.model_version)
    train_hard: list[tuple[np.ndarray, np.ndarray, float, int]] = []
    eval_hard: list[tuple[np.ndarray, np.ndarray, float, int]] = []
    generated = 0
    order = np.arange(len(incidents))
    rng.shuffle(order)
    split = max(
        1,
        min(
            len(incidents) - 1,
            round(len(incidents) * SELF_PLAY_TRAIN_FRACTION),
        ),
    )
    train_indices = {int(index) for index in order[:split]}
    evaluation_indices = [int(index) for index in order[split:]]
    if promote:
        minimum_incidents = _minimum_zero_event_sample(
            brain.config.max_self_play_evasion_rate,
            brain.config.promotion_confidence,
        )
        if len(evaluation_indices) < minimum_incidents:
            raise ValueError(
                f"promotion requires at least {minimum_incidents} held-out incident seeds "
                "for the configured confidence and evasion ceiling"
            )
    for incident_index, bundle in enumerate(incident_bundles):
        base_risk = brain.risk_for_features(bundle.vector, bundle.numeric)
        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
        for _ in range(candidates_per_incident):
            vector, numeric = _mutate_derived_features(
                bundle.vector,
                bundle.numeric,
                rng=rng,
                epsilon=epsilon,
            )
            candidates.append((brain.risk_for_features(vector, numeric), vector, numeric))
            generated += 1
        candidates.sort(key=lambda item: item[0])
        destination = train_hard if incident_index in train_indices else eval_hard
        for risk, vector, numeric in candidates[: max(4, candidates_per_incident // 8)]:
            destination.append((vector, numeric, min(base_risk, risk), incident_index))

    if not train_hard or not eval_hard:
        raise RuntimeError("self-play failed to construct disjoint train and evaluation groups")
    # NumPy accepts mutable sequences here, though its 2.2 stubs only describe arrays.
    rng.shuffle(cast(Any, train_hard))
    rng.shuffle(cast(Any, eval_hard))
    challenger = brain.classifier.clone()

    training: list[tuple[np.ndarray, bool]] = [(vector, True) for vector, _, _, _ in train_hard]
    challenger.fit(
        training,
        epochs=max(6, brain.config.classifier_epochs // 3),
        seed=brain.config.random_seed + 17 + brain.model_version,
    )

    before = [
        _risk_with(brain, brain.classifier, vector, numeric) for vector, numeric, _, _ in eval_hard
    ]
    after = [_risk_with(brain, challenger, vector, numeric) for vector, numeric, _, _ in eval_hard]
    benign_before = [
        _risk_with(brain, brain.classifier, bundle.vector, bundle.numeric)
        for bundle in benign_bundles
    ]
    benign_after = [
        _risk_with(brain, challenger, bundle.vector, bundle.numeric) for bundle in benign_bundles
    ]
    original_eval = [incident_bundles[index] for index in evaluation_indices]
    original_before = [
        _risk_with(brain, brain.classifier, bundle.vector, bundle.numeric)
        for bundle in original_eval
    ]
    original_after = [
        _risk_with(brain, challenger, bundle.vector, bundle.numeric) for bundle in original_eval
    ]
    threshold = brain.review_threshold
    evasion_before = _rate(value < threshold for value in original_before)
    evasion_after = _rate(value < threshold for value in original_after)
    false_positive_before = _rate(value >= threshold for value in benign_before)
    false_positive_after = _rate(value >= threshold for value in benign_after)
    false_positive_count = sum(value >= threshold for value in benign_after)
    incident_evasion_count = sum(value < threshold for value in original_after)
    evaluation_seed_ids = [seed_id for _, _, _, seed_id in eval_hard]
    hard_seed_before = _minimum_score_by_seed(before, evaluation_seed_ids)
    hard_seed_after = _minimum_score_by_seed(after, evaluation_seed_ids)
    hard_seed_evasion_before = _rate(value < threshold for value in hard_seed_before)
    hard_seed_evasion_after = _rate(value < threshold for value in hard_seed_after)
    hard_seed_evasion_count = sum(value < threshold for value in hard_seed_after)
    false_positive_upper = _binomial_upper_bound(
        false_positive_count,
        len(benign_after),
        brain.config.promotion_confidence,
    )
    incident_evasion_upper = _binomial_upper_bound(
        incident_evasion_count,
        len(original_after),
        brain.config.promotion_confidence,
    )
    hard_seed_evasion_upper = _binomial_upper_bound(
        hard_seed_evasion_count,
        len(hard_seed_after),
        brain.config.promotion_confidence,
    )
    hash_before = _classifier_hash(brain.classifier)
    hash_after = _classifier_hash(challenger)

    mean_before = float(np.mean(before))
    mean_after = float(np.mean(after))
    strictly_improved = (
        hash_after != hash_before
        and mean_after >= mean_before - 1e-9
        and evasion_after <= evasion_before
        and (mean_after >= mean_before + 0.01 or evasion_after < evasion_before - 1e-12)
    )
    incident_gate = _passes_incident_gate(
        ceiling=brain.config.max_self_play_evasion_rate,
        original_evasion_before=evasion_before,
        original_evasion_after=evasion_after,
        original_evasion_upper=incident_evasion_upper,
        original_mean_before=float(np.mean(original_before)),
        original_mean_after=float(np.mean(original_after)),
        hard_seed_evasion_before=hard_seed_evasion_before,
        hard_seed_evasion_after=hard_seed_evasion_after,
        hard_seed_evasion_upper=hard_seed_evasion_upper,
    )
    fpr_gate = false_positive_upper <= brain.config.target_false_positive_rate
    should_promote = bool(promote and strictly_improved and incident_gate and fpr_gate)
    if should_promote:
        reason = (
            "Challenger strictly improved held-out feature-space detection and met "
            "confidence-bounded absolute incident and benign gates."
        )
        brain.promote_classifier(
            challenger,
            expected_model_version=version_before,
        )
    else:
        if not promote:
            reason = "Evaluation-only run; promotion was disabled."
        elif not strictly_improved:
            reason = "Challenger failed the strict held-out improvement gate."
        elif not incident_gate:
            reason = (
                f"Challenger incident evasion upper bound {incident_evasion_upper:.3f} "
                f"or hard-variant seed upper bound {hard_seed_evasion_upper:.3f} "
                "exceeded the promotion ceiling."
            )
        else:
            reason = (
                f"Challenger benign false-positive upper bound {false_positive_upper:.3f} "
                "exceeded the promotion gate."
            )

    return SelfPlayReport(
        seed_incidents=len(incidents),
        generated_candidates=generated,
        hard_examples=len(train_hard) + len(eval_hard),
        mean_risk_before=mean_before,
        mean_risk_after=mean_after,
        evasion_rate_before=evasion_before,
        evasion_rate_after=evasion_after,
        benign_false_positive_rate_before=false_positive_before,
        benign_false_positive_rate_after=false_positive_after,
        benign_false_positive_upper_bound=false_positive_upper,
        operating_threshold=threshold,
        benign_holdout_examples=len(benign_holdout),
        incident_evaluation_examples=len(original_after),
        incident_evasion_upper_bound=incident_evasion_upper,
        hard_variant_seed_evasion_rate_before=hard_seed_evasion_before,
        hard_variant_seed_evasion_rate_after=hard_seed_evasion_after,
        hard_variant_seed_evasion_upper_bound=hard_seed_evasion_upper,
        classifier_hash_before=hash_before,
        classifier_hash_after=hash_after,
        incident_dataset_hash=incident_dataset_hash,
        benign_holdout_dataset_hash=benign_dataset_hash,
        promoted=should_promote,
        promotion_reason=reason,
        model_version_before=version_before,
        model_version_after=brain.model_version,
    )


def _mutate_derived_features(
    vector: np.ndarray,
    numeric: np.ndarray,
    *,
    rng: np.random.Generator,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_vector = vector.copy()
    candidate_numeric = numeric.copy()
    mutable = 25  # Missingness/provenance bits and categorical context are immutable.
    scale = epsilon * np.maximum(np.abs(candidate_numeric[:mutable]), 0.08)
    perturbation = rng.normal(0.0, scale)
    perturbation = np.clip(perturbation, -2.0 * scale, 2.0 * scale)
    candidate_numeric[:mutable] = np.clip(
        candidate_numeric[:mutable] + perturbation,
        -6.0,
        6.0,
    )
    candidate_vector[: numeric.size] = candidate_numeric
    return candidate_vector, candidate_numeric


def _risk_with(
    brain: AutonomousWeightSpaceBrain,
    classifier: OnlineRiskClassifier,
    features: np.ndarray,
    numeric: np.ndarray,
) -> float:
    novelty, _, _ = brain.novelty.scores(numeric)
    probability, _ = classifier.predict(features)
    if classifier.ready(brain.config):
        risk = 0.78 * probability + 0.22 * novelty
    else:
        risk = novelty
    return float(np.clip(risk, 0.0, 1.0))


def _rate(values: Iterable[bool]) -> float:
    materialized = list(values)
    return sum(bool(value) for value in materialized) / max(1.0, float(len(materialized)))


def _minimum_score_by_seed(scores: list[float], seed_ids: list[int]) -> list[float]:
    grouped: dict[int, float] = {}
    for score, seed_id in zip(scores, seed_ids, strict=True):
        grouped[seed_id] = min(score, grouped.get(seed_id, 1.0))
    return list(grouped.values())


def _passes_incident_gate(
    *,
    ceiling: float,
    original_evasion_before: float,
    original_evasion_after: float,
    original_evasion_upper: float,
    original_mean_before: float,
    original_mean_after: float,
    hard_seed_evasion_before: float,
    hard_seed_evasion_after: float,
    hard_seed_evasion_upper: float,
) -> bool:
    return bool(
        original_evasion_upper <= ceiling
        and original_evasion_after <= original_evasion_before
        and hard_seed_evasion_upper <= ceiling
        and hard_seed_evasion_after <= hard_seed_evasion_before
        and original_mean_after >= original_mean_before - 1e-9
    )


def _classifier_hash(classifier: OnlineRiskClassifier) -> str:
    payload = json.dumps(
        classifier.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _feature_hash(features: np.ndarray) -> str:
    canonical = np.ascontiguousarray(features, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def telemetry_dataset_hash(records: list[TransientLoRATelemetry]) -> str:
    canonical = [
        json.dumps(
            record.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for record in records
    ]
    rendered = "\n".join(sorted(canonical))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _minimum_zero_event_sample(limit: float, confidence: float) -> int:
    required = math.log1p(-confidence) / math.log1p(-limit)
    return math.ceil(required) if math.isfinite(required) else sys.maxsize


def _binomial_upper_bound(events: int, total: int, confidence: float) -> float:
    if total <= 0:
        return 1.0
    if events >= total:
        return 1.0
    target = 1.0 - confidence
    lower = events / total
    upper = 1.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if _binomial_cdf(events, total, midpoint) > target:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _binomial_cdf(events: int, total: int, probability: float) -> float:
    if probability <= 0:
        return 1.0
    if probability >= 1:
        return 0.0 if events < total else 1.0
    log_probability = math.log(probability)
    log_complement = math.log1p(-probability)
    terms = [
        math.lgamma(total + 1)
        - math.lgamma(index + 1)
        - math.lgamma(total - index + 1)
        + index * log_probability
        + (total - index) * log_complement
        for index in range(events + 1)
    ]
    maximum = max(terms)
    return math.exp(maximum) * sum(math.exp(term - maximum) for term in terms)


def _reject_near_duplicates(vectors: list[np.ndarray], *, label: str) -> None:
    for left_index, left in enumerate(vectors):
        for right in vectors[left_index + 1 :]:
            if np.allclose(left, right, rtol=1e-7, atol=1e-8):
                raise ValueError(f"{label} contain near-duplicate encoded examples")


def _reject_cross_near_duplicates(
    incidents: list[np.ndarray],
    benign: list[np.ndarray],
) -> None:
    for incident in incidents:
        for clean in benign:
            if np.allclose(incident, clean, rtol=1e-7, atol=1e-8):
                raise ValueError(
                    "incident seeds and benign holdout overlap in near-duplicate feature space"
                )
