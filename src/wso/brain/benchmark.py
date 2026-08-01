"""Deterministic defensive fixture for exercising the complete brain loop.

The fixture is interface evidence only. It does not model a deployed attack and must
never be cited as real-world detector performance.
"""

from __future__ import annotations

import json
import math
import uuid
from collections import Counter
from pathlib import Path

import numpy as np

from wso.brain.runtime import AutonomousWeightSpaceBrain
from wso.brain.self_play import run_defensive_self_play, telemetry_dataset_hash
from wso.brain.types import BrainConfig, BrainFeedback
from wso.telemetry.schemas import TransientLoRATelemetry


def generate_defensive_fixture(
    *,
    seed: int = 20260801,
    baseline_count: int = 96,
    labeled_per_class: int = 40,
    test_per_class: int = 32,
) -> tuple[
    list[TransientLoRATelemetry],
    list[TransientLoRATelemetry],
    list[TransientLoRATelemetry],
]:
    """Create trusted baseline, labeled training, and held-out audit streams."""

    rng = np.random.default_rng(seed)
    baseline = [
        _record(
            rng,
            index=index,
            family="ordinary_benign",
            label=False,
            split="baseline",
            fixture_seed=seed,
        )
        for index in range(baseline_count)
    ]
    labeled: list[TransientLoRATelemetry] = []
    audit: list[TransientLoRATelemetry] = []
    for index in range(labeled_per_class):
        labeled.append(
            _record(
                rng,
                index=index,
                family="benign_regime_shift",
                label=False,
                split="audit",
                fixture_seed=seed,
            )
        )
        attack_family = "gross_registered_stress" if index % 2 else "matched_spectral_stress"
        labeled.append(
            _record(
                rng,
                index=index,
                family=attack_family,
                label=True,
                split="audit",
                fixture_seed=seed,
            )
        )
    for index in range(test_per_class):
        audit.append(
            _record(
                rng,
                index=10_000 + index,
                family="benign_regime_shift",
                label=False,
                split="audit",
                fixture_seed=seed,
            )
        )
        attack_family = "matched_spectral_stress" if index % 3 else "gross_registered_stress"
        audit.append(
            _record(
                rng,
                index=20_000 + index,
                family=attack_family,
                label=True,
                split="audit",
                fixture_seed=seed,
            )
        )
    rng.shuffle(labeled)
    rng.shuffle(audit)
    return baseline, labeled, audit


def run_brain_demo(out_dir: str | Path, *, seed: int = 20260801) -> dict[str, object]:
    """Run bootstrap, autonomous decisions, feedback learning, and self-play."""

    out_dir = Path(out_dir)
    if (out_dir / "brain" / "brain_state.json").exists():
        raise FileExistsError(f"demo brain already exists under {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline, labeled, audit = generate_defensive_fixture(seed=seed, test_per_class=64)
    _write_telemetry(out_dir / "baseline.jsonl", baseline)
    _write_telemetry(out_dir / "labeled.jsonl", labeled)
    _write_telemetry(out_dir / "audit.jsonl", audit)

    config = BrainConfig(random_seed=seed, max_autonomous_action="deep_scan")
    brain = AutonomousWeightSpaceBrain(out_dir / "brain", config=config)
    bootstrap_status = brain.bootstrap(baseline=baseline, labeled=labeled)
    decisions = [brain.decide(record, mode="shadow") for record in audit]
    labels = [bool(record.provenance["ground_truth"]) for record in audit]
    scores = [decision.risk_probability for decision in decisions]
    predicted = [score >= brain.review_threshold for score in scores]
    benign_count = sum(not label for label in labels)
    false_positives = sum(prediction and not label for prediction, label in zip(predicted, labels))
    evaluation = {
        "schema_version": "1.0",
        "evidence_mode": "synthetic_fixture",
        "claim_boundary": (
            "This proves deterministic closed-loop mechanics only; it is not deployed-system "
            "detection evidence."
        ),
        "records": len(audit),
        "auroc": _binary_auroc(labels, scores),
        "brier_score": float(np.mean(np.square(np.asarray(scores) - np.asarray(labels)))),
        "precision_at_review_threshold": _binary_precision(labels, predicted),
        "recall_at_review_threshold": _binary_recall(labels, predicted),
        "false_positive_rate_at_review_threshold": false_positives / max(1, benign_count),
        "operating_review_threshold": brain.review_threshold,
        "recommended_actions": dict(Counter(d.recommended_action for d in decisions)),
        "executed_actions": dict(Counter(d.executed_action for d in decisions)),
        "bootstrap_status": bootstrap_status.model_dump(),
    }
    (out_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Keep self-play seeds disjoint from feedback learning. The benign audit
    # partition is never fitted and remains the frozen promotion gate.
    incidents = [record for record in audit if record.provenance["ground_truth"]]
    benign = [record for record in audit if not record.provenance["ground_truth"]]
    feedback_run_ids = {record.run_id for record in incidents[:16]}
    for record, decision, label in zip(audit, decisions, labels, strict=True):
        if record.run_id not in feedback_run_ids:
            continue
        brain.feedback(
            BrainFeedback(
                decision_id=decision.decision_id,
                incident=label,
                verified=True,
                actual_action=decision.executed_action,
                action_succeeded=True,
                containment_succeeded=False,
                source="offline_evaluation",
            )
        )
    self_play_incidents = [record for record in incidents if record.run_id not in feedback_run_ids]
    self_play = run_defensive_self_play(
        brain,
        incidents=self_play_incidents,
        benign_holdout=benign,
        candidates_per_incident=16,
        expected_incident_dataset_hash=telemetry_dataset_hash(self_play_incidents),
        expected_benign_holdout_hash=telemetry_dataset_hash(benign),
    )
    (out_dir / "self_play.json").write_text(self_play.model_dump_json(indent=2), encoding="utf-8")
    result = {
        **evaluation,
        "final_status": brain.status().model_dump(),
        "self_play": self_play.model_dump(),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def _record(
    rng: np.random.Generator,
    *,
    index: int,
    family: str,
    label: bool,
    split: str,
    fixture_seed: int,
) -> TransientLoRATelemetry:
    rank = 8
    if family == "ordinary_benign":
        spectrum = np.exp(-0.22 * np.arange(rank)) * rng.normal(1.0, 0.045, rank)
        drift = abs(float(rng.normal(0.08, 0.025)))
        sender_count = 2
    elif family == "benign_regime_shift":
        decay = float(rng.normal(0.155, 0.035))
        spectrum = np.exp(-decay * np.arange(rank)) * rng.lognormal(0.0, 0.10, rank)
        spectrum *= float(rng.normal(3.15, 0.22)) / max(1e-9, float(np.linalg.norm(spectrum)))
        drift = abs(float(rng.normal(0.40, 0.11)))
        sender_count = 4
    elif family == "gross_registered_stress":
        spectrum = np.asarray([4.8, 0.7, 0.52, 0.38, 0.24, 0.18, 0.12, 0.08])
        spectrum *= rng.normal(1.0, 0.055, rank)
        drift = abs(float(rng.normal(0.82, 0.08)))
        sender_count = 3
    elif family == "matched_spectral_stress":
        decay = float(rng.normal(0.18, 0.04))
        spectrum = np.exp(-decay * np.arange(rank)) * rng.lognormal(0.0, 0.11, rank)
        spectrum[0] *= float(rng.normal(1.12, 0.08))
        spectrum *= float(rng.normal(3.18, 0.24)) / max(1e-9, float(np.linalg.norm(spectrum)))
        drift = abs(float(rng.normal(0.45, 0.12)))
        sender_count = 4
    else:
        raise ValueError(f"unknown fixture family: {family}")

    spectrum = np.clip(spectrum, 1e-5, None)
    energy = np.square(spectrum)
    probabilities = energy / float(energy.sum())
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    effective_rank = math.exp(entropy)
    gate_logits = rng.normal(0.0, 0.45, sender_count)
    gate_probs = np.exp(gate_logits - float(gate_logits.max()))
    gate_probs /= float(gate_probs.sum())
    return TransientLoRATelemetry(
        run_id=f"fixture-{family}-{index}-{uuid.UUID(int=int(rng.integers(0, 2**63)))}",
        deployment_id="brain-fixture-deployment",
        module_id="receiver.layers.21.self_attn.q_proj",
        task_family="agent_tool_use",
        evidence_mode="fixture",
        collection_mode="tflow_fixture",
        source_kind="fixture",
        model_id="fixture-qwen3-4b",
        adapter_id=f"fixture-adapter-{index % 7}",
        split=split,  # type: ignore[arg-type]
        runtime_framework="fixture",
        observer_level="derived_weight_space_telemetry",
        layer_shape=[256, 256],
        sender_count=sender_count,
        rank_bound=sender_count * rank,
        sender_ids=[f"sender-{sender}" for sender in range(sender_count)],
        target_rank=rank,
        fusion_gate_logits=gate_logits.tolist(),
        fusion_gate_probs=gate_probs.tolist(),
        fusion_entropy=-float(np.sum(gate_probs * np.log(gate_probs))),
        frobenius_norm=float(np.linalg.norm(spectrum)),
        singular_values=spectrum.tolist(),
        effective_rank=effective_rank,
        spectral_gap=float(spectrum[0] - spectrum[1]),
        grassmann_drift=drift,
        provenance={
            "ground_truth": label,
            "fixture_family": family,
            "fixture_only": True,
            "seed": fixture_seed,
        },
    )


def _write_telemetry(path: Path, records: list[TransientLoRATelemetry]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def _binary_auroc(labels: list[bool], scores: list[float]) -> float:
    positive = [score for score, label in zip(scores, labels, strict=True) if label]
    negative = [score for score, label in zip(scores, labels, strict=True) if not label]
    if not positive or not negative:
        raise ValueError("AUROC requires both positive and negative examples")
    wins = sum(left > right for left in positive for right in negative)
    ties = sum(left == right for left in positive for right in negative)
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def _binary_precision(labels: list[bool], predicted: list[bool]) -> float:
    predicted_positive = sum(predicted)
    if predicted_positive == 0:
        return 0.0
    true_positive = sum(
        prediction and label for prediction, label in zip(predicted, labels, strict=True)
    )
    return true_positive / predicted_positive


def _binary_recall(labels: list[bool], predicted: list[bool]) -> float:
    positives = sum(labels)
    if positives == 0:
        return 0.0
    true_positive = sum(
        prediction and label for prediction, label in zip(predicted, labels, strict=True)
    )
    return true_positive / positives
