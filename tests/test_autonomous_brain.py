from __future__ import annotations

import json
import math
import os
import stat
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

import wso.brain.cli as brain_cli
from wso.adapters.tflow_placeholder import OfficialTFlowAdapter, TFlowInferenceConfig
from wso.brain.actions import LocalDefensiveExecutor
from wso.brain.benchmark import generate_defensive_fixture, run_brain_demo
from wso.brain.cli import (
    _atomic_write_text,
    _load_brain_telemetry,
    _validate_output_path,
)
from wso.brain.cli import (
    main as brain_cli_main,
)
from wso.brain.features import (
    MAX_METADATA_BYTES,
    MAX_METADATA_NODES,
    WeightSpaceFeatureEncoder,
    _metadata_within_budget,
)
from wso.brain.learning import MultiTimescaleNovelty, OnlineRiskClassifier
from wso.brain.runtime import AutonomousWeightSpaceBrain, telemetry_label
from wso.brain.self_play import (
    _minimum_zero_event_sample,
    _passes_incident_gate,
    run_defensive_self_play,
    telemetry_dataset_hash,
)
from wso.brain.types import ActionReceipt, BrainAction, BrainConfig, BrainFeedback
from wso.telemetry.schemas import TransientLoRATelemetry


def _fixture():
    return generate_defensive_fixture(
        seed=71,
        baseline_count=32,
        labeled_per_class=8,
        test_per_class=6,
    )


def _config(**updates: object) -> BrainConfig:
    values: dict[str, object] = {
        "min_supervised_examples": 12,
        "min_positive_examples": 2,
        "classifier_epochs": 8,
        "random_seed": 71,
    }
    values.update(updates)
    return BrainConfig.model_validate(values)


class _SpoofingExecutor:
    name = "spoofing-executor"

    def __init__(self) -> None:
        self.calls: list[BrainAction] = []

    def supports(self, action: BrainAction) -> bool:
        return True

    def execute(
        self,
        action: BrainAction,
        record: TransientLoRATelemetry,
    ) -> ActionReceipt:
        self.calls.append(action)
        return ActionReceipt(
            requested_action=action,
            executed_action="quarantine",
            executor=self.name,
            success=True,
            summary="spoofed receipt",
        )


class _CountingExecutor(LocalDefensiveExecutor):
    def __init__(self) -> None:
        self.calls: list[BrainAction] = []

    def execute(
        self,
        action: BrainAction,
        record: TransientLoRATelemetry,
    ) -> ActionReceipt:
        self.calls.append(action)
        return super().execute(action, record)


class _MalformedExecutor:
    name = "malformed-executor"

    def supports(self, action: BrainAction) -> bool:
        return True

    def execute(
        self,
        action: BrainAction,
        record: TransientLoRATelemetry,
    ) -> Any:
        return {"executed_action": "quarantine", "success": True}


def test_feature_encoder_is_deterministic_and_ignores_labels() -> None:
    baseline, _, _ = _fixture()
    record = baseline[0]
    encoder = WeightSpaceFeatureEncoder(_config())

    first = encoder.encode(record)
    relabeled = record.model_copy(
        update={
            "prompt_hash": "different-prompt-hash",
            "provenance": {
                **record.provenance,
                "ground_truth": True,
                "operator_note": "this must never become a model feature",
            },
        }
    )
    second = encoder.encode(relabeled)

    assert np.array_equal(first.vector, second.vector)
    assert np.isfinite(first.vector).all()
    assert len(first.vector) == _config().feature_dimension
    assert first.raw_numeric["leading_energy_concentration"] > 0


def test_brain_learns_persists_and_only_accepts_verified_feedback(tmp_path: Path) -> None:
    baseline, labeled, audit = _fixture()
    state_dir = tmp_path / "brain"
    brain = AutonomousWeightSpaceBrain(state_dir, config=_config())

    status = brain.bootstrap(baseline=baseline, labeled=labeled)
    assert status.ready_for_learned_risk
    assert status.positive_examples == 8
    assert status.negative_examples == 33

    incident = next(record for record in audit if telemetry_label(record) is True)
    decision = brain.decide(incident, mode="shadow")
    assert decision.executed_action == "observe"
    assert not decision.autonomous
    assert decision.action_receipt_id
    assert (state_dir / "brain_state.json").exists()
    assert (state_dir / "actions.jsonl").exists()

    version = brain.model_version
    brain.feedback(
        BrainFeedback(
            decision_id=decision.decision_id,
            incident=True,
            verified=False,
        )
    )
    assert brain.model_version == version
    assert brain.status().memory_count == 0

    verified = BrainFeedback(
        decision_id=decision.decision_id,
        incident=True,
        verified=True,
        actual_action=decision.executed_action,
        action_succeeded=True,
        containment_succeeded=False,
    )
    updated = brain.feedback(verified)
    assert updated.model_version == version + 1
    assert updated.feedback_examples == 1
    assert updated.memory_count == 1
    assert updated.pending_decisions == 0

    # Exact retries are idempotent, while a second/conflicting disposition is rejected.
    assert brain.feedback(verified).model_version == updated.model_version
    with pytest.raises(ValueError, match="already has a verified disposition"):
        brain.feedback(
            BrainFeedback(
                decision_id=decision.decision_id,
                incident=False,
                verified=True,
                actual_action=decision.executed_action,
                action_succeeded=True,
            )
        )

    reloaded = AutonomousWeightSpaceBrain(state_dir)
    assert reloaded.status() == brain.status()
    assert np.array_equal(reloaded.classifier.weights, brain.classifier.weights)


def test_risk_model_ranks_fixture_incidents_above_benign(tmp_path: Path) -> None:
    baseline, labeled, audit = _fixture()
    brain = AutonomousWeightSpaceBrain(tmp_path / "brain", config=_config())
    brain.bootstrap(baseline=baseline, labeled=labeled)

    incident_scores = []
    benign_scores = []
    for record in audit:
        bundle = brain.encoder.encode(record)
        score = brain.risk_for_features(bundle.vector, bundle.numeric)
        (incident_scores if telemetry_label(record) else benign_scores).append(score)

    # The fixture intentionally overlaps benign regime shift with matched stress;
    # it must provide signal without becoming a saturated perfect benchmark.
    assert np.mean(incident_scores) > np.mean(benign_scores) + 0.05


def test_unknown_context_and_missing_real_provenance_fail_closed(tmp_path: Path) -> None:
    baseline, labeled, audit = _fixture()
    brain = AutonomousWeightSpaceBrain(tmp_path / "brain", config=_config())
    brain.bootstrap(baseline=baseline, labeled=labeled)
    base = audit[0]

    unknown = base.model_copy(update={"module_id": "previously.unseen.module"})
    unknown_decision = brain.decide(unknown, mode="enforce")
    assert unknown_decision.recommended_action in {"deep_scan", "human_review"}
    assert unknown_decision.executed_action in {"deep_scan", "human_review"}
    assert unknown_decision.executed_action != "allow"

    incomplete_real = base.model_copy(
        update={
            "run_id": "incomplete-real",
            "evidence_mode": "real",
            "source_kind": None,
            "source_path_hashes": {},
            "collection_timestamp": None,
        }
    )
    real_decision = brain.decide(incomplete_real, mode="enforce")
    assert real_decision.recommended_action == "human_review"
    assert real_decision.executed_action == "human_review"
    assert not real_decision.autonomous

    valid_provenance = {
        "evidence_mode": "real",
        "source_kind": "tflow",
        "source_path_hashes": {"adapter": "a" * 64},
        "collection_timestamp": base.timestamp,
        "runtime_framework": "trusted-runtime",
        "collector_version": "1.0.0",
    }
    real_baseline = [record.model_copy(update=valid_provenance) for record in baseline]
    real_brain = AutonomousWeightSpaceBrain(tmp_path / "real-brain", config=_config())
    real_brain.bootstrap(baseline=real_baseline, labeled=labeled)

    malformed_claim = real_baseline[0].model_copy(
        update={
            "run_id": "malformed-real-provenance",
            "source_path_hashes": {"adapter": "not-a-sha256"},
        }
    )
    malformed_decision = real_brain.decide(
        malformed_claim,
        mode="enforce",
        provenance_verified=True,
    )
    assert malformed_decision.executed_action == "human_review"
    assert not malformed_decision.provenance_verified
    assert "incomplete" in " ".join(malformed_decision.reasons).lower()

    unverified_claim = real_baseline[0].model_copy(update={"run_id": "unverified-real-provenance"})
    assert real_brain.decide(unverified_claim, mode="enforce").executed_action == "human_review"
    verified_claim = real_baseline[0].model_copy(update={"run_id": "verified-real-provenance"})
    verified_decision = real_brain.decide(
        verified_claim,
        mode="enforce",
        provenance_verified=True,
    )
    assert verified_decision.provenance_verified
    assert "incomplete" not in " ".join(verified_decision.reasons).lower()


def test_action_ceiling_and_local_executor_never_quarantine(tmp_path: Path) -> None:
    baseline, labeled, audit = _fixture()
    brain = AutonomousWeightSpaceBrain(
        tmp_path / "brain",
        config=_config(max_autonomous_action="quarantine"),
        executor=LocalDefensiveExecutor(),
    )
    brain.bootstrap(baseline=baseline, labeled=labeled)

    action, autonomous, reason, receipt = brain._execution(
        recommended="quarantine",
        mode="enforce",
        record=audit[0],
    )
    assert action == "human_review"
    assert not autonomous
    assert "no quarantine capability" in reason.lower()
    assert receipt.executed_action == "human_review"

    action, autonomous, _, receipt = brain._execution(
        recommended="deep_scan",
        mode="enforce",
        record=audit[0],
    )
    assert action == "deep_scan"
    assert autonomous
    assert receipt.evidence["raw_tensor_persisted"] is False
    assert receipt.evidence["prompt_payload_used"] is False

    oversized = audit[0].model_copy(update={"singular_values": [1.0] * 4097})
    action, autonomous, _, receipt = brain._execution(
        recommended="deep_scan",
        mode="enforce",
        record=oversized,
    )
    assert action == "human_review"
    assert not autonomous
    assert receipt.executor == "brain_fail_closed_v2"


def test_self_play_emits_only_aggregate_report_and_uses_holdout_gate(tmp_path: Path) -> None:
    baseline, labeled, audit = _fixture()
    brain = AutonomousWeightSpaceBrain(tmp_path / "brain", config=_config())
    brain.bootstrap(baseline=baseline, labeled=labeled)
    incidents = [record for record in audit if telemetry_label(record) is True]
    benign = [record for record in audit if telemetry_label(record) is False]

    report = run_defensive_self_play(
        brain,
        incidents=incidents,
        benign_holdout=benign,
        candidates_per_incident=8,
        promote=False,
    )
    payload = report.model_dump()
    assert report.generated_candidates == len(incidents) * 8
    assert not report.promoted
    assert "Feature-space defensive hardening only" in report.safety_boundary
    assert "features" not in payload
    assert "candidates" not in payload
    assert report.hard_variant_seed_evasion_upper_bound >= 0.0
    assert not _passes_incident_gate(
        ceiling=0.20,
        original_evasion_before=0.0,
        original_evasion_after=0.0,
        original_evasion_upper=0.19,
        original_mean_before=0.7,
        original_mean_after=0.7,
        hard_seed_evasion_before=0.0,
        hard_seed_evasion_after=0.7,
        hard_seed_evasion_upper=0.9,
    )


def test_end_to_end_demo_writes_auditable_artifacts(tmp_path: Path) -> None:
    result = run_brain_demo(tmp_path / "demo", seed=83)

    assert result["evidence_mode"] == "synthetic_fixture"
    assert result["auroc"] >= 0.8
    assert result["final_status"]["feedback_examples"] > 0  # type: ignore[index]
    assert (tmp_path / "demo" / "brain" / "decisions.jsonl").exists()
    action = next(
        json.loads(line)
        for line in (tmp_path / "demo" / "brain" / "actions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if json.loads(line).get("phase") == "outcome"
    )
    assert "receipt" in action
    assert action["receipt"]["executor"] == "brain_shadow_v2"
    assert action["receipt"]["evidence"]["external_executor_invoked"] is False


def test_bootstrap_rejects_reinitialization_and_pseudo_labels(tmp_path: Path) -> None:
    baseline, _, audit = _fixture()
    pseudo = audit[0].model_copy(update={"provenance": {"model_prediction": "attack"}})
    brain = AutonomousWeightSpaceBrain(tmp_path / "brain", config=_config())
    status = brain.bootstrap(baseline=baseline, labeled=[pseudo])
    assert status.positive_examples == 0
    assert not status.ready_for_learned_risk
    with pytest.raises(FileExistsError):
        brain.bootstrap(baseline=baseline)


def test_extreme_finite_spectra_and_tied_baselines_remain_finite(tmp_path: Path) -> None:
    baseline, _, _ = _fixture()
    extreme = baseline[0].model_copy(
        update={
            "singular_values": [1e308, 1e308, 5e307],
            "frobenius_norm": 1e308,
            "effective_rank": 3.0,
            "spectral_gap": 0.0,
        }
    )
    bundle = WeightSpaceFeatureEncoder(_config()).encode(extreme)
    assert np.isfinite(bundle.vector).all()
    assert np.isfinite(bundle.numeric).all()

    extreme_baseline = [
        extreme.model_copy(update={"run_id": f"extreme-{index}"}) for index in range(12)
    ]
    brain = AutonomousWeightSpaceBrain(tmp_path / "extreme", config=_config())
    brain.bootstrap(baseline=extreme_baseline)
    reloaded = AutonomousWeightSpaceBrain(tmp_path / "extreme")
    assert np.isfinite(reloaded.classifier.weights).all()
    json.loads(
        (tmp_path / "extreme" / "brain_state.json").read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )

    novelty = MultiTimescaleNovelty(30, calibration_limit=128)
    identical = np.zeros(30, dtype=np.float64)
    novelty.fit_baseline([identical.copy() for _ in range(96)])
    score, _, _ = novelty.scores(identical)
    assert score < 0.01

    negative_rank = baseline[0].model_copy(update={"rank_bound": -1})
    with pytest.raises(ValueError, match="rank_bound"):
        WeightSpaceFeatureEncoder(_config()).encode(negative_rank)
    oversized_spectrum = baseline[0].model_copy(update={"singular_values": [1.0] * 4097})
    with pytest.raises(ValueError, match="singular_values"):
        WeightSpaceFeatureEncoder(_config()).encode(oversized_spectrum)

    class IterationTrap(list[object]):
        def __len__(self) -> int:
            return MAX_METADATA_NODES + 1

        def __iter__(self):
            raise AssertionError("oversized metadata children must not be queued")

    assert not _metadata_within_budget(IterationTrap())
    assert not _metadata_within_budget([1 << 1024] * (MAX_METADATA_NODES - 1))
    assert not _metadata_within_budget({"negative": -(1 << (MAX_METADATA_BYTES + 1))})
    assert not _metadata_within_budget({"expanded": frozenset(range(100_000))})
    assert not _metadata_within_budget({"unsupported": object()})
    assert not _metadata_within_budget({"escaped": "\x00" * (MAX_METADATA_BYTES // 6 + 1)})


def test_config_and_receipts_are_immutable_and_reject_unsafe_values() -> None:
    with pytest.raises(ValueError):
        BrainConfig(classifier_epochs=0)
    with pytest.raises(ValueError):
        BrainConfig(pending_decision_limit=-1)
    with pytest.raises(ValueError):
        BrainConfig(review_risk_threshold=0.80, high_risk_threshold=0.82)
    with pytest.raises(ValueError):
        BrainConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match="benign self-play records"):
        BrainConfig(target_false_positive_rate=0.01, promotion_confidence=0.95)
    with pytest.raises(ValueError, match="held-out self-play incident seeds"):
        BrainConfig(max_self_play_evasion_rate=0.02, promotion_confidence=0.95)
    with pytest.raises(ValueError, match="benign self-play records"):
        BrainConfig(target_false_positive_rate=1e-20)
    with pytest.raises(ValueError, match="held-out self-play incident seeds"):
        BrainConfig(max_self_play_evasion_rate=1e-20)
    smallest_rate = math.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="benign self-play records"):
        BrainConfig(target_false_positive_rate=smallest_rate)
    with pytest.raises(ValueError, match="held-out self-play incident seeds"):
        BrainConfig(max_self_play_evasion_rate=smallest_rate)
    assert _minimum_zero_event_sample(smallest_rate, 0.95) > MAX_METADATA_NODES

    config = BrainConfig()
    with pytest.raises(ValidationError):
        config.max_autonomous_action = "quarantine"  # type: ignore[misc]
    receipt = ActionReceipt(
        requested_action="observe",
        executed_action="observe",
        executor="test",
        success=True,
        summary="test",
    )
    with pytest.raises(ValidationError):
        receipt.success = False  # type: ignore[misc]


def test_context_identity_includes_adapter_source_runtime_and_only_baseline(tmp_path: Path) -> None:
    baseline, labeled, audit = _fixture()
    incident_only = next(
        record for record in labeled if telemetry_label(record) is True
    ).model_copy(update={"adapter_id": "incident-only-adapter"})
    brain = AutonomousWeightSpaceBrain(tmp_path / "brain", config=_config())
    brain.bootstrap(baseline=baseline, labeled=[*labeled, incident_only])

    changed = audit[0].model_copy(
        update={
            "run_id": "new-security-context",
            "adapter_id": "never-seen-adapter",
            "source_kind": "external-runtime",
            "runtime_framework": "new-runtime-v9",
        }
    )
    changed_decision = brain.decide(changed, mode="enforce")
    assert changed_decision.recommended_action in {"deep_scan", "human_review"}
    assert changed_decision.executed_action != "allow"

    incident_probe = incident_only.model_copy(update={"run_id": "incident-context-probe"})
    incident_decision = brain.decide(incident_probe, mode="enforce")
    assert incident_decision.recommended_action in {"deep_scan", "human_review"}
    assert "not present in the trusted baseline" in " ".join(incident_decision.reasons)


def test_shadow_never_invokes_executor_and_spoofed_receipts_fail_closed(
    tmp_path: Path,
) -> None:
    baseline, labeled, audit = _fixture()
    executor = _SpoofingExecutor()
    brain = AutonomousWeightSpaceBrain(
        tmp_path / "brain",
        config=_config(),
        executor=executor,
    )
    brain.bootstrap(baseline=baseline, labeled=labeled)

    shadow = brain.decide(audit[0], mode="shadow")
    assert shadow.executed_action == "observe"
    assert executor.calls == []
    shadow_retry = brain.decide(audit[0], mode="shadow")
    assert shadow_retry.decision_id == shadow.decision_id
    assert executor.calls == []

    action, autonomous, _, receipt = brain._execution(
        recommended="deep_scan",
        mode="enforce",
        record=audit[0],
    )
    assert executor.calls == ["deep_scan"]
    assert action == "human_review"
    assert not autonomous
    assert receipt.executor == "brain_fail_closed_v2"
    assert receipt.executed_action == "human_review"

    malformed = AutonomousWeightSpaceBrain(
        tmp_path / "malformed",
        config=_config(),
        executor=_MalformedExecutor(),
    )
    malformed.bootstrap(baseline=baseline, labeled=labeled)
    action, autonomous, reason, receipt = malformed._execution(
        recommended="deep_scan",
        mode="enforce",
        record=audit[0],
    )
    assert action == "human_review"
    assert not autonomous
    assert "failed validation" in reason
    assert receipt.executor == "brain_fail_closed_v2"

    ledger_before = (tmp_path / "brain" / "actions.jsonl").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="brain mode"):
        brain.decide(audit[0], mode="shdaow")  # type: ignore[arg-type]
    assert (tmp_path / "brain" / "actions.jsonl").read_text(encoding="utf-8") == ledger_before
    assert executor.calls == ["deep_scan"]


def test_stale_instances_serialize_feedback_without_lost_updates(tmp_path: Path) -> None:
    baseline, labeled, audit = _fixture()
    state_dir = tmp_path / "brain"
    brain = AutonomousWeightSpaceBrain(state_dir, config=_config())
    brain.bootstrap(baseline=baseline, labeled=labeled)
    first = brain.decide(audit[0], mode="shadow")
    second = brain.decide(audit[1], mode="shadow")
    stale_a = AutonomousWeightSpaceBrain(state_dir)
    stale_b = AutonomousWeightSpaceBrain(state_dir)

    first_label = bool(telemetry_label(audit[0]))
    second_label = bool(telemetry_label(audit[1]))
    stale_a.feedback(
        BrainFeedback(
            decision_id=first.decision_id,
            incident=first_label,
            actual_action=first.executed_action,
            action_succeeded=True,
            containment_succeeded=False if first_label else None,
        )
    )
    stale_b.feedback(
        BrainFeedback(
            decision_id=second.decision_id,
            incident=second_label,
            actual_action=second.executed_action,
            action_succeeded=True,
            containment_succeeded=False if second_label else None,
        )
    )

    final = AutonomousWeightSpaceBrain(state_dir).status()
    assert final.model_version == 3
    assert final.feedback_examples == 2
    assert final.pending_decisions == 0
    applied = [
        json.loads(line)
        for line in (state_dir / "feedback.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("phase") == "applied"
    ]
    assert len(applied) == 2


def test_torn_private_ledger_is_skipped_and_brain_files_are_private(tmp_path: Path) -> None:
    baseline, labeled, audit = _fixture()
    state_dir = tmp_path / "brain"
    brain = AutonomousWeightSpaceBrain(
        state_dir,
        config=_config(pending_decision_limit=1),
    )
    brain.bootstrap(baseline=baseline, labeled=labeled)
    first = brain.decide(audit[0], mode="shadow")
    second = brain.decide(audit[1], mode="shadow")
    brain.decide(audit[2], mode="shadow")

    private = state_dir / "learning_ledger.jsonl"
    lines = private.read_text(encoding="utf-8").splitlines()
    private.write_text("\n".join([lines[0], "{torn", *lines[1:]]) + "\n", encoding="utf-8")
    private.chmod(0o600)
    assert first.decision_id not in brain.pending
    assert second.decision_id not in brain.pending
    label = bool(telemetry_label(audit[1]))
    updated = brain.feedback(
        BrainFeedback(
            decision_id=second.decision_id,
            incident=label,
            actual_action=second.executed_action,
            action_succeeded=True,
            containment_succeeded=False if label else None,
        )
    )
    assert updated.feedback_examples == 1

    for filename in (
        "brain_state.json",
        "decisions.jsonl",
        "learning_ledger.jsonl",
        "feedback.jsonl",
        "actions.jsonl",
        ".brain.lock",
    ):
        assert stat.S_IMODE((state_dir / filename).stat().st_mode) == 0o600


def test_self_play_rejects_contamination_and_noop_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, labeled, audit = generate_defensive_fixture(
        seed=91,
        baseline_count=32,
        labeled_per_class=8,
        test_per_class=64,
    )
    incidents = [record for record in audit if telemetry_label(record) is True]
    benign = [record for record in audit if telemetry_label(record) is False]
    brain = AutonomousWeightSpaceBrain(tmp_path / "brain", config=_config(random_seed=91))
    brain.bootstrap(baseline=baseline, labeled=labeled)

    contaminated = benign.copy()
    contaminated[0] = contaminated[0].model_copy(
        update={"provenance": {**contaminated[0].provenance, "ground_truth": True}}
    )
    with pytest.raises(ValueError, match="explicit benign"):
        run_defensive_self_play(
            brain,
            incidents=incidents,
            benign_holdout=contaminated,
            candidates_per_incident=8,
            promote=False,
        )

    overlap = incidents[0].model_copy(
        update={"provenance": {**incidents[0].provenance, "ground_truth": False}}
    )
    with pytest.raises(ValueError, match="disjoint"):
        run_defensive_self_play(
            brain,
            incidents=incidents,
            benign_holdout=[overlap, *benign[1:]],
            candidates_per_incident=8,
            promote=False,
        )

    feature_overlap = incidents[0].model_copy(
        update={
            "run_id": "different-run-id-same-features",
            "provenance": {**incidents[0].provenance, "ground_truth": False},
        }
    )
    with pytest.raises(ValueError, match="encoded feature space"):
        run_defensive_self_play(
            brain,
            incidents=incidents,
            benign_holdout=[feature_overlap, *benign[1:]],
            candidates_per_incident=8,
            promote=False,
        )

    def no_op_fit(self: OnlineRiskClassifier, *args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(OnlineRiskClassifier, "fit", no_op_fit)
    version = brain.model_version
    report = run_defensive_self_play(
        brain,
        incidents=incidents,
        benign_holdout=benign,
        candidates_per_incident=8,
        promote=True,
        expected_incident_dataset_hash=telemetry_dataset_hash(incidents),
        expected_benign_holdout_hash=telemetry_dataset_hash(benign),
    )
    assert not report.promoted
    assert report.classifier_hash_before == report.classifier_hash_after
    assert brain.model_version == version
    assert report.benign_holdout_examples == len(benign)
    assert report.benign_false_positive_rate_after >= 0.0


def test_self_play_never_trains_on_benign_holdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, labeled, audit = _fixture()
    brain = AutonomousWeightSpaceBrain(tmp_path / "brain", config=_config())
    brain.bootstrap(baseline=baseline, labeled=labeled)
    incidents = [record for record in audit if telemetry_label(record) is True]
    benign = [record for record in audit if telemetry_label(record) is False]
    original_fit = OnlineRiskClassifier.fit
    captured: list[tuple[np.ndarray, bool]] = []

    def recording_fit(
        self: OnlineRiskClassifier,
        examples: list[tuple[np.ndarray, bool]],
        *,
        epochs: int,
        seed: int,
    ) -> None:
        captured.extend(examples)
        original_fit(self, examples, epochs=epochs, seed=seed)

    monkeypatch.setattr(OnlineRiskClassifier, "fit", recording_fit)
    run_defensive_self_play(
        brain,
        incidents=incidents,
        benign_holdout=benign,
        candidates_per_incident=8,
        promote=False,
    )
    assert captured
    assert all(label is True for _, label in captured)
    holdout_vectors = [brain.encoder.encode(record).vector for record in benign]
    assert not any(
        np.array_equal(training_vector, holdout_vector)
        for training_vector, _ in captured
        for holdout_vector in holdout_vectors
    )


def test_uninitialized_self_play_and_ambiguous_numeric_labels_fail_closed(
    tmp_path: Path,
) -> None:
    baseline, _, audit = _fixture()
    fresh = AutonomousWeightSpaceBrain(tmp_path / "fresh", config=_config())
    incidents = [record for record in audit if telemetry_label(record) is True]
    benign = [record for record in audit if telemetry_label(record) is False]
    with pytest.raises(RuntimeError, match="does not exist"):
        run_defensive_self_play(
            fresh,
            incidents=incidents,
            benign_holdout=benign,
            promote=False,
        )
    for invalid in (-1, 2, float("nan")):
        record = baseline[0].model_copy(update={"provenance": {"ground_truth": invalid}})
        assert telemetry_label(record) is None
    contradictory = baseline[0].model_copy(
        update={"provenance": {"ground_truth": True, "audit_label": False}}
    )
    assert telemetry_label(contradictory) is None


def test_classifier_precision_counts_unique_examples_not_epochs() -> None:
    examples = [(np.full(64, float(index % 3) / 3.0), bool(index % 2)) for index in range(20)]
    one_epoch = OnlineRiskClassifier(64, learning_rate=0.08, l2=0.002)
    many_epochs = OnlineRiskClassifier(64, learning_rate=0.08, l2=0.002)
    one_epoch.fit(examples, epochs=1, seed=1)
    many_epochs.fit(examples, epochs=24, seed=1)
    assert one_epoch.examples == many_epochs.examples == 20
    assert float(many_epochs.precision.sum()) < float(one_epoch.precision.sum()) * 2.0


def test_cli_output_guard_rejects_hardlinks_and_inputs(tmp_path: Path) -> None:
    baseline, labeled, _ = _fixture()
    state_dir = tmp_path / "brain"
    brain = AutonomousWeightSpaceBrain(state_dir, config=_config())
    brain.bootstrap(baseline=baseline, labeled=labeled)

    state_alias = tmp_path / "state-alias.json"
    os.link(state_dir / "brain_state.json", state_alias)
    with pytest.raises(ValueError, match="brain state"):
        _validate_output_path(state_alias, state_dir)

    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(baseline[0].model_dump_json() + "\n", encoding="utf-8")
    telemetry_alias = tmp_path / "telemetry-alias.jsonl"
    os.link(telemetry, telemetry_alias)
    with pytest.raises(ValueError, match="input telemetry"):
        _validate_output_path(telemetry_alias, state_dir, inputs=[telemetry])

    nested_transaction = state_dir / "action_attempts" / "aa" / "attempt.json"
    with pytest.raises(ValueError, match="state directory"):
        _validate_output_path(nested_transaction, state_dir)

    safe_parent = tmp_path / "safe-output"
    safe_parent.mkdir()
    parent_alias = tmp_path / "output-parent-alias"
    parent_alias.symlink_to(safe_parent, target_is_directory=True)
    raced_output = parent_alias / "brain_state.json"
    _validate_output_path(raced_output, state_dir)
    original_state = (state_dir / "brain_state.json").read_text(encoding="utf-8")
    parent_alias.unlink()
    parent_alias.symlink_to(state_dir, target_is_directory=True)
    with pytest.raises(ValueError, match="state directory"):
        _atomic_write_text(
            raced_output,
            "replaced\n",
            state_dir=state_dir,
        )
    assert (state_dir / "brain_state.json").read_text(encoding="utf-8") == original_state


def test_brain_cli_loader_enforces_record_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, _, _ = _fixture()
    telemetry = tmp_path / "too-many.jsonl"
    telemetry.write_text(
        "".join(record.model_dump_json() + "\n" for record in baseline[:3]),
        encoding="utf-8",
    )
    monkeypatch.setattr(brain_cli, "MAX_BRAIN_TELEMETRY_RECORDS", 2)
    with pytest.raises(ValueError, match="exceeds 2 records"):
        _load_brain_telemetry(telemetry)


def test_observation_idempotency_is_canonical_mode_aware_and_conflict_safe(
    tmp_path: Path,
) -> None:
    baseline, labeled, audit = _fixture()
    state_dir = tmp_path / "brain"
    executor = _CountingExecutor()
    brain = AutonomousWeightSpaceBrain(state_dir, config=_config(), executor=executor)
    brain.bootstrap(baseline=baseline, labeled=labeled)
    record = audit[0]

    shadow = brain.decide(record, mode="shadow")
    enforce = brain.decide(record, mode="enforce")
    assert shadow.decision_id != enforce.decision_id
    assert executor.calls
    calls = executor.calls.copy()

    canonical_retry = record.model_copy(
        update={
            "timestamp": time.time() + 60,
            "provenance": dict(reversed(list(record.provenance.items()))),
        }
    )
    retried = brain.decide(canonical_retry, mode="enforce")
    assert retried.decision_id == enforce.decision_id
    assert executor.calls == calls

    changed_payload = record.model_copy(update={"prompt_hash": "changed-payload"})
    with pytest.raises(ValueError, match="different telemetry payload"):
        brain.decide(changed_payload, mode="enforce")
    assert executor.calls == calls

    changed_source = record.model_copy(
        update={"source_path_hashes": {"adapter": "different-source-hash"}}
    )
    with pytest.raises(ValueError, match="different telemetry payload"):
        brain.decide(changed_source, mode="enforce")
    assert executor.calls == calls


def test_bootstrap_rejects_incident_and_ambiguous_baseline_labels(tmp_path: Path) -> None:
    baseline, labeled, _ = _fixture()
    incident = next(record for record in labeled if telemetry_label(record) is True)
    with pytest.raises(ValueError, match="labeled incident"):
        AutonomousWeightSpaceBrain(tmp_path / "incident", config=_config()).bootstrap(
            baseline=[incident]
        )

    ambiguous = baseline[0].model_copy(
        update={"provenance": {"ground_truth": False, "audit_label": True}}
    )
    with pytest.raises(ValueError, match="ambiguous"):
        AutonomousWeightSpaceBrain(tmp_path / "ambiguous", config=_config()).bootstrap(
            baseline=[ambiguous]
        )


def test_corrupt_persisted_learning_state_is_rejected(tmp_path: Path) -> None:
    baseline, labeled, audit = _fixture()
    state_dir = tmp_path / "brain"
    brain = AutonomousWeightSpaceBrain(state_dir, config=_config())
    brain.bootstrap(baseline=baseline, labeled=labeled)
    incident = next(record for record in audit if telemetry_label(record) is True)
    decision = brain.decide(incident, mode="shadow")
    brain.feedback(
        BrainFeedback(
            decision_id=decision.decision_id,
            incident=True,
            actual_action=decision.executed_action,
            action_succeeded=True,
            containment_succeeded=False,
        )
    )
    state_path = state_dir / "brain_state.json"
    original = json.loads(state_path.read_text(encoding="utf-8"))

    corrupted = json.loads(json.dumps(original))
    corrupted["classifier"]["learning_rate"] = float("nan")
    state_path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="learning rate"):
        AutonomousWeightSpaceBrain(state_dir)

    corrupted = json.loads(json.dumps(original))
    corrupted["classifier"]["examples"] = 999
    state_path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="counts"):
        AutonomousWeightSpaceBrain(state_dir)

    corrupted = json.loads(json.dumps(original))
    corrupted["memory"]["memories"][0]["centroid"] = [0.0]
    state_path.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="centroid dimension"):
        AutonomousWeightSpaceBrain(state_dir)


def test_action_outcome_recovers_after_state_save_failure_without_reexecution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, labeled, audit = _fixture()
    executor = _CountingExecutor()
    brain = AutonomousWeightSpaceBrain(
        tmp_path / "brain",
        config=_config(),
        executor=executor,
    )
    brain.bootstrap(baseline=baseline, labeled=labeled)
    original_save = brain._save

    def fail_save() -> None:
        raise OSError("injected state-save failure")

    monkeypatch.setattr(brain, "_save", fail_save)
    with pytest.raises(OSError, match="injected"):
        brain.decide(audit[0], mode="enforce")
    assert len(executor.calls) == 1

    monkeypatch.setattr(brain, "_save", original_save)
    recovered = brain.decide(audit[0], mode="enforce")
    assert recovered.run_id == audit[0].run_id
    assert len(executor.calls) == 1
    assert recovered.decision_id in brain.pending


def test_feedback_retries_repair_state_and_applied_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, labeled, audit = _fixture()
    state_dir = tmp_path / "brain"
    brain = AutonomousWeightSpaceBrain(state_dir, config=_config())
    brain.bootstrap(baseline=baseline, labeled=labeled)
    incident = next(record for record in audit if telemetry_label(record) is True)
    decision = brain.decide(incident, mode="shadow")
    feedback = BrainFeedback(
        decision_id=decision.decision_id,
        incident=True,
        actual_action=decision.executed_action,
        action_succeeded=True,
        containment_succeeded=False,
    )
    version = brain.model_version
    original_append = brain._append_jsonl

    def fail_applied(path: Path, payload: dict[str, Any]) -> None:
        if payload.get("phase") == "applied":
            raise OSError("injected applied-ledger failure")
        original_append(path, payload)

    monkeypatch.setattr(brain, "_append_jsonl", fail_applied)
    with pytest.raises(OSError, match="applied-ledger"):
        brain.feedback(feedback)
    assert brain.model_version == version + 1

    monkeypatch.setattr(brain, "_append_jsonl", original_append)
    repaired = brain.feedback(feedback)
    assert repaired.model_version == version + 1
    applied = [
        json.loads(line)
        for line in (state_dir / "feedback.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("phase") == "applied"
    ]
    assert len(applied) == 1


def test_interrupted_feedback_rebases_after_unrelated_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, labeled, audit = _fixture()
    state_dir = tmp_path / "brain"
    brain = AutonomousWeightSpaceBrain(state_dir, config=_config())
    brain.bootstrap(baseline=baseline, labeled=labeled)
    first_decision = brain.decide(audit[0], mode="shadow")
    second_decision = brain.decide(audit[1], mode="shadow")
    first_feedback = BrainFeedback(
        decision_id=first_decision.decision_id,
        incident=bool(telemetry_label(audit[0])),
    )
    second_feedback = BrainFeedback(
        decision_id=second_decision.decision_id,
        incident=bool(telemetry_label(audit[1])),
    )
    version = brain.model_version
    original_save = brain._save

    def fail_save() -> None:
        raise OSError("injected feedback state-save failure")

    monkeypatch.setattr(brain, "_save", fail_save)
    with pytest.raises(OSError, match="feedback state-save"):
        brain.feedback(first_feedback)
    monkeypatch.setattr(brain, "_save", original_save)

    brain.feedback(second_feedback)
    status = brain.feedback(first_feedback)
    assert status.model_version == version + 2
    assert status.feedback_examples == 2


def test_cli_feedback_id_supports_exact_retry(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline, labeled, audit = _fixture()
    state_dir = tmp_path / "brain"
    brain = AutonomousWeightSpaceBrain(state_dir, config=_config())
    brain.bootstrap(baseline=baseline, labeled=labeled)
    record = next(item for item in audit if telemetry_label(item) is True)
    decision = brain.decide(record, mode="shadow")
    argv = [
        "feedback",
        "--state-dir",
        str(state_dir),
        "--decision-id",
        decision.decision_id,
        "--feedback-id",
        "feedback-cli-stable-retry",
        "--incident",
    ]
    assert brain_cli_main(argv) == 0
    capsys.readouterr()
    assert brain_cli_main(argv) == 0
    capsys.readouterr()
    assert AutonomousWeightSpaceBrain(state_dir).status().feedback_examples == 1


def test_self_play_promotion_requires_hash_sample_power_and_absolute_incident_gate(
    tmp_path: Path,
) -> None:
    baseline, labeled, audit = generate_defensive_fixture(
        seed=117,
        baseline_count=32,
        labeled_per_class=8,
        test_per_class=20,
    )
    brain = AutonomousWeightSpaceBrain(tmp_path / "brain", config=_config(random_seed=117))
    brain.bootstrap(baseline=baseline, labeled=labeled)
    incidents = [record for record in audit if telemetry_label(record) is True]
    benign = [record for record in audit if telemetry_label(record) is False]
    with pytest.raises(ValueError, match="incident-dataset hash"):
        run_defensive_self_play(
            brain,
            incidents=incidents,
            benign_holdout=benign,
            candidates_per_incident=8,
            promote=True,
        )
    with pytest.raises(ValueError, match="benign-holdout hash"):
        run_defensive_self_play(
            brain,
            incidents=incidents,
            benign_holdout=benign,
            candidates_per_incident=8,
            promote=True,
            expected_incident_dataset_hash=telemetry_dataset_hash(incidents),
        )
    with pytest.raises(ValueError, match="at least 59 benign"):
        run_defensive_self_play(
            brain,
            incidents=incidents,
            benign_holdout=benign,
            candidates_per_incident=8,
            promote=True,
            expected_incident_dataset_hash=telemetry_dataset_hash(incidents),
            expected_benign_holdout_hash=telemetry_dataset_hash(benign),
        )


def test_self_play_rejects_unbounded_seed_and_candidate_work(tmp_path: Path) -> None:
    baseline, labeled, audit = _fixture()
    brain = AutonomousWeightSpaceBrain(tmp_path / "brain", config=_config())
    brain.bootstrap(baseline=baseline, labeled=labeled)
    incident = next(record for record in audit if telemetry_label(record) is True)
    benign = next(record for record in audit if telemetry_label(record) is False)
    oversized_incidents = [
        incident.model_copy(update={"run_id": f"oversized-incident-{index}"})
        for index in range(257)
    ]
    with pytest.raises(ValueError, match="at most 256 incident"):
        run_defensive_self_play(
            brain,
            incidents=oversized_incidents,
            benign_holdout=[benign],
            promote=False,
        )
    with pytest.raises(ValueError, match="at most 128 candidates"):
        run_defensive_self_play(
            brain,
            incidents=[
                incident.model_copy(update={"run_id": "candidate-bound-1"}),
                incident.model_copy(update={"run_id": "candidate-bound-2"}),
            ],
            benign_holdout=[benign],
            candidates_per_incident=129,
            promote=False,
        )


def test_tflow_capture_windows_are_process_serialized() -> None:
    module = ModuleType("fake_tflow_capture")

    def original(factor_dicts: list[dict[str, Any]], weights: list[Any]) -> dict[str, Any]:
        return {"factors": factor_dicts, "weights": weights}

    module.build_weighted_lora_entries = original  # type: ignore[attr-defined]
    adapter = OfficialTFlowAdapter(TFlowInferenceConfig())
    adapter._tflow_module = module  # type: ignore[assignment]
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first_window() -> None:
        with adapter._capture_weighted_entries([]):
            first_entered.set()
            assert release_first.wait(2.0)

    def second_window() -> None:
        assert first_entered.wait(2.0)
        with adapter._capture_weighted_entries([]):
            second_entered.set()

    first = threading.Thread(target=first_window)
    second = threading.Thread(target=second_window)
    first.start()
    second.start()
    assert first_entered.wait(2.0)
    assert not second_entered.wait(0.05)
    release_first.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)
    assert not first.is_alive()
    assert not second.is_alive()
    assert second_entered.is_set()
    assert module.build_weighted_lora_entries is original  # type: ignore[attr-defined]


def test_tflow_capture_rejects_same_thread_reentrancy() -> None:
    module = ModuleType("fake_reentrant_tflow_capture")

    def original(factor_dicts: list[dict[str, Any]], weights: list[Any]) -> dict[str, Any]:
        return {"factors": factor_dicts, "weights": weights}

    module.build_weighted_lora_entries = original  # type: ignore[attr-defined]
    adapter = OfficialTFlowAdapter(TFlowInferenceConfig())
    adapter._tflow_module = module  # type: ignore[assignment]
    outer: list[dict[str, Any]] = []
    with adapter._capture_weighted_entries(outer):
        module.build_weighted_lora_entries([{"outer": 1}], [1.0])  # type: ignore[attr-defined]
        with (
            pytest.raises(RuntimeError, match="nested TFlow capture"),
            adapter._capture_weighted_entries([]),
        ):
            pass
        module.build_weighted_lora_entries([{"outer": 2}], [1.0])  # type: ignore[attr-defined]
    assert len(outer) == 2
    assert module.build_weighted_lora_entries is original  # type: ignore[attr-defined]


def test_tflow_capture_restores_after_interrupted_hook_install() -> None:
    class OneShotInterruptModule(ModuleType):
        interrupt = False

        def __setattr__(self, name: str, value: object) -> None:
            super().__setattr__(name, value)
            if name == "build_weighted_lora_entries" and self.interrupt:
                self.interrupt = False
                raise KeyboardInterrupt

    module = OneShotInterruptModule("interrupting_tflow_capture")

    def original(factor_dicts: list[dict[str, Any]], weights: list[Any]) -> dict[str, Any]:
        return {"factors": factor_dicts, "weights": weights}

    module.build_weighted_lora_entries = original  # type: ignore[attr-defined]
    adapter = OfficialTFlowAdapter(TFlowInferenceConfig())
    adapter._tflow_module = module  # type: ignore[assignment]
    module.interrupt = True
    with pytest.raises(KeyboardInterrupt), adapter._capture_weighted_entries([]):
        pass
    assert module.build_weighted_lora_entries is original  # type: ignore[attr-defined]
    with adapter._capture_weighted_entries([]):
        pass
    assert module.build_weighted_lora_entries is original  # type: ignore[attr-defined]


def test_tflow_capture_retries_an_interrupted_hook_restore() -> None:
    class RestoreInterruptModule(ModuleType):
        original_hook: object | None = None
        interrupt_restore = False

        def __setattr__(self, name: str, value: object) -> None:
            if (
                name == "build_weighted_lora_entries"
                and value is self.original_hook
                and self.interrupt_restore
            ):
                self.interrupt_restore = False
                raise KeyboardInterrupt
            super().__setattr__(name, value)

    module = RestoreInterruptModule("restore_interrupt_tflow_capture")

    def original(factor_dicts: list[dict[str, Any]], weights: list[Any]) -> dict[str, Any]:
        return {"factors": factor_dicts, "weights": weights}

    module.build_weighted_lora_entries = original  # type: ignore[attr-defined]
    module.original_hook = original
    adapter = OfficialTFlowAdapter(TFlowInferenceConfig())
    adapter._tflow_module = module  # type: ignore[assignment]
    module.interrupt_restore = True
    with pytest.raises(KeyboardInterrupt), adapter._capture_weighted_entries([]):
        pass
    assert module.build_weighted_lora_entries is original  # type: ignore[attr-defined]
    with adapter._capture_weighted_entries([]):
        pass


def test_orphaned_action_intent_requires_operator_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, labeled, audit = _fixture()
    executor = _CountingExecutor()
    brain = AutonomousWeightSpaceBrain(
        tmp_path / "brain",
        config=_config(),
        executor=executor,
    )
    brain.bootstrap(baseline=baseline, labeled=labeled)
    original_execution = brain._execution

    def interrupt_execution(**kwargs: Any) -> Any:
        raise RuntimeError("injected crash before executor")

    monkeypatch.setattr(brain, "_execution", interrupt_execution)
    with pytest.raises(RuntimeError, match="injected crash"):
        brain.decide(audit[0], mode="enforce")
    monkeypatch.setattr(brain, "_execution", original_execution)
    with pytest.raises(RuntimeError, match="manual reconciliation"):
        brain.decide(audit[0], mode="enforce")

    blocked = brain.reconcile_action_attempt(audit[0], mode="enforce", attest_not_executed=False)
    assert blocked["phase"] == "blocked_unknown"
    with pytest.raises(RuntimeError, match="manual reconciliation"):
        brain.decide(audit[0], mode="enforce")

    # A separate observation demonstrates the explicit safe-to-retry path.
    monkeypatch.setattr(brain, "_execution", interrupt_execution)
    with pytest.raises(RuntimeError, match="injected crash"):
        brain.decide(audit[1], mode="enforce")
    monkeypatch.setattr(brain, "_execution", original_execution)
    abandoned = brain.reconcile_action_attempt(audit[1], mode="enforce", attest_not_executed=True)
    assert abandoned["phase"] == "abandoned_not_executed"
    retried = brain.decide(audit[1], mode="enforce")
    assert retried.run_id == audit[1].run_id
    assert len(executor.calls) == 1
