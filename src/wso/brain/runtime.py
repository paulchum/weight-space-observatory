"""Persistent observe-decide-act-learn loop for the Adapter Trust Gateway."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np

from wso.brain.actions import BrainActionExecutor, LocalDefensiveExecutor
from wso.brain.features import (
    FEATURE_SCHEMA,
    NUMERIC_FEATURE_NAMES,
    FeatureBundle,
    WeightSpaceFeatureEncoder,
    validate_telemetry_budget,
)
from wso.brain.learning import (
    ConstrainedContextualBandit,
    MultiTimescaleNovelty,
    OnlineRiskClassifier,
    StrategyBank,
    memory_incident_risk,
    outcome_reward,
)
from wso.brain.types import (
    ACTION_SEVERITY,
    ActionReceipt,
    BrainAction,
    BrainConfig,
    BrainDecision,
    BrainFeedback,
    BrainMode,
    BrainStatus,
    DecisionEnvelope,
)
from wso.telemetry.schemas import TransientLoRATelemetry

STATE_FILENAME = "brain_state.json"
DECISION_LEDGER = "decisions.jsonl"
PRIVATE_LEDGER = "learning_ledger.jsonl"
FEEDBACK_LEDGER = "feedback.jsonl"
ACTION_LEDGER = "actions.jsonl"
LOCK_FILENAME = ".brain.lock"
ACTION_ATTEMPT_DIR = "action_attempts"
FEEDBACK_DISPOSITION_DIR = "feedback_dispositions"
STATE_SCHEMA_VERSION = "2.1"


class AutonomousWeightSpaceBrain:
    """A bounded autonomous learner for adapter and transient-weight telemetry."""

    def __init__(
        self,
        state_dir: str | Path,
        *,
        config: BrainConfig | None = None,
        executor: BrainActionExecutor | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_path = self.state_dir / STATE_FILENAME
        self.executor = executor or LocalDefensiveExecutor()
        self.config: BrainConfig
        if self.state_path.exists():
            self._load()
            if config is not None and config != self.config:
                raise ValueError("supplied config does not match the persisted brain config")
        else:
            self.config = config or BrainConfig()
            self.encoder = WeightSpaceFeatureEncoder(self.config)
            self.classifier = OnlineRiskClassifier(
                self.config.feature_dimension,
                learning_rate=self.config.learning_rate,
                l2=self.config.l2,
            )
            self.novelty = MultiTimescaleNovelty(
                self.config.numeric_feature_count,
                calibration_limit=self.config.calibration_limit,
            )
            self.policy = ConstrainedContextualBandit(exploration=self.config.bandit_exploration)
            self.memory = StrategyBank(limit=self.config.memory_limit)
            self.model_version = 0
            self.baseline_examples = 0
            self.feedback_examples = 0
            self.review_threshold = self.config.review_risk_threshold
            self.known_contexts: set[str] = set()
            self.pending: dict[str, dict[str, Any]] = {}
            self.last_action_commit: dict[str, Any] | None = None
            self.last_feedback_commit: dict[str, Any] | None = None
            self.last_updated_at = time.time()

    @property
    def initialized(self) -> bool:
        return self.model_version > 0 and self.novelty.count > 0

    def bootstrap(
        self,
        *,
        baseline: list[TransientLoRATelemetry],
        labeled: list[TransientLoRATelemetry] | None = None,
    ) -> BrainStatus:
        """Initialize the slow baseline and supervised learner.

        Baseline records are explicitly trusted as benign. Extra records are used only
        when they carry an unambiguous ground-truth label in provenance.
        """

        if not baseline:
            raise ValueError("brain bootstrap requires trusted benign baseline telemetry")
        for record in baseline:
            label = telemetry_label(record)
            if label is True:
                raise ValueError("trusted baseline contains an explicitly labeled incident")
            if _has_explicit_label(record) and label is None:
                raise ValueError("trusted baseline contains an ambiguous or invalid label")

        with self._state_lock():
            if self.state_path.exists() or self.initialized:
                raise FileExistsError(
                    f"brain state already exists at {self.state_path}; create a new state directory"
                )

            baseline_bundles = [self.encoder.encode(record) for record in baseline]
            reference, calibration = _split_trusted_baseline(baseline_bundles)
            self.novelty.fit_baseline([bundle.numeric for bundle in reference])
            if calibration:
                self.novelty.calibrate([bundle.numeric for bundle in calibration])
            self.baseline_examples = len(baseline)
            training_bundles = [(bundle, False) for bundle in reference]
            self.known_contexts.update(_context_key(record) for record in baseline)

            for record in labeled or []:
                label = telemetry_label(record)
                if label is None:
                    continue
                bundle = self.encoder.encode(record)
                training_bundles.append((bundle, label))

            examples = [(bundle.vector, label) for bundle, label in training_bundles]
            self.classifier.fit(
                examples,
                epochs=self.config.classifier_epochs,
                seed=self.config.random_seed,
            )
            calibration_scores = [
                self.risk_for_features(bundle.vector, bundle.numeric) for bundle in calibration
            ]
            if calibration_scores:
                calibrated = float(
                    np.quantile(
                        calibration_scores,
                        1.0 - self.config.target_false_positive_rate,
                    )
                )
                upper = self.config.high_risk_threshold - 0.05
                self.review_threshold = min(
                    max(self.config.review_risk_threshold, calibrated), upper
                )
            self.model_version = 1
            self.last_updated_at = time.time()
            self._save()
            return self._status_unlocked()

    def decide(
        self,
        record: TransientLoRATelemetry,
        *,
        mode: BrainMode = "shadow",
        provenance_verified: bool = False,
    ) -> BrainDecision:
        """Score one observation, choose a bounded response, and persist its receipt."""

        if mode not in ("shadow", "enforce"):
            raise ValueError("brain mode must be 'shadow' or 'enforce'")
        if not isinstance(provenance_verified, bool):
            raise TypeError("provenance_verified must be an out-of-band boolean attestation")
        validate_telemetry_budget(record)
        validated_mode = cast(BrainMode, mode)
        observation_key = _observation_key(record)
        payload_digest = _observation_payload_digest(
            record,
            provenance_verified=provenance_verified,
        )

        with self._state_lock():
            if not self.state_path.exists():
                raise RuntimeError("brain state does not exist; run bootstrap first")
            self._load()
            self._require_initialized()
            self._repair_last_commits()
            for candidate_mode in ("shadow", "enforce"):
                candidate = self._load_action_attempt(observation_key, candidate_mode)
                if candidate is not None and candidate.get("payload_digest") != payload_digest:
                    raise ValueError(
                        "observation identity was reused with a different telemetry payload"
                    )
            prior = self._load_action_attempt(observation_key, validated_mode)
            if prior is not None:
                if prior.get("phase") == "committed":
                    existing = self._find_decision(str(prior["decision_id"]))
                    if existing is not None:
                        return BrainDecision.model_validate(existing)
                if prior.get("phase") == "outcome":
                    return self._recover_action_outcome(prior)
                if prior.get("phase") == "abandoned_not_executed":
                    prior = None
                else:
                    raise RuntimeError(
                        "a prior action attempt for this observation has no durable outcome; "
                        "manual reconciliation is required before retry"
                    )
            bundle = self.encoder.encode(record)
            retrieved = self.memory.retrieve(bundle.vector)
            memory_risk = memory_incident_risk(retrieved)
            novelty, drift, top_z = self.novelty.scores(bundle.numeric)
            learned_probability, learned_uncertainty = self.classifier.predict(bundle.vector)
            ready = self.classifier.ready(self.config)

            if ready:
                risk = 0.78 * learned_probability + 0.22 * novelty
                uncertainty = learned_uncertainty
            else:
                risk = novelty
                uncertainty = max(0.48, learned_uncertainty)
            if retrieved:
                risk = 0.88 * risk + 0.12 * memory_risk
            risk = float(np.clip(risk, 0.0, 1.0))
            uncertainty = float(
                np.clip(
                    uncertainty + 0.15 * drift + 0.12 / math_sqrt(self.baseline_examples),
                    0,
                    1,
                )
            )

            reasons = self._reasons(
                record=record,
                ready=ready,
                risk=risk,
                novelty=novelty,
                drift=drift,
                top_z=top_z,
                retrieved_count=len(retrieved),
                provenance_verified=provenance_verified,
            )
            unknown_context = _context_key(record) not in self.known_contexts
            missing_real_provenance = _missing_real_provenance(
                record,
                provenance_verified=provenance_verified,
            )
            candidates = self._admissible_actions(
                risk=risk,
                uncertainty=uncertainty,
                drift=drift,
                ready=ready,
                unknown_context=unknown_context,
                missing_real_provenance=missing_real_provenance,
            )
            context = self.policy.context(
                risk=risk,
                novelty=novelty,
                drift=drift,
                uncertainty=uncertainty,
                memory_risk=memory_risk,
            )
            recommended, scores = self.policy.select(candidates, context)
            decision_id = f"decision-{uuid.uuid4().hex}"
            scored_model_version = self.model_version
            action_intent = {
                "phase": "intent",
                "decision_id": decision_id,
                "observation_key": observation_key,
                "payload_digest": payload_digest,
                "timestamp": time.time(),
                "mode": validated_mode,
                "recommended_action": recommended,
                "model_version": scored_model_version,
            }
            self._write_action_attempt(observation_key, validated_mode, action_intent)
            self._append_jsonl(
                self.state_dir / ACTION_LEDGER,
                action_intent,
            )

        # External executors run outside the state lock. The durable intent above is
        # the idempotency barrier: concurrent or crash retries cannot execute twice.
        executed, autonomous, execution_reason, receipt = self._execution(
            recommended=recommended,
            mode=validated_mode,
            record=record,
        )
        reasons.append(execution_reason)
        decision = BrainDecision(
            decision_id=decision_id,
            run_id=record.run_id,
            deployment_id=record.deployment_id,
            module_id=record.module_id,
            task_family=record.task_family,
            risk_probability=risk,
            novelty_score=novelty,
            drift_score=drift,
            uncertainty=uncertainty,
            confidence=1.0 - uncertainty,
            recommended_action=recommended,
            executed_action=executed,
            mode=validated_mode,
            autonomous=autonomous,
            action_receipt_id=receipt.receipt_id,
            model_version=scored_model_version,
            feature_schema=FEATURE_SCHEMA,
            provenance_verified=bool(provenance_verified and not missing_real_provenance),
            reasons=reasons,
            policy_scores={key: round(value, 8) for key, value in scores.items()},
            retrieved_memories=retrieved,
        )
        envelope = DecisionEnvelope(
            decision=decision,
            features=bundle.vector.tolist(),
            numeric_features=bundle.numeric.tolist(),
            telemetry=_safe_telemetry(record),
        )

        with self._state_lock():
            if not self.state_path.exists():
                raise RuntimeError("brain state disappeared while an action was executing")
            self._load()
            self._require_initialized()
            self.pending[decision.decision_id] = envelope.model_dump()
            self._trim_pending()
            self.novelty.observe(bundle.numeric)
            self.last_updated_at = time.time()
            self._append_jsonl(self.state_dir / DECISION_LEDGER, decision.model_dump())
            self._append_jsonl(self.state_dir / PRIVATE_LEDGER, envelope.model_dump())
            self._append_jsonl(
                self.state_dir / ACTION_LEDGER,
                {
                    "phase": "outcome",
                    "decision_id": decision.decision_id,
                    "observation_key": observation_key,
                    "payload_digest": payload_digest,
                    "decision_timestamp": decision.timestamp,
                    "mode": validated_mode,
                    "recommended_action": recommended,
                    "autonomous": autonomous,
                    "model_version": scored_model_version,
                    "receipt": receipt.model_dump(),
                },
            )
            action_outcome = {
                **action_intent,
                "phase": "outcome",
                "decision_timestamp": decision.timestamp,
                "autonomous": autonomous,
                "receipt_id": receipt.receipt_id,
            }
            self._write_action_attempt(observation_key, validated_mode, action_outcome)
            committed_attempt = {**action_outcome, "phase": "committed"}
            self.last_action_commit = committed_attempt
            self._save()
            self._write_action_attempt(observation_key, validated_mode, committed_attempt)
            return decision

    def reconcile_action_attempt(
        self,
        record: TransientLoRATelemetry,
        *,
        mode: BrainMode,
        attest_not_executed: bool,
        provenance_verified: bool = False,
    ) -> dict[str, Any]:
        """Resolve an orphaned intent using an explicit operator attestation."""

        if mode not in ("shadow", "enforce"):
            raise ValueError("brain mode must be 'shadow' or 'enforce'")
        if not isinstance(provenance_verified, bool):
            raise TypeError("provenance_verified must be an out-of-band boolean attestation")
        validate_telemetry_budget(record)
        observation_key = _observation_key(record)
        payload_digest = _observation_payload_digest(
            record,
            provenance_verified=provenance_verified,
        )
        with self._state_lock():
            if not self.state_path.exists():
                raise RuntimeError("brain state does not exist; run bootstrap first")
            self._load()
            self._require_initialized()
            self._repair_last_commits()
            attempt = self._load_action_attempt(observation_key, mode)
            if attempt is None:
                raise KeyError("no action attempt exists for this observation and mode")
            if attempt.get("payload_digest") != payload_digest:
                raise ValueError("reconciliation telemetry does not match the action intent")
            if attempt.get("phase") != "intent":
                raise ValueError("only an unresolved intent can be reconciled")
            phase = "abandoned_not_executed" if attest_not_executed else "blocked_unknown"
            reconciled = {
                **attempt,
                "phase": phase,
                "reconciled_at": time.time(),
                "operator_attestation_not_executed": attest_not_executed,
            }
            self._write_action_attempt(observation_key, mode, reconciled)
            self._append_jsonl(
                self.state_dir / ACTION_LEDGER,
                {
                    "phase": "reconciliation",
                    "decision_id": attempt["decision_id"],
                    "observation_key": observation_key,
                    "mode": mode,
                    "resolution": phase,
                    "operator_attestation_not_executed": attest_not_executed,
                    "timestamp": reconciled["reconciled_at"],
                },
            )
            return reconciled

    def feedback(self, feedback: BrainFeedback) -> BrainStatus:
        """Learn from one verified disposition and consolidate its lesson."""

        with self._state_lock():
            if not self.state_path.exists():
                raise RuntimeError("brain state does not exist; run bootstrap first")
            self._load()
            self._require_initialized()
            self._repair_last_commits()
            fingerprint = _feedback_fingerprint(feedback)
            payload = self.pending.get(feedback.decision_id)
            if payload is None:
                payload = self._find_envelope(feedback.decision_id)
            if payload is None:
                raise KeyError(f"unknown decision_id: {feedback.decision_id}")
            envelope = DecisionEnvelope.model_validate(payload)
            if not feedback.verified:
                self._append_jsonl(
                    self.state_dir / FEEDBACK_LEDGER,
                    {"phase": "unverified", "feedback": feedback.model_dump()},
                )
                return self._status_unlocked()

            disposition = self._load_feedback_disposition(feedback.decision_id)
            if disposition is not None and disposition.get("phase") == "committed":
                self._ensure_feedback_applied(disposition)
                if (
                    disposition.get("feedback_id") == feedback.feedback_id
                    and disposition.get("fingerprint") == fingerprint
                ):
                    return self._status_unlocked()
                raise ValueError(
                    f"decision {feedback.decision_id} already has a verified disposition"
                )
            if disposition is not None:
                exact_incomplete_retry = (
                    disposition.get("phase") == "intent"
                    and disposition.get("feedback_id") == feedback.feedback_id
                    and disposition.get("fingerprint") == fingerprint
                )
                if not exact_incomplete_retry:
                    raise RuntimeError(
                        "an incomplete or conflicting feedback transaction requires "
                        "manual reconciliation"
                    )
                expected_version = disposition.get("expected_model_version")
                if not isinstance(expected_version, int) or expected_version > self.model_version:
                    raise RuntimeError("feedback transaction has an invalid model-version fence")
                if expected_version != self.model_version:
                    # A failed pre-commit attempt is safe to rebase after unrelated feedback:
                    # every durable application saves the same feedback as
                    # last_feedback_commit, which _repair_last_commits resolves first.
                    disposition = {
                        **disposition,
                        "expected_model_version": self.model_version,
                        "rebased_from_model_version": expected_version,
                    }
                    self._write_feedback_disposition(feedback.decision_id, disposition)
            else:
                disposition = {
                    "phase": "intent",
                    "feedback_id": feedback.feedback_id,
                    "decision_id": feedback.decision_id,
                    "fingerprint": fingerprint,
                    "expected_model_version": self.model_version,
                    "feedback": feedback.model_dump(),
                    "audit_intent": False,
                }
                self._write_feedback_disposition(feedback.decision_id, disposition)
            if disposition.get("audit_intent") is not True:
                self._append_jsonl(
                    self.state_dir / FEEDBACK_LEDGER,
                    {"phase": "intent", "feedback": feedback.model_dump()},
                )
                disposition = {**disposition, "audit_intent": True}
                self._write_feedback_disposition(feedback.decision_id, disposition)

            features = np.asarray(envelope.features, dtype=np.float64)
            numeric = np.asarray(envelope.numeric_features, dtype=np.float64)
            decision = envelope.decision
            actual_action = feedback.actual_action or decision.executed_action
            self.classifier.update(features, feedback.incident)
            if not feedback.incident:
                self.novelty.learn_benign(numeric)

            if feedback.has_action_outcome:
                reward = outcome_reward(
                    incident=feedback.incident,
                    impact=feedback.impact,
                    action_succeeded=feedback.action_succeeded,
                    containment_succeeded=feedback.containment_succeeded,
                    realized_cost=feedback.realized_cost,
                    disruption=feedback.disruption,
                    latency_ms=feedback.latency_ms,
                )
                memory_risk = memory_incident_risk(decision.retrieved_memories)
                context = self.policy.context(
                    risk=decision.risk_probability,
                    novelty=decision.novelty_score,
                    drift=decision.drift_score,
                    uncertainty=decision.uncertainty,
                    memory_risk=memory_risk,
                )
                self.policy.update(actual_action, context, reward)
                self.memory.consolidate(
                    decision=decision,
                    features=features,
                    incident=feedback.incident,
                    actual_action=actual_action,
                    reward=reward,
                )

            self.feedback_examples += 1
            self.model_version += 1
            self.pending.pop(feedback.decision_id, None)
            committed = {
                **disposition,
                "phase": "committed",
                "feedback_id": feedback.feedback_id,
                "decision_id": feedback.decision_id,
                "fingerprint": fingerprint,
                "incident": feedback.incident,
                "model_version": self.model_version,
            }
            self.last_feedback_commit = committed
            self.last_updated_at = time.time()
            self._save()
            self._write_feedback_disposition(feedback.decision_id, committed)
            self._ensure_feedback_applied(committed)
            return self._status_unlocked()

    def status(self) -> BrainStatus:
        with self._state_lock():
            if not self.state_path.exists():
                raise RuntimeError("brain state does not exist; run bootstrap first")
            self._load()
            self._require_initialized()
            self._repair_last_commits()
            return self._status_unlocked()

    def _status_unlocked(self) -> BrainStatus:
        ready = self.classifier.ready(self.config)
        return BrainStatus(
            model_version=self.model_version,
            baseline_examples=self.baseline_examples,
            supervised_examples=self.classifier.examples,
            positive_examples=self.classifier.positive_examples,
            negative_examples=self.classifier.negative_examples,
            feedback_examples=self.feedback_examples,
            calibration_examples=len(self.novelty.calibration),
            memory_count=len(self.memory.memories),
            pending_decisions=len(self.pending),
            ready_for_learned_risk=ready,
            ready_for_quarantine_recommendation=(
                ready and self.classifier.positive_examples >= self.config.min_positive_examples
            ),
            operating_review_threshold=self.review_threshold,
            max_autonomous_action=self.config.max_autonomous_action,
            last_updated_at=self.last_updated_at,
        )

    def refresh(self) -> int:
        """Reload a consistent initialized snapshot and return its version."""

        with self._state_lock():
            if not self.state_path.exists():
                raise RuntimeError("brain state does not exist; run bootstrap first")
            self._load()
            self._require_initialized()
            self._repair_last_commits()
            return self.model_version

    def risk_for_features(self, features: np.ndarray, numeric: np.ndarray) -> float:
        """Internal feature-space scorer used by the sealed self-play trainer."""

        novelty, _, _ = self.novelty.scores(numeric)
        probability, _ = self.classifier.predict(features)
        if self.classifier.ready(self.config):
            risk = 0.78 * probability + 0.22 * novelty
        else:
            risk = novelty
        return float(np.clip(risk, 0.0, 1.0))

    def promote_classifier(
        self,
        classifier: OnlineRiskClassifier,
        *,
        expected_model_version: int,
    ) -> None:
        """Install a challenger that has already passed an external promotion gate."""

        with self._state_lock():
            if not self.state_path.exists():
                raise RuntimeError("brain state does not exist; run bootstrap first")
            self._load()
            self._require_initialized()
            self._repair_last_commits()
            if self.model_version != expected_model_version:
                raise RuntimeError("brain changed during challenger evaluation; rerun self-play")
            if classifier.dimension != self.classifier.dimension:
                raise ValueError("challenger classifier dimension mismatch")
            self.classifier = classifier
            self.model_version += 1
            self.last_updated_at = time.time()
            self._save()

    def _admissible_actions(
        self,
        *,
        risk: float,
        uncertainty: float,
        drift: float,
        ready: bool,
        unknown_context: bool,
        missing_real_provenance: bool,
    ) -> tuple[BrainAction, ...]:
        if missing_real_provenance:
            return ("human_review",)
        if unknown_context:
            return ("deep_scan", "human_review")
        if risk < self.review_threshold:
            if uncertainty > 0.5 or drift > 0.45:
                return ("observe", "deep_scan")
            return ("allow", "observe")
        if risk < self.config.high_risk_threshold:
            return ("observe", "deep_scan", "human_review")
        if not ready or uncertainty > self.config.max_uncertainty_for_quarantine:
            return ("deep_scan", "human_review")
        if risk < self.config.critical_risk_threshold:
            return ("deep_scan", "human_review", "quarantine")
        return ("human_review", "quarantine")

    def _execution(
        self,
        *,
        recommended: BrainAction,
        mode: BrainMode,
        record: TransientLoRATelemetry,
    ) -> tuple[BrainAction, bool, str, ActionReceipt]:
        if mode not in ("shadow", "enforce"):
            raise ValueError("brain mode must be 'shadow' or 'enforce'")
        if mode == "shadow":
            receipt = _internal_receipt(
                action="observe",
                executor="brain_shadow_v2",
                summary="Recorded a shadow recommendation without invoking an executor.",
            )
            return (
                "observe",
                False,
                "Shadow mode recorded the recommendation without intervention.",
                receipt,
            )
        ceiling = ACTION_SEVERITY[self.config.max_autonomous_action]
        if recommended == "human_review":
            receipt, error = self._execute_validated("human_review", record)
            if error is not None:
                receipt = _internal_receipt(
                    action="human_review",
                    executor="brain_fail_closed_v2",
                    summary=f"Executor receipt rejected; internal review event recorded: {error}",
                )
            return (
                "human_review",
                False,
                "The policy abstained and requested human review.",
                receipt,
            )
        try:
            supported = self.executor.supports(recommended) is True
        except Exception:  # noqa: BLE001 - untrusted executor capability checks fail closed.
            supported = False
        executor_name = _executor_name(self.executor)
        if ACTION_SEVERITY[recommended] > ceiling or not supported:
            receipt = _internal_receipt(
                action="human_review",
                executor="brain_fail_closed_v2",
                summary="Autonomy policy converted the recommendation into a review event.",
            )
            constraint = (
                f"the immutable autonomy ceiling ({self.config.max_autonomous_action})"
                if ACTION_SEVERITY[recommended] > ceiling
                else f"executor {executor_name} has no {recommended} capability"
            )
            return (
                "human_review",
                False,
                f"{constraint.capitalize()} converted {recommended} into human review.",
                receipt,
            )
        receipt, error = self._execute_validated(recommended, record)
        if error is not None:
            fallback = _internal_receipt(
                action="human_review",
                executor="brain_fail_closed_v2",
                summary=f"Executor result rejected; internal review event recorded: {error}",
            )
            return (
                "human_review",
                False,
                (
                    f"The registered executor failed validation for {recommended}; "
                    "requested human review."
                ),
                fallback,
            )
        return (
            recommended,
            True,
            f"Executed the authorized {recommended} response autonomously.",
            receipt,
        )

    def _execute_validated(
        self,
        action: BrainAction,
        record: TransientLoRATelemetry,
    ) -> tuple[ActionReceipt, str | None]:
        """Invoke a registered executor and reject any receipt-contract deviation."""

        try:
            receipt = self.executor.execute(action, record)
        except Exception as exc:  # noqa: BLE001 - executor failures become review receipts.
            fallback = _internal_receipt(
                action="human_review",
                executor="brain_fail_closed_v2",
                summary="Registered executor raised; internal review event recorded.",
            )
            return fallback, f"executor raised {type(exc).__name__}"
        if not isinstance(receipt, ActionReceipt):
            fallback = _internal_receipt(
                action="human_review",
                executor="brain_fail_closed_v2",
                summary="Executor returned an invalid receipt type; internal review event recorded.",
            )
            return fallback, "executor returned a non-ActionReceipt result"
        problems: list[str] = []
        if receipt.requested_action != action:
            problems.append("requested_action mismatch")
        if receipt.executed_action != action:
            problems.append("executed_action mismatch")
        expected_executor = _executor_name(self.executor)
        if receipt.executor != expected_executor:
            problems.append("executor identity mismatch")
        if ACTION_SEVERITY[receipt.executed_action] > ACTION_SEVERITY[action]:
            problems.append("receipt exceeds requested severity")
        if not receipt.success:
            problems.append("executor reported failure")
        return receipt, "; ".join(problems) or None

    def _reasons(
        self,
        *,
        record: TransientLoRATelemetry,
        ready: bool,
        risk: float,
        novelty: float,
        drift: float,
        top_z: list[tuple[int, float]],
        retrieved_count: int,
        provenance_verified: bool,
    ) -> list[str]:
        reasons = [f"Combined learned risk is {risk:.3f}; baseline novelty is {novelty:.3f}."]
        notable = [
            (NUMERIC_FEATURE_NAMES[index], z)
            for index, z in top_z
            if z >= 2.0 and index < len(NUMERIC_FEATURE_NAMES)
        ]
        if notable:
            rendered = ", ".join(f"{name} ({z:.1f}σ)" for name, z in notable[:3])
            reasons.append(f"Largest trusted-baseline deviations: {rendered}.")
        if drift >= 0.35:
            reasons.append(f"Fast/slow memory disagreement indicates drift ({drift:.3f}).")
        if not ready:
            reasons.append("The supervised learner lacks enough positive and negative outcomes.")
        if retrieved_count:
            reasons.append(f"Retrieved {retrieved_count} verified similar experience(s).")
        if _context_key(record) not in self.known_contexts:
            reasons.append(
                "This task/module/model context was not present in the trusted baseline."
            )
        if _missing_real_provenance(record, provenance_verified=provenance_verified):
            reasons.append(
                "Real-evidence provenance is incomplete; the fail-closed policy abstained."
            )
        return reasons

    def _trim_pending(self) -> None:
        excess = len(self.pending) - self.config.pending_decision_limit
        if excess <= 0:
            return
        ordered = sorted(
            self.pending,
            key=lambda key: self.pending[key]["decision"]["timestamp"],
        )
        for decision_id in ordered[:excess]:
            self.pending.pop(decision_id, None)

    def _find_envelope(self, decision_id: str) -> dict[str, Any] | None:
        for payload in self._iter_jsonl(self.state_dir / PRIVATE_LEDGER):
            if payload.get("decision", {}).get("decision_id") == decision_id:
                return payload
        return None

    def _find_decision(self, decision_id: str) -> dict[str, Any] | None:
        for payload in self._iter_jsonl(self.state_dir / DECISION_LEDGER):
            if payload.get("decision_id") == decision_id:
                return payload
        return None

    def _load_action_attempt(
        self,
        observation_key: str,
        mode: BrainMode | str,
    ) -> dict[str, Any] | None:
        return _read_optional_json(self._action_attempt_path(observation_key, mode))

    def _write_action_attempt(
        self,
        observation_key: str,
        mode: BrainMode | str,
        payload: dict[str, Any],
    ) -> None:
        root = self.state_dir / ACTION_ATTEMPT_DIR
        root.mkdir(parents=True, exist_ok=True)
        root.chmod(0o700)
        _atomic_write_json(self._action_attempt_path(observation_key, mode), payload)

    def _action_attempt_path(self, observation_key: str, mode: BrainMode | str) -> Path:
        key = hashlib.sha256(f"{observation_key}:{mode}".encode()).hexdigest()
        return self.state_dir / ACTION_ATTEMPT_DIR / key[:2] / f"{key}.json"

    def _load_feedback_disposition(self, decision_id: str) -> dict[str, Any] | None:
        return _read_optional_json(self._feedback_disposition_path(decision_id))

    def _write_feedback_disposition(
        self,
        decision_id: str,
        payload: dict[str, Any],
    ) -> None:
        root = self.state_dir / FEEDBACK_DISPOSITION_DIR
        root.mkdir(parents=True, exist_ok=True)
        root.chmod(0o700)
        _atomic_write_json(self._feedback_disposition_path(decision_id), payload)

    def _feedback_disposition_path(self, decision_id: str) -> Path:
        key = hashlib.sha256(decision_id.encode("utf-8")).hexdigest()
        return self.state_dir / FEEDBACK_DISPOSITION_DIR / key[:2] / f"{key}.json"

    def _recover_action_outcome(self, attempt: dict[str, Any]) -> BrainDecision:
        decision_id = str(attempt.get("decision_id", ""))
        decision_payload = self._find_decision(decision_id)
        envelope_payload = self._find_envelope(decision_id)
        if decision_payload is None or envelope_payload is None:
            raise RuntimeError("action outcome is incomplete; manual reconciliation is required")
        decision = BrainDecision.model_validate(decision_payload)
        envelope = DecisionEnvelope.model_validate(envelope_payload)
        if decision_id not in self.pending:
            self.pending[decision_id] = envelope.model_dump()
            self._trim_pending()
            self.novelty.observe(np.asarray(envelope.numeric_features, dtype=np.float64))
        committed = {**attempt, "phase": "committed"}
        self.last_action_commit = committed
        self.last_updated_at = time.time()
        self._save()
        self._write_action_attempt(
            str(attempt["observation_key"]),
            str(attempt["mode"]),
            committed,
        )
        return decision

    def _repair_last_commits(self) -> None:
        if self.last_action_commit is not None:
            action = self.last_action_commit
            current = self._load_action_attempt(
                str(action["observation_key"]),
                str(action["mode"]),
            )
            if current is None or current.get("phase") != "committed":
                self._write_action_attempt(
                    str(action["observation_key"]),
                    str(action["mode"]),
                    {**action, "phase": "committed"},
                )
        if self.last_feedback_commit is not None:
            feedback = self.last_feedback_commit
            current = self._load_feedback_disposition(str(feedback["decision_id"]))
            if current is None or current.get("phase") != "committed":
                current = {**feedback, "phase": "committed"}
                self._write_feedback_disposition(str(feedback["decision_id"]), current)
            self._ensure_feedback_applied(current)

    def _ensure_feedback_applied(self, disposition: dict[str, Any]) -> None:
        if disposition.get("audit_applied") is True:
            return
        self._append_jsonl(
            self.state_dir / FEEDBACK_LEDGER,
            {
                "phase": "applied",
                "feedback_id": disposition["feedback_id"],
                "decision_id": disposition["decision_id"],
                "model_version": disposition["model_version"],
                "fingerprint": disposition["fingerprint"],
            },
        )
        updated = {**disposition, "audit_applied": True}
        self._write_feedback_disposition(str(disposition["decision_id"]), updated)

    def _iter_jsonl(self, path: Path) -> Iterator[dict[str, Any]]:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    yield payload

    def _require_initialized(self) -> None:
        if not self.initialized:
            raise RuntimeError("brain is not initialized; run bootstrap first")

    def _save(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": STATE_SCHEMA_VERSION,
            "feature_schema": FEATURE_SCHEMA,
            "config": self.config.model_dump(),
            "model_version": self.model_version,
            "baseline_examples": self.baseline_examples,
            "feedback_examples": self.feedback_examples,
            "review_threshold": self.review_threshold,
            "known_contexts": sorted(self.known_contexts),
            "pending": self.pending,
            "last_action_commit": self.last_action_commit,
            "last_feedback_commit": self.last_feedback_commit,
            "last_updated_at": self.last_updated_at,
            "classifier": self.classifier.to_dict(),
            "novelty": self.novelty.to_dict(),
            "policy": self.policy.to_dict(),
            "memory": self.memory.to_dict(),
            "safety": {
                "learn_only_from_verified_feedback": True,
                "raw_prompts_persisted": False,
                "raw_tensors_persisted": False,
                "autonomy_ceiling": self.config.max_autonomous_action,
            },
        }
        _atomic_write_json(self.state_path, payload)

    def _load(self) -> None:
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ValueError(f"unsupported brain state schema: {payload.get('schema_version')!r}")
        if payload.get("feature_schema") != FEATURE_SCHEMA:
            raise ValueError(f"unsupported brain feature schema: {payload.get('feature_schema')!r}")
        self.config = BrainConfig.model_validate(payload["config"])
        self.encoder = WeightSpaceFeatureEncoder(self.config)
        self.classifier = OnlineRiskClassifier.from_dict(payload["classifier"])
        self.novelty = MultiTimescaleNovelty.from_dict(payload["novelty"])
        self.policy = ConstrainedContextualBandit.from_dict(payload["policy"])
        self.memory = StrategyBank.from_dict(payload["memory"])
        self.model_version = int(payload["model_version"])
        self.baseline_examples = int(payload.get("baseline_examples", 0))
        self.feedback_examples = int(payload.get("feedback_examples", 0))
        self.review_threshold = float(
            payload.get("review_threshold", self.config.review_risk_threshold)
        )
        self.known_contexts = set(payload.get("known_contexts", []))
        self.pending = {
            str(key): DecisionEnvelope.model_validate(value).model_dump()
            for key, value in dict(payload.get("pending", {})).items()
        }
        self.last_action_commit = _optional_mapping(payload.get("last_action_commit"))
        self.last_feedback_commit = _optional_mapping(payload.get("last_feedback_commit"))
        self.last_updated_at = float(payload.get("last_updated_at", time.time()))
        if any(
            value < 0
            for value in (
                self.model_version,
                self.baseline_examples,
                self.feedback_examples,
            )
        ):
            raise ValueError("persisted brain counters must be non-negative")
        if not (
            math_is_finite(self.review_threshold)
            and math_is_finite(self.last_updated_at)
            and self.config.review_risk_threshold
            <= self.review_threshold
            <= self.config.high_risk_threshold - 0.05
        ):
            raise ValueError("persisted brain thresholds or timestamps are invalid")
        if not all(isinstance(key, str) for key in self.known_contexts):
            raise ValueError("persisted trusted contexts are invalid")
        if self.classifier.dimension != self.config.feature_dimension:
            raise ValueError("persisted classifier dimension does not match brain config")
        if not math_is_close(self.classifier.learning_rate, self.config.learning_rate):
            raise ValueError("persisted classifier learning rate does not match brain config")
        if not math_is_close(self.classifier.l2, self.config.l2):
            raise ValueError("persisted classifier regularization does not match brain config")
        if self.novelty.dimension != self.config.numeric_feature_count:
            raise ValueError("persisted novelty dimension does not match brain config")
        if self.novelty.calibration_limit != self.config.calibration_limit:
            raise ValueError("persisted novelty calibration limit does not match brain config")
        if not math_is_close(self.policy.exploration, self.config.bandit_exploration):
            raise ValueError("persisted policy exploration does not match brain config")
        if self.memory.limit != self.config.memory_limit:
            raise ValueError("persisted memory limit does not match brain config")
        if len(self.memory.memories) > self.memory.limit:
            raise ValueError("persisted strategy bank exceeds its configured limit")
        if any(
            len(memory.centroid) != self.config.feature_dimension for memory in self.memory.memories
        ):
            raise ValueError("persisted strategy memory has an invalid centroid dimension")
        if any(
            memory.success_count + memory.failure_count != memory.observations
            or memory.updated_at < memory.created_at
            or not -10.0 <= memory.mean_reward <= 10.0
            for memory in self.memory.memories
        ):
            raise ValueError("persisted strategy memory has inconsistent outcomes")
        _validate_transaction_marker(self.last_action_commit, kind="action")
        _validate_transaction_marker(self.last_feedback_commit, kind="feedback")

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        parent_existed = path.parent.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_existed = path.exists()
        rendered = json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC,
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        if not parent_existed:
            _fsync_directory(path.parent.parent)
        if not file_existed:
            _fsync_directory(path.parent)

    @contextmanager
    def _state_lock(self) -> Iterator[None]:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.chmod(0o700)
        path = self.state_dir / LOCK_FILENAME
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def telemetry_label(record: TransientLoRATelemetry) -> bool | None:
    """Read explicit ground truth without accepting model predictions as labels."""

    labels: list[bool] = []
    for key in (
        "ground_truth",
        "audit_label",
        "is_incident",
        "is_attack",
        "is_anomalous",
        "label",
    ):
        if key not in record.provenance:
            continue
        parsed = _parse_telemetry_label(record.provenance[key])
        if parsed is None:
            return None
        labels.append(parsed)
    if not labels or any(label != labels[0] for label in labels[1:]):
        return None
    return labels[0]


def _has_explicit_label(record: TransientLoRATelemetry) -> bool:
    return any(
        key in record.provenance
        for key in (
            "ground_truth",
            "audit_label",
            "is_incident",
            "is_attack",
            "is_anomalous",
            "label",
        )
    )


def _parse_telemetry_label(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math_is_finite(numeric) and numeric in (0.0, 1.0):
            return bool(numeric)
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {
            "1",
            "true",
            "yes",
            "incident",
            "attack",
            "anomalous",
            "positive",
        }:
            return True
        if normalized in {"0", "false", "no", "benign", "negative", "clean"}:
            return False
    return None


def _context_key(record: TransientLoRATelemetry) -> str:
    return json.dumps(
        [
            record.task_family or "unknown",
            record.module_id,
            record.collection_mode,
            record.model_id or "unknown",
            record.adapter_id or "unknown",
            record.source_kind or "unknown",
            record.runtime_framework or "unknown",
            record.observer_level,
            record.evidence_mode,
            record.deployment_id or "unknown",
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    )


def _missing_real_provenance(
    record: TransientLoRATelemetry,
    *,
    provenance_verified: bool = False,
) -> bool:
    if record.evidence_mode != "real":
        return False
    if not provenance_verified:
        return True
    if not (
        _valid_provenance_token(record.source_kind)
        and record.source_path_hashes
        and record.collection_timestamp is not None
        and record.collection_timestamp > 0
        and _valid_provenance_token(record.runtime_framework)
        and (
            _valid_provenance_token(record.collector_version)
            or _valid_git_commit(record.git_commit)
        )
    ):
        return True
    return not all(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
        for value in record.source_path_hashes.values()
    )


def _valid_provenance_token(value: str | None) -> bool:
    return bool(value and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/+@-]{0,127}", value))


def _valid_git_commit(value: str | None) -> bool:
    return bool(value and re.fullmatch(r"[0-9a-fA-F]{7,64}", value))


def _safe_telemetry(record: TransientLoRATelemetry) -> dict[str, object]:
    return {
        "run_id": record.run_id,
        "deployment_id": record.deployment_id,
        "module_id": record.module_id,
        "task_family": record.task_family,
        "collection_mode": record.collection_mode,
        "model_id": record.model_id,
        "adapter_id": record.adapter_id,
        "source_kind": record.source_kind,
        "runtime_framework": record.runtime_framework,
        "observer_level": record.observer_level,
        "evidence_mode": record.evidence_mode,
        "split": record.split,
        "timestamp": record.timestamp,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    parent_existed = path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        if not parent_existed:
            _fsync_directory(path.parent.parent)
            _fsync_directory(path.parent.parent.parent)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def math_sqrt(value: int) -> float:
    return max(1.0, float(value)) ** 0.5


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def math_is_finite(value: float) -> bool:
    return bool(np.isfinite(value))


def math_is_close(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=1e-12, atol=1e-15))


def _optional_mapping(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("persisted transaction marker is invalid")
    return dict(value)


def _validate_transaction_marker(value: dict[str, Any] | None, *, kind: str) -> None:
    if value is None:
        return
    required = (
        {"decision_id", "observation_key", "mode", "phase"}
        if kind == "action"
        else {"decision_id", "feedback_id", "fingerprint", "model_version", "phase"}
    )
    if not required.issubset(value) or value.get("phase") != "committed":
        raise ValueError(f"persisted {kind} transaction marker is invalid")


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid transaction record: {path}")  # noqa: TRY004
    return payload


def _split_trusted_baseline(
    bundles: list[FeatureBundle],
) -> tuple[list[FeatureBundle], list[FeatureBundle]]:
    if len(bundles) < 10:
        return bundles, []
    calibration = [bundle for index, bundle in enumerate(bundles) if index % 5 == 0]
    reference = [bundle for index, bundle in enumerate(bundles) if index % 5 != 0]
    return reference, calibration


def _feedback_fingerprint(feedback: BrainFeedback) -> str:
    rendered = feedback.model_dump_json(exclude={"timestamp"})
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _observation_key(record: TransientLoRATelemetry) -> str:
    """Stable event identity derived from the upstream run idempotency token."""

    identity = {
        "module_id": record.module_id,
        "run_id": record.run_id,
    }
    rendered = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _observation_payload_digest(
    record: TransientLoRATelemetry,
    *,
    provenance_verified: bool,
) -> str:
    payload = record.model_dump(mode="json")
    payload.pop("timestamp", None)
    payload["_brain_provenance_verified"] = provenance_verified
    rendered = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _internal_receipt(*, action: BrainAction, executor: str, summary: str) -> ActionReceipt:
    return ActionReceipt(
        requested_action=action,
        executed_action=action,
        executor=executor,
        success=True,
        summary=summary,
        evidence={"external_executor_invoked": False},
    )


def _executor_name(executor: object) -> str:
    try:
        name = getattr(executor, "name", None)
    except Exception:  # noqa: BLE001 - hostile executor properties fail closed.
        return "unidentified-executor"
    return name if isinstance(name, str) and name else "unidentified-executor"
