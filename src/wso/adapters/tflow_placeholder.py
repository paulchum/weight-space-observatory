"""Minimal adapter for the official TFlow inference path.

The official TFlow release is inference-only and exposes a ``TFlowMethod`` that
builds per-sender LoRA factors, groups them with
``build_weighted_lora_entries(...)``, and applies them to the receiver model for
one transient generation pass.  This module does not vendor TFlow or make it a
hard dependency.  Instead, it optionally imports a caller-provided TFlow checkout
and taps the weighted LoRA entries long enough to emit WSO derived telemetry.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import sys
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import torch

from wso.telemetry.hooks import (
    WeightedLoRAEntries,
    WeightedLoRAEntry,
    collect_batched_weighted_lora_event_telemetry,
    collect_weighted_lora_event_telemetry,
    materialize_weighted_lora_delta,
)
from wso.telemetry.schemas import EvidenceMode, TelemetrySplit, TransientLoRATelemetry

if TYPE_CHECKING:
    from wso.brain.runtime import AutonomousWeightSpaceBrain
    from wso.brain.types import BrainDecision, BrainMode


_TFLOW_CAPTURE_LOCK = threading.RLock()
_TFLOW_CAPTURE_ACTIVE = False


@dataclass(frozen=True)
class TFlowInferenceConfig:
    """Runtime settings for an optional official TFlow checkout."""

    tflow_repo_path: str | Path | None = None
    method: str = "tflow"
    dataset: str = "gsm8k"
    checkpoint: str | Path | None = None
    max_new_tokens: int | None = None
    generate_bs: int | None = 1
    config_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TFlowInferenceResult:
    """TFlow answer plus derived transient telemetry records."""

    answer: str
    metadata: dict[str, Any]
    telemetry: list[TransientLoRATelemetry]
    brain_decisions: list[BrainDecision] = field(default_factory=list)


class OfficialTFlowAdapter:
    """Wrap the official TFlow ``build_method(...).solve(...)`` inference path.

    The adapter is intentionally thin: it delegates setup and generation to the
    official modules, captures the transient weighted LoRA entries produced by
    the official TFlow method, and converts those entries into WSO telemetry via
    :func:`wso.telemetry.hooks.collect_delta_telemetry`.
    """

    def __init__(self, config: TFlowInferenceConfig | None = None) -> None:
        self.config = config or TFlowInferenceConfig()
        self._method: Any | None = None
        self._tflow_module: ModuleType | None = None
        self._resolved_config: dict[str, Any] | None = None

    @property
    def resolved_config(self) -> dict[str, Any] | None:
        return self._resolved_config

    def setup(self) -> None:
        """Import official TFlow modules and initialise the selected method."""

        configs_mod, methods_mod, tflow_mod = _import_official_tflow_modules(
            self.config.tflow_repo_path
        )
        cfg = configs_mod.get_config(self.config.method)
        cfg.setdefault("data", {})["dataset"] = self.config.dataset
        if self.config.generate_bs is not None:
            cfg.setdefault("evaluation", {})["generate_bs"] = self.config.generate_bs
        if self.config.max_new_tokens is not None:
            model_cfg = cfg.setdefault("model", {})
            model_cfg["max_new_tokens"] = self.config.max_new_tokens
            model_cfg.pop("max_new_tokens_by_dataset", None)
        if self.config.checkpoint is not None:
            if self.config.method != "tflow":
                raise ValueError("checkpoint is only valid for method='tflow'")
            cfg.setdefault("method_config", {}).setdefault("training", {})["eval_checkpoint"] = str(
                self.config.checkpoint
            )
        _deep_update(cfg, self.config.config_overrides)
        cfg["_mode"] = "inference"

        seed = int(cfg.get("seed", 42))
        with _official_import_path(self.config.tflow_repo_path):
            helpers = importlib.import_module("src.utils.helpers")
            helpers.set_seed(seed)
        method = methods_mod.build_method(cfg)
        method.setup()

        self._method = method
        self._tflow_module = tflow_mod
        self._resolved_config = cfg

    def solve(
        self,
        question: str,
        *,
        run_id: str | None = None,
        task_family: str | None = None,
        evidence_mode: EvidenceMode = "synthetic",
        source_path_hashes: dict[str, str] | None = None,
        dataset_id: str | None = None,
        dataset_sha256: str | None = None,
        split: TelemetrySplit | None = None,
        collector_version: str | None = None,
        git_commit: str | None = None,
        python_version: str | None = None,
        torch_version: str | None = None,
        device: str | None = None,
        collection_timestamp: float | None = None,
        module_ids: set[str] | None = None,
        baseline_deltas: Mapping[str, torch.Tensor] | None = None,
        deployment_id: str | None = None,
        adapter_id: str | None = "tflow-transient-lora",
        otel_resource: dict[str, Any] | None = None,
        brain: AutonomousWeightSpaceBrain | None = None,
        brain_mode: BrainMode = "shadow",
        brain_provenance_verified: bool = False,
    ) -> TFlowInferenceResult:
        """Run one official TFlow inference and return derived telemetry.

        When ``brain`` is supplied, the records are scored after generation. This is an
        asynchronous monitoring integration, not a pre-generation blocking claim.
        """

        if self._method is None or self._tflow_module is None:
            self.setup()
        assert self._method is not None
        assert self._tflow_module is not None

        resolved_run_id = run_id or str(uuid.uuid4())
        captured: list[WeightedLoRAEntries] = []
        with self._capture_weighted_entries(captured):
            result = self._method.solve(question)

        metadata = dict(result.get("metadata", {}))
        sender_count = int(metadata.get("num_senders") or 0)
        sender_ids = [f"agent_{idx}" for idx in range(sender_count)] or None
        model_id = _model_id_from_config(self._resolved_config)
        records: list[TransientLoRATelemetry] = []
        for weighted_entries in captured:
            records.extend(
                collect_tflow_weighted_lora_telemetry(
                    weighted_entries,
                    run_id=resolved_run_id,
                    task_family=task_family,
                    prompt_hash=hash_prompt(question),
                    evidence_mode=evidence_mode,
                    source_kind="tflow",
                    source_path_hashes=source_path_hashes,
                    model_path=(
                        str(self.config.tflow_repo_path)
                        if self.config.tflow_repo_path is not None
                        else None
                    ),
                    checkpoint_sha256=(
                        source_path_hashes.get(
                            str(Path(self.config.checkpoint).expanduser().resolve())
                        )
                        if source_path_hashes is not None and self.config.checkpoint is not None
                        else None
                    ),
                    dataset_id=dataset_id,
                    dataset_sha256=dataset_sha256,
                    split=split,
                    collector_version=collector_version,
                    git_commit=git_commit,
                    python_version=python_version,
                    torch_version=torch_version,
                    device=device,
                    collection_timestamp=collection_timestamp,
                    module_ids=module_ids,
                    sender_ids=sender_ids,
                    baseline_deltas=baseline_deltas,
                    deployment_id=deployment_id,
                    model_id=model_id,
                    adapter_id=adapter_id,
                    otel_resource=otel_resource,
                    provenance={
                        "official_tflow_method": self.config.method,
                        "official_tflow_dataset": self.config.dataset,
                        "official_tflow_repo_path": (
                            str(self.config.tflow_repo_path)
                            if self.config.tflow_repo_path is not None
                            else None
                        ),
                    },
                )
            )

        brain_decisions = (
            [
                brain.decide(
                    record,
                    mode=brain_mode,
                    provenance_verified=brain_provenance_verified,
                )
                for record in records
            ]
            if brain is not None
            else []
        )
        return TFlowInferenceResult(
            answer=str(result.get("answer", "")),
            metadata=metadata,
            telemetry=records,
            brain_decisions=brain_decisions,
        )

    @contextlib.contextmanager
    def _capture_weighted_entries(
        self,
        captured: list[WeightedLoRAEntries],
    ) -> Iterator[None]:
        global _TFLOW_CAPTURE_ACTIVE

        assert self._tflow_module is not None
        # The official hook is process-global module state. Serialize the complete
        # capture window so overlapping solves cannot mix entries or restore a stale
        # closure. Production integrations should replace monkey-patching with an
        # upstream request-scoped callback when TFlow exposes one.
        with _TFLOW_CAPTURE_LOCK:
            if _TFLOW_CAPTURE_ACTIVE:
                raise RuntimeError("nested TFlow capture windows are not supported")
            original = getattr(  # noqa: B009
                self._tflow_module,
                "build_weighted_lora_entries",
            )

            def capture(factor_dicts: list[dict[str, Any]], weights: list[Any]) -> dict[str, Any]:
                weighted_entries = original(factor_dicts, weights)
                captured.append(weighted_entries)
                return weighted_entries

            try:
                _TFLOW_CAPTURE_ACTIVE = True
                setattr(  # noqa: B010
                    self._tflow_module,
                    "build_weighted_lora_entries",
                    capture,
                )
                yield
            finally:
                try:
                    setattr(  # noqa: B010
                        self._tflow_module,
                        "build_weighted_lora_entries",
                        original,
                    )
                except BaseException:
                    # A one-shot asynchronous interruption can land before the
                    # restoration assignment. Retry the idempotent restore once,
                    # then preserve the original interruption for the caller.
                    setattr(  # noqa: B010
                        self._tflow_module,
                        "build_weighted_lora_entries",
                        original,
                    )
                    raise
                finally:
                    _TFLOW_CAPTURE_ACTIVE = False


def collect_tflow_weighted_lora_telemetry(
    weighted_entries: WeightedLoRAEntries,
    *,
    run_id: str,
    task_family: str | None = None,
    prompt_hash: str | None = None,
    evidence_mode: EvidenceMode = "synthetic",
    source_kind: str | None = None,
    source_path_hashes: dict[str, str] | None = None,
    model_path: str | None = None,
    adapter_path: str | None = None,
    checkpoint_sha256: str | None = None,
    dataset_id: str | None = None,
    dataset_sha256: str | None = None,
    split: TelemetrySplit | None = None,
    collector_version: str | None = None,
    git_commit: str | None = None,
    python_version: str | None = None,
    torch_version: str | None = None,
    device: str | None = None,
    collection_timestamp: float | None = None,
    module_ids: set[str] | None = None,
    sender_ids: Sequence[str] | None = None,
    fusion_gate_logits: torch.Tensor | None = None,
    baseline_deltas: Mapping[str, torch.Tensor] | None = None,
    deployment_id: str | None = None,
    model_id: str | None = None,
    adapter_id: str | None = None,
    observer_level: str = "derived_telemetry",
    otel_resource: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> list[TransientLoRATelemetry]:
    """Collect telemetry from official TFlow weighted LoRA entries.

    ``weighted_entries`` follows the official TFlow shape:
    ``{parameter_name.weight: [(A_i, B_i, scale_i, gate_weight_i), ...]}``.
    The materialized delta is computed in memory as
    ``sum_i gate_i * scale_i * (B_i @ A_i)`` and is not persisted.
    """

    return collect_weighted_lora_event_telemetry(
        weighted_entries,
        run_id=run_id,
        task_family=task_family,
        prompt_hash=prompt_hash,
        evidence_mode=evidence_mode,
        module_ids=module_ids,
        sender_ids=sender_ids,
        fusion_gate_logits=fusion_gate_logits,
        baseline_deltas=baseline_deltas,
        deployment_id=deployment_id,
        runtime_framework="official_tflow",
        model_id=model_id,
        adapter_id=adapter_id,
        collection_mode="tflow_inference_hook",
        source_kind=source_kind,
        source_path_hashes=source_path_hashes,
        model_path=model_path,
        adapter_path=adapter_path,
        checkpoint_sha256=checkpoint_sha256,
        dataset_id=dataset_id,
        dataset_sha256=dataset_sha256,
        split=split,
        collector_version=collector_version,
        git_commit=git_commit,
        python_version=python_version,
        torch_version=torch_version,
        device=device,
        collection_timestamp=collection_timestamp,
        observer_level=observer_level,
        otel_resource=otel_resource,
        provenance={
            "adapter_collector": (
                "wso.adapters.tflow_placeholder.collect_tflow_weighted_lora_telemetry"
            ),
            **(provenance or {}),
        },
    )


def collect_tflow_batched_weighted_lora_telemetry(
    weighted_entries: WeightedLoRAEntries,
    *,
    run_ids: Sequence[str],
    task_families: Sequence[str | None] | None = None,
    prompt_hashes: Sequence[str | None] | None = None,
    **kwargs: Any,
) -> list[TransientLoRATelemetry]:
    """Collect per-sample telemetry from official batched TFlow hook entries."""

    options = dict(kwargs)
    options.setdefault("runtime_framework", "official_tflow")
    options.setdefault("collection_mode", "tflow_inference_hook")
    provenance = options.pop("provenance", None) or {}
    options["provenance"] = {
        "adapter_collector": (
            "wso.adapters.tflow_placeholder.collect_tflow_batched_weighted_lora_telemetry"
        ),
        **provenance,
    }
    return collect_batched_weighted_lora_event_telemetry(
        weighted_entries,
        run_ids=run_ids,
        task_families=task_families,
        prompt_hashes=prompt_hashes,
        **options,
    )


def hash_prompt(prompt: str) -> str:
    """Stable short hash for prompt text without storing the prompt."""

    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _import_official_tflow_modules(
    repo_path: str | Path | None,
) -> tuple[ModuleType, ModuleType, ModuleType]:
    repo_root = Path(repo_path).expanduser().resolve() if repo_path is not None else None
    if repo_root is not None:
        _drop_conflicting_src_modules(repo_root)
    with _official_import_path(repo_root):
        try:
            configs_mod = importlib.import_module("src.configs")
            methods_mod = importlib.import_module("src.methods")
            tflow_mod = importlib.import_module("src.methods.tflow")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Could not import official TFlow modules. Provide a TFlow checkout "
                "path with TFlowInferenceConfig(tflow_repo_path=...)."
            ) from exc
    if repo_root is not None:
        _validate_module_under_repo(configs_mod, repo_root)
        _validate_module_under_repo(tflow_mod, repo_root)
    return configs_mod, methods_mod, tflow_mod


@contextlib.contextmanager
def _official_import_path(repo_path: str | Path | None) -> Iterator[None]:
    if repo_path is None:
        yield
        return
    root = str(Path(repo_path).expanduser().resolve())
    inserted = False
    if root not in sys.path:
        sys.path.insert(0, root)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(root)
            except ValueError:
                pass


def _drop_conflicting_src_modules(repo_root: Path) -> None:
    for name, module in list(sys.modules.items()):
        if name != "src" and not name.startswith("src."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        try:
            Path(module_file).resolve().relative_to(repo_root)
        except ValueError:
            sys.modules.pop(name, None)


def _validate_module_under_repo(module: ModuleType, repo_root: Path) -> None:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return
    try:
        Path(module_file).resolve().relative_to(repo_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Imported {module.__name__} from {module_file}, not from TFlow repo {repo_root}"
        ) from exc


def _deep_update(target: dict[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _model_id_from_config(config: Mapping[str, Any] | None) -> str | None:
    if config is None:
        return None
    model_cfg = config.get("model", {})
    if isinstance(model_cfg, Mapping):
        name = model_cfg.get("name")
        return str(name) if name is not None else None
    return None


__all__ = [
    "OfficialTFlowAdapter",
    "TFlowInferenceConfig",
    "TFlowInferenceResult",
    "WeightedLoRAEntries",
    "WeightedLoRAEntry",
    "collect_tflow_batched_weighted_lora_telemetry",
    "collect_tflow_weighted_lora_telemetry",
    "hash_prompt",
    "materialize_weighted_lora_delta",
]
