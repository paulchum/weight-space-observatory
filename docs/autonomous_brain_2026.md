# Autonomous Adapter Trust Gateway — research and build decision

Research cutoff: **August 1, 2026**

## Decision

The most valuable brain for this repository is not a generic chat agent. It is an
**autonomous adapter trust gateway**: a tenant-specific system that learns the geometry
of trusted LoRA and transient weight updates, estimates incident risk and uncertainty,
selects a bounded defensive response, remembers verified outcomes, and hardens a
challenger offline before any promotion.

This turns a one-time audit into a recurring control plane. Static registry scanning is
the entry product; runtime TFlow/adapter monitoring, bounded triage, and verified
feedback learning are the subscription and data moat. The word *autonomous* refers to
action selection inside an operator-owned envelope. It does not mean unrestricted
runtime control, self-authorized capabilities, or unsupervised policy rewriting.

## The one-year counterfactual

Anomaly detection, logistic regression, online learning, contextual bandits, and LoRA
all existed before August 1, 2025. The defensible claim is narrower: **between August 1,
2025 and August 1, 2026, query-specific weight-space communication, practical
weight-only adapter screening, production agent-monitoring patterns, continual agent
memory, and automated defensive self-play became credible enough to compose into a
new runtime control surface.** The constituent methods are not inventions of this
repository.

| Date | Primary evidence | Consequence for this product |
|---|---|---|
| Sep 29, 2025 | [ReasoningBank](https://arxiv.org/abs/2509.25140) introduced persistent learning from successful and failed agent trajectories; the [Google Research account](https://www.research.google/blog/reasoningbank-enabling-agents-to-learn-from-experience/) reports results on WebArena and SWE-Bench Verified. | Store compact operational lessons rather than replaying raw trajectories. |
| Nov 7, 2025 | Google introduced [Nested Learning and HOPE](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/), including multi-frequency continuum memory and a proof-of-concept self-modifying architecture. | Separate fast drift memory from slow trusted memory. HOPE itself is not implemented or claimed production-ready here. |
| Feb 16, 2026 | [Weight-space Detection of Backdoors in LoRA Adapters](https://arxiv.org/abs/2602.15195) reported 97% accuracy and under 2% false positives on a controlled benchmark of 500 adapters: 400 clean and 100 poisoned. | Derived weight statistics can support a learned screening layer, subject to independent validation. |
| May 13, 2026 | [TFlow](https://arxiv.org/abs/2605.13839) introduced query-specific inter-agent communication through transient LoRA perturbations and reported up to 83.27% fewer processed tokens and up to 4.6× lower wall-clock time than its text-based three-agent comparator. | A below-prompt, per-query communication surface now exists and needs an independent control layer. |
| May 17, 2026 | [ADR](https://arxiv.org/abs/2605.17380) reported a production agent-detection system at Uber with fast telemetry triage, context-aware reasoning, and hard-example generation. | A sensor → cheap learned triage → deeper diagnostic architecture is commercially credible. ADR monitors MCP/trajectory surfaces; weight geometry remains the differentiation here. |
| Jun 18, 2026 | Google DeepMind's [AI Control Roadmap](https://deepmind.google/blog/securing-the-future-of-ai-agents/) described analysis of one million coding-agent tasks, work used in a live Gemini Spark monitor, and detection plus prevention/response tiers. | Keep immutable capability and response ceilings outside the learned policy. |
| Jul 15, 2026 | OpenAI published [GPT-Red](https://openai.com/index/unlocking-self-improvement-gpt-red/), an automated red-team model used to adversarially train GPT-5.6; OpenAI reports six times fewer failures on its hardest direct prompt-injection benchmark than its best production model four months earlier. | Use sealed defensive hard-example generation, with disjoint evaluation and explicit promotion gates. |

These sources motivate the architecture; they do not validate this repository's
detector, transfer their reported metrics to this implementation, or prove that any
particular deployment is protected.

## Why this is the strongest commercial wedge

The demand signal is unusually explicit in 2026:

- The Cloud Security Alliance's April 2026 survey of 418 IT/security professionals
  reports that 82% had discovered unknown agents, 65% reported an agent-related
  incident in the prior year, and 79% considered context-aware controls important. It
  was vendor-commissioned, so this is demand evidence, not a neutral prevalence
  estimate. [CSA release and methodology](https://cloudsecurityalliance.org/press-releases/2026/04/21/new-cloud-security-alliance-survey-reveals-82-of-enterprises-have-unknown-ai-agents-in-their-environments)
- IBM's June 2026 study of 2,000 technology executives reports that 11% felt fully
  prepared for expected agent scale and 59% named security/compliance as a top barrier.
  Its reported associations are not causal product-performance evidence.
  [IBM study](https://newsroom.ibm.com/2026-06-08-new-ibm-study-finds-cios-and-ctos-face-growing-ai-control-gap-as-enterprise-deployment-scales)
- McKinsey's 2026 trust survey says nearly two-thirds of respondents named security and
  risk as the top barrier to scaling agentic AI.
  [McKinsey survey](https://www.mckinsey.com/capabilities/tech-and-ai/our-insights/tech-forward/state-of-ai-trust-in-2026-shifting-to-the-agentic-era)

Most agent-security products observe prompts, tools, identities, or endpoint actions.
WSO's defensible difference is the **weight and adapter surface below those logs**.

A sensible commercial sequence is:

1. Scan adapters in CI and private registries against a per-base-model trusted bank.
2. Shadow runtime TFlow/transient-update decisions inside the customer's VPC.
3. Add read-only deep diagnostics and SIEM/incident-system feedback.
4. Enable bounded automatic observation or throttling only after real holdout gates.
5. Keep quarantine and rollback behind a customer-owned executor and approval policy.

Pricing must come from design-partner willingness-to-pay and avoided-incident cost. No
revenue or ARR number in this document is presented as measured evidence.

## Implemented brain (state schema 2.1)

`wso-brain` is deliberately independent of the repository's theorem forest:

```text
derived LoRA/TFlow telemetry
        │
        ├── bounded-work + fail-closed real-provenance gate
        ├── exact trusted-context gate
        ├── spectral/geometry/context encoder
        ├── slow trusted baseline + fast drift memory
        ├── online incident classifier + uncertainty
        └── retrieved realized-outcome memory
        ▼
constrained contextual bandit
        │
operator-owned action ceiling
        │
per-mode durable action intent (`run_id` + `module_id`)
        │
typed executor → validated receipt or human-review fallback
        │
committed attempt or explicit operator reconciliation
        │
one verified disposition per decision
        ▼
classifier update; policy/memory update only from realized outcomes
        │
sealed feature-space challenger
        ▼
disjoint original/worst-hard-variant seed gates + separate benign confidence gate
```

### Enforced v2.1 contracts

**Trusted-baseline-only context admission.** Only records supplied through
`--baseline` add an authorized context. `--labeled` examples can train the classifier
when they contain an explicit label, but they never expand the context allowlist. The
key is an exact match over `task_family`, `module_id`, `collection_mode`, `model_id`,
`adapter_id`, `source_kind`, `runtime_framework`, `observer_level`, `evidence_mode`, and
`deployment_id`. An unseen full key can yield only `deep_scan` or `human_review`; it
cannot yield `allow` or `observe`.

**Real provenance is fail-closed and out of band.** Every `evidence_mode="real"` record
is forced to `human_review` by default, even if its JSON contains provenance-looking
fields. The brain CLI deliberately has no provenance-verification flag and calls the
runtime with the default `provenance_verified=False`; telemetry JSON cannot set the
function argument. Only a trusted in-process integration may call `decide(...)` or
`reconcile_action_attempt(...)` with `provenance_verified=True`, or pass
`brain_provenance_verified=True` through `OfficialTFlowAdapter.solve(...)`.

That assertion is necessary but not sufficient. A real record also needs a positive
`collection_timestamp`; `source_kind` and `runtime_framework` tokens, plus either a
`collector_version` token or a 7-to-64-digit hexadecimal `git_commit`; and a nonempty
`source_path_hashes` map whose every value is exactly 64 lowercase hexadecimal digits.
Structured tokens are 1 to 128 characters, start alphanumerically, and otherwise allow
only letters, digits, `_`, `.`, `:`, `/`, `+`, `@`, and `-`. These checks establish
shape and bind a trusted caller's assertion to a record. They do not verify a signature,
certificate, hash precommitment, caller identity, or the truth of the asserted
provenance, so production still needs authenticated ingestion and attestation.

**Slow trust and fast drift are separate.** Every scored observation may update fast
drift state. Only explicit bootstrap baseline data establish known contexts, and only
verified benign feedback updates the slow novelty baseline. The learner never treats
its own prediction as a label.

**Action identity is upstream identity, not mutable provenance.** The observation key
is the SHA-256 of canonical JSON containing only `run_id` and `module_id`. The complete
canonical telemetry payload, except its top-level timestamp, is hashed separately. A
retry may change that timestamp or dictionary insertion order, but reusing the identity
with any other payload change is rejected. This check runs against both shadow and
enforce records, so changes to `source_kind`, `source_path_hashes`, `model_id`,
`adapter_id`, `deployment_id`, or any other payload field cannot silently reuse the
prior identity. The payload digest also includes the out-of-band
`provenance_verified` boolean, so a previously unverified identity cannot be silently
replayed as verified. The code does not authenticate `run_id` or the attestation;
production ingestion must provide authenticated, globally unique upstream run IDs and
an authenticated provenance-verification boundary.

**Action attempts are per mode.** Shadow and enforce use separate attempt namespaces,
so the same observation can have distinct shadow and enforce decision IDs. Within a
mode, an exact retry of a committed attempt returns the existing decision without
invoking the executor again; only an attested `abandoned_not_executed` intent releases
a new attempt. An atomic intent record is written before execution; the executor runs
outside the state lock; then the outcome and state commit are written under the lock.
If a crash leaves only an intent, the code cannot know whether an external effect
occurred and blocks the retry.

**Orphaned intents require reconciliation.** `wso-brain reconcile-action` accepts one
record whose canonical payload matches the intent (the top-level timestamp may differ)
and the original mode. `--attest-not-executed` records an operator attestation and
permits one safe retry; `--keep-blocked` records that execution is uncertain and keeps
retries blocked. Neither option infers an external result.

**Locked, one-disposition feedback with exact retry/rebase.** State operations use an
exclusive POSIX advisory lock and reload state while holding it. A committed retry with
the same `feedback_id` and timestamp-independent payload fingerprint is a no-op. If an
exact transaction stopped after its durable intent but before its state commit, the
same retry may rebase its expected model-version fence after an unrelated commit and
then apply once. If state committed but the final disposition or applied audit record
did not, the persisted commit marker repairs those records without relearning. A
different or conflicting verified disposition is rejected; unverified feedback is
logged but changes no learned state. This exact-retry/rebase contract is available to
API callers that retain the same feedback object and to CLI callers that supply
`--feedback-id`. A caller who needs crash recovery must choose and durably persist that
stable ID before the first submission, then reuse the ID with the exact same payload.
Omitting the option still generates a new ID. A different or conflicting disposition
cannot be forced through this mechanism, and there is no arbitrary conflict-override or
`reconcile-feedback` CLI.

**Realized outcomes only for policy reward.** Every verified disposition trains the
incident classifier; verified benign dispositions also update the slow novelty
baseline. The contextual policy and retrieved strategy memory update only when an
observed action outcome is supplied. Outcome fields are `actual_action`,
`action_succeeded`, `containment_succeeded`, `realized_cost`, `disruption`, and
`latency_ms`. An incident needs a containment outcome; a benign case needs an action
success/failure outcome. The reward uses those observed values and does not assign
counterfactual rewards to actions that were not taken.

**Receipt validation fails closed locally, after execution.** Shadow mode never invokes
an executor and records an internal `observe` receipt. Enforce mode accepts an executor
receipt only when requested action, executed action, and executor identity exactly
match, the receipt does not escalate severity, and `success` is true. Exceptions raised
by the executor, unsupported actions, ceiling violations, mismatched `ActionReceipt`
fields, and failed receipts become an internal `human_review` event. A protocol-violating
executor that returns a non-`ActionReceipt` object also receives the internal
`human_review` fallback. The default executor is read-only and has no external
quarantine capability. However, validation runs only after `execute(...)` returns: if a
custom executor already performed a stronger side effect, rejecting its result and
recording `human_review` cannot undo that effect. Production must use capability-scoped
executor credentials plus independent external effect verification and
rollback/reconciliation controls.

**Telemetry and defensive work are explicitly bounded.** Every telemetry JSONL file
loaded by `wso-brain` is limited to 32 MiB, each line to 1 MiB, and each file to 4,096
nonblank records. The feature encoder and local spectral scan accept at most 4,096
singular values. Feature encoding additionally enforces these per-record budgets:

- at most 1,024 fusion-gate logits and 1,024 fusion-gate probabilities;
- at most 8 layer-shape dimensions, each between 0 and `2**31 - 1`;
- at most 1,024 sender identifiers;
- `sender_count`, `target_rank`, and `rank_bound` no greater than 1,000,000;
- at most 4,096 UTF-8 bytes for each context or sender-identifier string; and
- for each of `source_path_hashes`, `otel_resource`, and `provenance`, a metadata tree
  no deeper than 8, with at most 2,048 nodes and at most 65,536 accounted bytes.

Metadata is restricted to exact plain JSON-like shapes: an exact `dict` with exact
`str` keys, an exact `list`, or exact `str`, `None`, `bool`, `int`, and finite `float`
scalars. Container/scalar subclasses, tuples, sets, frozensets, bytes, non-string object
keys, arbitrary objects, and NaN or infinite floats are rejected. The byte budget
charges dictionary/list punctuation and separators plus string and scalar/numeric
serialization work; it is not merely a sum of string payload bytes.

Traversal also preflights container fanout before extending its work stack. For a
dictionary it charges both keys and values; for a list it charges every element. If the
nodes already visited, children already queued, and immediate children would exceed
2,048, the record is rejected before those children are iterated or queued. The node
limit therefore bounds the traversal queue as well as completed work.

Self-play separately accepts at most 256 incident seeds, 256 benign holdout records,
and 128 generated candidates per incident. These are deterministic memory/CPU work
bounds, not evidence that accepted telemetry or datasets are authentic or benign.

**Self-play separates seed groups and screens obvious overlap.** Incident and benign
records need explicit, consistent labels and unique run IDs. Encoded feature hashes
must also be unique and nonoverlapping, and exact/near-duplicate feature vectors are
rejected within and across the two datasets. Incident seed order is deterministically
shuffled, then split approximately 60/40. All hard variants from a seed remain in its
train or evaluation group; the separately supplied benign holdout is evaluated only
and is never fitted by that self-play call. There is no lineage registry proving that
an evaluation seed or benign record did not influence earlier bootstrap or feedback
learning; the cloned incumbent may therefore already contain information from it.
These checks reduce direct leakage inside this routine. They do **not** establish that
the holdout is statistically independent or i.i.d., authentic, or separated by source,
deployment, adapter, or time; those remain operator responsibilities.

**Promotion uses powered, absolute confidence gates.** Defaults are 95% promotion
confidence, a 5% benign-FPR ceiling, and a 20% incident-evasion ceiling. At those
defaults, promotion requires at least 59 benign records and at least 14 held-out
incident seed groups. Because the default seed split is approximately 60/40, 14 held-out
groups require at least 34 incident seeds overall. At these minimum sample sizes, any
observed false positive or evasion fails its respective gate.

`BrainConfig` rejects a configured confidence/rate pair when its zero-event requirement
cannot fit inside those fixed caps. Conceptually, for rate ceiling `r` and confidence
`c`, the required count is `ceil(log(1-c) / log(1-r))`. Feasibility is implemented
without that overflow-prone division: it compares
`log1p(-c) >= capacity * log1p(-r)`. This retains subnormal positive rates and safely
rejects their enormous requirements. The benign false-positive capacity is 256. The
incident-evasion capacity is 102 held-out seeds: the largest permitted 256-incident
input assigns `round(256 * 0.6) = 154` seeds to training and leaves 102 for evaluation.
This validation happens when the immutable config is constructed, before a self-play
run can request an impossible promotion sample.

For every run, self-play report schema `2.1` computes exact one-sided binomial upper
confidence bounds for post-training benign false positives, evasion on the original
held-out incident seeds, and seed-level hard-variant evasion. The last metric groups
held-out generated variants by their original seed, takes the minimum post-training
risk for each seed, and counts that seed as evaded when its worst variant is below the
operating review threshold. Promotion requires all three upper bounds—not only the
empirical rates—to be at or below their configured absolute ceilings. It also requires:

- a changed SHA-256 of serialized classifier state, rejecting a no-op;
- no regression in mean risk on held-out hard variants, empirical evasion on original
  held-out seeds, empirical worst-hard-variant evasion by seed, or mean risk on the
  original held-out seeds;
- either at least `0.01` higher mean risk on held-out hard variants or a strict fall in
  original-seed evasion;
- an unchanged brain model version when the promoted challenger is installed.

The report includes both incident empirical rates, all three exact upper bounds, sample
counts, operating threshold, classifier hashes, canonical incident and benign dataset
hashes, and model versions. Promotion additionally requires both
`--incident-dataset-hash` and
`--benign-holdout-hash` to match the canonical supplied datasets. Evaluation-only
`--no-promote` runs do not require those expected hashes or the promotion sample minima.

**Hashes identify content; they do not authenticate it.** Each required dataset-hash
check proves only equality with the value supplied to this invocation. Neither pin
proves that its value was precommitted earlier, signed, independently selected, or
uncontaminated. Canonical dataset hashes include record timestamps, unlike action
payload digests. A production protocol should pre-register and sign both incident and
benign datasets before training, then group their splits by source, deployment,
adapter, and time rather than relying only on record-level shuffling.

**Private local state.** `brain_state.json` is strictly loaded as state schema `2.1`;
other state versions are rejected rather than migrated. This is distinct from the
current `BrainConfig` schema (`2.0`), `SelfPlayReport` schema (`2.1`), and feature schema
(`weight-space-v2`). Managed JSON state, per-observation action attempts, per-decision
feedback dispositions, JSONL ledgers, and the lock file are written mode `0600`; state
and transaction directories are mode `0700`. JSONL writes use append mode, flush, and
`fsync`, and JSON state/transaction records use atomic replacement. The model state is
inspectable JSON rather than pickle, and raw prompts and tensors are excluded from brain
feature and learning-ledger payloads.

**Brain CLI output replacement pins the parent directory.** For `decide --out` and
`self-play --out`, the CLI first rejects paths inside the brain state tree and aliases
of protected state or input files. It then opens the resolved parent as a directory file
descriptor, rechecks the pinned directory's ancestry and input identity, creates and
`fsync`s a mode-`0600` temporary file relative to that descriptor, rechecks again, and
uses directory-descriptor-relative `replace` followed by a parent-directory `fsync`.
Holding the same parent descriptor across temporary creation and replacement closes the
symlink-parent swap race for these writes. This is local path and crash hardening, not
encryption, tamper evidence, or a transaction with another process.

### Commands

```bash
# Fully offline synthetic closed-loop fixture
wso-brain demo --out-dir artifacts/brain_demo
wso-brain status --state-dir artifacts/brain_demo/brain

# Real workflow: context admission comes only from this trusted baseline
wso-brain bootstrap \
  --state-dir brain_state \
  --baseline baseline_telemetry.jsonl \
  --labeled adjudicated_telemetry.jsonl

# Shadow records recommendations but never calls the executor
wso-brain decide \
  --state-dir brain_state \
  --telemetry audit_telemetry.jsonl \
  --mode shadow \
  --out decisions.jsonl

# Supply the action actually taken and its realized operational outcome.
# Persist this ID before submission and reuse the same ID + payload after a crash.
wso-brain feedback \
  --state-dir brain_state \
  --decision-id DECISION_ID \
  --feedback-id FEEDBACK_TXN_ID \
  --incident \
  --actual-action human_review \
  --action-succeeded \
  --not-contained \
  --realized-cost 1.0 \
  --disruption 0.2 \
  --latency-ms 120000

# Resolve one orphaned action intent. Keep it blocked when execution is uncertain.
wso-brain reconcile-action \
  --state-dir brain_state \
  --telemetry one_original_observation.jsonl \
  --mode enforce \
  --keep-blocked

# Use this alternative only after verifying that the external action did not occur.
wso-brain reconcile-action \
  --state-dir brain_state \
  --telemetry one_original_observation.jsonl \
  --mode enforce \
  --attest-not-executed

# Evaluate first; this does not require dataset pins or promotion minima.
wso-brain self-play \
  --state-dir brain_state \
  --incidents verified_incidents.jsonl \
  --benign-holdout frozen_benign_holdout.jsonl \
  --no-promote \
  --out self_play_report.json

# Promotion requires out-of-band expected canonical hashes for both datasets.
wso-brain self-play \
  --state-dir brain_state \
  --incidents preregistered_incidents.jsonl \
  --benign-holdout preregistered_benign_holdout.jsonl \
  --incident-dataset-hash EXPECTED_INCIDENT_SHA256 \
  --benign-holdout-hash EXPECTED_BENIGN_SHA256 \
  --out promotion_report.json
```

There is intentionally no CLI switch or telemetry-JSON field that can set
`provenance_verified`. Consequently, `wso-brain decide` sends real evidence to
`human_review` by default. A trusted in-process integration that takes responsibility
for the out-of-band attestation must call the API directly; reconciliation must reuse
the same attestation because its value is part of the original payload digest.

## Honest boundary and next proof

The checked-in demo is a **synthetic fixture**. It proves deterministic closed-loop
mechanics, schema behavior, local persistence, and gate execution only. Its AUROC,
false-positive rate, self-play result, and promotion outcome are not deployment
evidence, and the generated feature vectors are not real adapters, triggers, prompts,
payloads, or monitor-bypass artifacts.

Within that fixture, all 128 audit records are scored. Feedback learning uses 16 of the
64 incident records, and those run IDs are excluded from the remaining 48 incident
records used as self-play seeds. All 64 benign audit records remain unfitted: they are
not sent through feedback and are never challenger training examples, although scoring
them still updates the fast observational drift state. This removes direct incident
feedback/self-play reuse in the demo; it does not turn synthetic data into deployment
evidence or prove external dataset independence.

The remaining boundaries are concrete:

- The generic PEFT collector creates static adapter snapshots and does not execute the
  supplied prompts. It is not runtime behavioral evidence. The external TFlow adapter
  is the only current genuine transient-inference capture path, but it constructs
  telemetry and calls `brain.decide(...)` only after the official `solve(...)` call has
  returned. This is an asynchronous monitoring boundary relative to generation: it
  cannot block the response that produced the telemetry. Capture temporarily
  monkey-patches the official module's `build_weighted_lora_entries` function, so a
  process-global reentrant lock serializes cross-thread capture windows, including each
  official `solve(...)`. A second thread waits until the active window has restored the
  original hook. Because the lock is reentrant, a separate active-window guard rejects
  nested capture on the same thread with an error rather than installing a second
  monkey-patch. Hook installation and the yielded solve window are protected by nested
  `finally` cleanup: while Python can still unwind, an exception or interruption such
  as `KeyboardInterrupt` restores the original hook and clears the active-window flag,
  including when interruption occurs during hook installation. Cleanup cannot be
  guaranteed if the restoration assignment itself fails or the process is abruptly
  terminated. This prevents normal overlapping solves in one process from mixing
  captures or restoring stale hooks, but it also serializes in-process throughput. A
  production integration needs an upstream request-scoped callback or hook rather than
  process-global monkey-patching.
- A `--baseline` is trusted because the operator supplies it as such; v2.1 does not
  authenticate that trust decision. Exact context matching is a categorical allowlist,
  not statistical proof that a context is safe.
- The out-of-band `provenance_verified` or TFlow `brain_provenance_verified` boolean is
  available only to a trusted in-process caller; the CLI cannot derive it from JSON.
  It is included in action payload identity and still requires structured provenance
  fields and lowercase 64-hex source hashes. It is not a signature or cryptographic
  authentication of the caller, record, source, or hash contents.
- Self-play enforces unique run IDs, exact encoded-feature separation, near-duplicate
  screens, and disjoint seed-level train/evaluation groups. It does not verify i.i.d.,
  source, deployment, adapter, or time separation, nor dataset authenticity. Exact
  one-sided confidence bounds for benign false positives, original-seed evasion, and
  worst-hard-variant evasion by seed quantify sampling uncertainty conditional on the
  supplied records; they do not create deployment-generalization evidence.
- Classifier and dataset hashes identify canonical content in a report. Both expected
  dataset hashes are required for promotion, but neither is a signature or proof that
  the dataset was frozen before model training. Production needs authenticated
  pre-registration and signatures for both datasets and their grouped split manifest.
- Local action idempotency trusts the upstream `run_id` plus `module_id`. Without an
  authenticated globally unique run ID, an attacker or broken producer can create
  collisions, conflicts, or denial of service. Payload-digest checks detect changed
  source/model/adapter/deployment fields after identity reuse but do not authenticate
  the original values.
- POSIX `flock` coordinates cooperating processes on one filesystem. It is not a
  distributed transaction, multi-host consensus, or high-availability state store.
- Mode `0600` limits local Unix file access. It is not encryption, key management,
  tamper evidence, or OS-level immutability. Brain CLI `--out` files now use a private
  temporary file and pinned-parent atomic replacement, but the caller still selects the
  destination directory and other `wso-audit` outputs are outside this brain-specific
  path-hardening contract.
- CLI, feature, spectral, metadata, and self-play caps bound individual work units; they
  do not provide admission control, rate limiting across repeated invocations, dataset
  authenticity, or a proof that an accepted record is inexpensive for every downstream
  third-party integration.
- Receipt validation occurs in-process after executor code runs. It cannot prevent or
  undo a stronger side effect already performed by a faulty or hostile executor, and it
  does not prove that the declared external effect occurred. Capability-scoped
  credentials, external effect verification, idempotency, and rollback remain required.
  The executor receives the requested action and telemetry, but not the durable intent
  or decision ID, so the current interface does not provide external exactly-once
  execution.
- Exact feedback recovery requires reuse of the same `feedback_id` and fingerprint.
  The feedback CLI supports this through optional `--feedback-id`, but the caller must
  persist the ID before the first submission and repeat the exact payload. It does not
  authorize a conflicting second disposition, and no arbitrary feedback-conflict
  reconciliation is implemented.
- The policy learns only logged outcomes of actions actually reported. There is no
  causal or counterfactual policy evaluation and no reward for unobserved alternatives.

Before an autonomous production claim, the next evidence must include:

1. an upstream request-scoped TFlow capture hook that supports safe concurrent solves,
   plus synchronous scoring inside that hook before receiver generation where
   prevention is required, replacing the current serialized process-global
   monkey-patch and post-generation monitoring boundary;
2. per-projection Q/K/V/O metrics rather than only aggregate snapshots;
3. authenticated, pre-registered, signed incident and benign datasets, with seed groups
   split by deployment, adapter, source, and time before training;
4. calibration, recall at a fixed false-positive rate, confidence intervals, Brier
   score, subgroup checks, latency, and action-cost comparisons against fixed bands and
   always-scan baselines;
5. authenticated globally unique run IDs, replayable authenticated feedback IDs,
   signed candidate/champion artifacts, rollback, and poisoning/drift freezes;
6. capability-scoped customer-owned executor credentials plus external effect
   verification for any actual throttle, rollback, or quarantine action.

Until those exist, the accurate product statement is: **a functioning, bounded learned
control-plane prototype over derived weight-space telemetry, with synthetic end-to-end
validation and a production integration path—not a deployed or universally validated
detector.**
