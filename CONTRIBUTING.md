# Contributing to Weight-Space Observatory

Thanks for helping make below-prompt agent behavior observable.

The project is defensive by design. The most valuable contributions improve honest
measurement, reproducibility, runtime coverage, or operator control.

## High-value contributions

- collectors for real transient-adapter runtimes;
- frozen, provenance-rich evaluation packs;
- narrowly capability-scoped action executors;
- calibration, confidence-bound, and drift-analysis improvements;
- reproducible failure cases and false-positive reports;
- accessibility, documentation, and operator-experience improvements.

Please do not submit covert-channel encoders, monitor-bypass techniques, payload-hiding
systems, keyed extraction protocols, or operational evasion tooling. See
[SECURITY.md](SECURITY.md) for the complete boundary.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

ruff check .
mypy src/wso
pytest -q
```

Run the autonomous gateway directly:

```bash
wso-brain demo --out-dir artifacts/brain_demo
wso-brain status --state-dir artifacts/brain_demo/brain
```

## Pull requests

Keep changes focused and include:

1. the problem and intended operator impact;
2. the trust boundary or failure mode affected;
3. tests that cover the new behavior;
4. exact commands used for validation;
5. honest limits on what the evidence proves.

Changes to promotion criteria, provenance, action execution, or persistent state should
include adversarial tests and backward-compatibility notes.

## Reporting security issues

Do not open a public issue for a vulnerability that could enable unauthorized model
actions or sensitive-data exposure. Use GitHub's private vulnerability reporting when
available, and include a minimal defensive reproduction.
