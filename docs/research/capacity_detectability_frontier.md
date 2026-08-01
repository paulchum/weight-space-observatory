# Capacity–Detectability Frontier

This project measures whether task-independent signals can be detected or bounded in transient adapter telemetry.

Defensive measurements include:

- recoverability of synthetic, non-task labels from multi-metric telemetry summaries;
- task-conditioned anomaly scores;
- robustness to measured implementation noise;
- drift from benign subspace baselines.

This repository does not implement an operational payload-hiding system.

The v1 CLI command is:

```bash
wso-audit benchmark capacity --out frontier.csv --plot frontier.png
```

The command writes CSV and JSON frontier summaries. The PNG plot is generated locally
from aggregate recoverability AUC and anomaly rate, without storing prompts, raw
tensors, payloads, or extraction artifacts.
