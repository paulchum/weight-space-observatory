# Telemetry Schema

The core schema is `TransientLoRATelemetry`, defined in `src/wso/telemetry/schemas.py`.

It stores derived signatures rather than raw tensors:

- schema version, collection mode, model ID, adapter ID, baseline ID, and metric version;
- run ID, timestamp, task family, prompt hash, module ID, layer shape, sender count, and target rank;
- singular values;
- spectral gap;
- effective rank;
- Frobenius norm;
- fusion gate probabilities and entropy;
- Grassmannian drift;
- provenance metadata.

Related v1 schemas:

- `MetricBand`: calibrated quantile band for one metric.
- `BaselineProfile`: task/module/mode/model baseline profile.
- `AnomalyFinding`: per-record analysis result against a matched baseline.
- `CapacityFrontierReport`: report-ready synthetic capacity-detectability benchmark summary.
