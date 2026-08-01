"""Defensive capacity-detectability curve simulation.

This module estimates whether synthetic audit labels are statistically separable
from telemetry-like summaries before those summaries become anomalous under a
benign baseline. It is intentionally abstract: it does not encode payloads into
LoRA factors, define a key, or provide a receiver-side extraction protocol.
"""

from __future__ import annotations

import csv
import json
import random
import struct
import time
import zlib
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from statistics import mean

from wso.telemetry.schemas import CapacityFrontierPoint, CapacityFrontierReport, MetricBand


SYNTHETIC_METRICS = {
    "fusion_entropy": {"mean": 0.65, "std": 0.10, "direction": 1.0, "weight": 0.65},
    "frobenius_norm": {"mean": 9.50, "std": 1.20, "direction": 1.0, "weight": 0.80},
    "effective_rank": {"mean": 6.50, "std": 0.85, "direction": 1.0, "weight": 0.70},
    "spectral_gap": {"mean": 0.28, "std": 0.055, "direction": -1.0, "weight": 0.45},
    "grassmann_drift": {"mean": 0.55, "std": 0.14, "direction": 1.0, "weight": 1.00},
}

@dataclass(frozen=True)
class CapacityPoint:
    """One point on a defensive capacity-detectability frontier."""

    signal_strength: float
    recoverability_auc: float
    anomaly_rate: float
    benign_lower: float
    benign_upper: float
    metric_aucs: dict[str, float] = field(default_factory=dict)
    metric_anomaly_rates: dict[str, float] = field(default_factory=dict)
    benign_bands: dict[str, MetricBand] = field(default_factory=dict)


def simulate_frontier(
    signal_strengths: list[float],
    *,
    samples_per_strength: int = 400,
    benign_samples: int = 1000,
    noise_std: float = 1.0,
    seed: int = 7,
) -> list[CapacityPoint]:
    """Return synthetic frontier points for defensive measurement planning.

    The generated metrics are stand-ins for derived telemetry summaries such as
    spectral gap, subspace drift, and fusion-gate statistics. The synthetic label
    is an offline audit annotation used only to measure separability.
    """

    rng = random.Random(seed)
    benign_rows = [_sample_metric_row(rng, 0.0, 0, noise_std) for _ in range(benign_samples)]
    benign_scores = [_aggregate_score(row) for row in benign_rows]
    lower = _quantile(benign_scores, 0.05)
    upper = _quantile(benign_scores, 0.95)
    benign_bands = _benign_metric_bands(benign_rows)

    points: list[CapacityPoint] = []
    for strength in signal_strengths:
        aggregate_scored: list[tuple[float, int]] = []
        metric_scored: dict[str, list[tuple[float, int]]] = {
            metric: [] for metric in SYNTHETIC_METRICS
        }
        anomaly_hits = 0
        metric_anomaly_hits = {metric: 0 for metric in SYNTHETIC_METRICS}

        for idx in range(samples_per_strength):
            label = idx % 2
            row = _sample_metric_row(rng, strength, label, noise_std)
            score = _aggregate_score(row)
            aggregate_scored.append((score, label))

            row_is_anomalous = False
            for metric, value in row.items():
                config = SYNTHETIC_METRICS[metric]
                signed_value = value * float(config["direction"])
                metric_scored[metric].append((signed_value, label))
                if not benign_bands[metric].contains(value):
                    metric_anomaly_hits[metric] += 1
                    row_is_anomalous = True
            if row_is_anomalous:
                anomaly_hits += 1

        auc = _rank_auc(aggregate_scored)
        anomaly_rate = anomaly_hits / float(samples_per_strength)
        points.append(
            CapacityPoint(
                signal_strength=strength,
                recoverability_auc=auc,
                anomaly_rate=anomaly_rate,
                benign_lower=lower,
                benign_upper=upper,
                metric_aucs={
                    metric: _rank_auc(scored) for metric, scored in metric_scored.items()
                },
                metric_anomaly_rates={
                    metric: hits / float(samples_per_strength)
                    for metric, hits in metric_anomaly_hits.items()
                },
                benign_bands=benign_bands,
            )
        )
    return points


def default_signal_strengths(max_strength: float = 3.0, steps: int = 13) -> list[float]:
    """Return evenly spaced signal strengths for a compact frontier sweep."""

    if steps < 2:
        return [0.0]
    step = max_strength / float(steps - 1)
    return [round(i * step, 6) for i in range(steps)]


def write_frontier_csv(points: list[CapacityPoint], output: Path) -> None:
    """Write frontier points to CSV."""

    metric_fields = [
        f"{metric}_{suffix}"
        for metric in SYNTHETIC_METRICS
        for suffix in ("auc", "anomaly_rate")
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "signal_strength",
                "recoverability_auc",
                "anomaly_rate",
                "benign_lower",
                "benign_upper",
                *metric_fields,
            ],
        )
        writer.writeheader()
        for point in points:
            row = {
                "signal_strength": point.signal_strength,
                "recoverability_auc": point.recoverability_auc,
                "anomaly_rate": point.anomaly_rate,
                "benign_lower": point.benign_lower,
                "benign_upper": point.benign_upper,
            }
            for metric in SYNTHETIC_METRICS:
                row[f"{metric}_auc"] = point.metric_aucs.get(metric, 0.0)
                row[f"{metric}_anomaly_rate"] = point.metric_anomaly_rates.get(metric, 0.0)
            writer.writerow(row)


def build_frontier_report(
    points: list[CapacityPoint],
    *,
    samples_per_strength: int,
    benign_samples: int,
    noise_std: float,
    seed: int,
) -> CapacityFrontierReport:
    """Convert dataclass points to a JSON/report schema."""

    benchmark_id = f"capacity-{int(time.time())}-{seed}"
    return CapacityFrontierReport(
        benchmark_id=benchmark_id,
        samples_per_strength=samples_per_strength,
        benign_samples=benign_samples,
        noise_std=noise_std,
        seed=seed,
        points=[
            CapacityFrontierPoint(
                signal_strength=point.signal_strength,
                recoverability_auc=point.recoverability_auc,
                anomaly_rate=point.anomaly_rate,
                metric_aucs=point.metric_aucs,
                metric_anomaly_rates=point.metric_anomaly_rates,
                benign_bands=point.benign_bands,
            )
            for point in points
        ],
    )


def write_frontier_json(report: CapacityFrontierReport, output: Path) -> None:
    """Write a report-ready frontier summary as JSON."""

    output.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")


def write_frontier_plot(points: list[CapacityPoint], output: Path) -> None:
    """Write a small PNG plot using only the Python standard library."""

    width, height = 900, 520
    margin_left, margin_right, margin_top, margin_bottom = 72, 32, 36, 70
    pixels = bytearray([255] * width * height * 3)

    def set_pixel(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            idx = (y * width + x) * 3
            pixels[idx : idx + 3] = bytes(color)

    def draw_line(
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    set_pixel(x0 + ox, y0 + oy, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    if not points:
        _write_png(output, width, height, pixels)
        return

    x_min = min(point.signal_strength for point in points)
    x_max = max(point.signal_strength for point in points)
    if x_min == x_max:
        x_max = x_min + 1.0

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def project(signal: float, value: float) -> tuple[int, int]:
        x = margin_left + int((signal - x_min) / (x_max - x_min) * plot_width)
        y = margin_top + int((1.0 - max(0.0, min(1.0, value))) * plot_height)
        return x, y

    axis_color = (31, 41, 55)
    grid_color = (229, 231, 235)
    draw_line((margin_left, margin_top), (margin_left, height - margin_bottom), axis_color)
    draw_line((margin_left, height - margin_bottom), (width - margin_right, height - margin_bottom), axis_color)
    for idx in range(1, 5):
        y = margin_top + int(idx * plot_height / 5)
        draw_line((margin_left, y), (width - margin_right, y), grid_color)

    sorted_points = sorted(points, key=lambda point: point.signal_strength)
    auc_points = [project(point.signal_strength, point.recoverability_auc) for point in sorted_points]
    anomaly_points = [project(point.signal_strength, point.anomaly_rate) for point in sorted_points]
    for left, right in zip(auc_points, auc_points[1:]):
        draw_line(left, right, (37, 99, 235))
    for left, right in zip(anomaly_points, anomaly_points[1:]):
        draw_line(left, right, (220, 38, 38))
    for point in auc_points:
        draw_line((point[0] - 3, point[1]), (point[0] + 3, point[1]), (37, 99, 235))
        draw_line((point[0], point[1] - 3), (point[0], point[1] + 3), (37, 99, 235))
    for point in anomaly_points:
        draw_line((point[0] - 3, point[1]), (point[0] + 3, point[1]), (220, 38, 38))
        draw_line((point[0], point[1] - 3), (point[0], point[1] + 3), (220, 38, 38))

    _write_png(output, width, height, pixels)


def summarize_frontier(points: list[CapacityPoint]) -> str:
    """Return a short text summary for terminal use."""

    if not points:
        return "No frontier points generated."
    lowest_anomaly = min(points, key=lambda p: (p.anomaly_rate, -p.recoverability_auc))
    highest_auc = max(points, key=lambda p: p.recoverability_auc)
    return (
        "Capacity-detectability summary\n"
        f"- points: {len(points)}\n"
        f"- benign band: [{points[0].benign_lower:.3f}, {points[0].benign_upper:.3f}]\n"
        f"- highest recoverability AUC: {highest_auc.recoverability_auc:.3f} "
        f"at signal={highest_auc.signal_strength:.3f}, "
        f"anomaly_rate={highest_auc.anomaly_rate:.3f}\n"
        f"- lowest anomaly point: signal={lowest_anomaly.signal_strength:.3f}, "
        f"AUC={lowest_anomaly.recoverability_auc:.3f}, "
        f"anomaly_rate={lowest_anomaly.anomaly_rate:.3f}"
    )


def _sample_metric_row(
    rng: random.Random,
    signal_strength: float,
    label: int,
    noise_std: float,
) -> dict[str, float]:
    row: dict[str, float] = {}
    for metric, config in SYNTHETIC_METRICS.items():
        base_mean = float(config["mean"])
        base_std = float(config["std"]) * noise_std
        direction = float(config["direction"])
        weight = float(config["weight"])
        shift = direction * signal_strength * weight * base_std if label else 0.0
        row[metric] = rng.gauss(base_mean + shift, base_std)
    return row


def _aggregate_score(row: dict[str, float]) -> float:
    signed_z_scores = []
    for metric, value in row.items():
        config = SYNTHETIC_METRICS[metric]
        base_mean = float(config["mean"])
        base_std = float(config["std"])
        direction = float(config["direction"])
        signed_z_scores.append(direction * (value - base_mean) / base_std)
    return float(mean(signed_z_scores))


def _benign_metric_bands(rows: list[dict[str, float]]) -> dict[str, MetricBand]:
    bands: dict[str, MetricBand] = {}
    for metric in SYNTHETIC_METRICS:
        values = [row[metric] for row in rows]
        bands[metric] = MetricBand(
            metric=metric,
            lower=_quantile(values, 0.05),
            upper=_quantile(values, 0.95),
            sample_count=len(values),
        )
    return bands


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a quantile for an empty list")
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def _rank_auc(scored_labels: list[tuple[float, int]]) -> float:
    """Compute binary AUC with average ranks for ties."""

    positives = sum(label == 1 for _, label in scored_labels)
    negatives = sum(label == 0 for _, label in scored_labels)
    if positives == 0 or negatives == 0:
        raise ValueError("AUC requires at least one positive and one negative label")

    ordered = sorted(scored_labels, key=lambda item: item[0])
    rank_sum = 0.0
    rank = 1
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        average_rank = (rank + rank + (j - i) - 1) / 2.0
        rank_sum += average_rank * sum(label == 1 for _, label in ordered[i:j])
        rank += j - i
        i = j

    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _write_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    rows = bytearray()
    stride = width * 3
    for y in range(height):
        rows.append(0)
        start = y * stride
        rows.extend(pixels[start : start + stride])

    png = bytearray()
    png.extend(b"\x89PNG\r\n\x1a\n")
    png.extend(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
    png.extend(chunk(b"IDAT", zlib.compress(bytes(rows), level=9)))
    png.extend(chunk(b"IEND", b""))
    path.write_bytes(bytes(png))
