from wso.experiments.capacity_curve import (
    build_frontier_report,
    default_signal_strengths,
    simulate_frontier,
    summarize_frontier,
)


def test_default_signal_strengths_include_bounds() -> None:
    strengths = default_signal_strengths(max_strength=2.0, steps=5)
    assert strengths == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_recoverability_increases_with_synthetic_signal() -> None:
    points = simulate_frontier([0.0, 2.5], samples_per_strength=800, seed=11)
    assert points[1].recoverability_auc > points[0].recoverability_auc
    assert "grassmann_drift" in points[1].metric_aucs
    assert "frobenius_norm" in points[1].metric_anomaly_rates


def test_summary_mentions_defensive_frontier() -> None:
    points = simulate_frontier([0.0], samples_per_strength=100, seed=3)
    assert "Capacity-detectability summary" in summarize_frontier(points)


def test_build_frontier_report_contains_safety_boundary() -> None:
    points = simulate_frontier([0.0, 1.0], samples_per_strength=100, seed=3)
    report = build_frontier_report(
        points,
        samples_per_strength=100,
        benign_samples=1000,
        noise_std=1.0,
        seed=3,
    )
    assert report.points[0].metric_aucs
    assert "payload encoders" in report.safety_boundary
