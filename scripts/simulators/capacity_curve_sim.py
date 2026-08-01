"""CLI wrapper for the defensive capacity-detectability simulator."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from wso.experiments.capacity_curve import (  # noqa: E402
    build_frontier_report,
    default_signal_strengths,
    simulate_frontier,
    summarize_frontier,
    write_frontier_csv,
    write_frontier_json,
    write_frontier_plot,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a defensive capacity-detectability frontier simulation."
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--plot", type=Path, default=None, help="Optional PNG plot output path")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--max-strength", type=float, default=3.0, help="Maximum signal strength")
    parser.add_argument("--steps", type=int, default=13, help="Number of frontier points")
    parser.add_argument(
        "--samples-per-strength",
        type=int,
        default=400,
        help="Synthetic audit samples per signal strength",
    )
    parser.add_argument("--benign-samples", type=int, default=1000, help="Benign baseline samples")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    points = simulate_frontier(
        default_signal_strengths(args.max_strength, args.steps),
        samples_per_strength=args.samples_per_strength,
        benign_samples=args.benign_samples,
        seed=args.seed,
    )
    print(summarize_frontier(points))
    if args.output is not None:
        write_frontier_csv(points, args.output)
        print(f"Wrote frontier CSV: {args.output}")
    if args.json_output is not None:
        report = build_frontier_report(
            points,
            samples_per_strength=args.samples_per_strength,
            benign_samples=args.benign_samples,
            noise_std=1.0,
            seed=args.seed,
        )
        write_frontier_json(report, args.json_output)
        print(f"Wrote frontier JSON: {args.json_output}")
    if args.plot is not None:
        write_frontier_plot(points, args.plot)
        print(f"Wrote frontier plot: {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
