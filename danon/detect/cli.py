"""Reproduce command for the Phase 1 benchmark.

    python -m danon.detect.cli --seeds 11 23 37 --out outputs_real/phase1_benchmark

Runs the change-point detector over held-out synthetic scenarios, scores
every run against sealed ground truth, and writes the machine-readable
record (JSON + Parquet), the markdown report, and optional Plotly PNGs.
"""

from __future__ import annotations

import argparse
from typing import List

from danon.detect.benchmark import (
    BenchmarkResults,
    run_benchmark,
    write_json_record,
    write_parquet_record,
)
from danon.detect.report import BenchmarkReport


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m danon.detect.cli",
        description="Phase 1 change-detection benchmark (synthetic, research-only).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[11, 23, 37],
        help="scenario seeds to sweep (default: 11 23 37)",
    )
    parser.add_argument(
        "--n-cases", type=int, default=20, help="cases per scenario (default 20)"
    )
    parser.add_argument(
        "--out", type=str, default="outputs_real/phase1_benchmark",
        help="output directory (default outputs_real/phase1_benchmark)",
    )
    args = parser.parse_args(argv)

    from danon.detect.synthetic import ScenarioConfig

    cfg = ScenarioConfig(n_cases=args.n_cases)
    results: BenchmarkResults = run_benchmark(args.seeds, cfg)
    print(results.summary_line())
    write_json_record(results, f"{args.out}/benchmark_record.json")
    write_parquet_record(results, f"{args.out}/benchmark_runs.parquet")
    report = BenchmarkReport(results)
    report.write_markdown(f"{args.out}/benchmark_report.md")
    plots = report.write_plots(args.out)
    print(f"record + report written under {args.out}/")
    if plots:
        print(f"plots: {', '.join(plots)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
