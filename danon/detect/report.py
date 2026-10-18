"""Benchmark report: plots and honest prose, generated from the record.

Two Plotly figures (sensitivity/specificity across seeds, and detection
delay distribution), the aggregate summary, the run table, and a
limitations section imported VERBATIM from the record — the report cannot
claim what the record does not.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from danon.detect.benchmark import (
    BenchmarkResults,
    record_to_dict,
    _limitations_text,
)


class BenchmarkReport:
    """An in-memory report built from a BenchmarkResults object."""

    def __init__(self, results: BenchmarkResults) -> None:
        self.results = results

    def markdown(self) -> str:
        r = self.results
        lines: List[str] = []
        lines.append("# EvoTwin Phase 1 — change-point detection benchmark")
        lines.append("")
        lines.append("RESEARCH-ONLY. Synthetic data, no clinical content.")
        lines.append("")
        lines.append("## Aggregate")
        lines.append("")
        lines.append(f"- {self.results.summary_line()}")
        lines.append("")
        lines.append("## Per-seed runs")
        lines.append("")
        lines.append("| seed | sensitivity | specificity | precision | FAR/1000d | delay | insuff. |")
        lines.append("|---|---|---|---|---|---|---|")
        for run in self.results.runs:
            ev = run.evaluation
            prec = f"{ev['precision']:.3f}" if ev["precision"] is not None else "n/a"
            delay = f"{ev['median_delay_days']:.1f}" if ev["median_delay_days"] is not None else "n/a"
            lines.append(
                f"| {run.seed} | {ev['sensitivity']:.3f} | {ev['specificity']:.3f} | "
                f"{prec} | {ev['false_alarms_per_1000_days']:.2f} | {delay} | "
                f"{run.n_insufficient} |"
            )
        lines.append("")

        miss = [(run.seed, c) for run in self.results.runs for c in run.missed_cases]
        fa = [(run.seed, c) for run in self.results.runs for c in run.false_alarm_cases]
        if miss:
            lines.append("## Missed changes (named, not averaged away)")
            lines.append("")
            for seed, c in miss:
                lines.append(f"- seed {seed}: {c}")
            lines.append("")
        if fa:
            lines.append("## False alarms (named, not averaged away)")
            lines.append("")
            for seed, c in fa:
                lines.append(f"- seed {seed}: {c}")
            lines.append("")
        if not miss and not fa:
            lines.append("## Failures")
            lines.append("")
            lines.append("None recorded in this sweep.")
            lines.append("")

        lines.append("## Limitations (verbatim from the record)")
        lines.append("")
        for lim in _limitations_text():
            lines.append(f"- {lim}")
        lines.append("")
        return "\n".join(lines)

    def plotly_figures(self) -> Dict[str, "object"]:
        """Build the report's figures. Plotly is optional at import time."""
        import plotly.graph_objects as go

        figures: Dict[str, object] = {}
        seeds = [r.seed for r in self.results.runs]
        sens = [r.evaluation["sensitivity"] for r in self.results.runs]
        spec = [r.evaluation["specificity"] for r in self.results.runs]
        # delay points from run records
        delay_days: List[float] = []
        planted_days: List[float] = []
        for run in self.results.runs:
            for d in run.delayed_cases:
                planted_days.append(d["planted_day"])
                delay_days.append(d["detector_day"] - d["planted_day"])

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=seeds, y=sens, mode="markers+lines", name="sensitivity"))
        fig1.add_trace(go.Scatter(x=seeds, y=spec, mode="markers+lines", name="specificity"))
        fig1.update_layout(
            title="Sensitivity and specificity by scenario seed",
            xaxis_title="seed",
            yaxis_title="fraction",
            yaxis=dict(range=[0, 1.02]),
        )
        figures["rates_by_seed"] = fig1

        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                x=[f"seed {s}" for s in seeds],
                y=[r.evaluation["median_delay_days"] or 0.0 for r in self.results.runs],
                name="median delay (d)",
            )
        )
        fig2.update_layout(
            title="Median detection delay by scenario seed",
            xaxis_title="seed",
            yaxis_title="days",
        )
        figures["delay_by_seed"] = fig2
        return figures

    def write_json(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record_to_dict(self.results), fh, indent=2, default=str)

    def write_markdown(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.markdown(), encoding="utf-8")

    def write_plots(self, directory: str) -> List[str]:
        """Render the figures to PNG where kaleido is available; skip silently."""
        written: List[str] = []
        try:
            figures = self.plotly_figures()
        except ImportError:
            return written
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            try:
                target = out_dir / f"{name}.png"
                fig.write_image(str(target))
                written.append(str(target))
            except Exception:
                # kaleido missing is not a benchmark failure; the HTML/JSON
                # record still carries the underlying numbers.
                continue
        return written


class ReportBuilder:
    """Convenience alias retained for the CLI."""

    def __init__(self, results: BenchmarkResults) -> None:
        self._report = BenchmarkReport(results)

    def markdown(self) -> str:
        return self._report.markdown()

    def write_json(self, path: str) -> None:
        self._report.write_json(path)

    def write_markdown(self, path: str) -> None:
        self._report.write_markdown(path)

    def write_plots(self, directory: str) -> List[str]:
        return self._report.write_plots(directory)


def write_report(results: BenchmarkResults, directory: str) -> Dict[str, str]:
    """Write the full report: markdown + JSON record + optional PNGs."""
    rep = BenchmarkReport(results)
    out: Dict[str, str] = {}
    md_path = str(Path(directory) / "benchmark_report.md")
    js_path = str(Path(directory) / "benchmark_record.json")
    rep.write_markdown(md_path)
    rep.write_json(js_path)
    out["markdown"] = md_path
    out["json"] = js_path
    pngs = rep.write_plots(directory)
    if pngs:
        out["plots"] = ", ".join(pngs)
    return out
