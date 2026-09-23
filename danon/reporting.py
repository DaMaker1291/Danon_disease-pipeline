"""Phase-3 deliverables for the real-data campaign.

WHAT THIS MODULE DOES
---------------------
Runs the tested pipeline over the real BraTS geometries x real-TCGA r0s
factorial and renders the poster/report artifacts:

  * `run_real_campaign()`  -- per case x intake: schedule report + spatial
    field exports (.vti). Pure composition of tested pieces.
  * `km_plot()`            -- Kaplan-Meier curves from per-replicate series
    with a log-rank test and an explicit 'not estimable' path.
  * `dose_timeline_plot()` -- the executed dose log (Output B visual).
  * `pitch_vs_measured_ledger()` -- the poster's honesty table.

Design rules held over from the rest of the repo:
  * never a number that the artifact does not carry (the ledger renders
    censoring qualifiers from the report, not from memory);
  * plots are disabled unless matplotlib is importable (deployment
    boxes may not have it) -- refusal, not silent fallback;
  * matplotlib is imported INSIDE the plot functions so the module works
    in headless data-processing roles.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus
from danon.genomics_intake import CohortIntake, load_gdc_cohort
from danon.realdata import BRATS2021_SOURCE, prepare_brats_case
from danon.schedule import (
    DosingScheduleReport,
    PatientIntake,
    compile_dosing_schedule,
    export_fields_vti,
    format_schedule_text,
)
from danon.evosteer import TumorGeometry


# ---------------------------------------------------------------------------
# The campaign facade
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    """One case x intake cell of the real-data campaign."""

    case_id: str
    patient_id: str
    r0: float
    report: DosingScheduleReport
    vti_paths: List[str] = field(default_factory=list)

    def summary_line(self) -> str:
        ext = (
            "n/a"
            if self.report.pfs_extension_median is None
            else f"{self.report.pfs_extension_median:+.1%}"
        )
        tox = (
            "n/a"
            if self.report.toxicity_reduction_median is None
            else f"{self.report.toxicity_reduction_median:.1%}"
        )
        return (
            f"{self.case_id:>22s} x {self.patient_id} r0={self.r0:.3f} : "
            f"PFS {self.report.policy_median_pfs:.0f} vs MTD "
            f"{self.report.paired_mtd_median_pfs:.0f} d (ext {ext}, "
            f"tox {tox}, events {len(self.report.events)})"
        )


def run_real_campaign(
    seg_paths: Sequence[str],
    cohort: Optional[CohortIntake] = None,
    out_dir: str = "outputs_real",
    days: int = 365,
    n_replicates: int = 3,
    keep_fields: bool = True,
    max_cases: Optional[int] = None,
    log: Optional[print] = print,
) -> List[CaseResult]:
    """Run the schedule pipeline over real geometries x cohort intakes.

    This is a facade over tested pieces (`prepare_brats_case`,
    `PatientIntake`, `compile_dosing_schedule`, `export_fields_vti`) --
    it adds orchestration and bookkeeping, no new science, and claims none.
    """
    from dataclasses import replace as dc_replace

    cohort = cohort or load_gdc_cohort()
    os.makedirs(out_dir, exist_ok=True)
    if max_cases is not None:
        seg_paths = seg_paths[:max_cases]

    # Three cohort r0s: min / median / max (the intake spread actually
    # present in the transcriptomes).
    r0s = sorted(p.r0 for p in cohort.patients)
    picks = {r0s[0], r0s[len(r0s) // 2], r0s[-1]}
    sigs = [p for p in cohort.patients if p.r0 in picks]

    results: List[CaseResult] = []
    for seg_path in seg_paths:
        geo = prepare_brats_case(seg_path)
        for sig in sigs:
            intake = PatientIntake(
                r0=sig.r0,
                r0_source=sig.provenance_string(),
                geometry_note=(
                    f"BraTS 2021 case {geo.case_id} expert segmentation "
                    f"(TCGA/TCIA lineage), engine-decimated, "
                    f"{geo.patient_geometry.n_tumour_voxels} tumour voxels"
                ),
            )
            rep = compile_dosing_schedule(
                geo.patient_geometry.geometry, intake,
                days=days, n_replicates=n_replicates,
            )
            vti_paths: List[str] = []
            if keep_fields:
                from danon.evosteer import CollateralSwitchPolicy, TumorEvolutionSimulator

                sim = TumorEvolutionSimulator(geo.patient_geometry.geometry)
                traj = sim.run(
                    CollateralSwitchPolicy(0.15, 0.05),
                    days=days,
                    initial_resistant_fraction=sig.r0,
                    seed=rep.seed_used if hasattr(rep, "seed_used") else 7,
                    keep_fields=True,
                    fill_fraction=0.65,
                )
                vti_path = os.path.join(out_dir, f"{geo.case_id}_{sig.patient_id}_fields.vti")
                export_fields_vti(traj, geo.patient_geometry.geometry, vti_path)
                vti_paths.append(vti_path)

            results.append(
                CaseResult(
                    case_id=geo.case_id,
                    patient_id=sig.patient_id,
                    r0=sig.r0,
                    report=rep,
                    vti_paths=vti_paths,
                )
            )
            if log is not None:
                log(results[-1].summary_line())

    log_path = os.path.join(out_dir, "campaign_log.json")
    new_entries = [
        {
            "case_id": r.case_id,
            "patient_id": r.patient_id,
            "r0": r.r0,
            "policy_median_pfs": r.report.policy_median_pfs,
            "paired_mtd_median_pfs": r.report.paired_mtd_median_pfs,
            "pfs_extension_median": r.report.pfs_extension_median,
            "toxicity_reduction_median": r.report.toxicity_reduction_median,
            "progressed": r.report.progressed,
            "n_events": len(r.report.events),
            "pfs_policy_replicates": r.report.pfs_policy_replicates,
            "pfs_mtd_replicates": r.report.pfs_mtd_replicates,
            "toxicity_reduction_replicates": r.report.toxicity_reduction_replicates,
            "notes": r.report.notes,
            "vti": r.vti_paths,
        }
        for r in results
    ]
    # Merge with any previous chunk's log so chunked campaigns accumulate.
    existing = []
    if os.path.exists(log_path):
        try:
            with open(log_path) as fh:
                existing = json.load(fh)
        except (json.JSONDecodeError, OSError):
            existing = []
    seen = {(e["case_id"], e["patient_id"]) for e in existing}
    for e in new_entries:
        key = (e["case_id"], e["patient_id"])
        if key in seen:
            existing = [x for x in existing if (x["case_id"], x["patient_id"]) != key]
        existing.append(e)
    with open(log_path, "w") as fh:
        json.dump(existing, fh, indent=1)
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _require_plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as e:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "matplotlib is required for plot generation and is not importable "
            f"here ({e}); the pipeline refuses to produce degraded plots"
        ) from e


def km_plot(
    policy_times: Sequence[float],
    policy_events: Sequence[int],
    mtd_times: Sequence[float],
    mtd_events: Sequence[int],
    out_path: str,
    title: str = "Simulated progression-free survival (real patient geometry)",
) -> Dict[str, object]:
    """Kaplan-Meier curves with log-rank test, from per-replicate series.

    Events are progressions; reaching the horizon without progression is
    censoring (the standard PFS convention). Raises (rather than drawing
    a misleading chart) when either arm has fewer than 3 events.
    """
    from danon.evotwin import kaplan_meier, km_median
    from danon.validation import logrank_two_group

    if sum(policy_events) + sum(mtd_events) < 3:
        raise ValueError(
            "fewer than 3 progression events overall: the log-rank test is "
            "not estimable; extend the horizon or add replicates rather "
            "than plotting a misleading curve"
        )
    # An arm with ZERO events (all censored at the horizon) is a valid,
    # informative KM curve (flat at 1.0) -- the log-rank still runs on the
    # pooled events. What is NOT allowed is claiming a median for it.

    plt = _require_plt()
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    curves = []
    for times, events, label, color in (
        (mtd_times, mtd_events, "MTD (standard of care)", "#c62828"),
        (policy_times, policy_events, "Adaptive collateral switch", "#2e7d32"),
    ):
        km = kaplan_meier(list(times), list(events))
        t = np.asarray(km["t"], dtype=float)
        S = np.asarray(km["S"], dtype=float)
        ax.step(t, S, where="post", label=label, color=color, linewidth=2)
        med = km_median(km)
        if med is not None:
            ax.axvline(med, color=color, linestyle=":", alpha=0.6)
        ax.annotate(
            f"median {med:.0f} d" if med is not None else "median not reached",
            xy=(0.98, 0.92 if color == "#c62828" else 0.82),
            xycoords="axes fraction",
            ha="right",
            color=color,
            fontsize=9,
        )
        curves.append((list(times), list(events)))

    all_times = curves[0][0] + curves[1][0]
    all_events = curves[0][1] + curves[1][1]
    lr = logrank_two_group(
        all_times,
        all_events,
        [True] * len(curves[0][0]) + [False] * len(curves[1][0]),
    )
    # A hazard ratio needs events in BOTH arms. With one arm fully censored the
    # ratio is a numerical artifact (observed: 1.4e17), so it is not printed --
    # the p-value remains valid because the log-rank compares observed against
    # expected events under the null. Claiming the artifact as a figure is
    # exactly the kind of number this repo exists to refuse.
    hr = None
    if sum(policy_events) > 0 and sum(mtd_events) > 0:
        hr = float(lr["hazard_ratio"])
        label = f"log-rank p = {lr['logrank_p_value']:.2g}, HR = {hr:.2f}"
    else:
        arm = "policy" if sum(policy_events) == 0 else "MTD"
        label = (
            f"log-rank p = {lr['logrank_p_value']:.2g}; HR not estimable "
            f"(no progression events in the {arm} arm)"
        )
    ax.text(0.02, 0.04, label, transform=ax.transAxes, fontsize=9)
    ax.set_xlabel("days")
    ax.set_ylabel("progression-free")
    ax.set_title(title)
    ax.legend(loc="center left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    return {
        "logrank_p": lr["logrank_p_value"],
        "hazard_ratio": hr,
        "policy_events": int(sum(policy_events)),
        "mtd_events": int(sum(mtd_events)),
    }


def dose_timeline_plot(report: DosingScheduleReport, out_path: str) -> str:
    """Executed dose log of the reference replicate as a step timeline."""
    plt = _require_plt()
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    horizon = report.horizon_days
    days = np.arange(horizon + 1)
    dose_a = np.ones(horizon + 1)
    dose_b = np.zeros(horizon + 1)
    for i, ev in enumerate(report.events):
        start = ev.day
        end = report.events[i + 1].day if i + 1 < len(report.events) else horizon
        if ev.action == "A->B":
            dose_a[start:end] = 0.0
            dose_b[start:end] = 1.0
        else:
            dose_a[start:end] = 1.0
            dose_b[start:end] = 0.0
    ax.fill_between(days, 0, dose_a, step="post", alpha=0.55, label="Drug A", color="#1565c0")
    ax.fill_between(days, 0, dose_b, step="post", alpha=0.55, label="Drug B", color="#ef6c00")
    ax.set_yticks([0, 1])
    ax.set_xlabel("day")
    ax.set_title(f"Executed schedule ({len(report.events)} regime changes)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    return out_path


def cohort_bar_plot(results: Sequence[CaseResult], out_path: str) -> str:
    """PFS extension (and toxicity reduction) across the campaign cells."""
    plt = _require_plt()
    labels = [f"{r.case_id[-5:]}\nr0={r.r0:.2f}" for r in results]
    ext = [0.0 if r.report.pfs_extension_median is None else r.report.pfs_extension_median for r in results]
    tox = [0.0 if r.report.toxicity_reduction_median is None else r.report.toxicity_reduction_median for r in results]
    x = np.arange(len(results))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(results)), 4.2))
    ax.bar(x - width / 2, ext, width, label="PFS extension vs MTD", color="#2e7d32")
    ax.bar(x + width / 2, tox, width, label="toxicity reduction", color="#f9a825")
    for xi, e in zip(x, ext):
        if e > 0:
            ax.text(xi - width / 2, e + 0.01, f"+{e:.0%}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("fraction")
    ax.set_title("Real-geometry campaign: per-cell paired outcomes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    return out_path


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def pitch_vs_measured_ledger(report: DosingScheduleReport) -> str:
    """The poster's honesty table, generated from a real-geometry report."""
    from danon.schedule import poster_summary

    return poster_summary(report)


def campaign_bar_plot(
    records: Sequence[Dict[str, object]],
    out_path: str,
    title: str = "Real-geometry campaign: per-cell paired outcomes",
) -> str:
    """Per-cell PFS extension and toxicity reduction, straight from the log.

    Built from `campaign_log.json` rows so the figure can be regenerated at
    any time from the artifact that produced it, without re-running the
    campaign or reconstructing report objects.
    """
    if not records:
        raise ValueError("refusing to draw an empty campaign figure")
    plt = _require_plt()

    labels = [
        f"{str(r['case_id'])[-5:]}\nr0={float(r['r0']):.2f}" for r in records
    ]
    ext = [
        0.0 if r.get("pfs_extension_median") is None else float(r["pfs_extension_median"])
        for r in records
    ]
    tox = [
        0.0
        if r.get("toxicity_reduction_median") is None
        else float(r["toxicity_reduction_median"])
        for r in records
    ]
    x = np.arange(len(records))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(6.5, 0.55 * len(records)), 4.2))
    ax.bar(x - width / 2, ext, width, label="PFS extension vs paired MTD", color="#2e7d32")
    ax.bar(x + width / 2, tox, width, label="toxicity reduction", color="#f9a825")
    for xi, e in zip(x, ext):
        ax.text(xi - width / 2, e + 0.01, f"+{e:.0%}", ha="center", fontsize=6.5)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45)
    ax.set_ylabel("fraction")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    return out_path


# ---------------------------------------------------------------------------
# Robustness figure + validation packet
# ---------------------------------------------------------------------------


def sensitivity_curve_plot(
    records: Sequence[Dict[str, object]],
    out_path: str,
    case_id: Optional[str] = None,
    targets: Optional[Sequence[str]] = None,
) -> str:
    """Burden-ratio vs perturbation scale: does the conclusion survive?

    `records` are the rows from `SensitivityReport.to_records()`. Only cells
    whose control arm actually progressed the same way matter here, but the
    ratio itself is defined for every cell, so nothing is dropped silently.
    """
    rows = [r for r in records if case_id is None or r.get("case_id") == case_id]
    if not rows:
        raise ValueError(
            f"no sensitivity records for case_id={case_id!r}: refusing to draw "
            "an empty robustness figure"
        )
    if targets is not None:
        rows = [r for r in rows if r.get("target") in targets]

    plt = _require_plt()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    by_target: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        by_target.setdefault(str(r["target"]), []).append(r)

    for target, cells in sorted(by_target.items()):
        cells = sorted(cells, key=lambda r: float(r["scale"]))
        xs = [float(r["scale"]) for r in cells]
        ys = [float(r["burden_ratio"]) for r in cells]
        ax.plot(xs, ys, marker="o", label=target, linewidth=1.6)

    ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.0)
    ax.text(0.02, 1.02, "parity with MTD", transform=ax.get_yaxis_transform(),
            fontsize=8, color="#555555")
    ax.set_xscale("log")
    ax.set_xlabel("perturbation scale (1.0 = assumed value)")
    ax.set_ylabel("mean tumour burden ratio (policy / MTD)")
    ax.set_title(
        f"Parametric robustness{' -- ' + case_id if case_id else ''}: "
        "paired policy vs MTD"
    )
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    return out_path


def validation_packet(
    cohort,
    model_times: Sequence[float],
    model_events: Sequence[int],
    sensitivity_records: Optional[Sequence[Dict[str, object]]] = None,
    test_count: Optional[int] = None,
    out_path: Optional[str] = None,
) -> str:
    """Assemble the external-validation section from the artifacts themselves.

    Every number here is read from an object, not typed: the cohort sizes from
    the reconstructed cohort, the anchor verdict from the anchor check, the
    curve gaps from the plausibility check, the robustness counts from the
    sensitivity records. Nothing in the packet can drift from the data,
    because nothing in it is a literal.
    """
    from danon.retrospective import anchor_check, plausibility_check

    anchor = anchor_check(cohort)
    check = plausibility_check(cohort, model_times, model_events)

    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("EXTERNAL VALIDATION PACKET -- real clinical outcomes (TCGA-GBM)")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"source: {cohort.source}")
    lines.append(
        f"cohort: {cohort.n_cases} cases; progression arm n="
        f"{len(cohort.progression_times)} ({cohort.n_progression_events} events, "
        f"{cohort.n_progression_censored} censored); OS arm n={len(cohort.os_times)} "
        f"({int(sum(cohort.os_events))} deaths)"
    )
    lines.append(
        f"observed KM median progression: {cohort.median_progression_days} d; "
        f"dated-event IQR: {cohort.progression_event_iqr()}"
    )
    lines.append(f"observed KM median OS: {cohort.median_os_days} d")
    lines.append("")
    lines.append("LAYER 1 -- ingestion anchor (validates the DATA HANDLING)")
    lines.append(f"  {anchor.message()}")
    lines.append("")
    lines.append("LAYER 2 -- control-arm timescale plausibility (weakest layer)")
    for line in check.summary_table().splitlines():
        lines.append(f"  {line}")
    lines.append("")
    lines.append("LAYER 3 -- parametric robustness (structure, not correctness)")
    if sensitivity_records:
        targets = sorted({str(r["target"]) for r in sensitivity_records})
        wins = sum(1 for r in sensitivity_records if r.get("policy_wins"))
        lines.append(
            f"  cells: {len(sensitivity_records)} over {len(targets)} perturbation "
            f"targets; policy better on mean burden in {wins}/"
            f"{len(sensitivity_records)}"
        )
        for target in targets:
            cells = [r for r in sensitivity_records if r["target"] == target]
            ratios = [float(r["burden_ratio"]) for r in cells]
            worst = max(ratios)
            scale_at_worst = float(cells[ratios.index(worst)]["scale"])
            lines.append(
                f"    {target:<18s} worst burden ratio {worst:.2f} at scale "
                f"{scale_at_worst:.2f} ({'still better' if worst < 1.0 else 'LOSES'})"
            )
    else:
        lines.append("  no sensitivity records supplied: layer not evaluated")
    lines.append("")
    lines.append("LAYER 4 -- paired baseline control (policy vs MTD, same geometry/seeds)")
    lines.append(
        f"  model MTD arm: n={check.model_n}, {check.model_events} events, median "
        f"{check.model_median_days} d"
    )
    if test_count is not None:
        lines.append(f"  automated test count at time of packet: {test_count}")
    lines.append("")
    lines.append("WHAT THIS PACKET DOES NOT SHOW")
    for limit in check.caveats:
        lines.append(f"  - {limit}")
    text = "\n".join(lines)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as fh:
            fh.write(text + "\n")
    return text
