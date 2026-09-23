"""
The 365-day outcome card: exact per-day engine state, exported for the UI.

WHAT THIS PRODUCES
------------------
For each real patient geometry (BraTS 2021 segmentation, TCGA/TCIA lineage)
and each cohort-derived initial resistant fraction, this module runs the
adaptive collateral-switch arm and the paired MTD control at the same seeds
and exports:

  * the full daily series of both arms (sensitive burden, resistant burden,
    resistant fraction, per-day dose A and dose B), and
  * an EXACT day-365 readout: total burden, resistant fraction, cumulative
    toxicity, progression status, progression day -- per replicate and as
    medians across replicates.

Every number in the export comes from the engine. Nothing is interpolated,
smoothed, or summarised into a headline the engine did not produce.

WHAT IT IS NOT
--------------
Not a patient prediction. The day-365 readout is the state of a model at day
365 under ASSUMED constants; `notes` (carried into every card and rendered by
the UI) says so, and `outcome_export_provenance()` records which parts are
measured and which are assumptions.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus
from danon.realdata import (
    realdata_provenance,
    realdata_what_this_model_cannot_know,
)
from danon.evosteer import (
    DEFAULT_CLONES,
    DEFAULT_DRUGS,
    RANO_PROGRESSION_FRACTION,
    CollateralSwitchPolicy,
    MTDPolicy,
    TumorEvolutionSimulator,
    TumorGeometry,
    what_this_model_cannot_know,
)


# ---------------------------------------------------------------------------
# One arm, one replicate
# ---------------------------------------------------------------------------


@dataclass
class ReplicateRun:
    """One engine run's exact record (daily series retained when asked)."""

    seed: int
    days: int
    burden_S: np.ndarray
    burden_R: np.ndarray
    dose_A: np.ndarray
    dose_B: np.ndarray
    pfs_day: Optional[int]
    progressed: bool
    toxicity: float
    final_resistant_fraction: float

    # -- exact day-N readout ------------------------------------------------

    def day(self, day: int) -> Dict[str, float]:
        """Exact engine state at the end of `day` (1-indexed day of the run)."""
        if day < 1 or day > self.days:
            raise ValueError(f"day {day} outside the 1..{self.days} run window")
        i = day - 1
        s = float(self.burden_S[i])
        r = float(self.burden_R[i])
        total = s + r
        return {
            "day": day,
            "burden_sensitive": s,
            "burden_resistant": r,
            "burden_total": total,
            "resistant_fraction": (r / total) if total > 0 else 0.0,
            "dose_A_today": float(self.dose_A[i]),
            "dose_B_today": float(self.dose_B[i]),
            "dose_A_cumulative": float(np.sum(self.dose_A[: day])),
            "dose_B_cumulative": float(np.sum(self.dose_B[: day])),
        }


@dataclass
class ArmRecord:
    label: str
    policy: str
    replicates: List[ReplicateRun] = field(default_factory=list)
    daily_seed: Optional[int] = None

    # -- derived, all from the engine's own numbers -------------------------

    @property
    def pfs_values(self) -> List[float]:
        return [
            float(r.pfs_day) if r.pfs_day is not None else float(r.days)
            for r in self.replicates
        ]

    @property
    def median_pfs(self) -> float:
        return float(np.median(self.pfs_values)) if self.replicates else float("nan")

    @property
    def n_progressed(self) -> int:
        return sum(1 for r in self.replicates if r.progressed)

    @property
    def median_toxicity(self) -> float:
        return (
            float(np.median([r.toxicity for r in self.replicates]))
            if self.replicates
            else float("nan")
        )

    def day_readouts(self, day: int) -> List[Dict[str, float]]:
        return [r.day(day) for r in self.replicates]

    def median_day(self, day: int) -> Dict[str, float]:
        """Median across replicates of every day-`day` quantity."""
        rows = self.day_readouts(day)
        out: Dict[str, float] = {"day": day}
        for key in (
            "burden_sensitive",
            "burden_resistant",
            "burden_total",
            "resistant_fraction",
            "dose_A_cumulative",
            "dose_B_cumulative",
        ):
            out[key] = float(np.median([row[key] for row in rows]))
        return out

    def daily(self) -> Dict[str, List[float]]:
        """The reference replicate's daily series (the UI's line chart)."""
        ref = self.replicates[0]
        total = ref.burden_S + ref.burden_R
        with np.errstate(divide="ignore", invalid="ignore"):
            rfrac = np.where(total > 0, ref.burden_R / total, 0.0)
        return {
            "seed": ref.seed,
            "burden_total": [round(float(x), 6) for x in total],
            "burden_sensitive": [round(float(x), 6) for x in ref.burden_S],
            "burden_resistant": [round(float(x), 6) for x in ref.burden_R],
            "resistant_fraction": [round(float(x), 6) for x in rfrac],
            "dose_A": [round(float(x), 4) for x in ref.dose_A],
            "dose_B": [round(float(x), 4) for x in ref.dose_B],
        }


# ---------------------------------------------------------------------------
# Running a case
# ---------------------------------------------------------------------------


# The engine's own limitation list still contains the phantom-era entry about
# geometry not being an MRI reconstruction. On these cards the geometry IS a
# real expert segmentation, so that note is replaced rather than printed -- a
# stale caveat is as misleading as a missing one.
_STALE_PHANTOM_NOTE = "the phantom is a stand-in, not an MRI reconstruction"


_REAL_GEOMETRY_NOTE = (
    "geometry IS real: BraTS 2021 expert segmentation (TCGA/TCIA lineage), but it "
    "is majority-vote decimated to the engine grid, so sub-voxel anatomy is gone "
    "and the vasculature is the null-uniform proxy -- sanctuary dynamics are not "
    "studyable on these cases without supplied perfusion maps"
)


def _card_notes() -> List[str]:
    engine_notes = [
        n for n in what_this_model_cannot_know() if _STALE_PHANTOM_NOTE not in n
    ]
    return [_REAL_GEOMETRY_NOTE] + engine_notes + list(
        realdata_what_this_model_cannot_know()
    )


def _run_arm(
    geometry: TumorGeometry,
    arm: str,
    r0: float,
    seeds: Sequence[int],
    days: int,
) -> ArmRecord:
    if arm not in ("policy", "mtd"):
        raise ValueError("arm must be 'policy' or 'mtd'")
    if not seeds:
        raise ValueError("need at least one seed")
    if days < 2:
        raise ValueError("need at least 2 days")

    sim = TumorEvolutionSimulator(geometry)
    label = (
        "Adaptive collateral switch"
        if arm == "policy"
        else "MTD (maximum tolerated dose)"
    )
    rec = ArmRecord(label=label, policy=arm)
    for seed in seeds:
        policy_obj = (
            CollateralSwitchPolicy(0.15, 0.05) if arm == "policy" else MTDPolicy()
        )
        traj = sim.run(
            policy_obj, days=days, initial_resistant_fraction=r0, seed=int(seed)
        )
        rec.replicates.append(
            ReplicateRun(
                seed=int(seed),
                days=int(traj.days),
                burden_S=np.asarray(traj.burden_S, dtype=float),
                burden_R=np.asarray(traj.burden_R, dtype=float),
                dose_A=np.asarray(traj.dose_A, dtype=float),
                dose_B=np.asarray(traj.dose_B, dtype=float),
                pfs_day=(int(traj.pfs_day) if traj.pfs_day is not None else None),
                progressed=bool(traj.progressed),
                toxicity=float(traj.toxicity),
                final_resistant_fraction=float(traj.final_resistant_fraction),
            )
        )
    rec.daily_seed = int(seeds[0])
    return rec


def _headline(
    policy: ArmRecord, mtd: ArmRecord, day: int, initial_burden: float
) -> Dict[str, object]:
    """The exact day-`day` comparison, computed from engine state only."""
    p = policy.median_day(day)
    m = mtd.median_day(day)
    pb, mb = p["burden_total"], m["burden_total"]
    frac_of_initial_p = (pb / initial_burden) if initial_burden > 0 else None
    frac_of_initial_m = (mb / initial_burden) if initial_burden > 0 else None

    # Two honesty flags the UI must render, because both would otherwise read
    # as results: a fully-censored policy arm makes the PFS extension a LOWER
    # BOUND, and a near-zero burden makes the resistant fraction numerically
    # defined but biologically meaningless (the post-eradication artifact).
    censored = policy.n_progressed == 0 and len(policy.replicates) > 0
    clearance = (
        "near-complete clearance at day {}: median burden is {:.4%} of the seeded "
        "tumour, so the resistant fraction ({:.3f}) is numerically defined but not "
        "interpretable at this magnitude".format(day, frac_of_initial_p or 0.0,
                                                 p["resistant_fraction"])
        if (frac_of_initial_p is not None and frac_of_initial_p < 1e-3)
        else None
    )
    pfs_ratio = (
        (policy.median_pfs / mtd.median_pfs)
        if mtd.median_pfs and mtd.median_pfs > 0
        else None
    )
    tox_red = (
        (mtd.median_toxicity - policy.median_toxicity) / mtd.median_toxicity
        if mtd.median_toxicity > 0
        else None
    )
    return {
        "day": day,
        "burden_policy": pb,
        "burden_mtd": mb,
        "burden_policy_over_mtd": (pb / mb) if mb > 0 else None,
        "burden_reduction_vs_mtd": ((mb - pb) / mb) if mb > 0 else None,
        "resistant_fraction_policy": p["resistant_fraction"],
        "resistant_fraction_mtd": m["resistant_fraction"],
        "progression_free_policy": policy.n_progressed < len(policy.replicates),
        "progression_free_mtd": mtd.n_progressed < len(mtd.replicates),
        "n_progressed_policy": policy.n_progressed,
        "n_progressed_mtd": mtd.n_progressed,
        "median_pfs_policy": policy.median_pfs,
        "median_pfs_mtd": mtd.median_pfs,
        "pfs_ratio": pfs_ratio,
        "pfs_extension": (pfs_ratio - 1.0) if pfs_ratio is not None else None,
        "toxicity_reduction": tox_red,
        "n_replicates": len(policy.replicates),
        "initial_burden": initial_burden,
        "burden_fraction_of_initial_policy": frac_of_initial_p,
        "burden_fraction_of_initial_mtd": frac_of_initial_m,
        "pfs_extension_is_lower_bound": censored,
        "pfs_extension_label": (
            None
            if pfs_ratio is None
            else (
                f">= {(pfs_ratio - 1.0):+.1%} (policy arm never progressed; censored "
                f"at the {day} d horizon)"
                if censored
                else f"{(pfs_ratio - 1.0):+.1%}"
            )
        ),
        "clearance_note": clearance,
    }


def run_case_outcome(
    geometry: TumorGeometry,
    r0: float,
    case_id: str,
    patient_id: str,
    r0_source: str,
    geometry_note: str,
    seeds: Sequence[int] = (101, 102, 103),
    days: int = 365,
) -> Dict[str, object]:
    """Run both arms and assemble one outcome card (plain JSON-able dict)."""
    if not math.isfinite(r0) or not 0.0 <= r0 <= 1.0:
        raise ValueError(f"r0 must be a fraction in [0, 1], got {r0!r}")

    policy = _run_arm(geometry, "policy", r0, seeds, days)
    mtd = _run_arm(geometry, "mtd", r0, seeds, days)
    initial_burden = float(
        np.median([float(r.burden_S[0] + r.burden_R[0]) for r in policy.replicates])
    )

    card: Dict[str, object] = {
        "case_id": case_id,
        "patient_id": patient_id,
        "intake": {
            "r0": float(r0),
            "r0_source": r0_source,
            "r0_note": (
                "initial resistant fraction derived from a real TCGA-GBM STAR "
                "expression profile (danon.genomics_intake); it is a calibrated "
                "mapping from transcriptome to fraction, not a measured clonal "
                "fraction"
            ),
        },
        "geometry": {
            "note": geometry_note,
            "grid": list(geometry.tumor_mask.shape),
            "tumour_voxels": int(np.count_nonzero(geometry.tumor_mask)),
            "voxel_mm": round(float(geometry.voxel_mm), 3),
        },
        "seeds": [int(s) for s in seeds],
        "horizon_days": int(days),
        "arms": {
            "policy": {
                "label": policy.label,
                "median_pfs_days": policy.median_pfs,
                "n_progressed": policy.n_progressed,
                "median_toxicity": policy.median_toxicity,
                "daily": policy.daily(),
                "replicate_day_readouts": [
                    {"seed": r.seed, "pfs_day": r.pfs_day,
                     "progressed": r.progressed, "toxicity": r.toxicity,
                     "day": r.day(days)}
                    for r in policy.replicates
                ],
            },
            "mtd": {
                "label": mtd.label,
                "median_pfs_days": mtd.median_pfs,
                "n_progressed": mtd.n_progressed,
                "median_toxicity": mtd.median_toxicity,
                "daily": mtd.daily(),
                "replicate_day_readouts": [
                    {"seed": r.seed, "pfs_day": r.pfs_day,
                     "progressed": r.progressed, "toxicity": r.toxicity,
                     "day": r.day(days)}
                    for r in mtd.replicates
                ],
            },
        },
        "day_365": _headline(policy, mtd, days, initial_burden),
        "notes": _card_notes(),
        "data_provenance": {k: v.value for k, v in realdata_provenance().items()},
    }
    return card


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_outcome_cards(
    cases: Sequence[Dict[str, object]],
    out_path: str,
    title: str = "EvoSteer day-365 outcome cards",
    mirror_path: Optional[str] = None,
) -> Dict[str, object]:
    """Write the outcome-card JSON the UI reads.

    `cases` entries are the kwargs for `run_case_outcome` (geometry, r0,
    case_id, patient_id, r0_source, geometry_note, plus optional seeds/days).

    `mirror_path` writes an identical copy (the frontend reads its own
    `public/evotwin/outcome_cards.json` under the app's base path), so the
    dashboard and the artifact can never diverge: they are the same file.
    """
    if not cases:
        raise ValueError("refusing to write an outcome export with no cases")

    cards: List[Dict[str, object]] = []
    for spec in cases:
        spec = dict(spec)
        required = ("geometry", "r0", "case_id", "patient_id", "r0_source", "geometry_note")
        missing = [k for k in required if k not in spec]
        if missing:
            raise ValueError(f"case spec missing required fields: {missing}")
        cards.append(run_case_outcome(**spec))

    payload = {
        "title": title,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "EvoSteer: 3D spatial ABM with diffusing drugs and clonal competition",
        "rano_progression_fraction": RANO_PROGRESSION_FRACTION,
        "switch_rule": "Drug A -> Drug B when resistant fraction >= 15%, back at <= 5%",
        "engine_constants": {
            "growth_S_per_day": DEFAULT_CLONES.growth_S,
            "growth_R_per_day": DEFAULT_CLONES.growth_R,
            "competition": DEFAULT_CLONES.competition,
            "drug_sensitivity_scale_R": DEFAULT_CLONES.drug_sensitivity_scale_R,
            "kill_max_drug_A_on_S_R": list(DEFAULT_DRUGS.kill_max_S),
            "kill_max_drug_B_on_R": list(DEFAULT_DRUGS.kill_max_R),
            "ec50": list(DEFAULT_DRUGS.ec50),
            "diffusion": list(DEFAULT_DRUGS.diffusion),
            "clearance": list(DEFAULT_DRUGS.clearance),
            "provenance": "ASSUMED - bench calibration is the missing tier "
                          "(see danon.validation, VALIDATION.md layer 3)",
        },
        "provenance": {
            **{k: v.value for k, v in realdata_provenance().items()},
            **{k: v.value for k, v in outcome_export_provenance().items()},
        },
        "cases": cards,
    }

    blob = json.dumps(payload, indent=1)
    for target in (out_path, mirror_path):
        if not target:
            continue
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w") as fh:
            fh.write(blob)
    return payload


def load_cards(path: str) -> Dict[str, object]:
    with open(path) as fh:
        payload = json.load(fh)
    if not payload.get("cases"):
        raise ValueError(f"outcome export at {path} has no cases")
    return payload


def outcome_export_provenance() -> Dict[str, EvidenceStatus]:
    return {
        "engine_arithmetic": EvidenceStatus.ESTABLISHED,       # covered by the test suite
        "geometry_real_brats_segmentation": EvidenceStatus.ESTABLISHED,
        "r0_from_real_transcriptome": EvidenceStatus.UNKNOWN,  # calibrated mapping, not a measurement
        "engine_constants": EvidenceStatus.UNKNOWN,            # ASSUMED
        "day_365_readout_is_a_patient_prediction": EvidenceStatus.PROVEN_FAILED,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


_DASHBOARD_CSS = """
:root{--bg:#05070d;--panel:rgba(15,20,34,.72);--panel2:rgba(22,29,48,.6);
--line:rgba(120,160,220,.16);--txt:#e8eefc;--dim:#8ea2c6;--dim2:#63748f;
--acc:#38bdf8;--acc2:#a78bfa;--good:#34d399;--warn:#fbbf24;--bad:#fb7185}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1200px 800px at 15% 0%,rgba(56,189,248,.10),transparent 60%),
radial-gradient(900px 700px at 90% 20%,rgba(167,139,250,.10),transparent 60%),var(--bg);
color:var(--txt);font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial}
.wrap{max-width:1280px;margin:0 auto;padding:28px 20px 60px}
header{display:flex;justify-content:space-between;align-items:flex-end;gap:20px;flex-wrap:wrap;margin-bottom:22px}
h1{margin:0;font-size:26px;letter-spacing:-.4px}
h1 small{display:block;font-size:12px;font-weight:400;color:var(--dim);letter-spacing:.6px;text-transform:uppercase;margin-top:6px}
.badges{display:flex;gap:8px;flex-wrap:wrap}
.badge{border:1px solid var(--line);background:var(--panel2);padding:5px 10px;border-radius:999px;font-size:11px;color:var(--dim);letter-spacing:.3px}
.badge.live{color:var(--good);border-color:rgba(52,211,153,.35)}
.badge.warn{color:var(--warn);border-color:rgba(251,191,36,.35)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px 20px;
backdrop-filter:blur(14px);box-shadow:0 18px 50px rgba(0,0,0,.45);margin-bottom:18px}
.panel h2{margin:0 0 4px;font-size:12px;letter-spacing:1.4px;text-transform:uppercase;color:var(--dim)}
.panel .sub{margin:0 0 16px;color:var(--dim2);font-size:12px}
.grid{display:grid;gap:18px}
.h360{grid-template-columns:1.15fr .85fr}
.h365{grid-template-columns:repeat(auto-fit,minmax(165px,1fr))}
.tscroll{overflow-x:auto}
table{min-width:520px}
@media (max-width:900px){
  .h360{grid-template-columns:1fr}
  .wrap{padding:20px 14px 48px}
  h1{font-size:21px}
  .panel{padding:15px 15px;border-radius:14px}
  .controls select{min-width:200px}
}
.kpi{background:linear-gradient(160deg,rgba(56,189,248,.10),rgba(15,20,34,.55));
border:1px solid var(--line);border-radius:14px;padding:14px 16px}
.kpi .k{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--dim2)}
.kpi .v{font-size:27px;font-weight:650;letter-spacing:-.6px;margin:6px 0 2px;font-variant-numeric:tabular-nums}
.kpi .m{font-size:11px;color:var(--dim)}
.kpi.good .v{color:var(--good)}.kpi.warn .v{color:var(--warn)}.kpi.bad .v{color:var(--bad)}
.kpi.acc .v{color:var(--acc)}.kpi.acc2 .v{color:var(--acc2)}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:13px}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left;color:var(--dim)}
thead th{font-size:10.5px;text-transform:uppercase;letter-spacing:1px;color:var(--dim2);font-weight:600}
tbody tr:hover{background:rgba(120,160,220,.05)}
tfoot td{font-weight:650;color:var(--txt);border-bottom:none}
select,input[type=range]{accent-color:var(--acc)}
select{background:var(--panel2);color:var(--txt);border:1px solid var(--line);border-radius:10px;padding:8px 10px;font:inherit;min-width:260px}
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.controls label{color:var(--dim);font-size:12px}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--dim);font-size:12px;margin-top:10px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px}
svg{display:block;width:100%;height:auto;overflow:visible}
.note{color:var(--dim2);font-size:12px;line-height:1.65}
.note b{color:var(--warn);font-weight:600}
.foot{margin-top:22px;color:var(--dim2);font-size:11.5px;line-height:1.7;border-top:1px solid var(--line);padding-top:16px}
code{background:rgba(120,160,220,.10);padding:1px 6px;border-radius:6px;font-size:12px}
.scrub{display:flex;align-items:center;gap:12px}
.scrub .dv{font-variant-numeric:tabular-nums;font-size:20px;font-weight:650;color:var(--acc);min-width:96px}
.flag{border:1px solid rgba(251,191,36,.35);background:rgba(251,191,36,.08);color:#fde68a;
border-radius:12px;padding:10px 14px;font-size:12.5px;margin:0 0 14px;line-height:1.55}
.flag.info{border-color:rgba(56,189,248,.3);background:rgba(56,189,248,.07);color:#bae6fd}
"""


def regenerate_ui_artifacts(
    payload: Dict[str, object],
    dashboard_path: str = "outputs_real/day365_dashboard.html",
    title: Optional[str] = None,
) -> Tuple[str, str]:
    """Re-render the standalone dashboard from an existing export.

    Returns (dashboard_path, cards_path_of_the_payload_source) so a caller can
    log both. Rendering never invents data: it re-reads the payload it is
    given, which is why the artifact can be regenerated without re-running the
    engine.
    """
    path = render_dashboard_html(payload, dashboard_path, title=title)
    return path, str(payload.get("title") or "")


def render_dashboard_html(
    payload: Dict[str, object],
    out_path: str,
    title: Optional[str] = None,
) -> str:
    """Render a self-contained dashboard (data embedded, no server needed).

    Everything -- data, styles, chart code -- is inlined so the file can be
    opened directly, attached to a report, or served by any static host
    without a build step, and so the numbers can never drift from the JSON
    that produced them.
    """
    cases = payload.get("cases") or []
    if not cases:
        raise ValueError("refusing to render a dashboard with no outcome cards")

    heading = title or str(payload.get("title") or "EvoSteer day-365 outcome")
    data_json = json.dumps(payload, separators=(",", ":"))
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{heading}</title><style>{_DASHBOARD_CSS}</style></head>
<body><div class="wrap">
<header>
  <h1>{heading}<small>Exact day-365 engine state &middot; adaptive collateral switch vs MTD &middot; real BraTS / TCGA-GBM patient geometry</small></h1>
  <div class="badges">
    <span class="badge live">&bull; ENGINE OUTPUT</span>
    <span class="badge warn">CONSTANTS ASSUMED</span>
    <span class="badge">NOT A PATIENT PREDICTION</span>
  </div>
</header>

<div class="panel">
  <h2>Case</h2>
  <p class="sub" id="caseSub"></p>
  <div class="controls">
    <label>Patient geometry &times; intake</label>
    <select id="caseSel"></select>
    <span class="badge" id="seedBadge"></span>
  </div>
</div>

<div class="panel">
  <h2>Day 365 &mdash; exact predicted outcome</h2>
  <p class="sub" id="headSub"></p>
  <div id="flags"></div>
  <div class="grid h365" id="kpis"></div>
</div>

<div class="grid h360">
  <div class="panel">
    <h2>Tumour burden over 365 days</h2>
    <p class="sub">Total cell density (sum of voxels), reference replicate &mdash; median across replicates shown in the table below.</p>
    <div id="burdenChart"></div>
    <div class="legend">
      <span><i style="background:#34d399"></i>adaptive switch</span>
      <span><i style="background:#fb7185"></i>MTD control</span>
      <span><i style="background:rgba(120,160,220,.35)"></i>RANO progression threshold</span>
    </div>
  </div>
  <div class="panel">
    <h2>Clonal composition</h2>
    <p class="sub">Resistant fraction R/(S+R). The switch rule acts on this quantity &mdash; which is not clinically observable in real time.</p>
    <div id="fracChart"></div>
    <div class="legend">
      <span><i style="background:#a78bfa"></i>adaptive switch</span>
      <span><i style="background:#fbbf24"></i>MTD control</span>
    </div>
  </div>
</div>

<div class="panel">
  <h2>Executed dosing</h2>
  <p class="sub">Adaptive arm: Drug A &rarr; Drug B on resistant-fraction thresholds. Control arm: continuous Drug A.</p>
  <div id="doseChart"></div>
  <div class="legend">
    <span><i style="background:#38bdf8"></i>Drug A</span>
    <span><i style="background:#fb923c"></i>Drug B</span>
  </div>
</div>  <div class="panel">
    <h2>Exact state at day 365, per replicate</h2>
    <p class="sub">Every value below is read from the engine at day 365 of that replicate &mdash; no interpolation.</p>
    <div class="tscroll" id="repTable"></div>
  </div>

  <div class="panel">
    <h2>Cross-check</h2>
    <p class="sub">Median across replicates, both arms, day 365 of the same run.</p>
    <div class="tscroll" id="cmpTable"></div>
  </div>

<div class="panel">
  <h2>What this does not show</h2>
  <p class="sub">Carried verbatim from the engine's own limitation record.</p>
  <div class="note" id="limits"></div>
</div>

<div class="foot" id="foot"></div>
</div>
<script>
const DATA = {data_json};
const CASES = DATA.cases;
const sel = document.getElementById('caseSel');
CASES.forEach((c,i)=>{{ const o=document.createElement('option'); o.value=i;
  o.textContent = c.case_id+' \\u00d7 '+c.patient_id+'  (r0 = '+c.intake.r0.toFixed(3)+')'; sel.appendChild(o); }});
sel.addEventListener('change', ()=>render(+sel.value));

const fmt = (x,d=2)=> (x===null||x===undefined||Number.isNaN(x)) ? '\\u2014' :
  (Math.abs(x)>=1e5 ? x.toExponential(2) :
   (Math.abs(x)>0 && Math.abs(x)<1e-3 ? x.toExponential(2) : x.toFixed(d)));
const pct = (x,d=1)=> (x===null||x===undefined) ? '\\u2014' : (x*100).toFixed(d)+'%';
const num = (x,d=0)=> fmt(x,d).replace(/\\B(?=(\\d{{3}})+(?!\\d))/g, ',');

function line(series, color, w, h, pad) {{
  const n = series.length; let max = 0;
  series.forEach(s=>s.points.forEach(p=>{{ if(p[1]>max) max=p[1]; }}));
  max = max>0 ? max : 1;
  const X = i => pad.l + (w-pad.l-pad.r) * (i/(n-1));
  const Y = v => h-pad.b - (h-pad.b-pad.t) * (v/max);
  let g = '';
  for (let k=0;k<=4;k++) {{ const y=pad.t+(h-pad.t-pad.b)*k/4;
    g += `<line x1="${{pad.l}}" x2="${{w-pad.r}}" y1="${{y}}" y2="${{y}}" stroke="rgba(120,160,220,.13)"/>`;
    g += `<text x="${{pad.l-8}}" y="${{y+4}}" fill="#63748f" font-size="10" text-anchor="end">${{fmt(max*(1-k/4),0)}}</text>`; }}
  let paths = series.map(s=>{{
    const d = s.points.map((p,i)=>`${{i?'L':'M'}}${{X(p[0]).toFixed(1)}},${{Y(p[1]).toFixed(1)}}`).join(' ');
    const area = `${{d}} L${{X(n-1).toFixed(1)}},${{(h-pad.b)}} L${{X(0).toFixed(1)}},${{(h-pad.b)}} Z`;
    return `<path d="${{area}}" fill="${{s.color}}" opacity=".10"/><path d="${{d}}" fill="none" stroke="${{s.color}}" stroke-width="2"/>`;
  }}).join('');
  let th = '';
  if (series[0].threshold) {{ const y=Y(series[0].threshold);
    th = `<line x1="${{pad.l}}" x2="${{w-pad.r}}" y1="${{y}}" y2="${{y}}" stroke="rgba(120,160,220,.5)" stroke-dasharray="5 4"/>
          <text x="${{w-pad.r}}" y="${{y-6}}" fill="#8ea2c6" font-size="10" text-anchor="end">RANO threshold</text>`; }}
  const ticks = [0,91,182,273,364].map(i=>`<text x="${{X(i)}}" y="${{h-pad.b+16}}" fill="#63748f" font-size="10" text-anchor="middle">day ${{i+1}}</text>`).join('');
  return `<svg viewBox="0 0 ${{w}} ${{h}}" preserveAspectRatio="xMidYMid meet">${{g}}${{th}}${{paths}}${{ticks}}</svg>`;
}}

function stepChart(A, B, w, h, pad) {{
  const n = A.length;
  const X = i => pad.l + (w-pad.l-pad.r) * (i/(n-1));
  const y0 = h-pad.b, y1 = pad.t;
  const band = (v, color) => {{
    let d = '', open = false, start = 0;
    for (let i=0;i<n;i++) {{ if (v[i]>0.5 && !open) {{ open=true; start=i; }}
      else if (v[i]<=0.5 && open) {{ open=false;
        d += `M${{X(start).toFixed(1)}},${{y0}} L${{X(start).toFixed(1)}},${{y1}} L${{X(i-1).toFixed(1)}},${{y1}} L${{X(i-1).toFixed(1)}},${{y0}} Z`; }} }}
    if (open) d += `M${{X(start).toFixed(1)}},${{y0}} L${{X(start).toFixed(1)}},${{y1}} L${{X(n-1).toFixed(1)}},${{y1}} L${{X(n-1).toFixed(1)}},${{y0}} Z`;
    return `<path d="${{d}}" fill="${{color}}" opacity=".55"/>`;
  }};
  const ticks = [0,91,182,273,364].map(i=>`<text x="${{X(i)}}" y="${{h-4}}" fill="#63748f" font-size="10" text-anchor="middle">day ${{i+1}}</text>`).join('');
  return `<svg viewBox="0 0 ${{w}} ${{h}}" preserveAspectRatio="xMidYMid meet">${{band(B,'#fb923c')}}${{band(A,'#38bdf8')}}${{ticks}}</svg>`;
}}

function render(idx) {{
  const c = CASES[idx];
  const pol = c.arms.policy, mtd = c.arms.mtd, h = c.day_365;
  const pD = pol.daily, mD = mtd.daily;
  document.getElementById('caseSub').textContent =
    c.geometry.note + ' \\u2014 engine grid ' + c.geometry.grid.join('\\u00d7') +
    ', ' + num(c.geometry.tumour_voxels,0) + ' tumour voxels at ' + c.geometry.voxel_mm + ' mm. Intake: ' + c.intake.r0_source;
  document.getElementById('seedBadge').textContent = 'replicates: seeds ' + c.seeds.join(', ') + ' \\u00b7 horizon ' + c.horizon_days + ' d';
  document.getElementById('headSub').textContent =
    'Median across ' + h.n_replicates + ' replicates, both arms, identical geometry and seeds. Progression-free at day ' + h.day + ': adaptive ' +
    (h.progression_free_policy ? 'yes' : 'no') + ', MTD ' + (h.progression_free_mtd ? 'yes' : 'no') + '.';

  const red = h.burden_reduction_vs_mtd;
  const kpis = [
    ['PROGRESSION-FREE AT DAY 365', h.progression_free_policy ? 'YES' : 'NO',
      'adaptive switch \\u00b7 ' + h.n_progressed_policy + '/' + h.n_replicates + ' replicates progressed',
      h.progression_free_policy ? 'good' : 'bad'],
    ['MTD CONTROL AT DAY 365', h.progression_free_mtd ? 'yes' : 'NO',
      h.n_progressed_mtd + '/' + h.n_replicates + ' replicates progressed', h.progression_free_mtd ? '' : 'bad'],
    ['BURDEN REDUCTION VS MTD', red===null?'\\u2014':pct(red,1),
      'policy '+num(h.burden_policy)+' vs MTD '+num(h.burden_mtd)+' cell-density units', 'acc'],
    ['MEDIAN PFS', num(h.median_pfs_policy,0)+' d',
      (h.pfs_extension_label===null
        ? 'vs MTD '+num(h.median_pfs_mtd,0)+' d (extension not estimable)'
        : 'vs MTD '+num(h.median_pfs_mtd,0)+' d \\u00b7 '+h.pfs_extension_label),
      'acc2'],
    ['TOXICITY REDUCTION', h.toxicity_reduction===null?'\\u2014':pct(h.toxicity_reduction,1),
      'cumulative weighted dose proxy vs MTD', 'good'],
    ['RESISTANT FRACTION AT 365', pct(h.resistant_fraction_policy,1),
      h.clearance_note ? 'not interpretable at this burden' : 'MTD control: '+pct(h.resistant_fraction_mtd,1),
      h.clearance_note ? 'warn' : (h.resistant_fraction_policy>0.5?'warn':'good')],
  ];

  document.getElementById('flags').innerHTML =
    (h.clearance_note ? `<div class="flag">\\u26a0 ${{h.clearance_note}}</div>` : '') +
    (h.pfs_extension_is_lower_bound
      ? `<div class="flag info">The median-PFS extension is a LOWER BOUND: no replicate in the adaptive arm progressed inside the ${{h.day}}-day horizon, so the true extension cannot be measured here &mdash; only bounded below.</div>`
      : '');
  document.getElementById('kpis').innerHTML = kpis.map(([k,v,m,cls])=>
    `<div class="kpi ${{cls}}"><div class="k">${{k}}</div><div class="v">${{v}}</div><div class="m">${{m}}</div></div>`).join('');

  const pB = pD.burden_total, mB = mD.burden_total;
  const threshold = pB[0] * (1 + DATA.rano_progression_fraction);
  document.getElementById('burdenChart').innerHTML = line([
    {{points: pB.map((v,i)=>[i,v]), color:'#34d399'}},
    {{points: mB.map((v,i)=>[i,v]), color:'#fb7185', threshold: threshold}}
  ], '#34d399', 620, 260, {{l:56,r:16,t:14,b:28}});
  document.getElementById('fracChart').innerHTML = line([
    {{points: pD.resistant_fraction.map((v,i)=>[i,v]), color:'#a78bfa'}},
    {{points: mD.resistant_fraction.map((v,i)=>[i,v]), color:'#fbbf24'}}
  ], '#a78bfa', 620, 260, {{l:56,r:16,t:14,b:28}});
  document.getElementById('doseChart').innerHTML = stepChart(pD.dose_A, pD.dose_B, 760, 150, {{l:8,r:8,t:10,b:20}});

  const rows = (arm, name) => arm.replicate_day_readouts.map(r =>
    `<tr><td>${{name}} \\u00b7 seed ${{r.seed}}</td><td>${{r.pfs_day===null?'not reached':num(r.pfs_day,0)+' d'}}</td>
     <td>${{r.progressed?'YES':'no'}}</td><td>${{num(r.day.burden_total)}}</td>
     <td>${{pct(r.day.resistant_fraction)}}</td><td>${{num(r.day.dose_A_cumulative)}}</td>
     <td>${{num(r.day.dose_B_cumulative)}}</td><td>${{fmt(r.toxicity,1)}}</td></tr>`).join('');
  document.getElementById('repTable').innerHTML =
    `<table><thead><tr><th>arm \\u00b7 replicate</th><th>PFS</th><th>progressed</th>
     <th>burden @365</th><th>R fraction</th><th>\\u03a3 dose A</th><th>\\u03a3 dose B</th><th>toxicity</th></tr></thead>
     <tbody>${{rows(pol,'adaptive')}}${{rows(mtd,'MTD')}}</tbody></table>`;

  const i365 = Math.min(365, pD.burden_total.length) - 1;
  const cmp = [
    ['Tumour burden (cell density)', h.burden_policy, h.burden_mtd],
    ['Resistant clone burden', pD.burden_resistant[i365], mD.burden_resistant[i365]],
    ['Sensitive clone burden', pD.burden_sensitive[i365], mD.burden_sensitive[i365]],
    ['Resistant fraction', h.resistant_fraction_policy, h.resistant_fraction_mtd],
    ['Cumulative dose, Drug A', pD.dose_A.slice(0,i365+1).reduce((a,b)=>a+b,0), mD.dose_A.slice(0,i365+1).reduce((a,b)=>a+b,0)],
    ['Cumulative dose, Drug B', pD.dose_B.slice(0,i365+1).reduce((a,b)=>a+b,0), mD.dose_B.slice(0,i365+1).reduce((a,b)=>a+b,0)],
    ['Median PFS (days)', h.median_pfs_policy, h.median_pfs_mtd],
    ['Median toxicity proxy', pol.median_toxicity, mtd.median_toxicity],
  ];
  document.getElementById('cmpTable').innerHTML =
    `<table><thead><tr><th>quantity at day 365</th><th>adaptive switch</th><th>MTD control</th><th>policy / MTD</th></tr></thead><tbody>` +
    cmp.map(([k,a,b])=>{{ const r = (b>0)? (a/b) : null;
      return `<tr><td>${{k}}</td><td>${{fmt(a,3)}}</td><td>${{fmt(b,3)}}</td><td>${{r===null?'\\u2014':fmt(r,3)}}</td></tr>`; }}).join('') +
    `</tbody></table>`;

  document.getElementById('limits').innerHTML = c.notes.map(n=>'&bull; '+n).join('<br>');
  document.getElementById('foot').innerHTML =
    'Model: '+DATA.model+' &middot; engine constants: <b>'+DATA.engine_constants.provenance+'</b><br>' +
    'Generated '+DATA.generated_utc+' from '+CASES.length+' real-geometry cases. ' +
    'Endpoint: RANO-fraction burden threshold of the simulator, not a radiological read. ' +
    'No number on this page is a prediction for the patient the geometry came from, and none of it is a validated therapy.';
}}
render(0);
</script></body></html>
"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_path
