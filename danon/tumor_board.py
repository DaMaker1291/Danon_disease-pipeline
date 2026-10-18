"""The tumour-board view: five panels, and every field either has a source or is refused.

WHY THIS EXISTS
---------------
A clinical decision-support dashboard is the easiest place in a project to
publish an invented number. A biomarker chip, a recurrence date, a compliance
badge and a test count all LOOK like data whatever produced them, and the screen
gives them the authority of the surrounding real ones.

The mock-up this module answers proposed exactly that: a patient ID, an MGMT
beta of 0.82, a Ki-67 index of 28%, a 5-ALA residual of 0.42 cm3, recurrence on
day 144 under MTD and "day 365+" under the adaptive schedule, a badge reading
972 passing tests and 28/28 claims, an "FDA-PCCP bounds: NOMINAL" status, and a
cryptographic hash of the model. Nearly all of it is unavailable here, and the
one part that IS available -- the paired MTD-versus-adaptive outcome -- is
available with numbers that are not those.

So the artifact carries three kinds of field, never mixed and never implied:

  * value   -- a number read from a recorded engine run, with the run named;
  * refused -- a field this repository has no data for, with the reason and the
               module that would have to produce it;
  * quoted  -- a figure a mock-up showed, carried here ONLY as the thing being
               refused, so a reader comparing the dashboard to this artifact can
               see which numbers have no source behind them.

WHAT IT IS NOT
--------------
It is not a medical device, a certified model, or a regulatory submission. It
carries no FDA status of any kind, because this project has no submission and no
clearance to report the status OF. It also carries no patient: `patient_id` here
is a de-identified cohort label from the committed BraTS-2021 case, and the only
patient-level data in it are the ones a published segmentation carries.

Read out of recorded artifacts, not recomputed: `outputs_real/outcome_cards.json`,
`outputs_real/app_artifacts/evolution.json`, `outputs_real/perf.json` and the
recorded claims audit. Recomputing the outcome cohort inside a UI build would
mean the screen and the record could disagree, which is the failure the artifact
envelope exists to prevent.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from danon.cure import EvidenceStatus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTCOME_CARDS = os.path.join("outputs_real", "outcome_cards.json")
PERF_RECORD = os.path.join("outputs_real", "perf.json")
EVOLUTION_RECORD = os.path.join("outputs_real", "app_artifacts", "evolution.json")
CLAIMS_RECORD = os.path.join("outputs_real", "app_artifacts", "claims.json")

#: The engine files the audit-bar hash covers. Named individually rather than
#: globbed, because a hash over `danon/*.py` would change when an unrelated
#: module is edited and would then claim the MODEL changed.
ENGINE_SOURCE_FILES = (
    "danon/evosteer.py",
    "danon/accel.py",
    "danon/anisotropy.py",
    "danon/uq.py",
)


def _load(rel: str) -> Dict[str, Any]:
    path = os.path.join(ROOT, rel)
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        raise FileNotFoundError(
            f"{rel} is not readable ({exc}); build it first -- this artifact "
            f"reports recorded output and has nothing to report without it"
        )


def engine_source_hash() -> Dict[str, Any]:
    """A hash of the engine's own source, and an explicit statement of what that is.

    A "cryptographic hash of the simulation model" is only meaningful if the
    bytes hashed are named. These are the four modules that hold the numeric
    core: the day loop, the accelerated kernels, the transport operator and the
    replicate runner. Hashing a binary that does not exist here would produce a
    string with no content behind it.
    """
    digest = hashlib.sha256()
    files = []
    for rel in ENGINE_SOURCE_FILES:
        path = os.path.join(ROOT, rel)
        with open(path, "rb") as fh:
            blob = fh.read()
        per_file = hashlib.sha256(blob).hexdigest()
        files.append({"path": rel, "sha256": per_file, "bytes": len(blob)})
        digest.update(rel.encode("utf-8"))
        digest.update(blob)
    return {
        "algorithm": "sha256",
        "scope": "the engine's source files, not a built or certified artifact",
        "files": files,
        "combined_sha256": digest.hexdigest(),
        "verify_with": (
            "sha256 over each file above, concatenated in that order, each "
            "prefixed with its path -- reproducible with hashlib on any machine"
        ),
    }


def _test_function_count() -> Dict[str, Any]:
    """Count test functions the way the registry does, so the two cannot disagree.

    A BUILD-time count, dated, not a remembered one: every run of
    `build_tumor_board()` recounts, so anything reading this artifact live gets
    the tree's current number, and what is checked into `outputs_real/` is a
    snapshot with the day it was taken. The mock-up's "972 tests passing" is a
    badge with no counter behind it; this one names the counter.
    """
    from danon.claims import _suite_function_count

    total, classes = _suite_function_count()
    return {
        "test_functions": total,
        "in_classes": classes,
        "counted_by": "danon.claims._suite_function_count",
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": (
            "functions, not pytest's collected count: a parametrised test is one "
            "function here and many collected cases, and quoting the larger "
            "number as 'tests' would overstate the suite. The count is taken when "
            "this artifact is BUILT, so the committed copy is dated."
        ),
    }


def _tipping_day(curve: List[float]) -> Optional[int]:
    """First day the resistant fraction exceeds 0.5, or None if it never does."""
    for i, value in enumerate(curve):
        if value > 0.5:
            return i
    return None


def build_tumor_board(claims_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Assemble the five panels from recorded output. Raises if a record is missing.

    `claims_payload` is the artifact envelope of an audit taken in THIS process,
    when there is one. Passing it matters: the claims registry is what the audit
    bar reports, and a full build writes the claims artifact and this one in the
    same run, so reading the FILE here would print the previous build's count on
    the newest screen -- a staleness the envelope's own contract forbids. The
    file is still the fallback, for a standalone build.
    """
    cards = _load(OUTCOME_CARDS)
    perf = _load(PERF_RECORD)
    claims = claims_payload if claims_payload is not None else _load(CLAIMS_RECORD)
    try:
        evolution = _load(EVOLUTION_RECORD)["data"]
    except (KeyError, FileNotFoundError):
        evolution = None

    test_count = _test_function_count()
    case = cards["cases"][0]
    day365 = case["day_365"]
    arms = case["arms"]
    geometry = case["geometry"]

    patient_panel = {
        "recorded": [
            {
                "field": "case",
                "value": case["case_id"],
                "source": "committed BraTS-2021 segmentation, engine-ready grid",
            },
            {
                "field": "cohort label",
                "value": case["patient_id"],
                "source": "the same segmentation's de-identified file label",
            },
            {
                "field": "engine grid",
                "value": " x ".join(str(n) for n in geometry["grid"]),
                "source": "recorded geometry",
            },
            {
                "field": "tumour voxels at t0",
                "value": geometry["tumour_voxels"],
                "source": "recorded geometry",
            },
            {
                "field": "voxel edge",
                "value": f"{geometry['voxel_mm']} mm",
                "source": "recorded geometry",
            },
            {
                "field": "resistant fraction at t0",
                "value": case["intake"]["r0"],
                "source": case["intake"]["r0_source"],
            },
            {
                "field": "replicates / horizon",
                "value": f"{len(case['seeds'])} seeds x {case['horizon_days']} days",
                "source": "recorded outcome card",
            },
        ],
        "not_recorded": [
            {
                "field": "MGMT promoter methylation beta",
                "status": "no patient-level value in this repository",
                "detail": (
                    "danon/mgmt.py models methylation status from published "
                    "per-probe beta distributions and a declared cutoff; it "
                    "refuses to report an observed beta for a patient, because "
                    "no methylation array for this case is present"
                ),
                "would_come_from": "a patient methylation array or a pathology report",
                "quoted_in_the_dashboard_mockup": "0.82 (methylated)",
            },
            {
                "field": "Ki-67 index",
                "status": "refused by danon.histology.ki67_fraction",
                "detail": (
                    "a Ki-67 fraction needs a Ki-67 immunohistochemistry slide, "
                    "and no file in the cached cohort is one; the module raises "
                    "rather than estimating the fraction from H&E staining"
                ),
                "would_come_from": "a Ki-67 IHC slide (or a digital pathology tile with the stain identified)",
                "quoted_in_the_dashboard_mockup": "28%",
            },
            {
                "field": "5-ALA fluorescent residual volume",
                "status": "no digitized resection cavity in this repository",
                "detail": (
                    "danon/clinical_twin.py can take an initial_state map (the "
                    "5-ALA use case) and the engine honours it exactly, but no "
                    "5-ALA volume or resection cavity is committed here"
                ),
                "would_come_from": "an intraoperative 5-ALA acquisition or a post-op cavity mask",
                "quoted_in_the_dashboard_mockup": "0.42 cm3",
            },
            {
                "field": "age, sex, KPS",
                "status": "not present in the committed data",
                "detail": "BraTS-2021 ships imaging and segmentation, not a clinical table",
                "would_come_from": "the treatment institution's record",
                "quoted_in_the_dashboard_mockup": "58 M, KPS 90",
            },
        ],
    }

    volume_panel: Dict[str, Any] = {
        "note": (
            "the frames are the engine's exact end-of-day spatial state on the "
            "recorded days; the committed day-120 volume is separately available "
            "as the VTI the ParaView screen downloads"
        ),
        "arm_frames": {},
    }
    if evolution is not None:
        volume_panel["frame_days"] = evolution["frame_days"]
        volume_panel["grid"] = evolution["grid"]
        volume_panel["spacing_mm"] = evolution["spacing_mm"]
        for arm in ("policy", "mtd"):
            entry = evolution["arms"][arm]
            volume_panel["arm_frames"][arm] = {
                "seed": entry["seed"],
                "n_frames": len(entry["frames_vti_b64"]),
                "pfs_day": entry["pfs_day"],
                "progressed": entry["progressed"],
            }
        kinetics: Dict[str, Any] = {
            "days": list(range(len(evolution["resistant_fraction"]["policy"]))),
            "resistant_fraction": evolution["resistant_fraction"],
            "tipping_day": {
                arm: _tipping_day(evolution["resistant_fraction"][arm])
                for arm in ("policy", "mtd")
            },
            "note": (
                "R/(S+R) is an engine state, not a clinical measurement: the "
                "engine records it per day, and the switch rule acts on it, but "
                "no assay in this repository observes it"
            ),
        }
    else:
        kinetics = {
            "resistant_fraction": {},
            "tipping_day": {},
            "note": (
                "no recorded evolution artifact: build it with "
                "danon.app_artifacts.write_app_artifacts(keys=['evolution'])"
            ),
        }

    dosing_panel = {
        "switch_rule": cards["switch_rule"],
        "rano_progression_fraction": cards["rano_progression_fraction"],
        "arms": {
            arm: {
                "label": entry["label"],
                "median_pfs_days": entry["median_pfs_days"],
                "n_progressed": entry["n_progressed"],
                "n_replicates": len(case["seeds"]),
                "median_toxicity": entry["median_toxicity"],
                "cumulative_drug_A": float(sum(entry["daily"]["dose_A"])),
                "cumulative_drug_B": float(sum(entry["daily"]["dose_B"])),
            }
            for arm, entry in arms.items()
        },
        "day_365": {
            "burden_policy": day365["burden_policy"],
            "burden_mtd": day365["burden_mtd"],
            "burden_reduction_vs_mtd": day365["burden_reduction_vs_mtd"],
            "progression_free_policy": day365["progression_free_policy"],
            "progression_free_mtd": day365["progression_free_mtd"],
            "toxicty_note": "the toxicity figure is a cumulative weighted dose proxy, not a grade",
        },
        "pfs_extension_is_lower_bound": bool(day365.get("pfs_extension_is_lower_bound")),
        "no_recurrence_date": (
            "this panel deliberately carries no 'recurrence date'. The engine "
            "reports progression against its own RANO burden rule per replicate; "
            "a single date under each arm would be a point estimate over "
            "stochastic runs presented as an outcome"
        ),
    }

    measured = {
        "laplacian_ratio_by_grid": {
            " x ".join(str(n) for n in op["shape"]): op["laplacian_ratio"]
            for op in perf["operators"]
        },
        "divergence_ratio_by_grid": {
            " x ".join(str(n) for n in op["shape"]): op["divergence_ratio"]
            for op in perf["operators"]
        },
        "day_ratio_scalar": perf["day"]["scalar"]["ratio"],
        "day_ratio_heterogeneous": perf["day"]["heterogeneous"]["ratio"],
        "batch_per_replicate_ratio": perf["throughput"]["batch_per_replicate_streams"]["ratio"],
        "batch_one_stream_ratio": perf["throughput"]["batch_one_stream"]["ratio"],
        "process_ratio_by_workers": {
            k: v["ratio"] for k, v in perf["throughput"]["processes"]["by_workers"].items()
        },
        "bit_identity_holds": {
            "fields": all(r["exact_equal"] for r in perf["bit_identity"]["fields"]),
            "trajectory": all(
                v
                for row in perf["bit_identity"]["trajectory"].values()
                for v in row.values()
                if isinstance(v, bool)
            ),
        },
        "measured_at": perf["measured_at"],
        "statistic": perf["method"]["statistic"],
        "not_measured": perf["what_is_not_measured"],
    }

    audit_panel = {
        "claims": {
            "passed": claims["data"]["n_passed"],
            "total": claims["data"]["n_claims"],
            "failed": claims["data"]["n_failed"],
            "failed_ids": [
                row["id"] for row in claims["data"]["claims"] if not row["supported"]
            ],
            "source": "danon.claims.audit_claims(), recorded by danon/app_artifacts.py",
        },
        **test_count,
        "engine_source_hash": engine_source_hash(),
        "acceleration": measured,
        "no_regulatory_status": (
            "this panel reports no FDA, CE or PCCP status. There is no "
            "submission and no clearance behind this software, so a green "
            "compliance chip would be an assertion about a regulator that has "
            "never seen it"
        ),
    }

    refusals = [
        {
            "shown_in_the_mockup": "Recurrence: Day 144 (MTD) vs Day 365+ (adaptive)",
            "why_refused": (
                "a recurrence date per arm is a point estimate over stochastic "
                "replicates; the engine reports a progression day per replicate "
                "and the adaptive arm's median is a LOWER BOUND when no replicate "
                "progresses"
            ),
            "what_is_recorded_instead": {
                "mtd": {
                    "median_pfs_days": arms["mtd"]["median_pfs_days"],
                    "n_progressed": arms["mtd"]["n_progressed"],
                },
                "adaptive": {
                    "median_pfs_days": arms["policy"]["median_pfs_days"],
                    "n_progressed": arms["policy"]["n_progressed"],
                },
            },
        },
        {
            "shown_in_the_mockup": "972 tests passing",
            "why_refused": (
                "a badge with no counter behind it. The suite is recounted "
                "whenever this artifact is built and never remembered: as of "
                f"{test_count['as_of']} the tree holds "
                f"{test_count['test_functions']} test functions"
            ),
            "what_is_recorded_instead": test_count,
        },
        {
            "shown_in_the_mockup": "28/28 claims verified",
            "why_refused": (
                "the registry is the count, and the recorded audit is "
                f"{claims['data']['n_passed']}/{claims['data']['n_claims']}"
            ),
            "what_is_recorded_instead": {
                "passed": claims["data"]["n_passed"],
                "total": claims["data"]["n_claims"],
            },
        },
        {
            "shown_in_the_mockup": "45 s -> 12 ms per PDE time-step on an RTX 4090 / A100",
            "why_refused": (
                "there is no CUDA device, no JAX and no CuPy in this "
                "repository's environment, so no GPU figure here is measured; "
                "what IS measured is the same operators on the CPU, on this "
                "machine, in outputs_real/perf.json"
            ),
            "what_is_recorded_instead": {
                "cpu_operator_ratios": measured["laplacian_ratio_by_grid"],
                "at_the_clinical_grid": "256 x 256 x 256",
            },
        },
        {
            "shown_in_the_mockup": "Fourier neural operator surrogate, R^2 >= 0.95, 8 ms forecasts",
            "why_refused": (
                "no surrogate is trained or evaluated in this repository, so "
                "there is no accuracy and no latency to report"
            ),
            "what_is_recorded_instead": None,
        },
        {
            "shown_in_the_mockup": "10,000-trajectory Monte Carlo via Ray on CUDA streams",
            "why_refused": (
                "Ray is not installed and no GPU exists here. Widening the "
                "arrays instead (one batched tensor of replicates) was measured "
                "at or below the serial rate; running replicates in OS "
                "processes was measured above it, and cannot change a number "
                "because every worker runs the unmodified serial engine"
            ),
            "what_is_recorded_instead": {
                "batch_per_replicate_streams": measured["batch_per_replicate_ratio"],
                "batch_one_stream": measured["batch_one_stream_ratio"],
                "processes": measured["process_ratio_by_workers"],
            },
        },
        {
            "shown_in_the_mockup": "Zero-copy WebGPU streaming at 60 FPS",
            "why_refused": (
                "this frontend renders recorded frames with three.js from "
                "artifacts, and no WebGPU path exists; a frame rate was never "
                "measured, and the app has no streaming pipeline to measure"
            ),
            "what_is_recorded_instead": None,
        },
        {
            "shown_in_the_mockup": "FDA-PCCP bounds: NOMINAL",
            "why_refused": (
                "there is no FDA submission, no PCCP document and no bound to "
                "be nominal against; the engine's own bound CLAIMS are audited "
                "in danon/claims.py instead"
            ),
            "what_is_recorded_instead": {
                "claims_passed": claims["data"]["n_passed"],
                "claims_total": claims["data"]["n_claims"],
            },
        },
    ]

    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "case": {
            "case_id": case["case_id"],
            "patient_id": case["patient_id"],
            "grid": geometry["grid"],
            "voxel_mm": geometry["voxel_mm"],
            "tumour_voxels": geometry["tumour_voxels"],
            "geometry_note": geometry["note"],
            "r0": case["intake"]["r0"],
            "r0_source": case["intake"]["r0_source"],
        },
        "patient_panel": patient_panel,
        "volume_panel": volume_panel,
        "kinetics_panel": kinetics,
        "dosing_panel": dosing_panel,
        "audit_panel": audit_panel,
        "refusals": refusals,
        "model": cards["model"],
    }


def tumor_board_provenance() -> Dict[str, EvidenceStatus]:
    """What class of number each panel carries, so the UI can chip it."""
    return {
        "patient_panel.recorded": EvidenceStatus.ESTABLISHED,
        "patient_panel.not_recorded": EvidenceStatus.UNKNOWN,
        "volume_panel": EvidenceStatus.ESTABLISHED,
        "kinetics_panel": EvidenceStatus.ESTABLISHED,
        "dosing_panel": EvidenceStatus.ESTABLISHED,
        "dosing_panel.no_recurrence_date": EvidenceStatus.UNKNOWN,
        "audit_panel.acceleration": EvidenceStatus.ESTABLISHED,
        "audit_panel.no_regulatory_status": EvidenceStatus.UNKNOWN,
        "engine_parameters": EvidenceStatus.UNKNOWN,
        "any_gpu_figure": EvidenceStatus.PROVEN_FAILED,
        "any_fno_surrogate_figure": EvidenceStatus.PROVEN_FAILED,
        "any_webgpu_frame_rate": EvidenceStatus.PROVEN_FAILED,
        "any_fda_status": EvidenceStatus.PROVEN_FAILED,
    }


def what_a_tumor_board_cannot_show() -> List[str]:
    return [
        "nothing on these panels is a patient prediction: the engine runs on "
        "declared constants, and the one patient in the artifact is a "
        "segmentation, not a clinical record",
        "the panels carry no biomarker, no molecular status and no "
        "intraoperative residual, because this repository holds none of those "
        "for this case",
        "no figure here is a GPU measurement, a surrogate model's accuracy, or "
        "a rendered frame rate; none of those things exist in this repository",
        "the audit bar reports the state of THIS repository's checks. It is not "
        "a regulatory status and must not be read as one",
    ]
