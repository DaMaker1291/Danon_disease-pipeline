"""
Claims audit: every pitch claim paired with the code that substantiates it.

WHY THIS EXISTS
---------------
A project pitch drifts from its code, and the drift is always in one direction:
towards the more impressive claim. Three drifts were found in the ISEF pitch
this repository is being written for, and each would have been caught by a judge
opening the repository:

* **"provenance registry (ESTABLISHED, DERIVED, PROVEN_FAILED)"** -- `DERIVED`
  is not a member of `EvidenceStatus`. It appears in this repository as prose
  (a *category* alongside CALIBRATED and ASSUMED), and the enum that actually
  gates provenance has five different members. Using one vocabulary while
  citing the other is how a provenance claim becomes decoration.
* **"deterministic math and physics, not a black box"** -- the engine is
  stochastic by design (mutation draw and multiplicative growth noise), and the
  repository ships `torch.nn` modules plus an ESM-based fitness scorer wired to
  live `/api/esm/*` endpoints. The defensible claim is narrower and stronger:
  the *scheduler* is mechanistic; a pretrained protein language model
  contributes capsid-fitness scoring in the *design* arm, and is labelled.
* **"528 tests"** -- a number written once and never recomputed.

So the claims are declared here, each bound to a live check that reads the
repository and can **fail**. A claim whose check fails is reported as failed.
`audit_claims()` is the function a poster should be generated from, and
`render_claims_markdown()` prints the table.

WHAT THIS IS NOT
----------------
An audit is not evidence of efficacy. It shows the code supports the sentence
next to it; it says nothing about whether the underlying biology is right.
`what_these_claims_cannot_establish()` states that in the envelope.
"""

from __future__ import annotations

import json
import os
import re
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from danon.cure import EvidenceStatus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Live checks: each one reads the repository and can fail
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    supported: bool
    detail: str


def _read(relpath: str) -> str:
    with open(os.path.join(ROOT, relpath)) as fh:
        return fh.read()


def check_evidence_status_vocabulary() -> CheckResult:
    """The provenance enum must be described with its own member names."""
    members = [m.name for m in EvidenceStatus]
    has_derived = "DERIVED" in members
    return CheckResult(
        supported=not has_derived,
        detail=(
            f"EvidenceStatus members: {', '.join(members)}"
            + ("" if not has_derived else " (pitch says DERIVED, which is not a member)")
        ),
    )


def check_neural_components_are_declared() -> CheckResult:
    """The repo does contain learned components; a claim of 'no black box' fails."""
    text = _read("danon/vector_design/ml_scorer.py")
    modules = re.findall(r"^class (\w+)\(nn\.Module\)", text, flags=re.M)
    esm = os.path.exists(os.path.join(ROOT, "danon", "esm_inference.py"))
    api = "esm" in _read("api_server.py")
    return CheckResult(
        supported=bool(modules),
        detail=(
            f"torch modules present: {', '.join(modules) or 'none'}; "
            f"esm_inference.py present: {esm}; /api/esm endpoints wired: {api} "
            f"-- a claim of 'no black box anywhere' is false as worded"
        ),
    )


def check_seeded_reproducibility() -> CheckResult:
    """Runs are stochastic; the honest claim is seeded reproducibility."""
    from danon.evosteer import MTDPolicy, TumorEvolutionSimulator, make_gbm_phantom

    geom = make_gbm_phantom(size=8, seed=0)
    sim = TumorEvolutionSimulator(geom)
    a = sim.run(MTDPolicy(), days=15, initial_resistant_fraction=0.05, seed=7)
    b = sim.run(MTDPolicy(), days=15, initial_resistant_fraction=0.05, seed=7)
    c = sim.run(MTDPolicy(), days=15, initial_resistant_fraction=0.05, seed=8)
    same = bool((a.burden_S == b.burden_S).all())
    differs = not bool((a.burden_S == c.burden_S).all())
    return CheckResult(
        supported=same and differs,
        detail=(
            f"same seed reproduces exactly: {same}; different seed diverges: "
            f"{differs} -- the model is stochastic and seeded, not deterministic"
        ),
    )


def check_structure_is_deposited() -> CheckResult:
    """The capsid geometry must be the deposited structure, digest-verified."""
    from danon.vector_design.structure import EXPECTED_SHA256, ensure_structure_file, file_sha256

    path, _ = ensure_structure_file(allow_download=False)
    digest = file_sha256(path)
    return CheckResult(
        supported=digest == EXPECTED_SHA256,
        detail=f"PDB file {os.path.basename(path)} sha256 {digest[:16]}... matches the pin",
    )


def check_wild_type_from_deposit() -> CheckResult:
    """The wild-type capsid must equal VP1 263-736 of that structure."""
    from danon.vector_design.aav_generator import VP1_FRAME_OFFSET, WILD_TYPE_AAV9_CAPSID
    from danon.vector_design.structure import load_aav9_capsid

    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
        "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
        "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
        "TYR": "Y", "VAL": "V",
    }
    capsid = load_aav9_capsid(allow_download=False)
    derived = "".join(
        three_to_one[r.aa] for r in capsid.residues if r.resseq >= VP1_FRAME_OFFSET
    )
    return CheckResult(
        supported=derived == WILD_TYPE_AAV9_CAPSID,
        detail=(
            f"{len(WILD_TYPE_AAV9_CAPSID)} residues, derived from the deposit: "
            f"{derived == WILD_TYPE_AAV9_CAPSID}"
        ),
    )


def check_mechanistic_core() -> CheckResult:
    """Fickian transport and replicator competition must be in the engine."""
    text = _read("danon/evosteer.py")
    has_diffusion = "diffusion" in text and "_laplacian" in text
    has_competition = "competition" in text and "carrying_capacity" in text
    has_kill = "ec50" in text and "kill_max" in text
    return CheckResult(
        supported=has_diffusion and has_competition and has_kill,
        detail=(
            f"PDE transport (diffusion + Laplacian): {has_diffusion}; "
            f"Lotka-Volterra competition with carrying capacity: {has_competition}; "
            f"saturating (Hill/EC50) kill kinetics: {has_kill} -- the scheduler is "
            f"mechanistic"
        ),
    )


def check_parameters_are_declared_assumptions() -> CheckResult:
    """Rate constants must be labelled, since they are not measured here."""
    text = _read("danon/evosteer.py")
    marked = text.count("CALIBRATED")
    assumed = "ASSUMED" in text
    return CheckResult(
        supported=assumed and marked > 0,
        detail=(
            f"{marked} parameters marked CALIBRATED, 'ASSUMED' section present: "
            f"{assumed} -- the mechanism is mechanistic, the constants are not "
            f"measurements"
        ),
    )


def _suite_function_count() -> Tuple[int, int]:
    """Test functions in the suite the README's own command runs.

    Counts test *functions*, at module level and inside classes: an anchored
    `^def test_` misses every method in a test class, which understated this
    suite by more than half the first time it ran. Both roots are covered
    because the documented command is `pytest tests/ danon/multiagent/`; a
    count over `tests/` alone is not the number that command collects.
    """
    total = 0
    classes = 0
    for root in ("tests", "danon/multiagent"):
        abs_root = os.path.join(ROOT, root)
        for name in sorted(os.listdir(abs_root)):
            if not (name.startswith("test_") and name.endswith(".py")):
                continue
            text = _read(os.path.join(root, name))
            total += len(re.findall(r"^\s*def test_", text, flags=re.M))
            classes += len(re.findall(r"^class Test", text, flags=re.M))
    return total, classes


def check_test_suite_is_counted_live() -> CheckResult:
    """Never quote a test count that was not recomputed -- including in README.

    Two failure modes, both of which happened. Counting: an anchored
    `^def test_` understated the suite by more than half. Quoting: the README's
    suite row advertised 1068 passed while the tree it describes held 1059 test
    functions -- a remembered number for a quantity that changes every time a
    test is added, so the row could only be kept true by hand. The row now
    defers to this check and the number is published where it is derived.
    """
    total, classes = _suite_function_count()
    try:
        readme = _read("README.md")
    except OSError as exc:
        return CheckResult(supported=False, detail=f"README unreadable: {exc}")
    if "check_test_suite_is_counted_live" not in readme:
        return CheckResult(
            supported=False,
            detail=(
                "the README's suite row does not defer to this check: a "
                "remembered count is how 1068 came to be advertised for a "
                "1059-function suite"
            ),
        )
    return CheckResult(
        supported=total > 0,
        detail=(
            f"{total} test functions in {classes} classes across tests/ and "
            f"danon/multiagent/ (pytest collects more than this wherever a "
            f"test is parametrised); README defers to this check rather than "
            f"quoting a count"
        ),
    )


def check_voxel_scale() -> CheckResult:
    """The engine grid is millimetres, not micrometres.

    A pitch describing 100 um voxels was off by ~40x per axis against what this
    repository actually builds: real BraTS cases are majority-vote decimated to
    an engine grid around 4 mm. Reporting a finer grid than the code produces is
    the kind of number a judge can check by asking one question.
    """
    from danon.realdata import DecimationSpec

    spec = DecimationSpec()
    factors = [float(f) for f in spec.factors]
    voxel_mm = sum(factors) / len(factors) * 1.0     # source grid is 1 mm
    text = _read("danon/realdata.py")
    return CheckResult(
        supported=voxel_mm >= 0.5,
        detail=(
            f"engine voxel ~{voxel_mm:.2f} mm from decimation factors {spec.factors} "
            f"(source 1 mm); ~{voxel_mm * 1000:.0f} um, not 100 um; decimation is "
            f"declared as a lossy step in the module: "
            f"{'documented lossy step' in text}"
        ),
    )


def check_runtime_365_days() -> CheckResult:
    """A 365-day run is measured here, not asserted on a poster."""
    import time

    from danon.evosteer import MTDPolicy, TumorEvolutionSimulator, make_gbm_phantom

    geom = make_gbm_phantom(size=24, seed=0)
    sim = TumorEvolutionSimulator(geom)
    t0 = time.time()
    sim.run(MTDPolicy(), days=365, initial_resistant_fraction=0.05, seed=0)
    seconds = time.time() - t0
    return CheckResult(
        supported=seconds < 60.0,
        detail=(
            f"one 365-day arm on the engine's {int(geom.tumor_mask.sum())}-voxel "
            f"phantom: {seconds:.2f} s on this machine (re-measured at audit time; "
            f"a full paired case with replicates on real geometry costs ~20 s)"
        ),
    )


def check_parameters_are_bulk_not_single_cell() -> CheckResult:
    """Inputs are bulk transcriptomes; single-cell inference is refused."""
    text = _read("danon/genomics_intake.py")
    uses_bulk = "bulk RNA-seq" in text
    refuses_single_cell = "single_cell_clone_fractions_from_bulk" in text and "PROVEN_FAILED" in text
    return CheckResult(
        supported=uses_bulk and refuses_single_cell,
        detail=(
            f"computational inputs are bulk RNA-seq: {uses_bulk}; deriving "
            f"single-cell clone fractions from bulk is marked PROVEN_FAILED in "
            f"the module: {refuses_single_cell} -- a pitch claiming scRNA-seq "
            f"parameters would be describing data this pipeline never reads"
        ),
    )


def check_engine_c_index_is_not_claimed() -> CheckResult:
    """The audit must confirm the engine declares its own missing metric."""
    from danon.predictive import engine_discrimination_verdict

    v = engine_discrimination_verdict()
    return CheckResult(
        supported=(not v.estimable and v.c_index is None),
        detail=f"engine C-index declared not estimable: {not v.estimable}",
    )


def check_spatial_agreement_is_not_claimed() -> CheckResult:
    """No Dice/segmentation result may be claimed while the verdict refuses.

    This check is deliberately *coupled* to the data verdict: if a collection
    with masks and a second timepoint ever appears, this check starts failing
    until the poster is updated to state the new result. A claim that cannot go
    stale is a claim nobody is maintaining.
    """
    from danon.multimodal import load_evidence, spatial_dice_verdict

    v = spatial_dice_verdict()
    imaging = [
        m for m in load_evidence()
        if m.key.startswith("tcia_") and m.key != "tcia_nbia_api"
    ]
    checked = sum(
        int(m.get("patients_checked_for_timepoints") or 0) for m in imaging
    )
    masks = sum(int(m.get("n_seg_series") or 0) for m in imaging)
    return CheckResult(
        supported=not v.supported,
        detail=(
            f"spatial Dice test {'SUPPORTED' if v.supported else 'REFUSED'}; "
            f"{len(imaging)} imaging collection(s) probed, "
            f"{masks} segmentation series, "
            f"{checked} patient(s) checked for a second timepoint"
        ),
    )


def check_potency_is_measured() -> CheckResult:
    """The potency figures must come from real measured records."""
    from danon.pharmacology import load_glioma_potency

    try:
        mol, s = load_glioma_potency()
    except Exception as exc:
        return CheckResult(False, f"real potency intake unavailable: {exc}")
    ok = s.n_exact >= 30 and s.n_records >= 50 and s.most_potent_exact_nM is not None
    return CheckResult(
        supported=ok,
        detail=(
            f"{mol.chembl_id} {s.standard_type}: {s.n_exact} exact of "
            f"{s.n_records} glioma-line record(s), median "
            f"{s.median_exact_nM:,.0f} nM, most potent "
            f"{s.most_potent_exact_nM:,.0f} nM, {s.n_right_censored} right-censored"
        ),
    )


def check_ec50_unit_is_undeclared() -> CheckResult:
    """The engine's EC50 has no unit, so no potency calibration may be implied."""
    from danon.pharmacology import (
        compare_engine_ec50,
        engine_ec50_unit,
        load_glioma_potency,
    )

    unit = engine_ec50_unit()
    try:
        _, s = load_glioma_potency()
        cmp = compare_engine_ec50(s, 0.30, unit)
        refused = not cmp.comparable
        detail = (
            f"engine EC50 unit declared: {unit!r}; comparison "
            f"{'REFUSED' if refused else 'permitted'} -- "
            f"{cmp.reason if refused else cmp.describe()}"
        )
    except Exception as exc:
        refused = unit is None
        detail = f"EC50 unit {unit!r}; potency intake unavailable: {exc}"
    return CheckResult(supported=refused, detail=detail)


def check_censoring_is_preserved() -> CheckResult:
    """The cohort must contain censored patients, and gains must be measured.

    A pitch quoting a held-out progression C-index has to be quoted against an
    endpoint that includes censoring: discarding every censored patient deletes
    the longest survivors and inflated the progression C-index from 0.534 to
    0.590. This check is coupled to the real numbers, so if the loader regresses
    the poster's sentence stops verifying.
    """
    from danon.predictive import kfold_holdout_c_index, load_cohort_rows

    rows, notes = load_cohort_rows(endpoint="progression")
    events = sum(r.event for r in rows)
    censored = len(rows) - events
    if censored <= 0:
        return CheckResult(
            False,
            f"progression cohort has no censored patients ({len(rows)} events, "
            f"0 censored): every patient is an event and the C-index is biased upward",
        )
    rep = kfold_holdout_c_index(rows, endpoint="progression")
    beaten_by_age = rep.mean_multivariable <= rep.mean_age_only
    return CheckResult(
        supported=True,
        detail=(
            f"progression endpoint: {len(rows)} patients, {events} events, "
            f"{censored} censored at last contact; held-out C = "
            f"{rep.mean_multivariable:.3f} +/- {rep.sd_multivariable:.3f} against "
            f"{rep.mean_age_only:.3f} for age alone -- the multivariable model "
            f"{'is' if beaten_by_age else 'is not'} beaten by age",
        ),
    )


def check_no_inflated_estimand_is_claimed() -> CheckResult:
    """No proposed 0.90+ target may be reported as a measured score.

    Coupled to the data, not to prose: the spatial target must stay refused
    while the live imaging verdict refuses, no responder label may appear, and
    the GDSC IC50s must still be mostly extrapolations. If any of those change,
    this check starts failing until the poster states the new number.
    """
    from danon.eval_targets import (
        GDSC2_FITTED_PATH,
        TargetStatus,
        audit_responder_label,
        in_vitro_target,
        spatial_expansion_target,
    )

    spatial = spatial_expansion_target()
    label = audit_responder_label()
    in_vitro = in_vitro_target()

    no_label = label.n_best_overall_response == 0
    refused = (
        spatial.status is TargetStatus.REFUSED_NO_DATA
        and in_vitro.status is TargetStatus.REFUSED_INCOMMENSURABLE
    )
    no_measured_score = not (spatial.has_number or in_vitro.has_number)
    return CheckResult(
        supported=no_label and refused and no_measured_score,
        detail=(
            f"`best_overall_response` records: {label.n_best_overall_response} of "
            f"{label.n_cases} cases (empty: {no_label}); spatial Dice target "
            f"{spatial.status.value}; in-vitro target {in_vitro.status.value}; "
            f"no target reports a measured score: {no_measured_score}. "
            f"Reachable response flag: {label.n_positive} positives in "
            f"{label.n_clean_label} cases, majority-class accuracy "
            f"{100 * label.majority_accuracy:.2f}%"
        ),
    )


def check_gdsc_ic50s_are_not_all_measurements() -> CheckResult:
    """In-vitro target values must be mostly extrapolations, and say so."""
    import os

    from danon.eval_targets import GDSC2_FITTED_PATH, load_gdsc_glioma

    if not os.path.exists(GDSC2_FITTED_PATH):
        return CheckResult(
            False,
            f"GDSC fitted release not cached at {GDSC2_FITTED_PATH}, so the "
            f"extrapolation claim cannot be verified here",
        )
    audit = load_gdsc_glioma()
    all_tmz_extrapolated = audit.temozolomide_above_max_tested == audit.temozolomide_rows
    return CheckResult(
        supported=audit.extrapolated_fraction > 0.5 and all_tmz_extrapolated,
        detail=(
            f"{audit.n_gbm_rows} glioma rows over {audit.n_gbm_drugs} drugs and "
            f"{audit.n_gbm_lines} lines: {100 * audit.extrapolated_fraction:.1f}% of "
            f"IC50s above the highest concentration tested; temozolomide "
            f"{audit.temozolomide_above_max_tested} of {audit.temozolomide_rows} "
            f"extrapolated. An R^2 against these would be a fit to a lower bound"
        ),
    )


def check_untestable_tiers_are_refused() -> CheckResult:
    """The RL tier and the EGFRvIII feature must stay refused, with reasons."""
    from danon.eval_targets import probe_egfrviii_reachability, rl_policy_target

    rl = rl_policy_target()
    egfr = probe_egfrviii_reachability()
    rl_refused = rl.measured is None and "no reinforcement-learning agent" in rl.detail
    egfr_refused = not egfr.reachable_open
    return CheckResult(
        supported=rl_refused and egfr_refused,
        detail=(
            f"no RL agent present: {rl_refused}; the metric is self-defined, so it "
            f"cannot be falsified. EGFRvIII: {egfr.n_structural_variation_files} "
            f"structural-variation files of which "
            f"{egfr.n_structural_variation_controlled} are controlled, against "
            f"{egfr.n_open_maf_files} open SNV/indel files that cannot carry an "
            f"exon 2-7 deletion: reachable from open data = {egfr.reachable_open}"
        ),
    )


def check_dice_degenerate_case_is_not_perfect() -> CheckResult:
    """An empty prediction must not score 1.0 against an empty truth."""
    import numpy as np

    from danon.eval_targets import dice_similarity

    z = np.zeros((4, 4, 4))
    both_empty = dice_similarity(z, z)
    t = np.zeros((4, 4, 4))
    t[1, 1, 1] = 1
    miss = dice_similarity(z, t)
    return CheckResult(
        supported=(not both_empty.estimable) and miss.coefficient == 0.0,
        detail=(
            f"both masks empty -> estimable={both_empty.estimable} "
            f"(coefficient {both_empty.coefficient}); empty prediction against a "
            f"non-empty truth -> {miss.coefficient:.1f}. The conventional "
            f"`total_volume == 0 -> 1.0` shortcut would score the first case at "
            f"1.0, i.e. a perfect score for making no prediction"
        ),
    )


def check_diffusion_units_and_operator() -> CheckResult:
    """Diffusion units are explicit, and the two transport operators agree.

    Two things must hold together: the mm^2/day to voxel^2/day bridge must be
    present with the repository's own voxel size (not a third value), and the
    heterogeneous operator must reduce to the engine's scalar Laplacian on
    constant D. If the second ever fails, the heterogeneous path is a different
    model rather than a generalisation of the first.
    """
    import numpy as np

    from danon.anisotropy import (
        REAL_VOXEL_MM,
        d_to_engine,
        d_to_physical,
        divergence_d_grad,
        unit_audit,
    )
    from danon.evosteer import _laplacian
    from danon.realdata import DecimationSpec

    spec = DecimationSpec()
    repo_voxel = float(np.mean([1.0 * f for f in spec.factors]))
    voxel_agrees = abs(repo_voxel - REAL_VOXEL_MM) < 1e-9

    rng = np.random.default_rng(0)
    c = rng.random((7, 7, 7))
    d = 0.35
    identity_holds = bool(
        np.allclose(
            divergence_d_grad(c, np.full_like(c, d)),
            d * _laplacian(c),
            atol=1e-12,
        )
    )
    row = next(r for r in unit_audit() if "drug diffusion" in r.name)
    physical = d_to_physical(d, REAL_VOXEL_MM)
    return CheckResult(
        supported=voxel_agrees and identity_holds,
        detail=(
            f"voxel size {REAL_VOXEL_MM:.3f} mm from decimation {spec.factors} "
            f"matches realdata's mean convention: {voxel_agrees}; "
            f"div(D grad C) reduces to D*lap(C) on constant D: {identity_holds}; "
            f"the engine's D={d:g} voxel^2/day is {row.phantom_mm2_per_day:.4g} "
            f"mm^2/day on the phantom but {physical:.4g} mm^2/day on real geometry "
            f"({row.inflation_factor:.2f}x inflation); "
            f"1 voxel^2/day = {1.0 / d_to_engine(1.0, 1.0):.4g} mm^2/day at 1 mm"
        ),
    )


def check_covariate_gains_are_measured() -> CheckResult:
    """Covariate gains are computed on the real cohort, never transcribed.

    Two things must hold at once: the module must contain no authored gain
    figure, and the leakage contrast must actually show a post-treatment KPS
    outranking a pre-treatment one -- that is the measurement that justifies the
    timepoint gate, and it is only meaningful while it holds.
    """
    from danon.cohort_features import (
        KPS_BASELINE_TIMEPOINTS,
        KPS_POST_BASELINE_TIMEPOINTS,
        kps_leakage_contrast,
    )
    from danon.predictive import load_cohort_rows

    source = _read("danon/cohort_features.py")
    # Any of the roadmap's estimated deltas appearing in the module would mean
    # an estimate had been pasted in beside the measurements.
    authored = [s for s in ("+0.05 to +0.08", "0.780 - 0.820", "+0.04 to +0.06") if s in source]
    rows, _ = load_cohort_rows(endpoint="os")
    contrast = kps_leakage_contrast(rows, _clinical_cases())
    gate_disjoint = not (KPS_BASELINE_TIMEPOINTS & KPS_POST_BASELINE_TIMEPOINTS)
    leaks = contrast.post_baseline_c > contrast.baseline_c
    return CheckResult(
        supported=not authored and gate_disjoint and leaks,
        detail=(
            f"authored gain figures in the module: {authored or 'none'}; "
            f"baseline and post-treatment timepoint sets disjoint: {gate_disjoint}; "
            f"post-treatment KPS outranks pre-treatment: {leaks} "
            f"({contrast.post_baseline_c:.3f} vs {contrast.baseline_c:.3f}) "
            f"-- a covariate gain is a measurement here, not an estimate"
        ),
    )


def _clinical_cases() -> List[dict]:
    """The cached clinical export, for checks that need the raw case records."""
    from danon.predictive import GDC_CLINICAL_PATH

    try:
        with open(GDC_CLINICAL_PATH) as fh:
            return list(json.load(fh).get("hits", []))
    except (OSError, ValueError):
        return []


def check_holdout_emulator_is_measured() -> CheckResult:
    """The 0.90+ emulator claim must exist as a one-shot holdout measurement.

    Delegates to the single reusable measurement (locked feature subset, one
    fit, one prediction). A re-run below 0.90 fails the claim rather than
    letting the sentence survive as stale.
    """
    import os as _os

    from danon.high_ceiling import HOLDOUT_PATH, LOCKED_LAMBDA, holdout_measurement

    train_path = _os.path.join(ROOT, "data/synthetic_emulator_dataset.json")
    hold_path = _os.path.join(ROOT, HOLDOUT_PATH)
    if not (_os.path.exists(train_path) and _os.path.exists(hold_path)):
        return CheckResult(
            supported=False,
            detail="emulator train/holdout datasets missing; the 0.90 claim has no measurement",
        )
    m = holdout_measurement()
    return CheckResult(
        supported=m.meets_claim,
        detail=(
            f"one-shot holdout R^2 = {m.r2:.4f} (95% bootstrap CI "
            f"{m.ci[0]:.4f}-{m.ci[1]:.4f}) on {m.n_test} untouched "
            f"configurations (features locked on the training half, single "
            f"refit, lambda={LOCKED_LAMBDA}); claim band R^2 >= 0.90"
        ),
    )


def check_mgmt_status_is_measured_not_faked() -> CheckResult:
    """The MGMT classifier result must be the real measured AUC, low as it is.

    Pins the honest miss: from age/sex/diagnosis the held-out AUC is near
    chance, and the claim recorded here is that the measurement exists and is
    labelled, not that it is high.
    """
    from danon.high_ceiling import mgmt_status_target

    m = mgmt_status_target()
    v = m.verdict
    low = np.isfinite(m.auc) and m.auc < 0.70
    chance_in_ci = m.auc_ci[0] <= 0.5 <= m.auc_ci[1]
    return CheckResult(
        supported=bool(np.isfinite(m.auc) and low and chance_in_ci and v.n >= 100),
        detail=(
            f"real-patient MGMT-status classifier: OOF AUC {m.auc:.4f} "
            f"(CI {m.auc_ci[0]:.4f}-{m.auc_ci[1]:.4f}) on {v.n} patients; "
            f"reported as indistinguishable from chance, NOT as a 0.90+ result"
        ),
    )


def check_covariate_upgrades_are_measured() -> CheckResult:
    """The four-requested-upgrades report must exist, freshly measured.

    Re-measures the ceiling on the live cache: the all-covariates model must
    still fall below age-alone discrimination on the subset it keeps, and the
    artifact must record each of the four upgrades with a paired delta.
    """
    import json as _json
    import os as _os

    from danon.cohort_features import (
        GDC_CLINICAL_PATH,
        TREATMENT_RECORDS_PATH,
        baseline_kps,
        predictive_ceiling,
    )
    from danon.mgmt import load_mgmt_beta_panel, load_published_mgmt_calls
    from danon.predictive import load_cohort_rows

    clinical = _os.path.join(ROOT, GDC_CLINICAL_PATH)
    treatments = _os.path.join(ROOT, TREATMENT_RECORDS_PATH)
    artifact = _os.path.join(ROOT, "outputs_real/covariate_upgrades.txt")
    if not (_os.path.exists(clinical) and _os.path.exists(treatments)):
        return CheckResult(
            supported=False,
            detail="cached GDC clinical/treatment exports absent; the upgrade report has no data",
        )

    rows, _ = load_cohort_rows(endpoint="os")
    ids = [r.submitter_id for r in rows]
    calls = load_published_mgmt_calls()
    panel = load_mgmt_beta_panel()
    feats = [
        ("kps_baseline", baseline_kps(_clinical_cases())),
        ("mgmt_methylated", calls.binaries(ids)),
        ("mgmt_beta", panel.betas_of(ids)),
    ]
    ceiling = predictive_ceiling(rows, feats, n_folds=5, seed=0, n_bootstrap=100)
    artifact_ok = _os.path.exists(artifact) and all(
        marker in _read("outputs_real/covariate_upgrades.txt")
        for marker in (
            "upgrade 3: pre-treatment KPS",
            "upgrade 1: MGMT promoter methylation",
            "upgrade 4: restrict to patients recording RT",
            "upgrade 2: 3D sub-region segmentation",
        )
    )
    return CheckResult(
        supported=bool(ceiling.pooled_c < ceiling.age_only_c and artifact_ok),
        detail=(
            f"four covariate upgrades measured on the live cache: all-covariates "
            f"ceiling pooled C {ceiling.pooled_c:.4f} (n={ceiling.n_patients}) vs "
            f"age-alone {ceiling.age_only_c:.4f} on the same kept subset -- stacking "
            f"unavailable-for-most covariates costs discrimination; the artifact "
            f"records each upgrade with its paired delta: {artifact_ok}"
        ),
    )


def check_mgmt_resistance_sweep_is_measured() -> CheckResult:
    """The beta->resistance mapping must move the engine in the methylated direction.

    Re-runs a small two-beta MTD sweep live and requires the fully methylated
    arm not to progress earlier than the unmethylated one, and the committed
    artifact to carry the full quantile measurement it reports.
    """
    from danon.evosteer import MTDPolicy
    from danon.mgmt import mgmt_resistance_sweep

    sweep = mgmt_resistance_sweep(
        (0.0, 0.8),
        beta_source="claims audit",
        geometry_size=8,
        policies=[MTDPolicy()],
        n_replicates=2,
        seed0=0,
    )
    arms = {a.beta: a for a in sweep.by_policy["mtd"]}
    direction_ok = arms[0.8].median_pfs_days >= arms[0.0].median_pfs_days
    direction_word = "OK" if direction_ok else "REVERSED"
    artifact_path = os.path.join(ROOT, "outputs_real/mgmt_resistance.txt")
    artifact = _read("outputs_real/mgmt_resistance.txt") if os.path.exists(artifact_path) else ""
    artifact_ok = "r0 = Baseline_Resistance * (1 - beta)" in artifact and "beta=" in artifact
    return CheckResult(
        supported=bool(direction_ok and artifact_ok),
        detail=(
            f"live two-beta sweep: MTD median PFS {arms[0.0].median_pfs_days:.0f}d "
            f"(unmethylated) -> {arms[0.8].median_pfs_days:.0f}d (fully methylated), "
            f"direction {direction_word}; committed quantile artifact present: {artifact_ok}"
        ),
    )


def check_treatment_restriction_separates_membership_from_signal() -> CheckResult:
    """The RT+temozolomide restriction must report cohort selection separately.

    Re-measures on the live treatment cache: the kept subset's own baseline and
    the full-cohort baseline must both be present, and the cohort-membership
    delta must be reported against the full-cohort fold sd so a change in n is
    never reported as a change in signal.
    """
    import json as _json
    import os as _os

    from danon.cohort_features import (
        TREATMENT_RECORDS_PATH,
        treatment_exposure,
        treatment_restriction_effect,
    )
    from danon.predictive import load_cohort_rows

    tpath = _os.path.join(ROOT, TREATMENT_RECORDS_PATH)
    if not _os.path.exists(tpath):
        return CheckResult(supported=False, detail="treatment cache absent")
    hits = _json.load(open(tpath))["hits"]
    rows, _ = load_cohort_rows(endpoint="os")
    r = treatment_restriction_effect(
        rows, hits, covariates=[], n_folds=5, seed=0, exposure=treatment_exposure(hits)
    )
    kept = 0 < r.n_restricted < r.n_cohort
    baseline_finite = np.isfinite(r.base_c_full) and np.isfinite(r.base_c_restricted)
    delta_in_sd = abs(r.cohort_membership_delta) <= r.base_sd_full
    return CheckResult(
        supported=bool(kept and baseline_finite and delta_in_sd),
        detail=(
            f"restriction keeps {r.n_restricted}/{r.n_cohort} patients; base C "
            f"{r.base_c_full:.4f} (all) -> {r.base_c_restricted:.4f} (restricted); "
            f"cohort-membership delta {r.cohort_membership_delta:+.4f} inside one "
            f"fold sd ({r.base_sd_full:.4f}) -- selection is reported separately "
            f"from any covariate signal"
        ),
    )


# ---------------------------------------------------------------------------
# Clinical-twin stream checks (the four-hospital-datastream module)
# ---------------------------------------------------------------------------


def check_resistance_is_emergent() -> CheckResult:
    """Resistance must come from the mechanism, not from the initial state.

    This is the falsification that exposed two defects in this engine: with
    no seeded clone the clustered seed still placed ~9 cells (so mutation did
    no work and mu=0 matched mu=5e-5 exactly), and with no extinction floor a
    sub-single-cell remnant regrew deterministically while R/(S+R) read 1.00.
    It is checked live here rather than quoted, so the fix cannot silently
    regress.
    """
    import dataclasses as _dc

    from danon.evosteer import (
        DEFAULT_CLONES,
        MTDPolicy,
        TumorEvolutionSimulator,
        make_gbm_phantom,
    )

    geom = make_gbm_phantom()
    no_mu = _dc.replace(DEFAULT_CLONES, mutation_rate=0.0)

    def run(clones, ifr: float, days: int):
        return TumorEvolutionSimulator(geom, clones=clones).run(
            MTDPolicy(), days=days, initial_resistant_fraction=ifr, seed=101
        )

    sterile = run(no_mu, 0.0, 200)
    seeded = run(DEFAULT_CLONES, 0.005, 250)

    # (1) no mutation + no seed => no resistant cell may exist at all.
    no_ghost = float(sterile.burden_R.max()) == 0.0
    # (2) the floor makes eradication exact rather than asymptotic.
    eradicated = float(sterile.burden_S[-1]) == 0.0
    # (3) the verdict must track the ASSUMED pre-existing clone.
    dependent = seeded.pfs_day is not None and sterile.pfs_day is None
    return CheckResult(
        supported=no_ghost and eradicated and dependent,
        detail=(
            f"mutation off + no seeded clone -> max resistant burden "
            f"{float(sterile.burden_R.max()):.3e} (no ghost resistance: {no_ghost}); "
            f"sensitive population extinguished exactly: {eradicated}; "
            f"verdict tracks the assumed pre-existing clone "
            f"(MTD PFS {seeded.pfs_day} d seeded vs {sterile.pfs_day} unseeded): "
            f"{dependent}"
        ),
    )


def check_stream_identities_hold() -> CheckResult:
    """Each stream's neutral value must reproduce the engine default exactly."""
    from danon.clinical_twin import KI67_REF, stream2_growth_rate, stream2_clones, build_stream3_clearance, stream1_initial_state
    from danon.evosteer import DEFAULT_CLONES

    g = stream2_growth_rate(KI67_REF)
    c = stream2_clones(KI67_REF)
    ki67_ok = abs(g - DEFAULT_CLONES.growth_S) < 1e-12 and (
        abs(c.growth_R - DEFAULT_CLONES.growth_R) < 1e-12
    )
    rcbv = np.full((4, 4, 4), 0.5)
    pattern = build_stream3_clearance(rcbv)
    clr_ok = bool(np.allclose(pattern, 1.0))
    mask = np.ones((3, 3, 3), dtype=bool)
    fluo = np.ones((3, 3, 3))
    ns, nr = stream1_initial_state(fluo, mask, r0=0.05)
    total = float(ns.sum() + nr.sum())
    expected = 0.65 * 1.0 * int(mask.sum())
    ala_ok = abs(total - expected) < 1e-9
    return CheckResult(
        supported=ki67_ok and clr_ok and ala_ok,
        detail=(
            f"Ki67={KI67_REF} -> calibrated growth exactly: {ki67_ok}; "
            f"uniform rCBV -> uniform clearance: {clr_ok}; 5-ALA map preserves "
            f"the engine fill burden ({total:.4f} == {expected:.4f}): {ala_ok}"
        ),
    )


def check_stream3_dsc_perfusion_is_measured() -> CheckResult:
    """Stream 3's measured form: a real DSC series read into a relative-CBV map.

    Four things are verified live, not restated. The pipeline recovers a
    planted 3:1 CBV ratio under signal noise (a chain that cannot recover a
    known ratio has no business reporting an unknown one). Every committed case
    has a bolus window inside the acquisition, a drift fraction strictly between
    0 and 1 (the head-to-tail line was subtracted, not added), and a transport
    field with more than one value. The endpoint is labelled for what it is: PFS
    is censored in this cohort, so the burden readout is the one with resolution
    and the claim may not rest on PFS. And the archive label stays corrected --
    the CaPTk series are structural, not perfusion maps.
    """
    from danon.perfusion import planted_cbv_recovery

    recovery = planted_cbv_recovery(true_ratio=3.0, noise=0.002, seed=1)
    recovery_ok = (
        recovery["relative_error"] < 0.05
        and 3 <= recovery["window_frames"] <= 20
        and recovery["peak_is_after_baseline"] == 1.0
    )

    path = os.path.join(ROOT, "outputs_real", "perfusion_cohort.json")
    try:
        rows = json.load(open(path))
    except (OSError, ValueError) as exc:
        return CheckResult(supported=False, detail=f"perfusion cohort unreadable: {exc}")
    if not isinstance(rows, list) or not rows:
        return CheckResult(supported=False, detail="perfusion cohort holds no cases")

    problems: List[str] = []
    contrasts: List[float] = []
    burdens: List[float] = []
    censored = 0
    for row in rows:
        cbv, engine = row.get("cbv", {}), row.get("engine", {})
        if not (cbv.get("arrival_frame", 99) < cbv.get("peak_frame", -1) < cbv.get("end_frame", -1)):
            problems.append(f"{row.get('patient_id')}: bolus window out of order")
        if not 0.0 < float(cbv.get("drift_peak_ratio", 0.0)) < 1.0:
            problems.append(f"{row.get('patient_id')}: drift fraction not in (0, 1)")
        for policy, arm in (engine.get("arms") or {}).items():
            contrasts.append(float(arm.get("field_contrast_ratio", 0.0)))
            burdens.append(float(arm.get("max_abs_relative_burden_difference", 0.0)))
            if int(arm.get("pfs_informed_pairs", 0)) == 0:
                censored += 1
            if not np.isfinite(arm.get("max_abs_relative_burden_difference", float("nan"))):
                problems.append(f"{row.get('patient_id')}/{policy}: burden difference missing")

    if not contrasts or min(contrasts) <= 1.0:
        problems.append("no arm reports a transport contrast above 1x")
    if censored == 0:
        problems.append(
            "no arm reports a censored PFS endpoint; the claim's endpoint "
            "labelling would then be wrong"
        )

    entry = _read("danon/multimodal.py")
    if "PROCESSED_CAPTK_TOKENS" not in entry or "CaPTk-processed perfusion maps" in entry:
        problems.append(
            "danon/multimodal.py still calls the CaPTk-processed structural "
            "series perfusion maps"
        )

    supported = recovery_ok and not problems
    return CheckResult(
        supported=supported,
        detail=(
            f"planted 3:1 CBV recovered to {recovery['recovered_ratio']:.3f} "
            f"({recovery['relative_error']:.1%} error, {recovery['window_frames']:.0f}-frame "
            f"window); {len(rows)} downloaded series: field contrast "
            f"{min(contrasts):.1f}-{max(contrasts):.1f}x, final-burden effect up to "
            f"{max(burdens):.1%}, {censored} arm-sets with a censored PFS endpoint"
            + ("; " + "; ".join(problems) if problems else "")
        ),
    )


def check_stream3_diffusion_is_measured() -> CheckResult:
    """The rCBV -> D(x) arm: field stats re-derived live, engine arms consistent.

    The field statistics are recomputed from the cached BraTS case on every
    check (cheap, no engine time). The engine arms are read from the committed
    cohort and required to be self-consistent, because re-running six paired
    engine trials inside an audit is minutes of compute, not milliseconds.
    """
    import json

    from danon.clinical_twin import (
        DIFFUSION_GAMMA,
        STREAM3_DIFFUSION_COHORT_PATH,
        build_stream3_diffusion_field,
        build_stream3_vessels,
    )

    if not os.path.exists(STREAM3_DIFFUSION_COHORT_PATH):
        return CheckResult(
            supported=False,
            detail="stream-3 diffusion cohort missing; rebuild via "
            "_stream3_diffusion_cohort.py",
        )
    records = json.load(open(STREAM3_DIFFUSION_COHORT_PATH))
    consistent = bool(records)
    for rec in records:
        for arm in rec["arms"].values():
            consistent = consistent and (
                arm["gamma"] == DIFFUSION_GAMMA
                and len(arm["pfs_uniform"]) == len(arm["pfs_perfusion"])
                and arm["median_pfs_delta_days"]
                == float(np.median(arm["pfs_perfusion"]) - np.median(arm["pfs_uniform"]))
                and arm["n_pairs_in_disagreement"]
                == sum(a != b for a, b in zip(arm["pfs_uniform"], arm["pfs_perfusion"]))
            )

    # Live re-derivation of the field itself from the real cached case.
    case = "01017"
    t1ce = os.path.join(ROOT, "data", "brats", f"BraTS2021_{case}_t1ce.nii.gz")
    seg = os.path.join(ROOT, "data", "brats", f"BraTS2021_{case}_seg.nii.gz")
    live_ok = False
    live_detail = "cached case missing"
    if os.path.exists(t1ce) and os.path.exists(seg):
        vessels = build_stream3_vessels(t1ce, seg)
        field = build_stream3_diffusion_field(vessels["vessels"])
        committed = [r for r in records if r["case_id"] == case]
        if committed:
            arm = committed[0]["arms"]["mtd"]
            live_ok = (
                abs(arm["field_min_mm2_per_day"] - float(field.d_mm2_per_day[field.d_mm2_per_day > 0].min())) < 1e-9
                and abs(arm["field_max_mm2_per_day"] - float(field.d_mm2_per_day.max())) < 1e-9
                and arm["voxels_active"] == int(np.count_nonzero(vessels["vessels"]))
            )
            live_detail = (
                f"case {case} live field D(x) = "
                f"{float(field.d_mm2_per_day[field.d_mm2_per_day > 0].min()):.4g}-"
                f"{float(field.d_mm2_per_day.max()):.4g} mm^2/day "
                f"({field.contrast_ratio:.1f}x), {int(np.count_nonzero(vessels['vessels']))} "
                f"perfused voxels; matches committed arms: {live_ok}"
            )

    mtd = [r["arms"]["mtd"]["median_pfs_delta_days"] for r in records]
    st = [r["arms"]["steering"]["median_pfs_delta_days"] for r in records]
    disagree = [
        r["arms"][pol]["n_pairs_in_disagreement"]
        for r in records
        for pol in ("mtd", "steering")
    ]
    return CheckResult(
        supported=consistent and live_ok,
        detail=(
            f"{len(records)} cases x 2 policies x 3 seeds, paired transport-only: "
            f"MTD paired PFS delta {min(mtd):+.0f}..{max(mtd):+.0f} d (medians "
            f"{np.median(mtd):+.1f} d), steering {min(st):+.0f}..{max(st):+.0f} d; "
            f"{sum(disagree)}/{len(disagree)} policy-arms move at all; "
            f"cohort self-consistent: {consistent}; {live_detail}"
        ),
    )


def check_stream3_cohort_is_measured() -> CheckResult:
    """The committed stream-3 cohort file must match a live re-run of case 01017."""
    from danon.clinical_twin import STREAM3_COHORT_PATH, stream3_report

    if not os.path.exists(STREAM3_COHORT_PATH):
        return CheckResult(
            supported=False,
            detail="stream-3 cohort file missing; rebuild via _stream3_cohort.py",
        )
    report = stream3_report()
    n_ok = report["n_cases"] == 6 and report["n_paired_runs_per_arm"] == 18
    mean_d = report["mtd_pfs_delta_mean_days"]
    # Re-measure one paired seed live to confirm the committed numbers derive
    # from the code, not from a stale file.
    from danon.clinical_twin import build_stream3_vessels
    from danon.imaging import from_nifti_mask
    from danon.evosteer import MTDPolicy, TumorEvolutionSimulator

    case = "01017"
    vessels = os.path.join(
        os.path.dirname(STREAM3_COHORT_PATH),
        "..", "data", "engine_ready", f"BraTS2021_{case}_vessels_t1ce.nii.gz",
    )
    live_ok = False
    live_delta = None
    if os.path.exists(vessels):
        pg_r = from_nifti_mask(
            os.path.join(ROOT, "data", "engine_ready", f"BraTS2021_{case}_mask.nii.gz"),
            proliferative_path=os.path.join(
                ROOT, "data", "engine_ready", f"BraTS2021_{case}_prolif.nii.gz"
            ),
            vessels_path=vessels,
        )
        pg_n = from_nifti_mask(
            os.path.join(ROOT, "data", "engine_ready", f"BraTS2021_{case}_mask.nii.gz"),
            proliferative_path=os.path.join(
                ROOT, "data", "engine_ready", f"BraTS2021_{case}_prolif.nii.gz"
            ),
        )
        a = TumorEvolutionSimulator(pg_r.geometry).run(
            MTDPolicy(), days=365, initial_resistant_fraction=0.005, seed=101
        )
        b = TumorEvolutionSimulator(pg_n.geometry).run(
            MTDPolicy(), days=365, initial_resistant_fraction=0.005, seed=101
        )
        live_delta = (a.pfs_day or 365) - (b.pfs_day or 365)
        committed = report["cases"][case]["mtd_pfs_median_real"]
        live_ok = (a.pfs_day or 365) == committed
    return CheckResult(
        supported=n_ok and live_ok,
        detail=(
            f"cohort 6 cases / 18 paired runs: {n_ok}; live re-run of 01017 "
            f"seed 101 reproduces the committed PFS: {live_ok} "
            f"(paired delta {live_delta} d; committed mean {mean_d:+.2f} d)"
        ),
    )


def check_stream2_capacity_mapping_is_measured() -> CheckResult:
    """Stream 2's cellularity reaches the engine's density ceiling, and is measured.

    The numbers in the claim sentence are DERIVED here rather than restated: the
    sweep's ratios must equal the slide record's own crop extremes over its own
    pooled density, the K x1.0 arm must reproduce its control exactly (the
    pairing assertion), and the MTD arms' per-seed burden ratio must equal the
    capacity ratio they were run at. If any of those move, the check fails and
    the sentence cannot sit there stale.
    """
    import json

    from danon.clinical_twin import (
        CAPACITY_COHORT_PATH,
        HISTOLOGY_PROBE_PATH,
        stream2_capacity_clones,
        stream2_carrying_capacity,
    )
    from danon.evosteer import DEFAULT_CLONES

    # (1) The mapping: identity at the reference, the ratio carried through, and
    # the absolute route refused with the reason the archive supplies.
    identity = stream2_carrying_capacity(636.0, 636.0)
    identity_ok = (
        identity.ratio == 1.0
        and identity.k == DEFAULT_CLONES.carrying_capacity
        and "section thickness" in identity.absolute_refusal
        and abs(stream2_capacity_clones(954.0, 636.0).carrying_capacity - 1.5) < 1e-12
        and stream2_capacity_clones(954.0, 636.0).growth_S == DEFAULT_CLONES.growth_S
    )

    # (2) The sweep's endpoints come from the committed slide record.
    try:
        probe = json.load(open(os.path.join(ROOT, HISTOLOGY_PROBE_PATH)))
        density = probe["slide_measured"]["density"]
        reference = float(density["nuclei_per_mm2"])
        spread = density["per_crop_spread"]
        # Deduped, because a perfectly uniform slide measures the same density in
        # its smallest and largest crop: the sweep then legitimately collapses to
        # one arm, and demanding exactly three would fail a valid cohort.
        expected = sorted(
            {
                (spread["min"]) / reference,
                1.0,
                (spread["max"]) / reference,
            }
        )
    except (OSError, ValueError, KeyError) as exc:
        return CheckResult(
            supported=False, detail=f"histology probe unreadable: {exc}"
        )

    try:
        rows = json.load(open(os.path.join(ROOT, CAPACITY_COHORT_PATH)))
    except (OSError, ValueError) as exc:
        return CheckResult(
            supported=False,
            detail=f"stream-2 capacity cohort unreadable ({exc}); rebuild via "
            "_stream2_capacity_cohort.py",
        )
    if not isinstance(rows, list) or not rows:
        return CheckResult(supported=False, detail="capacity cohort holds no cases")

    problems: List[str] = []
    mtd_departures: List[float] = []
    steering_departures: List[float] = []
    pfs_moves = 0
    pfs_total = 0
    for row in rows:
        if float(row.get("reference_nuclei_per_mm2", 0.0)) != reference:
            problems.append(
                f"{row.get('case_id')}: reference density is not the slide record's"
            )
        for policy, arm in (row.get("arms") or {}).items():
            ratios = sorted(float(k) for k in arm.get("k_by_ratio", {}))
            if not (1 <= len(ratios) <= 3) or len(ratios) != len(expected) or any(
                abs(a - b) > 1e-6 for a, b in zip(ratios, expected)
            ):
                problems.append(
                    f"{row.get('case_id')}/{policy}: K ratios are not the slide's "
                    f"measured crop extremes"
                )
            unit = arm["burden_capacity"].get("1.000000")
            if unit != arm["burden_uniform"] or arm["pfs_capacity"].get(
                "1.000000"
            ) != arm["pfs_uniform"]:
                problems.append(
                    f"{row.get('case_id')}/{policy}: the K x1.0 arm does not "
                    f"reproduce its control -- the pairing is broken"
                )
            for key, k in arm.get("k_by_ratio", {}).items():
                pairs = list(
                    zip(arm["burden_uniform"], arm["burden_capacity"].get(key, []))
                )
                if len(pairs) != len(arm["burden_uniform"]):
                    problems.append(
                        f"{row.get('case_id')}/{policy}: arm and control burden "
                        f"lists differ in length"
                    )
                    continue
                for control, capacity in pairs:
                    if control <= 0:
                        problems.append(
                            f"{row.get('case_id')}/{policy}: zero control burden"
                        )
                        continue
                    departure = abs(capacity / control - k)
                    (mtd_departures if policy == "mtd" else steering_departures).append(
                        departure
                    )
            for delta in (arm.get("median_pfs_delta_days") or {}).values():
                pfs_total += 1
                if delta != 0:
                    pfs_moves += 1
                    if abs(delta) > 1.0:
                        problems.append(
                            f"{row.get('case_id')}/{policy}: PFS moved {delta} d, "
                            f"which the sentence does not claim"
                        )

    if not identity_ok:
        problems.append("the density->K mapping is not the identity at its reference")
    if not mtd_departures or max(mtd_departures) > 0.01:
        problems.append(
            "under MTD the burden ratio no longer equals the capacity ratio, so "
            "the covariance reading in the sentence is wrong"
        )
    if not steering_departures or max(steering_departures) <= max(mtd_departures or [0.0]):
        problems.append(
            "the steering arm no longer departs further from the capacity ratio "
            "than the MTD arm does, so the floor-gating reading is unsupported"
        )
    if pfs_moves > pfs_total / 2:
        problems.append(
            "most policy-arms now move PFS; the sentence's 'does not move PFS' "
            "would be false"
        )

    supported = not problems
    return CheckResult(
        supported=supported,
        detail=(
            f"mapping identity at the reference: {identity_ok}; sweep ratios match "
            f"the slide record's crops ({', '.join(f'{r:.4f}' for r in expected)}); "
            f"{len(rows)} real cases x 2 seeds paired: MTD |burden ratio - K| max "
            f"{max(mtd_departures or [0.0]):.2e} (pure scale), steering max "
            f"{max(steering_departures or [0.0]):.3f} (a floor gates the switch); "
            f"PFS moved in {pfs_moves} of {pfs_total} policy-arms"
            + ("; " + "; ".join(problems) if problems else "")
        ),
    )


def check_stream2_histology_measured_and_ki67_refused() -> CheckResult:
    """Stream 2: what the biopsy archive yields, and the ratio it cannot give.

    Four things are verified live. (a) The probe record is readable and its
    slide is still on disk -- a nuclei density with no slide behind it is a
    number with nothing under it. (b) The stain really is non-DAB: the
    committed brown fraction is ~0 and the called stain is not 'dab-positive',
    so the Ki-67 numerator is absent rather than merely uncomputed. (c) The
    refusal is exercised, not described -- `ki67_fraction` is called on the
    record's own stain and must raise. (d) The census agrees with the record
    and still refuses stream 2 as an INPUT: the downloaded slide is a sample,
    and a haematoxylin-only sample must not flip the stream to 'available'.
    """
    import json

    from danon.histology import HistologyError, ki67_fraction

    path = os.path.join(ROOT, "outputs_real", "histology_probe.json")
    try:
        record = json.load(open(path))
    except (OSError, ValueError) as exc:
        return CheckResult(supported=False, detail=f"histology probe unreadable: {exc}")

    census = record.get("census") or {}
    slide = record.get("slide_measured") or {}
    density = slide.get("density") or {}
    stain = slide.get("stain") or {}
    problems: List[str] = []

    if not density.get("nuclei_per_mm2") or not density.get("nuclei"):
        problems.append("the record carries no measured nuclei density")
    if int(density.get("crops_used", 0)) < 2:
        problems.append("the density rests on a single crop")
    if not (density.get("slide") or {}).get("mpp_um"):
        problems.append("the record reports a density without an MPP")
    # The slide lives under data/tcia/, which is gitignored: the committed record
    # is the evidence and the slide is a re-downloadable cache. So a cache miss is
    # reported, not failed -- but a present slide must match the recorded size, so
    # a truncated or swapped file cannot sit behind a density it does not contain.
    if not slide.get("file"):
        problems.append("the record names no slide")
    else:
        sample = os.path.join(ROOT, "data", "tcia", "wsiprode", str(slide["file"]))
        if os.path.exists(sample):
            if os.path.getsize(sample) != int(slide.get("bytes") or 0):
                problems.append(
                    f"the cached slide's size ({os.path.getsize(sample)} B) does not "
                    f"match the recorded {slide.get('bytes')} B"
                )
            slide_note = "cached slide size matches the record"
        else:
            slide_note = "slide not cached (re-download via _histology_probe.py)"

    if stain.get("stain") == "dab-positive":
        problems.append(
            "the slide is called dab-positive, so the Ki-67 numerator may exist "
            "and the refusal's premise would be wrong"
        )
    if float(stain.get("brown_fraction", 1.0)) > 0.15:
        problems.append("the committed brown fraction is not negligible")

    # Exercise the refusal on the record's own stain rather than trusting prose.
    refusal_ok = False
    refusal_excerpt = ""
    try:
        ki67_fraction(stain)
    except HistologyError as exc:
        refusal_excerpt = str(exc)
        refusal_ok = "Ki-67" in refusal_excerpt
    if not refusal_ok:
        problems.append("ki67_fraction did not refuse the archive's own stain")
    if "Ki-67" not in str(slide.get("ki67_refusal") or ""):
        problems.append("the recorded refusal does not name the missing numerator")

    # The retrieval claim must be a live count, not a remembered one.
    if int(census.get("n_slide_images") or 0) <= 0:
        problems.append("the probe recorded no retrievable slide images")
    if int(census.get("n_named_ki67") or 0) != 0:
        problems.append(
            "the index now names Ki-67 slides; the archive should be re-probed "
            "rather than keeping this refusal"
        )

    live = _histology_stream2_census()
    if live is None:
        problems.append("the census carries no whole-slide evidence block")
    else:
        if live.get("status") != "REFUSED" or int(live.get("n_files", -1)) != 0:
            problems.append(
                "the census no longer refuses stream 2 as an input; a sample "
                "slide must not count as a Ki-67 input"
            )
        recorded = (live.get("wsi") or {})
        if recorded.get("n_named_ki67") != census.get("n_named_ki67"):
            problems.append("the census and the probe disagree on the Ki-67 count")
        if recorded.get("nuclei_per_mm2") != density.get("nuclei_per_mm2"):
            problems.append("the census and the probe disagree on the density")
        if "Ki-67" not in str(live.get("refused_for") or ""):
            problems.append("the census does not name what stream 2 refuses")

    supported = not problems
    return CheckResult(
        supported=supported,
        detail=(
            f"{census.get('n_slide_images')} TCGA-GBM slide images retrievable, "
            f"{census.get('n_named_ki67')} named Ki-67; the downloaded slide is "
            f"{stain.get('stain')} (brown {stain.get('brown_fraction')}) at MPP "
            f"{(slide.get('slide') or {}).get('mpp_um')}, giving "
            f"{density.get('nuclei_per_mm2')} nuclei/mm^2 over "
            f"{density.get('crops_used')} crops; {slide_note}; Ki-67 refusal "
            f"exercised and stream 2 still refused as an input"
            + ("; " + "; ".join(problems) if problems else "")
        ),
    )


def _histology_stream2_census() -> Optional[dict]:
    try:
        from danon.clinical_twin import clinical_stream_census

        return clinical_stream_census()["stream2_histology"]
    except Exception:  # pragma: no cover - census failure is reported by its own check
        return None


def check_stream_census_refuses_honestly() -> CheckResult:
    """Streams 1/2/4 must refuse with live counts, and stream 4 must state how
    much is retrievable rather than implying the data does not exist."""
    from danon.clinical_twin import clinical_stream_census

    census = clinical_stream_census()
    s1 = census["stream1_5ala"]
    s2 = census["stream2_histology"]
    s3 = census["stream3_perfusion"]
    s4 = census["stream4_sequential"]
    archive = s4.get("archive") or {}
    ok = (
        s1["status"] == "REFUSED" and s1["n_files"] == 0
        and s2["status"] == "REFUSED" and s2["n_files"] == 0
        and s4["status"] == "REFUSED" and s4["n_pairs_cached"] == 0
        and s3["status"] == "MEASURABLE"
        # the retrievable side is read from the recorded probe, not asserted
        # from memory: a census that says "none" where 41 exist is the defect
        # this check exists to prevent.
        and s4["n_pairs_retrievable"] == 41
        and archive.get("n_patients_dsc_perfusion") == 456
        and archive.get("n_patients_processed_perfusion") == 628
    )
    return CheckResult(
        supported=ok,
        detail=(
            "census: stream1 REFUSED({}), stream2 REFUSED({}), stream3 "
            "MEASURABLE({} t1ce files; archive carries {} DSC perfusion "
            "patients), stream4 REFUSED({} cached / {} retrievable pairs)".format(
                s1["n_files"], s2["n_files"], s3["n_t1ce_files"],
                s3.get("n_archive_dsc_patients"), s4["n_pairs_cached"],
                s4["n_pairs_retrievable"],
            )
        ),
    )


def check_availability_census_is_measured() -> CheckResult:
    """The longitudinal cohort counts must come from the committed manifest.

    Also checks the one thing a cohort census is most likely to overstate: a
    count of what a manifest RECORDS is not a count of what has been measured.
    The images are not downloaded, so the census must say so.
    """
    from danon.availability import CFB_GBM_MANIFEST, mri_availability_census

    if not os.path.exists(CFB_GBM_MANIFEST):
        return CheckResult(
            supported=False,
            detail="committed CFB-GBM availability manifest missing",
        )
    c = mri_availability_census()
    ok = (
        c.n_patients == 264
        and c.n_patients_multi_timepoint == 170
        and c.n_usable_pairs == 393
        and c.n_patients_with_usable_pair == 168
        and c.n_cross_machine_pairs == 37
        and "10.7937/v9pn-2f72" in c.source
        and "not downloaded" in " ".join(c.notes)
    )
    return CheckResult(
        supported=ok,
        detail=(
            f"CFB-GBM manifest: {c.n_patients} patients, "
            f"{c.n_patients_multi_timepoint} with >1 timepoint, "
            f"{c.n_patients_with_usable_pair} with a usable "
            f"{c.sequence} pair ({c.n_usable_pairs} pairs, "
            f"{c.n_cross_machine_pairs} spanning two scanners); images NOT "
            f"downloaded, so this is availability, not a measurement"
        ),
    )


def check_stream4_resolution_is_a_design_calculation() -> CheckResult:
    """The stream-4 floor must be reproducible, and labelled as a design calc.

    Two failure modes are ruled out here. The first is a floor that was never
    computed: the full-cohort C is re-measured live and must match the committed
    artifact, so a stale file cannot masquerade as a calculation. The second is
    the one this project keeps hitting -- a design parameter read as a result.
    The artifact must say what it is, and must not claim to have measured any
    stream.
    """
    from danon.availability import mri_availability_census
    from danon.clinical_twin import stream4_resolution_report

    try:
        payload = stream4_resolution_report()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return CheckResult(
            supported=False,
            detail=f"stream-4 design calculation unreadable: {type(exc).__name__}: {exc}",
        )
    res = payload["resolution"]

    # Re-measure the quantity the artifact's reference point rests on.
    from danon.cohort_features import _cv
    from danon.predictive import load_cohort_rows

    rows, _ = load_cohort_rows()
    live = _cv(rows, None, n_folds=5, seed=0)
    live_matches = (
        abs(live.mean_c - payload["source_base_c"]) < 1e-9
        and abs(live.sd_c - payload["source_base_fold_sd"]) < 1e-9
    )

    n_manifest = mri_availability_census().n_patients_with_usable_pair
    floor_ok = (
        res["n_patients"] == n_manifest
        and res["n_draws"] == 24
        and 0.0 < res["floor_optimistic"] < res["floor_conservative"] < 0.2
    )
    labelled = (
        "DESIGN calculation" in payload["what_this_is"]
        and "not a measurement of any datastream" in payload["what_this_is"]
        and len(payload["what_this_cannot_show"]) >= 3
    )

    return CheckResult(
        supported=bool(live_matches and floor_ok and labelled),
        detail=(
            f"n={res['n_patients']} (manifest's usable pairs), {res['n_draws']} draws: "
            f"floor {res['floor_optimistic']:.4f} (1 SE) / "
            f"{res['floor_conservative']:.4f} (fold yardstick); reference cohort "
            f"re-measured live at C={live.mean_c:.4f} +/- {live.sd_c:.4f} "
            f"(matches artifact: {live_matches}); labelled as a design "
            f"calculation: {labelled}"
        ),
    )


def check_readme_claim_count_matches_the_registry() -> CheckResult:
    """The README's claim count must be the registry's, not a remembered one.

    This exists because it drifted: the README advertised 31/31 verified while
    the registry held 33, so a published number had been sitting stale for two
    commits. A count that nothing re-derives is the same failure mode the rest
    of this registry exists to prevent, applied to the registry's own headline.
    """
    import re

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md"
    )
    try:
        text = open(path, encoding="utf-8").read()
    except OSError as exc:
        return CheckResult(supported=False, detail=f"README unreadable: {exc}")
    # EVERY place the README states this count, not just the bold matrix row.
    # The first version of this check matched only `**n/n verified**`, and the
    # same number written as a plain comment (`# 38/38 bound claims`, above the
    # audit command) escaped it entirely and sat stale while the registry held
    # 40 -- a duplicate of the counted fact that the guard was blind to purely
    # because of its formatting.
    pairs = re.findall(r"(\d+)/(\d+)\s*(?:verified|bound claims)", text)
    if not pairs:
        return CheckResult(
            supported=False,
            detail="README states no 'n/n verified' claim count to check",
        )
    live = len(CLAIMS)
    stale = [f"{a}/{b}" for a, b in pairs if int(a) != live or int(b) != live]
    return CheckResult(
        supported=not stale,
        detail=(
            f"README states the claim count {len(pairs)}x "
            f"({', '.join(f'{a}/{b}' for a, b in pairs)}); the registry holds "
            f"{live} claims"
            + ("" if not stale else f" -- stale: {', '.join(stale)}")
        ),
    )


def check_stream4_agreement_is_a_validation_target() -> CheckResult:
    """Stream 4 must be a comparison, and must refuse without a reference.

    Three things are ruled out. A comparator that reports agreement it did not
    earn: the identical-field case must be exactly 1.0 and the misaligned case
    must be measurably worse. A refusal that hard-codes its own count: the
    refusal's numbers are read back against the census, so the two cannot
    drift apart. And any drift of language into "input": the returned record
    must carry its role, because calling a validation target a covariate is how
    this stream came to be described as raising a C-index in the first place.
    """
    import numpy as np

    from danon.clinical_twin import clinical_stream_census, stream4_velocity_agreement

    empty = stream4_velocity_agreement()
    if empty.status != "REFUSED" or empty.pearson_r is not None:
        return CheckResult(
            supported=False,
            detail=(
                f"stream-4 agreement returned status {empty.status!r} with "
                f"pearson {empty.pearson_r!r}; an unmeasured comparison must "
                "refuse and carry no statistic"
            ),
        )
    stream4 = clinical_stream_census().get("stream4_sequential", {})
    cached = stream4.get("n_pairs_cached")
    retrievable = stream4.get("n_pairs_retrievable")
    counts_live = (
        cached is not None
        and retrievable is not None
        and f"{cached} cached" in empty.refusal
        and f"{retrievable} retrievable" in empty.refusal
    )

    # Two active voxels, not one: a correlation over a single point has no
    # variance, and the function correctly reports no r rather than a 1.0 it
    # did not earn. Asserting r==1.0 on a one-voxel field would be testing the
    # test.
    base = np.zeros((5, 5, 5))
    grown = base.copy()
    grown[2, 2, 2] = 1.0
    grown[1, 1, 1] = 0.5
    same = stream4_velocity_agreement(measured=(base, grown), simulated=(base, grown))
    shifted = base.copy()
    shifted[3, 3, 3] = 1.0
    shifted[1, 1, 1] = -0.5
    other = stream4_velocity_agreement(measured=(base, grown), simulated=(base, shifted))
    # numpy's corrcoef returns 0.9999999999999999 for a field against itself,
    # so this compares within floating-point slack rather than by equality.
    identical_r = same.pearson_r is not None and abs(same.pearson_r - 1.0) < 1e-12
    discriminates = (
        same.status == "MEASURED"
        and identical_r
        and same.sign_agreement == 1.0
        and other.pearson_r is not None
        and other.pearson_r < 1.0
    )
    role_labelled = same.to_json_dict().get("role", "").startswith("validation target")
    supported = bool(counts_live and discriminates and role_labelled)
    return CheckResult(
        supported=supported,
        detail=(
            f"refusal cites the census live ({cached} cached / {retrievable} "
            f"retrievable): {counts_live}; identical fields r={same.pearson_r}, "
            f"misaligned r={other.pearson_r} -> discriminates: {discriminates}; "
            f"role labelled a validation target: {role_labelled}"
        ),
    )


CHECKS: Dict[str, Callable[[], CheckResult]] = {
    "evidence_status_vocabulary": check_evidence_status_vocabulary,
    "readme_claim_count_current": check_readme_claim_count_matches_the_registry,
    "availability_census_measured": check_availability_census_is_measured,
    "stream4_resolution_is_design_only": check_stream4_resolution_is_a_design_calculation,
    "censoring_is_preserved": check_censoring_is_preserved,
    "covariate_gains_are_measured": check_covariate_gains_are_measured,
    "diffusion_units_and_operator": check_diffusion_units_and_operator,
    "no_inflated_estimand": check_no_inflated_estimand_is_claimed,
    "gdsc_ic50s_not_all_measured": check_gdsc_ic50s_are_not_all_measurements,
    "untestable_tiers_refused": check_untestable_tiers_are_refused,
    "holdout_emulator_measured": check_holdout_emulator_is_measured,
    "mgmt_status_measured_not_faked": check_mgmt_status_is_measured_not_faked,
    "covariate_upgrades_measured": check_covariate_upgrades_are_measured,
    "mgmt_resistance_sweep_measured": check_mgmt_resistance_sweep_is_measured,
    "restriction_membership_vs_signal": check_treatment_restriction_separates_membership_from_signal,
    "dice_degenerate_case": check_dice_degenerate_case_is_not_perfect,
    "spatial_agreement_not_claimed": check_spatial_agreement_is_not_claimed,
    "potency_is_measured": check_potency_is_measured,
    "ec50_unit_undeclared": check_ec50_unit_is_undeclared,
    "neural_components_declared": check_neural_components_are_declared,
    "seeded_reproducibility": check_seeded_reproducibility,
    "structure_is_deposited": check_structure_is_deposited,
    "wild_type_from_deposit": check_wild_type_from_deposit,
    "mechanistic_core": check_mechanistic_core,
    "parameters_declared": check_parameters_are_declared_assumptions,
    "test_suite_counted": check_test_suite_is_counted_live,
    "engine_c_index_absent": check_engine_c_index_is_not_claimed,
    "voxel_scale": check_voxel_scale,
    "runtime_365": check_runtime_365_days,
    "bulk_not_single_cell": check_parameters_are_bulk_not_single_cell,
    "resistance_is_emergent": check_resistance_is_emergent,
    "stream_identities": check_stream_identities_hold,
    "stream3_cohort_measured": check_stream3_cohort_is_measured,
    "stream3_diffusion_measured": check_stream3_diffusion_is_measured,
    "stream3_dsc_perfusion_measured": check_stream3_dsc_perfusion_is_measured,
    "stream_census_refuses": check_stream_census_refuses_honestly,
    "stream2_histology_measured": check_stream2_histology_measured_and_ki67_refused,
    "stream2_capacity_measured": check_stream2_capacity_mapping_is_measured,
    "stream4_agreement_is_a_validation_target": check_stream4_agreement_is_a_validation_target,
}


# ---------------------------------------------------------------------------
# The claims
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Claim:
    """One sentence a poster may carry, and what backs it."""

    id: str
    sentence: str
    tier: EvidenceStatus
    substantiated_by: str
    check: str
    limit: str
    # Which study of this repository the claim belongs to. The app routes only
    # the tumour twin, so the vector-design claims are marked and hidden from
    # its default view rather than silently presented inside an oncology tool.
    study: str = "tumour_twin"


CLAIMS: Tuple[Claim, ...] = (
    Claim(
        "mechanistic_scheduler",
        "The dosing scheduler is mechanistic: drugs move by a discretised "
        "Fickian diffusion PDE and clones compete by Lotka-Volterra dynamics "
        "with saturating (EC50) kill kinetics.",
        EvidenceStatus.ESTABLISHED,
        "danon/evosteer.py::TumorEvolutionSimulator",
        "mechanistic_core",
        "The mechanism is mechanistic; the rate constants are declared "
        "assumptions, not measurements from this project.",
    ),
    Claim(
        "resistance_is_not_ghost",
        "Resistance in this engine is produced by the model, not carried by "
        "the initial state: with mutation off and no seeded clone the resistant "
        "burden is exactly zero, and with the extinction floor the sensitive "
        "population is extinguished exactly rather than lingering as a "
        "sub-single-cell density. The engine's qualitative verdict (does "
        "continuous MTD control the tumour?) tracks whether a resistant "
        "subclone exists at diagnosis, which is an ASSUMED input here, not a "
        "measured one.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_resistance_is_emergent",
        "resistance_is_emergent",
        "This verifies the mechanism is doing the work; it says nothing about "
        "whether the assumed constants are right, and the verdict's dependence "
        "on the pre-existing fraction is a limitation to state, not a result "
        "to publish.",
    ),
    Claim(
        "not_a_classifier",
        "The scheduling core is not a trained classifier: it solves transport "
        "and population dynamics rather than learning a mapping from images to "
        "labels. A pretrained protein language model contributes capsid-fitness "
        "scoring in the separate design arm, and is labelled as such wherever it "
        "is used.",
        EvidenceStatus.ESTABLISHED,
        "danon/evosteer.py (scheduler) vs danon/vector_design/ml_scorer.py + danon/vector_design/esm_inference.py (design arm)",
        "neural_components_declared",
        "A claim of 'no machine learning anywhere' is false: torch modules and "
        "an ESM scorer ship in the design arm.",
    ),
    Claim(
        "seeded_reproducibility",
        "Runs are stochastic (mutation timing and growth noise) and fully "
        "seeded: identical seeds reproduce a trajectory exactly, and results are "
        "reported as distributions over replicates.",
        EvidenceStatus.ESTABLISHED,
        "danon/evosteer.py::TumorEvolutionSimulator.run(seed=...)",
        "seeded_reproducibility",
        "Reproducible is not the same as deterministic, and neither is the same "
        "as correct.",
    ),
    Claim(
        "deposited_structure",
        "Capsid geometry is the deposited AAV9 structure (PDB 3UX1), assembled "
        "with the entry's own REMARK 350 symmetry operators and pinned by "
        "SHA-256 verified on every load.",
        EvidenceStatus.ESTABLISHED,
        "danon/vector_design/structure.py::load_aav9_capsid",
        "structure_is_deposited",
        "C-alpha trace only: it is not an atomic surface, and it says nothing "
        "about conformational ensembles.",
        study="vector_design",
    ),
    Claim(
        "real_wild_type",
        "The engineered capsid's wild type is VP1 residues 263-736 of that same "
        "deposited structure, re-derived from the PDB on every test run.",
        EvidenceStatus.ESTABLISHED,
        "danon/vector_design/aav_generator.py::WILD_TYPE_AAV9_CAPSID",
        "wild_type_from_deposit",
        "One deposited model, not a consensus sequence across AAV9 isolates.",
        study="vector_design",
    ),
    Claim(
        "provenance_registry",
        "Every output carries a provenance tier from a fixed vocabulary: "
        f"{', '.join(m.name for m in EvidenceStatus)} -- including permanent "
        "PROVEN_FAILED entries for claims this project made and withdrew.",
        EvidenceStatus.ESTABLISHED,
        "danon/cure.py::EvidenceStatus",
        "evidence_status_vocabulary",
        "A tier is an assertion about evidence; it is only as good as the "
        "citation behind the entry using it.",
    ),
    Claim(
        "declared_assumptions",
        "Rate constants are labelled in the source as CALIBRATED or ASSUMED, "
        "with the benchmark measurement that would replace each one identified.",
        EvidenceStatus.ESTABLISHED,
        "danon/evosteer.py (CALIBRATION section) + danon/validation.py tier 3",
        "parameters_declared",
        "Being labelled is not being measured: every rate is still an assumption "
        "until a bench run replaces it.",
    ),
    Claim(
        "engineering_rigor",
        "The repository carries an automated test suite covering the physics, "
        "the statistics and the refusal paths, with the count computed live "
        "and the README's own suite row checked against it rather than "
        "maintained by hand.",
        EvidenceStatus.ESTABLISHED,
        "tests/",
        "test_suite_counted",
        "Tests verify the implementation. They are not evidence about biology.",
    ),
    Claim(
        "heldout_discrimination",
        "A Cox model of clinical covariates reaches C = 0.64 on held-out "
        "TCGA-GBM patients, validating the held-out statistical machinery on "
        "real data.",
        EvidenceStatus.EMERGING_HUMAN,
        "danon/predictive.py::kfold_holdout_c_index",
        "engine_c_index_absent",
        "That C-index is about age and grade, not about this engine. The engine "
        "itself has no estimable patient-level C-index.",
    ),
    Claim(
        "multiscale_coarse_graining",
        "The engine is a coarse-grained continuum--agent hybrid, not a discrete "
        "cell-by-cell simulation: clinical voxels hold clone densities and drug "
        "concentration, so transport is solved on a mesh rather than per cell.",
        EvidenceStatus.ESTABLISHED,
        "danon/evosteer.py::TumorEvolutionSimulator (density fields) + danon/realdata.py::DecimationSpec",
        "voxel_scale",
        "The engine grid is ~4 mm, not the 100 um voxels this pitch originally "
        "claimed, and the decimation from a 1 mm scan is a documented lossy step.",
    ),
    Claim(
        "runtime_is_measured",
        "A 365-day forecast is a sub-second-to-second computation on the engine's "
        "phantom (and ~20 s for a full paired case on real geometry), re-measured "
        "whenever this audit runs rather than quoted.",
        EvidenceStatus.ESTABLISHED,
        "danon/evosteer.py::TumorEvolutionSimulator.run(days=365)",
        "runtime_365",
        "Runtime is machine-dependent; the audit prints the machine's own number.",
    ),
    Claim(
        "inputs_are_bulk",
        "Molecular inputs are real bulk RNA-seq (GDC open STAR counts), and the "
        "pipeline states that single-cell clone fractions cannot be recovered from "
        "them.",
        EvidenceStatus.ESTABLISHED,
        "danon/genomics_intake.py (bulk RNA-seq -> r0, calibrated mapping)",
        "bulk_not_single_cell",
        "A claim of scRNA-seq-derived parameters would describe data this pipeline "
        "never reads, and `PROVEN_FAILED` in the same module says so.",
    ),
    Claim(
        "no_spatial_agreement_result",
        "No spatial-agreement (Dice) result is claimed: no reachable glioma "
        "imaging collection carries both an expert tumour mask and a second "
        "timepoint, so a predicted field cannot be scored against an observed "
        "recurrence scan. The refusal is computed from recorded probe counts and "
        "flips automatically if such a collection appears.",
        EvidenceStatus.PROVEN_FAILED,
        "danon/multimodal.py::spatial_dice_verdict over data/tcia_probe/multimodal_evidence.json",
        "spatial_agreement_not_claimed",
        "Absence of a test is not a negative result: nothing here is evidence "
        "that the engine's spatial predictions are wrong, only that the public "
        "data reachable from this environment cannot test them.",
    ),
    Claim(
        "potency_from_measured_records",
        "Drug potency is taken from real measured activity records (ChEMBL) "
        "rather than authored: 31 exact glioma-line IC50s for temozolomide, "
        "median 135 uM and most potent 6.4 uM, spanning a 496x between-model "
        "range, with 19 right-censored records counted separately and their units "
        "preserved.",
        EvidenceStatus.ESTABLISHED,
        "danon/pharmacology.py::load_glioma_potency (cache data/pharmacology/)",
        "potency_is_measured",
        "2D monolayer IC50 at a single timepoint is not 3D tumour kill behind a "
        "blood-brain barrier, and the records come from heterogeneous assays "
        "whose readout and exposure time are largely unrecorded.",
    ),
    Claim(
        "engine_potency_scale_not_anchored",
        "The engine's EC50 carries no declared concentration unit, so it is not "
        "compared to measured potency: any ratio between them would be "
        "unitless. The measured values are reported instead.",
        EvidenceStatus.UNKNOWN,
        "danon/pharmacology.py::engine_ec50_unit + compare_engine_ec50 (refusal)",
        "ec50_unit_undeclared",
        "This is a defect, not a feature. Until the engine's concentration scale "
        "is declared, its potency constant cannot be anchored to anything "
        "measurable, and no calibration claim should be made about it.",
    ),
    Claim(
        "censored_endpoint",
        "Held-out discrimination is reported on an endpoint that keeps its "
        "censored patients: the progression cohort is 594 patients with 372 "
        "events and 222 censored at last contact, and OS is 595 patients with "
        "491 events and 104 censored. An earlier version of the loader searched "
        "follow-up records for a field that only exists on the diagnosis node, "
        "so no patient was ever censored and the progression C-index read 0.590 "
        "instead of 0.534.",
        EvidenceStatus.ESTABLISHED,
        "danon/predictive.py::load_cohort_rows",
        "censoring_is_preserved",
        "Censoring makes the number smaller and more honest. It is still not "
        "clinical validation.",
    ),
    Claim(
        "covariates_measured_not_estimated",
        "Every covariate proposed for the real cohort is reported with a "
        "measured delta-C on the same patients, and Karnofsky status is drawn "
        "only from pre-treatment timepoints. A post-treatment KPS is measurably "
        "leaky: as a lone predictor it scores C = 0.684 against 0.597 for the "
        "pre-treatment value, because it is partly determined by the outcome.",
        EvidenceStatus.ESTABLISHED,
        "danon/cohort_features.py::covariate_effect, kps_leakage_contrast",
        "covariate_gains_are_measured",
        "A measured gain of +0.026 inside a fold sd of 0.048 is reported as 'no "
        "measurable gain'. MGMT is reachable but caps the cohort at 142 patients, "
        "and sub-region segmentation is refused outright.",
    ),
    Claim(
        "transport_units_explicit",
        "Diffusion is measured in voxel^2/day, and the bridge to physical units is "
        "explicit. The engine's D = 0.35 is 0.35 mm^2/day on the 1 mm phantom but "
        "4.71 mm^2/day on the real 3.67 mm decimated grid -- 13.44x, with "
        "`voxel_mm` documented as bookkeeping only, so nothing raised. The "
        "spatially heterogeneous transport term is div(D(x) grad C) with harmonic "
        "face conductances, which reduces exactly to the engine's scalar "
        "D * laplacian(C) on constant D.",
        EvidenceStatus.ESTABLISHED,
        "danon/anisotropy.py::d_to_engine, divergence_d_grad",
        "diffusion_units_and_operator",
        "The heterogeneous operator is slower or redirected relative to a scalar "
        "field of the same mean, never faster: no delivery improvement can be "
        "claimed from this change alone. Tissue D values are literature "
        "constants and no DTI series is present.",
    ),
    Claim(
        "no_inflated_estimand",
        "No proposed high-scoring target is reported as a measurement. The "
        "spatial Dice target is refused for want of paired imaging with masks; "
        "`best_overall_response` is empty for all 617 cases, so no responder "
        "label exists and the nearest flag is 4.3% positive and runs backwards "
        "against survival; and the in-vitro R^2 is incommensurable because the "
        "engine's EC50 has no concentration unit.",
        EvidenceStatus.PROVEN_FAILED,
        "danon/eval_targets.py::target_verdicts",
        "no_inflated_estimand",
        "Refusing three estimands does not raise the fourth: overall-survival "
        "C stays 0.656 and progression C stays 0.534.",
    ),
    Claim(
        "gdsc_ic50s_mostly_extrapolated",
        "GDSC2's glioma IC50s are largely extrapolations rather than "
        "measurements: 76.0% lie above the highest concentration tested, and for "
        "temozolomide every one of the 34 glioma values does. The ChEMBL records "
        "already in this repository agree independently, with 19 of 50 glioma "
        "temozolomide records right-censored.",
        EvidenceStatus.ESTABLISHED,
        "danon/eval_targets.py::load_gdsc_glioma",
        "gdsc_ic50s_not_all_measured",
        "Two independent sources agreeing that this quantity is mostly a lower "
        "bound is a reason to refuse the fit, not a caveat on it.",
    ),
    Claim(
        "no_rl_agent",
        "No reinforcement-learning agent exists. Every dosing policy is a "
        "closed-form rule -- MTDPolicy returns the constant (1.0, 0.0) for every "
        "state -- so no policy is learned, trained, or converged, and a claim of "
        "'95.4% normalised reward retention over training episodes' describes "
        "software that is not here.",
        EvidenceStatus.PROVEN_FAILED,
        "danon/evosteer.py::MTDPolicy, AdaptiveDosePolicy, CollateralSwitchPolicy",
        "untestable_tiers_refused",
        "The deeper problem is the metric: a self-defined reward and its "
        "normalization are both chosen by the author, so the number cannot be "
        "contradicted by any measurement. The repo's comparable quantity is "
        "anchored externally instead (paired MTD control, RANO threshold, "
        "censored lower bound).",
    ),
    Claim(
        "egfrviii_not_in_open_data",
        "EGFRvIII is an in-frame deletion of exons 2-7 -- a structural variant -- "
        "so it cannot appear in the open Masked Somatic Mutation files. All 1,847 "
        "structural-variation files for TCGA-GBM are controlled access, so "
        "EGFRvIII status is not obtainable from open data and cannot be a feature "
        "of a classifier built here.",
        EvidenceStatus.PROVEN_FAILED,
        "danon/eval_targets.py::probe_egfrviii_reachability",
        "untestable_tiers_refused",
        "Inferring the deletion from RNA-seq junction reads is a legitimate "
        "alternative and would be an inference, not a mutation call.",
    ),
    Claim(
        "no_engine_c_index",
        "This engine is not reported to predict individual patients: no dataset "
        "here pairs baseline tumour state, administered schedule and outcome, and "
        "one curve per simulation means a C-index would be 0.5 by construction.",
        EvidenceStatus.UNKNOWN,
        "danon/predictive.py::engine_discrimination_verdict",
        "engine_c_index_absent",
        "Stating the absence is the honest position; it is not a finding.",
    ),
    Claim(
        "holdout_emulator_090",
        "A physics-feature ridge surrogate predicts the engine's final tumour "
        "burden at one-shot holdout R^2 = 0.9211 (95% bootstrap CI 0.8615-"
        "0.9536) on 48 configurations never used for feature selection or "
        "fitting: the engine's outcome surface is predictable from its inputs "
        "to better than 0.90. This is a property of the SIMULATOR, not an "
        "accuracy on patients, and is bound to the one-shot re-measurement.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_holdout_emulator_is_measured",
        "holdout_emulator_measured",
        "Synthetic ground truth (the engine itself). Never present beside "
        "patient numbers as if it raised them.",
    ),
    Claim(
        "mgmt_status_not_predictable_from_open_covariates",
        "Predicting the published MGMT promoter call from age, sex and "
        "diagnosis yields out-of-fold AUC 0.5387 (95% CI 0.4850-0.5964) on "
        "396 real TCGA-GBM patients -- indistinguishable from chance. The "
        "0.90 band on this target requires per-patient molecular features the "
        "open cache does not carry; the measurement is reported at its real "
        "value instead of being dropped or inflated.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_mgmt_status_is_measured_not_faked",
        "mgmt_status_measured_not_faked",
        "A real miss with a real CI is worth more than a claimed number.",
    ),
    Claim(
        "mgmt_beta_moves_the_engine_in_the_methylated_direction",
        "The requested mapping r0 = Baseline_Resistance * (1 - beta), with "
        "Baseline_Resistance derived from the calibrated clones (0.75), moves "
        "the engine: across the real TCGA-GBM beta range (quantiles of 140 "
        "patients, 2 probes), MTD median progression improves 144 -> 167 days "
        "and the resistant-clone dominance day moves 22 -> 23, while the "
        "steering policy saturates at 365 days at every beta (zero dynamic "
        "range under this mapping). Simulated, not clinical.",
        EvidenceStatus.ESTABLISHED,
        "danon/mgmt.py::mgmt_resistance_sweep on the cached cBioPortal panel "
        "(outputs_real/mgmt_resistance.txt)",
        "mgmt_resistance_sweep_measured",
        "The term reaches only the sensitivity-scale component of resistance; "
        "the kill ceiling is untouched. No patient outcome is claimed.",
    ),
    Claim(
        "covariate_upgrades_measured_not_estimated",
        "The four covariate upgrades, each measured against a matched baseline "
        "on the patients it keeps: KPS beside age/sex HR=0.9810 per point "
        "(3.3 SE) with delta_C=+0.0264 vs fold sd 0.0476 (n=247); MGMT binary "
        "HR=0.8326 with delta_C=-0.0006 (n=409); continuous beta HR=0.6627 "
        "with delta_C=+0.0006 (n=137); RT+temozolomide restriction keeps "
        "356/595 with cohort-membership delta -0.0112 inside fold sd 0.0247. "
        "None produces a measurable held-out gain; stacking all of them drops "
        "the ceiling below age alone (0.5785 vs 0.6203 at n=68).",
        EvidenceStatus.ESTABLISHED,
        "danon/cohort_features.py::covariate_upgrade_report "
        "(outputs_real/covariate_upgrades.txt)",
        "covariate_upgrades_measured",
        "Directionally consistent with trial literature; the cohort cannot "
        "resolve it. A change in n is never reported as a change in signal.",
    ),
    Claim(
        "clinical_streams_wired_with_identity_guarantees",
        "The four-stream clinical datastream is implemented with identity "
        "guarantees: Ki-67 at the declared reference (0.15) returns the "
        "calibrated growth rates exactly; uniform rCBV yields uniform "
        "clearance; a uniform 5-ALA map preserves the engine's fill burden. "
        "Each stream's constants are declared (KI67_REF=0.15, "
        "CLEARANCE_GAMMA=1.0), and stream 4 is an observation, not an input.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_stream_identities_hold",
        "stream_identities",
        "Mappings are re-parameterisations, not rival models; the neutral "
        "value is the calibrated default.",
    ),
    Claim(
        "stream3_real_vasculature_measured_small_effect",
        "Stream 3 (perfusion) is measured on real cached data: t1ce contrast "
        "volumes block-averaged through the segmentations' exact engine "
        "transform for 6 BraTS cases (tumour mean enhancement 0.37-0.49), "
        "then run PAIRED (same geometry, same seeds) against the null-uniform "
        "vasculature: MTD PFS moves by a mean of -1.17 days (sd 1.01, max 3, "
        "18 paired runs; 5/6 cases earlier), steering stays horizon-censored "
        "in both geometries (paired delta exactly 0). At engine scale the "
        "vessel-source variation contributes about one day. This claim "
        "previously ended by asserting that the sanctuary structure 'lives "
        "in D(x), which both geometries share' -- a speculation stated as a "
        "finding, with no check behind it. It has since been MEASURED: "
        "varying D(x) by 3.3x does not restore the difference either (see "
        "stream3_diffusion_measured).",
        EvidenceStatus.ESTABLISHED,
        "danon/clinical_twin.py::stream3_report over "
        "outputs_real/stream3_cohort.json (rebuildable via _stream3_cohort.py)",
        "stream3_cohort_measured",
        "A wired-and-measured stream whose measured effect is small is a "
        "result, not a failure; t1ce is a vascular proxy, not Ktrans.",
    ),
    Claim(
        "stream3_diffusion_measured",
        "Stream 3's diffusion arm (D_chemo(x) = D0*(1+2.5*rCBV(x)), the "
        "specification's own formula) is implemented against the engine's "
        "`diffusion_field` hook and measured on real cached BraTS-2021 cases, "
        "PAIRED transport-only: identical mask, proliferation, vessels source "
        "and seeds, differing only in the transport field. The mapping "
        "produces a 3.35-3.45x contrast in D(x) (1543-2673 perfused voxels per "
        "case), and across 3 cases x 2 policies x 3 seeds (18 paired runs) "
        "only 1 of 6 policy-arms moves AT ALL: MTD paired deltas -1, 0, 0 days "
        "(one seed pair in one case), steering exactly 0 in every case. That "
        "is within seed noise at this cohort size -- a 3.4x transport contrast "
        "does NOT eliminate a delivery blind spot here, and the effect is of "
        "the same order as the vasculature arm. The arm is transport-only by construction "
        "because the simulator reduces a supplied field to a mean-one "
        "relative pattern, so D0 and the unit conversion cancel exactly and "
        "gamma=2.5 is this parameter's only live constant: no absolute "
        "diffusivity claim rides on this mapping.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_stream3_diffusion_is_measured over "
        "outputs_real/stream3_diffusion_cohort.json (rebuildable via "
        "_stream3_diffusion_cohort.py), field stats re-derived live each audit",
        "stream3_diffusion_measured",
        "The engine arms are the committed measurement, not re-run inside the "
        "audit; the field statistics ARE recomputed live. An rCBV map is "
        "scalar, so this mapping is isotropic per voxel -- it is not a "
        "diffusion tensor, and no anisotropy is fabricated from perfusion.",
    ),
    Claim(
        "stream3_dsc_perfusion_measured",
        "Stream 3 consumes MEASURED perfusion, not only a contrast proxy: raw DSC "
        "series download from TCIA (UPENN-GBM), `danon.dicomio` reads their "
        "JPEG-2000-encapsulated pixels, and `danon.perfusion` integrates the "
        "first pass into a relative-CBV map. 3 series are downloaded and "
        "measured -- selected as the three smallest by listed FileSize, a rule "
        "fixed before any outcome was seen -- across three different "
        "acquisitions (TR 1.83-2.0 s, TE 45-51 ms, 700-900 files each). A "
        "head-to-tail drift line is subtracted before integration because 30-51% "
        "of the raw volume-mean peak on these series is drift, not bolus; "
        "without it the first real series called frame 0 the arrival and "
        "integrated 45 of 50 frames. The pipeline recovers a planted 3:1 CBV "
        "ratio to 0.9% under 0.2% signal noise; on the real maps the transport "
        "field spans 15.3-21.6x, and the only endpoint with resolution is the "
        "final burden, which differs between transport arms by at most "
        "0.18-8.81% (steering the more sensitive of the two policies). PFS is "
        "censored in every arm of this cohort and is reported as censored.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_stream3_dsc_perfusion_is_measured over "
        "outputs_real/perfusion_cohort.json (rebuildable via "
        "_perfusion_cohort.py)",
        "stream3_dsc_perfusion_measured",
        "The maps are relative CBV by first-pass integration: no arterial input "
        "function is fitted, so this is NOT absolute CBV and NOT Ktrans. The "
        "region the engine runs on is rCBV >= 1.5x the map median -- a "
        "perfusion-defined ROI, NOT a tumour segmentation, since no "
        "segmentation is co-registered to the DSC grid. No result here is about "
        "a patient's tumour, and nothing links a series to a survival outcome.",
    ),
    Claim(
        "longitudinal_availability_is_counted_from_the_manifest",
        "Stream 4's cohort is counted from the committed CFB-GBM availability "
        "manifest (TCIA, DOI 10.7937/v9pn-2f72): 264 patients, 170 with more "
        "than one timepoint, and 168 with a usable T1-Gd pair at least 4 weeks "
        "apart (393 pairs, 37 of them spanning two scanners). The manifest "
        "records that a series EXISTS; the images are not downloaded, so no "
        "registration, artefact or comparability check has been performed and "
        "no velocity field has been measured. The CFB-GBM collection is not "
        "served by the NBIA API this repo probes, so activating the stream "
        "requires downloading the collection, not querying it.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_availability_census_is_measured",
        "availability_census_measured",
        "Counts are read from the file on every check, so replacing the "
        "manifest with a newer release moves every number at once.",
    ),
    Claim(
        "readme_claim_count_is_derived_not_remembered",
        "The README's bound-claims row is checked against the registry on every "
        "audit rather than maintained by hand. It had drifted: the README "
        "advertised 31/31 verified while the registry held 33, so the project's "
        "own headline number was stale for two commits -- the exact failure "
        "mode the registry exists to catch, applied to the registry itself.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_readme_claim_count_matches_the_registry",
        "readme_claim_count_current",
        "It checks that the README agrees with the registry, not that the "
        "registry is right: claims can be wrong together and still agree.",
    ),
    Claim(
        "stream4_cohort_resolution_is_a_design_calculation",
        "Whether the manifest's cohort could resolve a gain is computed rather "
        "than assumed: resampling 168-patient cohorts from the real cached "
        "cohort (n=595, overall survival) 24 times moves the held-out C by "
        "0.0219 (1 SE), mean C 0.6429 (age alone 0.6469), with a mean fold sd "
        "of 0.0610; the full cohort scores 0.6559 +/- 0.0247. Two consequences "
        "follow. (a) Stream 4 is an OBSERVATION -- it calibrates the engine's "
        "derived velocity and cannot move a C-index at all until a hazard "
        "mapping is declared, so no download of CFB-GBM images would by itself "
        "raise the ceiling. (b) At that n any declared mapping would have to "
        "clear 0.0219 to be distinguishable from cohort noise, and the "
        "covariate effects already measured on the larger cohort (KPS +0.0264, "
        "MGMT -0.0006) sit inside the conservative fold yardstick of 0.0610. "
        "This measures the resolution of a cohort, not perfusion, "
        "fluorescence, Ki-67 or delta-volume.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_stream4_resolution_is_a_design_calculation over "
        "outputs_real/stream4_resolution.json (rebuildable via "
        "_stream4_resolution.py)",
        "stream4_resolution_is_design_only",
        "A design calculation bounds what a study could show; it is not "
        "evidence that any stream carries a signal, and the reference cohort "
        "is scored on overall survival, which the manifest does not carry.",
    ),
    Claim(
        "stream4_agreement_is_a_validation_target",
        "Stream 4 is implemented as a COMPARISON, not an input: "
        "stream4_velocity_agreement reports a measured delta-volume field "
        "against the engine's simulated one (rate ratio, Pearson r, sign "
        "agreement, peak-growth voxel) and refuses with the census's live "
        "counts when either side is absent. Identical fields score r=1.0; a "
        "field whose growth sits in a different voxel scores lower, so the "
        "statistic can be bad. No measured velocity is ever fed to the engine "
        "-- injecting a dR/dt into a model that derives its own would "
        "double-count the rate -- and no agreement, however good, can move a "
        "discrimination metric.",
        EvidenceStatus.ESTABLISHED,
        "danon/clinical_twin.py::stream4_velocity_agreement",
        "stream4_agreement_is_a_validation_target",
        "The comparison has never been run on a real measured pair: 0 are "
        "cached, so what a real agreement would score is unknown. This binds "
        "what the function IS, not what it would find.",
    ),
    Claim(
        "clinical_stream_census_refuses_with_live_counts",
        "Streams 1 (5-ALA fluorescence), 2 (whole-slide Ki-67) and 4 "
        "(sequential MRI) refuse on their CACHED counts -- no fluorescence "
        "captures, no WSIs, zero cached sequential pairs -- while their "
        "mappings are implemented and unit-tested, activating when real files "
        "are supplied. No stream is back-filled. The census also reports what "
        "is RETRIEVABLE without new code, so a refusal is not mistaken for "
        "absence: 41 of 630 probed UPENN-GBM patients carry a >=28-day "
        "follow-up interval, and the archive carries raw DSC perfusion for 456 "
        "patients. (Its 628 patients' worth of CaPTk-processed series are "
        "co-registered STRUCTURAL volumes, not rCBV/Ktrans maps: an earlier "
        "version of this sentence said perfusion maps, and a downloaded file "
        "settled it -- 'AX T1 PRE : Processed_CaPTk', TR 777 ms, one frame "
        "per file.)",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_stream_census_refuses_honestly",
        "stream_census_refuses",
        "The census is computed live from the data directory and the recorded "
        "probe, so acquiring the files flips the refusal to 'available' "
        "automatically.",
    ),
    Claim(
        "stream2_capacity_is_measured_and_is_a_scale_under_mtd",
        "Stream 2's measured cellularity reaches the engine's per-voxel density "
        "ceiling and is measured there, PAIRED on real cached BraTS-2021 geometry "
        "(same mask, proliferation, seeds, policy and initial resistant "
        "fraction; the ONLY difference is clones.carrying_capacity). The sweep's "
        "endpoints are read from the slide record rather than chosen: its own "
        "measured crop extremes over its own pooled density. What it measures is "
        "a scale and not a lever. Under continuous MTD the per-seed burden ratio "
        "equals the capacity ratio to within a fraction of a percent, so the "
        "reported burden is rescaled by K and the tumour's course is unchanged, "
        "and PFS does not move. Under the adaptive steering policy the ratio "
        "DEPARTS from K, because the engine's switch is gated on ABSOLUTE floors "
        "(MIN_VIABLE_DENSITY, DETECTION_BURDEN_DENSITY) that a K rescale does not "
        "move with it. So measured packing can matter only where an absolute "
        "floor gates a decision, and the ~100%+ burden differences reported at "
        "high K are that rescaling, not a treatment effect. An ABSOLUTE "
        "cells/mm^3 ceiling is refused: a whole-slide image counts per unit area "
        "and the archive's slides record MPP and AppMag with no section "
        "thickness to divide by.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_stream2_capacity_mapping_is_measured over "
        "outputs_real/stream2_capacity_cohort.json (rebuildable via "
        "_stream2_capacity_cohort.py); ratios and reference re-derived from "
        "outputs_real/histology_probe.json on every audit",
        "stream2_capacity_measured",
        "Three cases x two seeds per arm is a small paired cohort, and the "
        "steering departure is seed-dependent, so the MECHANISM is measured while "
        "the magnitude is not: no confidence interval is claimed here. The "
        "density is one slide (TCGA-06-5418) held fixed across the geometry of "
        "three different BraTS cases, so this is a sensitivity of the engine to K, "
        "not a statement about any patient's tumour. Nothing here moves a clinical "
        "endpoint: PFS moved 0 days in all but one policy-arm.",
    ),
    Claim(
        "restriction_never_mistaken_for_signal",
        "Restricting to patients recording RT plus temozolomide is measured as "
        "cohort selection first: kept 356 of 595, membership delta -0.0112 C "
        "against the full-cohort fold sd 0.0247, with within-restricted-set "
        "covariate deltas (KPS -0.0043, MGMT +0.0046) reported separately and "
        "below their own fold sd.",
        EvidenceStatus.ESTABLISHED,
        "danon/cohort_features.py::treatment_restriction_effect",
        "restriction_membership_vs_signal",
        "A record is an exposure, not an ordered protocol: first-line cannot "
        "be separated from salvage in this export.",
    ),
    Claim(
        "stream2_biopsy_measures_cellularity_and_refuses_ki67",
        "The biopsy stream measures what the archive actually holds and refuses "
        "what it does not. TCGA-GBM carries 2,053 whole-slide images at the live "
        "GDC count, none of them named as Ki-67 or IHC; the smallest by listed "
        "file size (6.01 MB) was downloaded and read -- an Aperio SVS, "
        "9352x4666, MPP 2.0037 um/px, AppMag 5, haematoxylin-only with a brown "
        "fraction of 0.0 over the sampled crop. From it danon/histology.py "
        "reports 636 nuclei per mm^2 over 4 crops (3,588 nuclei, per-crop "
        "spread 460-1579, median nucleus area 76.3 um^2). That is CELLULARITY. "
        "Stream 2's mapping needs the Ki-67 labelling index, whose numerator "
        "does not exist in this source, so ki67_fraction raises and no r0 is "
        "derived: the growth mapping stays unit-tested at KI67_REF=0.15 until "
        "a real Ki-67 slide is supplied. Substituting a nuclei density for the "
        "labelling index would fabricate the growth rate it feeds.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_stream2_histology_measured_and_ki67_refused "
        "over outputs_real/histology_probe.json (rebuildable via "
        "_histology_probe.py)",
        "stream2_histology_measured",
        "The 636 nuclei/mm^2 is ONE slide, the smallest by listed size -- not a "
        "cohort statistic. The zero-Ki-67 finding generalises only as far as the "
        "file index plus this one slide's pixels: the index carries barcodes, "
        "not markers, so none being named is a lower bound on identifiability, "
        "and the remaining 2,052 slides were not downloaded or read. The slide "
        "itself is a gitignored cache under data/tcia/ (re-downloadable via "
        "_histology_probe.py); the committed record is the evidence, and when "
        "the cache is present its size is checked against the record. The stain "
        "call is a hue-fraction heuristic, not a validated assay, and the nuclei "
        "count comes from connected components in a hue band with an area filter "
        "-- clumped nuclei are dropped rather than split, which is part of why "
        "crops spread 3.4x. No pathologist count anchors any of it, and nothing "
        "here is a grade, a WHO criterion, or a statement about a patient.",
    ),
)


@dataclass
class ClaimAudit:
    claim: Claim
    supported: bool
    detail: str


def audit_claims(claims: Optional[Tuple[Claim, ...]] = None) -> List[ClaimAudit]:
    """Run every claim's check against the repository as it is right now."""
    out: List[ClaimAudit] = []
    for claim in claims if claims is not None else CLAIMS:
        fn = CHECKS.get(claim.check)
        if fn is None:
            out.append(ClaimAudit(claim, False, f"unknown check {claim.check!r}"))
            continue
        try:
            res = fn()
        except Exception as exc:  # a check that cannot run has not passed
            out.append(ClaimAudit(claim, False, f"check raised: {exc}"))
            continue
        out.append(ClaimAudit(claim, bool(res.supported), res.detail))
    return out


def render_claims_markdown(audits: Optional[List[ClaimAudit]] = None) -> str:
    """A poster-ready table. Failed checks are printed, not dropped."""
    audits = audits if audits is not None else audit_claims()
    lines = [
        "| claim | tier | supported by | verified | limit |",
        "|---|---|---|---|---|",
    ]
    for a in audits:
        mark = "yes" if a.supported else "**NO**"
        lines.append(
            f"| {a.claim.sentence} | `{a.claim.tier.name}` | "
            f"`{a.claim.substantiated_by}` | {mark} | {a.claim.limit} |"
        )
    failures = [a for a in audits if not a.supported]
    lines.append("")
    if failures:
        lines.append(f"{len(failures)} of {len(audits)} claims failed their check:")
        for a in failures:
            lines.append(f"- **{a.claim.id}**: {a.detail}")
    else:
        lines.append(f"All {len(audits)} claims verified against the repository.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Honesty envelope
# ---------------------------------------------------------------------------


def claims_provenance() -> Dict[str, EvidenceStatus]:
    return {
        "claim_matches_code": EvidenceStatus.ESTABLISHED,
        "mechanism_is_mechanistic": EvidenceStatus.ESTABLISHED,
        "parameters_are_measured": EvidenceStatus.PROVEN_FAILED,
        "engine_predicts_patients": EvidenceStatus.UNKNOWN,
        "clinical_benefit": EvidenceStatus.UNKNOWN,
    }


def what_these_claims_cannot_establish() -> List[str]:
    return [
        "that a mechanistic model is more likely to be right than a learned one: "
        "mechanism makes a model falsifiable and interpretable, not correct",
        "that any rate constant in this repository is measured: they are "
        "calibrated or assumed, and the tiers say which",
        "that passing the audit means a judge should believe the biology: the "
        "audit checks the code against its own sentences",
        "that the engine predicts a patient's course, or that any schedule here "
        "extends a life -- both require prospective trials",
    ]
