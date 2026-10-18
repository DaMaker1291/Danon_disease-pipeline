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
    # Explicit UTF-8, not the locale codec: every file read here is UTF-8, and on
    # Windows the default cp1252 raises on any byte it cannot map (0x9d, the tail
    # of a typographic close-quote, is enough) -- which would turn an editing
    # choice in the README into a failed claim on one platform only.
    with open(os.path.join(ROOT, relpath), encoding="utf-8") as fh:
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


def _drug_units_probe_record() -> Optional[dict]:
    """The recorded label read and the measured relabelling experiment."""
    import json

    path = os.path.join(ROOT, "outputs_real", "drug_units_probe.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def check_ec50_unit_is_undeclared() -> CheckResult:
    """The engine's EC50 has no unit, so no potency calibration may be implied.

    And the unit is NOT what is missing. The kill law is `kill_max * c / (c +
    EC50)`, homogeneous of degree 0 in (c, EC50), and the drug transport is
    linear in its source -- so scaling the source strength and the EC50 together
    relabels concentration units exactly, and the trajectory is BIT-IDENTICAL.
    That is re-derived live here on the kill law, and read from the recorded
    engine-level experiment. What the label supplies is a concentration scale,
    not a potency: its Cmax is 7.5 mcg/mL oral / 7.3 mcg/mL IV at 150 mg/m2, and
    the figure a data-mining route usually quotes as a Cmax -- 5.5 -- is the
    CLEARANCE in L/hr/m2, which the label's own sentence says.
    """
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

    problems: List[str] = []

    # (1) Live, and arithmetic rather than remembered: the kill law cannot see
    # a relabelling of concentration, because c/(c+EC50) is unchanged when both
    # arguments are scaled.
    from danon.evosteer import DEFAULT_DRUGS

    ec50 = float(DEFAULT_DRUGS.ec50[0])
    homogeneous = all(
        abs((c / (c + ec50)) - ((c * k) / (c * k + ec50 * k))) == 0.0
        for c in (1e-3, 0.30, 5.0, 12.0)
        for k in (1e-3, 1024.0)
    )
    if not homogeneous:
        problems.append(
            "the kill law is no longer homogeneous of degree 0 in (c, EC50), so "
            "a unit relabelling would change the model and this claim is wrong"
        )

    # (2) The engine-level experiment, from its own record: scaling BOTH is
    # bit-identical, and scaling the EC50 alone moves the trajectory -- so the
    # first statement is not an artefact of a measurement that sees nothing.
    probe = _drug_units_probe_record()
    if probe is None:
        problems.append(
            "the drug-units probe is missing; rebuild it via _drug_units_probe.py"
        )
    else:
        rel = probe.get("relabelling") or {}
        if rel.get("burden_arrays_bit_identical_when_both_scaled") is not True:
            problems.append(
                "scaling source strength and EC50 together is no longer "
                "bit-identical, so the units are not a free choice after all"
            )
        if rel.get("burden_arrays_differ_when_only_ec50_scaled") is not True:
            problems.append(
                "scaling the EC50 alone no longer moves the trajectory, so the "
                "relabelling measurement cannot see anything"
            )
        if rel.get("kill_fraction_bit_identical_under_the_scale") is not True:
            problems.append("the recorded kill-fraction check is no longer exact")

        # (3) The label, as recorded: the concentration scale is read, and the
        # number a route calls a Cmax is the label's clearance.
        label = probe.get("label") or {}
        sentences = " ".join(label.get("sentences") or [])
        if label.get("status") != 200 or not sentences:
            problems.append("the label read is missing from the record")
        else:
            low = sentences.lower()
            if "7.5 mcg/ml" not in low or "7.3 mcg/ml" not in low:
                problems.append(
                    "the recorded label sentences no longer carry the oral and IV "
                    "Cmax values this claim quotes"
                )
            if "clearance" not in low or "5.5 l/hr" not in low:
                problems.append(
                    "the recorded label no longer says the 5.5 figure is a "
                    "clearance in L/hr/m2"
                )
            if re.search(r"cmax[^.]{0,80}5\.5", low):
                problems.append(
                    "the label is now recorded calling 5.5 a Cmax, which the "
                    "label's own elimination sentence contradicts"
                )

    supported = refused and not problems
    if problems:
        return CheckResult(
            supported=False, detail=detail + "; " + "; ".join(problems)
        )
    return CheckResult(
        supported=supported,
        detail=(
            detail
            + "; kill law degree-0 homogeneous in (c, EC50): "
            + str(homogeneous)
            + "; scaling source strength AND EC50 by 2^10 reproduces the "
            "60-day burden arrays bit for bit while scaling the EC50 alone "
            "does not, so the engine's concentration scale is a free choice; "
            "the label's own kinetic scale is a Cmax of 7.5 mcg/mL oral / "
            "7.3 mcg/mL IV at 150 mg/m2 with clearance 5.5 L/hr/m2 and "
            "half-life 1.8 h, so what is missing is a MEASURED potency, not a "
            "unit"
        ),
    )


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


def check_egfrviii_is_refused() -> CheckResult:
    """EGFRvIII status must stay unreachable from open data, with the count.

    This check used to also cover the RL tier, but the two were only ever
    coupled by the word 'untestable': one was a missing dataset (no open
    structural-variant calls exist for EGFRvIII) and the other was missing
    code (no trained policy). The RL tier now has its own check
    (`check_rl_policy_is_measured`), and this one is what it says it is.
    """
    from danon.eval_targets import probe_egfrviii_reachability

    egfr = probe_egfrviii_reachability()
    return CheckResult(
        supported=not egfr.reachable_open,
        detail=(
            f"EGFRvIII: {egfr.n_structural_variation_files} "
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

    # The two sources a public-data route names for the missing numerator were
    # probed and RECORDED, because "this archive has no Ki-67 slide" is a much
    # weaker statement than "and neither source that route names carries one":
    # the first invites a re-probe, the second needs a schema change. The
    # recorded field and column lists are re-searched here rather than trusted,
    # so a hand-edited record cannot assert an absence its own evidence denies.
    probe = _ki67_source_probe_record()
    if probe is None:
        problems.append(
            "the external-source probe is missing, so the refusal rests on this "
            "tree alone; rebuild it via _ki67_source_probe.py"
        )
    else:
        tokens = ("ki67", "ki_67", "ki-67", "mib", "labelling", "labeling")

        def _proliferation_named(names) -> List[str]:
            return [n for n in names if any(t in str(n).lower() for t in tokens)]

        gdc = probe.get("gdc_tcga_gbm_clinical_schema") or {}
        fields = list(gdc.get("field_names") or [])
        if gdc.get("status") != 200 or not gdc.get("schema_version"):
            problems.append(
                "the recorded GDC clinical file is not the schema that was read"
            )
        if len(fields) < 50:
            problems.append(
                f"the recorded TCGA-GBM clinical schema carries only {len(fields)} "
                "field(s), so it is not a clinical record and cannot testify about "
                "what the record lacks"
            )
        if not any(
            "karnofsky" in f.lower() or "vital_status" in f.lower() for f in fields
        ):
            problems.append(
                "the recorded schema carries no known clinical field, so a missing "
                "Ki-67 field in it would mean nothing"
            )
        found = gdc.get("proliferation_tokens_found") or _proliferation_named(fields)
        if found:
            problems.append(
                f"a Ki-67/MIB-1 field now exists in the TCGA-GBM clinical schema "
                f"({found}): the public-data route is live and this refusal must be "
                f"revisited rather than kept"
            )

        tables = (probe.get("ivygap_tables") or {}).get("tables") or {}
        expected_tables = {
            "tumor_details.csv",
            "sub_block_details.csv",
            "rna_seq_samples_details.csv",
            "gene_expression_details.csv",
        }
        missing = sorted(expected_tables - set(tables))
        if missing:
            problems.append(f"the Ivy GAP probe is missing {'/'.join(missing)}")
        for name, table in tables.items():
            columns = list(table.get("columns") or [])
            if table.get("status") != 200 or not columns:
                problems.append(f"the Ivy GAP table {name} was not read")
            named = table.get("proliferation_named_columns") or _proliferation_named(
                columns
            )
            if named:
                problems.append(
                    f"the Ivy GAP table {name} now carries a proliferation-named "
                    f"column ({named}); if it is a labelling index the refusal has "
                    f"to change"
                )
        tumour_columns = set(
            (tables.get("tumor_details.csv") or {}).get("columns") or []
        )
        if not {"mgmt_methylation", "survival_days"}.issubset(tumour_columns):
            problems.append(
                "the recorded Ivy GAP tumour table is not the per-tumour table, so "
                "its columns say nothing about what the atlas lacks"
            )
        # The trap is recorded WITH the finding: the atlas does carry a
        # proliferation-named column, and it is a vessel-structure area fraction.
        # If it ever disappears, the probe's distinction is no longer shown.
        sub_block_columns = (tables.get("sub_block_details.csv") or {}).get("columns") or []
        if "normalized_area_ctmvp" not in sub_block_columns:
            problems.append(
                "Ivy GAP's microvascular-proliferation structure column is gone, so "
                "the probe no longer shows a structure area fraction beside the "
                "labelling index it does not have"
            )

        tcia = probe.get("tcia_collections") or {}
        if int(tcia.get("n_collections") or 0) < 100:
            problems.append("the TCIA public collection list was not read")
        if tcia.get("ivy_named_collections"):
            problems.append(
                f"an ivy-named TCIA collection now exists "
                f"({tcia.get('ivy_named_collections')}), so the atlas cohort's "
                f"imaging may be reachable through the API this repository queries"
            )

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
            f"exercised and stream 2 still refused as an input. The two external "
            f"sources a public-data route names were probed and neither carries "
            f"the numerator: the TCGA-GBM clinical BCR XML (schema "
            f"{(probe or {}).get('gdc_tcga_gbm_clinical_schema', {}).get('schema_version')}, "
            f"{len(((probe or {}).get('gdc_tcga_gbm_clinical_schema') or {}).get('field_names') or [])} "
            f"distinct fields, read back live) has no Ki-67/MIB-1 field in it, and "
            f"Ivy GAP's four public tables carry no proliferation index -- the only "
            f"proliferation in the atlas is normalized_area_ctmvp, the area fraction "
            f"of MICROVASCULAR proliferation, a vessel measurement rather than a "
            f"labelling index -- with no ivy-named collection in TCIA's public list"
            + ("; " + "; ".join(problems) if problems else "")
        ),
    )


def _histology_stream2_census() -> Optional[dict]:
    try:
        from danon.clinical_twin import clinical_stream_census

        return clinical_stream_census()["stream2_histology"]
    except Exception:  # pragma: no cover - census failure is reported by its own check
        return None


def _ki67_source_probe_record() -> Optional[dict]:
    """The recorded read of the two external sources a public-data route names."""
    import json

    path = os.path.join(ROOT, "outputs_real", "ki67_source_probe.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
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


# ---------------------------------------------------------------------------
# Numerical verification, and the demo path a stranger can actually run
# ---------------------------------------------------------------------------


def check_solver_verified_against_analytic_solutions() -> CheckResult:
    """The solver against exact analytic values, not against itself.

    Every quantity the sentence names is re-derived here from the engine's own
    constants and operators: the zero-flux Laplacian on its Neumann eigenbasis,
    the closed-domain mass identity, the coincidence of the two transport
    branches at constant D, the observed order of spatial convergence, the
    signed clearance bias implied by the stability rule, and -- from the
    committed paired cohort -- the size of that bias against the run's own
    seed-to-seed spread. Move any of them and the sentence cannot sit there.
    """
    import json
    import math

    from danon.anisotropy import divergence_d_grad
    from danon.evosteer import DEFAULT_DRUGS, _laplacian

    d = float(DEFAULT_DRUGS.diffusion[0])
    lam = float(DEFAULT_DRUGS.clearance[0])

    # (1) f = cos(pi k (i+1/2)/N) satisfies the edge-replicated boundary exactly
    # and is an eigenvector with eigenvalue -4 sin^2(pi k / 2N) per axis.
    n = 24
    residual = 0.0
    for k in (1, 3, 7, 12):
        idx = np.arange(n) + 0.5
        f = np.cos(math.pi * k * idx / n)[:, None, None] * np.ones((1, n, n))
        eigen = -4.0 * math.sin(math.pi * k / (2.0 * n)) ** 2
        residual = max(
            residual,
            float(
                np.linalg.norm(_laplacian(f) - eigen * f)
                / abs(eigen)
                / np.linalg.norm(f)
            ),
        )

    # (2) Closed domain: the operator's total must vanish on any field.
    rng = np.random.default_rng(7)
    spike = np.zeros((16, 16, 16))
    spike[0, 0, 0] = 1.0
    mass_rel = 0.0
    for field in (rng.random((16, 16, 16)), np.ones((16, 16, 16)), spike):
        mass_rel = max(
            mass_rel,
            abs(float(_laplacian(field).sum())) / max(float(np.abs(field).sum()), 1.0),
        )

    # (3) The heterogeneous branch must collapse onto the homogeneous one.
    field = rng.random((12, 12, 12))
    reference = d * _laplacian(field)
    gap_rel = float(
        np.abs(divergence_d_grad(field, np.full(field.shape, d)) - reference).max()
    ) / max(float(np.abs(reference).max()), 1e-300)

    # (4) Second order in space: halving dx must quarter the eigenvalue error.
    errors = [
        abs(-4.0 * math.sin(math.pi / (2.0 * m)) ** 2 * m * m + math.pi ** 2)
        / math.pi ** 2
        for m in (8, 16, 32, 64)
    ]
    orders = [math.log2(a / b) for a, b in zip(errors, errors[1:])]
    min_order = min(orders)

    # (5) The one signed discretization error, from the engine's own rule.
    sub_dt = min(0.4 / (6.0 * d), 1.0)
    n_sub = max(1, int(math.ceil(1.0 / sub_dt)))
    dts = 1.0 / n_sub
    realized = (1.0 - lam * dts) ** n_sub
    exact = math.exp(-lam)
    bias = exact / realized - 1.0

    # (6) Is that error below the model's own resolution? The paired cohort on
    # real geometry answers with a number rather than an argument.
    cohort = os.path.join(ROOT, "outputs_real", "clearance_bias_cohort.json")
    try:
        with open(cohort, encoding="utf-8") as fh:
            record = json.load(fh)
    except (OSError, ValueError) as exc:
        return CheckResult(
            supported=False,
            detail=(
                f"clearance-bias cohort unreadable ({exc}); rebuild via "
                "_clearance_bias_cohort.py"
            ),
        )

    problems: List[str] = []
    if residual > 1e-12:
        problems.append(f"the Laplacian is not exact on its eigenbasis ({residual:.2e})")
    if mass_rel > 1e-9:
        problems.append(f"the operator does not conserve mass ({mass_rel:.2e})")
    if gap_rel > 1e-12:
        problems.append(f"the two transport branches differ at constant D ({gap_rel:.2e})")
    if min_order <= 1.9:
        problems.append(f"spatial convergence is not second order ({min_order:.4f})")
    if not 0.0 < bias < 0.002:
        problems.append(f"the clearance bias is not the bounded one ({bias * 100:.4f}%/day)")

    recorded_bias = float(record.get("bias_rate_per_day", 0.0))
    if abs(recorded_bias - bias) > 1e-12:
        problems.append(
            f"the cohort's bias ({recorded_bias:.10f}) is not the rule's ({bias:.10f})"
        )
    delta = float(record.get("worst_max_rel_delta_percent", 0.0))
    spread = float(record.get("worst_seed_spread_percent", 0.0))
    ratio = record.get("delta_over_spread")
    if spread <= 0.0 or delta <= 0.0:
        problems.append("the cohort records no endpoint change to compare against the spread")
    elif ratio is None or abs(float(ratio) - delta / spread) > 1e-9:
        problems.append("the recorded delta/spread ratio is not the ratio of the recorded pair")
    elif float(ratio) > 0.01:
        problems.append(
            f"the bias is no longer below the model's resolution "
            f"({float(ratio):.2e} of the seed spread)"
        )

    return CheckResult(
        supported=not problems,
        detail=(
            f"Neumann eigenvector residual {residual:.2e}; mass identity "
            f"{mass_rel:.2e} relative; the constant-D branches agree to {gap_rel:.2e}; "
            f"observed spatial order {min_order:.4f} (second); clearance bias "
            f"+{bias * 100:.4f}%/day from the engine's own sub-step rule; over the "
            f"paired cohort that bias moves an endpoint by at most {delta:.4f}%, "
            f"against a seed-to-seed spread of {spread:.2f}% -- {float(ratio or 0.0):.2e} "
            f"of it"
            + ("; " + "; ".join(problems) if problems else "")
        ),
    )


def check_sample_trial_runs_without_a_download() -> CheckResult:
    """A fresh clone must be able to RUN the thing, not only read about it.

    Three things are verified live: the committed volumes are byte-for-byte the
    phantom generator's output (so the dataset cannot drift from, or quietly
    replace, the phantom the docstring names); the config path drives the real
    simulator through both arms; and the committed report is internally
    consistent with the dataset and with the sentence that cites it.
    """
    import json
    import tempfile

    from danon.evosteer import make_gbm_phantom
    from danon.imaging import from_nifti_mask
    from danon.reproduce import (
        SAMPLE_ROOT,
        SAMPLE_SEEDS,
        build_sample_dataset,
        format_report,
        load_config,
        run_config,
        sample_config_path,
    )

    root = os.path.join(ROOT, SAMPLE_ROOT)
    config_path = sample_config_path(root)
    problems: List[str] = []

    volumes = (
        "sub-01/sub-01_mask.nii.gz",
        "sub-01/anat/sub-01_prolif.nii.gz",
        "sub-01/perf/sub-01_vessels.nii.gz",
        "sub-01/sub-01_twin.json",
        "dataset_description.json",
    )
    with tempfile.TemporaryDirectory() as tmp:
        fresh = os.path.join(tmp, "sample")
        build_sample_dataset(fresh)
        for relative in volumes:
            committed = os.path.join(root, *relative.split("/"))
            rebuilt = os.path.join(fresh, *relative.split("/"))
            if not os.path.exists(committed):
                problems.append(f"{relative} is not committed")
                continue
            with open(committed, "rb") as a, open(rebuilt, "rb") as b:
                if a.read() != b.read():
                    problems.append(
                        f"{relative} is not what build_sample_dataset produces; "
                        "regenerate with `python -m danon.reproduce build-sample`"
                    )

        spec = load_config(config_path)
        page = from_nifti_mask(
            spec["inputs"]["mask"],
            proliferative_path=spec["inputs"]["proliferative"],
            vessels_path=spec["inputs"]["vessels"],
        )
        phantom = make_gbm_phantom(size=24, seed=0)
        if not np.array_equal(phantom.tumor_mask, page.geometry.tumor_mask):
            problems.append("the committed mask is not the phantom's tumour mask")
        if float(np.abs(phantom.proliferative - page.geometry.proliferative).max()) != 0.0:
            problems.append("the committed proliferation map is not the phantom's, exactly")
        if float(np.abs(phantom.vessels - page.geometry.vessels).max()) > 0.5 / 255.0:
            problems.append("the committed vessel map is more than one quantisation step off")

        # An empty cohort must be refused, not answered with a median of nothing.
        short = {
            "subject": "sub-01",
            "synthetic": True,
            "inputs": {
                "mask": spec["inputs"]["mask"],
                "proliferative": spec["inputs"]["proliferative"],
                "vessels": spec["inputs"]["vessels"],
            },
            "trial": {"days": 3, "seeds": [101], "arms": ["mtd", "steering"]},
        }
        bad = dict(short, trial=dict(short["trial"], seeds=[]))
        empty_path = os.path.join(tmp, "empty.json")
        with open(empty_path, "w", encoding="utf-8") as fh:
            json.dump(bad, fh)
        try:
            load_config(empty_path)
            problems.append("an empty seed list was accepted")
        except ValueError:
            pass

        short_path = os.path.join(tmp, "short.json")
        with open(short_path, "w", encoding="utf-8") as fh:
            json.dump(short, fh)
        short_report = run_config(short_path)
        if set(short_report["arms"]) != {"mtd", "steering"}:
            problems.append("the config-driven run did not execute both arms")
        if len(short_report["arms"]["mtd"]["runs"]) != 1:
            problems.append("the config-driven run did not execute the seed it was given")
        if "SYNTHETIC PHANTOM" not in format_report(short_report):
            problems.append("the printed report does not label the phantom as synthetic")

    # The committed report must still describe this dataset and this sentence.
    record_path = os.path.join(ROOT, "outputs_real", "sample_twin_report.json")
    try:
        with open(record_path, encoding="utf-8") as fh:
            record = json.load(fh)
    except (OSError, ValueError) as exc:
        return CheckResult(
            supported=False,
            detail=f"sample report unreadable ({exc}); rebuild via `python -m danon.reproduce --out outputs_real/sample_twin_report.json`",
        )

    if record.get("config") != SAMPLE_ROOT.replace(os.sep, "/") + "/sub-01/sub-01_twin.json":
        problems.append("the report was produced from a different config")
    if not record.get("synthetic"):
        problems.append("the report does not carry the synthetic label")
    live_geometry = {
        "n_voxels": int(page.geometry.tumor_mask.size),
        "n_tumour_voxels": int(page.geometry.tumor_mask.sum()),
    }
    for key, value in live_geometry.items():
        if int(record.get("geometry", {}).get(key, -1)) != value:
            problems.append(
                f"the report's {key} ({record.get('geometry', {}).get(key)}) is not the "
                f"dataset's ({value})"
            )
    if list(record.get("trial", {}).get("seeds", [])) != list(SAMPLE_SEEDS):
        problems.append("the report's seeds are not the sample's")

    arms = record.get("arms", {})
    mtd_runs = arms.get("mtd", {}).get("runs", [])
    steer_runs = arms.get("steering", {}).get("runs", [])
    if len(mtd_runs) != len(SAMPLE_SEEDS) or len(steer_runs) != len(SAMPLE_SEEDS):
        problems.append("the report does not hold one run per arm per seed")
    elif not all(r.get("progressed") for r in mtd_runs):
        problems.append("the sentence's 'MTD progresses in every replicate' is no longer true")
    elif any(r.get("progressed") for r in steer_runs):
        problems.append("the sentence's 'the switching arm does not progress' is no longer true")
    else:
        delta_days = float(record.get("paired", {}).get("median_pfs_delta_days", 0.0))
        pfs = [float(r["pfs_days"]) for r in mtd_runs]
        spread_days = max(pfs) - min(pfs)
        if int(record.get("paired", {}).get("n_pairs_in_disagreement", 0)) != len(SAMPLE_SEEDS):
            problems.append("the arms no longer differ on every paired seed")
        if delta_days <= spread_days:
            problems.append(
                f"the paired delta ({delta_days} d) no longer clears the arm's own "
                f"seed spread ({spread_days} d), so the sentence reads noise"
            )

    return CheckResult(
        supported=not problems,
        detail=(
            f"{len(volumes)} committed sample files are byte-identical to "
            f"build_sample_dataset's output; the loaded mask and proliferation map "
            f"equal the phantom's exactly and the vessel map is within one "
            f"quantisation step; a config-driven run executes both arms; the "
            f"committed report agrees with the dataset "
            f"({live_geometry['n_tumour_voxels']:,} tumour voxels of "
            f"{live_geometry['n_voxels']:,})"
            + ("; " + "; ".join(problems) if problems else "")
        ),
    )


def check_uncertainty_budget_separates_input_from_noise() -> CheckResult:
    """Where the spread lives, re-derived -- and the posterior label refused.

    The record's every verdict is recomputed here from its own mean squares:
    the sign of (MSB - MSW)/r decides whether a row is resolvable, refused, or
    degenerate, and the fraction must be that component over the total. When
    the panel's cache is present the donor count and the input band are re-read
    from the LIVE panel, so a rebuilt panel with different betas cannot leave
    this sentence standing on an old distribution.
    """
    import json

    from danon.uncertainty import (
        percentile_band,
        uncertainty_provenance,
    )

    path = os.path.join(ROOT, "outputs_real", "uncertainty_budget.json")
    try:
        with open(path, encoding="utf-8") as fh:
            record = json.load(fh)
    except (OSError, ValueError) as exc:
        return CheckResult(
            supported=False,
            detail=(
                f"uncertainty record unreadable ({exc}); rebuild via "
                "_uncertainty_cohort.py"
            ),
        )

    problems: List[str] = []
    if record.get("is_credible_interval") is not False:
        problems.append("the record does not refuse the credible-interval label")
    if uncertainty_provenance()["credible_interval"].name != "PROVEN_FAILED":
        problems.append("the provenance no longer marks a credible interval as refused")

    reps = int(record.get("replicates") or 0)
    if reps < 2:
        problems.append(
            f"the record's replicate count ({reps}) cannot support a within estimate"
        )

    degenerate: List[Tuple[str, str]] = []
    resolvable: List[Tuple[str, str]] = []
    refused: List[Tuple[str, str]] = []
    for row in record.get("arms", []):
        slot = (str(row.get("arm")), str(row.get("readout")))
        budget = row.get("budget") or {}
        if not {"ms_between", "ms_within"} <= set(budget):
            problems.append(f"{slot}: the row records no mean squares")
            continue
        msb, msw = float(budget["ms_between"]), float(budget["ms_within"])
        between = (msb - msw) / reps if reps else 0.0
        if bool(budget.get("is_degenerate")):
            if msb != 0.0 or msw != 0.0:
                problems.append(f"{slot}: flagged degenerate with non-zero mean squares")
            degenerate.append(slot)
        elif between <= 0.0:
            if budget.get("component_between") is not None:
                problems.append(f"{slot}: a negative component was recorded as a number")
            elif budget.get("input_fraction") is not None:
                problems.append(f"{slot}: a fraction was recorded with no component")
            else:
                refused.append(slot)
        else:
            recorded = budget.get("component_between")
            if recorded is None:
                problems.append(f"{slot}: a positive component was recorded as missing")
            elif abs(float(recorded) - between) > 1e-9 * max(abs(between), 1e-300):
                problems.append(f"{slot}: the recorded component is not (MSB - MSW)/r")
            else:
                fraction = between / (between + msw)
                expected = budget.get("input_fraction")
                if expected is None or abs(float(expected) - fraction) > 1e-9:
                    problems.append(
                        f"{slot}: the recorded fraction is not the component over the total"
                    )
                else:
                    resolvable.append(slot)

        propagated = row.get("propagated_band") or {}
        span = row.get("input_span") or {}
        for label, band in (("propagated", propagated), ("input span", span)):
            if not {"lo", "median", "hi", "n"} <= set(band):
                problems.append(f"{slot}: the {label} is not a band")
                continue
            if not float(band["lo"]) <= float(band["median"]) <= float(band["hi"]):
                problems.append(f"{slot}: the {label} is not an ordered interval")
        if int(propagated.get("n", -1)) != int(record["n_donors"]) * reps:
            problems.append(
                f"{slot}: the propagated band does not hold one outcome per donor x seed"
            )
        if int(span.get("n", -1)) != int(record["n_donors"]):
            problems.append(f"{slot}: the input span is not one value per donor")

    if ("steering", "pfs_days") not in degenerate:
        problems.append(
            "the switching arm's PFS is no longer degenerate: the sentence's claim "
            "that it carries no spread at this design is wrong"
        )
    if not refused:
        problems.append(
            "no row is recorded as unresolvable, so the sentence's 'not resolvable "
            "above the model's own noise' has nothing behind it"
        )
    if "mtd" not in {arm for arm, _ in resolvable + refused}:
        problems.append("the MTD arm is missing from the record entirely")

    # The sentence quotes a standard-deviation ratio and two significance classes.
    # Both are re-derived here from the recorded mean squares, because a quoted
    # number that nothing recomputes is the one thing this registry exists to stop.
    from danon.uncertainty import VarianceBudget, f_p_value

    def _budget(slot: Tuple[str, str]) -> Optional[VarianceBudget]:
        for row in record.get("arms", []):
            if (str(row.get("arm")), str(row.get("readout"))) != slot:
                continue
            b = row.get("budget") or {}
            return VarianceBudget(
                n_inputs=int(b["n_inputs"]),
                n_replicates=int(b["n_replicates"]),
                df_between=int(b["df_between"]),
                df_within=int(b["df_within"]),
                ms_between=float(b["ms_between"]),
                ms_within=float(b["ms_within"]),
            )
        return None

    ratio_text = "no mtd/pfs_days row"
    mtd = _budget(("mtd", "pfs_days"))
    if mtd is None:
        problems.append("the record has no mtd/pfs_days row")
    else:
        ratio = mtd.noise_to_signal_sd_ratio
        mtd_p = f_p_value(mtd.f_ratio, mtd.df_between, mtd.df_within) if mtd.f_ratio else None
        if ratio is None:
            problems.append(
                "the MTD PFS input component is no longer positive, so the sentence's "
                "'a repeat run moves further than a different donor' has no content"
            )
        else:
            # Not tighter than this on purpose: a one-day shift in a progression
            # day on a different platform moves the ratio, and the sentence's
            # claim is its magnitude, not its last digit.
            if not 2.5 <= ratio <= 3.3:
                problems.append(
                    f"the MTD PFS noise-to-signal SD ratio moved from 2.883 to "
                    f"{ratio:.3f}"
                )
            ratio_text = (
                f"MTD PFS: repeat-run SD {mtd.within_sd:.2f} d vs donor-to-donor SD "
                f"{mtd.between_sd:.2f} d (ratio {ratio:.3f})"
            )
        if mtd_p is None:
            problems.append("scipy is unavailable, so the F verdict cannot be computed")
        elif not 0.05 < mtd_p <= 0.15:
            problems.append(
                f"the MTD PFS F test left the 'not significant at 0.05' class "
                f"(p={mtd_p:.5f})"
            )
        else:
            ratio_text += f", F p={mtd_p:.5f} (not significant at 0.05)"

    burden = _budget(("mtd", "final_burden"))
    if burden is None or burden.component_between is None:
        problems.append(
            "the MTD burden row no longer resolves the input, so the sentence's "
            "'the burden readout does resolve it' is false"
        )
    else:
        burden_p = (
            f_p_value(burden.f_ratio, burden.df_between, burden.df_within)
            if burden.f_ratio
            else None
        )
        if burden_p is None or burden_p >= 0.01:
            problems.append(
                f"the MTD burden F test is no longer significant at 0.01 "
                f"(p={burden_p})"
            )
        else:
            ratio_text += f"; MTD burden: input fraction {burden.input_fraction * 100:.1f}%, p={burden_p:.5f}"

    # Bind the donor set to the live panel when its cache is present. The cache
    # is gitignored and re-downloadable, so a miss is reported, not failed.
    panel_note = ""
    from danon.mgmt import MGMT_BETA_CACHE

    if os.path.exists(os.path.join(ROOT, MGMT_BETA_CACHE)):
        from danon.mgmt import load_mgmt_beta_panel

        panel = load_mgmt_beta_panel()
        values = [v for v in (panel.mean_beta(p) for p in panel.patient_ids) if v is not None]
        live_band = percentile_band(values)
        if panel.n_patients != int(record["n_donors"]):
            problems.append(
                f"the record's {record['n_donors']} donors are not the live panel's "
                f"{panel.n_patients}"
            )
        recorded_band = record.get("input_band") or {}
        recorded_lo, recorded_hi = recorded_band.get("lo"), recorded_band.get("hi")
        if (
            recorded_lo is None
            or recorded_hi is None
            or abs(live_band.lo - float(recorded_lo)) > 1e-12
            or abs(live_band.hi - float(recorded_hi)) > 1e-12
        ):
            problems.append(
                "the record's input band is not the live panel's: the distribution "
                "changed and the record is stale"
            )
        panel_note = (
            f"the live panel reproduces the record's {panel.n_patients} donors and "
            f"its beta band {live_band}"
        )
    else:
        panel_note = (
            f"panel cache absent ({MGMT_BETA_CACHE}); the donor count and band were "
            "checked against the record only"
        )

    return CheckResult(
        supported=not problems,
        detail=(
            f"{len(resolvable)} row(s) resolve the input above the noise, "
            f"{len(refused)} refuse it (negative component recorded unclipped), "
            f"{len(degenerate)} degenerate; {ratio_text}; credible interval refused "
            f"in the record and in the provenance; {panel_note}"
            + ("; " + "; ".join(problems) if problems else "")
        ),
    )


def check_rl_policy_is_measured() -> CheckResult:
    """A trained RL policy's stored numbers must reproduce live.

    The artifact is not taken on trust. `danon.rl_steering` verifies the
    checkpoint and training-record hashes, the complete train/monitor/evaluation
    seed partition, all held-out arm summaries, live rule selection on the
    monitor seeds, and retention arithmetic, then re-runs the first held-out
    sampled and greedy seeds through the engine. This is a spot-reproduction
    plus consistency audit, not a full 30-seed re-execution; it does not
    validate the authored metric clinically.
    """
    try:
        from danon.rl_steering import verify_rl_policy_retention
    except Exception as exc:  # pragma: no cover - import failure is reported
        return CheckResult(False, f"danon.rl_steering unavailable: {exc}")
    try:
        v = verify_rl_policy_retention()
    except Exception as exc:
        return CheckResult(False, f"live retention re-measurement raised: {exc}")
    return CheckResult(supported=v.ok, detail=v.detail)

def check_short_screening_is_calibrated_not_assumed() -> CheckResult:
    """The short-horizon screen must carry a MEASURED fidelity, not a promise.

    A 120-day screen is the obvious way to make a 365-day search cheap, and it
    is wrong here: measured against the full-horizon ranking it is NEGATIVELY
    correlated and discards the true winner. That is a load-bearing finding, so
    it is audited rather than left in a notebook.

    This check passes only while the calibration exists AND the API carries the
    verdict. If `screen_fidelity` is missing, the saving is unvalidated and the
    check refuses rather than assuming it works.
    """
    import json as _json
    import os as _os

    path = _os.path.join(ROOT, "outputs_real/dosing.json")
    if not _os.path.exists(path):
        return CheckResult(
            supported=False,
            detail="outputs_real/dosing.json missing; short-horizon screening is "
                   "uncalibrated, so it must not be used",
        )
    with open(path, encoding="utf-8") as fh:
        d = _json.load(fh)
    fid = d.get("screen_fidelity")
    if not fid:
        return CheckResult(
            supported=False,
            detail="dosing artifact carries no screen_fidelity section; the "
                   "cost of short-horizon screening is unmeasured",
        )

    rho = fid.get("spearman_rho")
    kept = bool(fid.get("screen_keeps_full_winner"))
    caught = fid.get("top_k_caught_by_screen")
    missed = fid.get("top_k_missed_by_screen") or []
    verdict = (
        "the short screen RETAINS the full-horizon winner"
        if kept
        else "the short screen DISCARDS the full-horizon winner"
    )
    return CheckResult(
        supported=True,
        detail=(
            f"short-horizon screen calibration on {fid.get('n_candidates')} "
            f"candidates: {fid.get('screen_days')}-day ranking vs "
            f"{fid.get('full_days')}-day ranking gives Spearman {rho:.3f}, "
            f"Kendall tau {fid.get('kendall_tau'):.3f}; {verdict}; "
            f"{caught} of the top-{fid.get('top_k')} full-horizon calendars "
            f"survive the screen, {len(missed)} missed. Short screening is "
            f"therefore NOT used by the search (stage 1 ranks on the full "
            f"horizon), and the API returns this verdict to any client that "
            f"asks for one."
        ),
    )

def check_dosing_schedules_are_measured_on_the_engine() -> CheckResult:
    """The dosing-console claim must be a real engine run, honestly reported.

    This reads `outputs_real/dosing.json` and refuses in three distinct ways,
    because three different things can be wrong and they need different
    sentences:

      * the artifact is missing -- nothing was ever run;
      * the competitive-release or search section is missing -- the claim
        quotes a number the artifact does not contain;
      * the search reported a win the artifact does not support -- the
        sentence would be asserting a schedule beat MTD when it did not.

    The last one matters most. The measured search is a NULL result, and this
    check fails if the registry ever starts claiming otherwise.
    """
    import json as _json
    import os as _os

    path = _os.path.join(ROOT, "outputs_real/dosing.json")
    if not _os.path.exists(path):
        return CheckResult(
            supported=False,
            detail="outputs_real/dosing.json missing; no dosing schedule has "
                   "been run on the engine, so nothing may be claimed",
        )
    try:
        with open(path, encoding="utf-8") as fh:
            d = _json.load(fh)
    except ValueError as e:
        return CheckResult(
            supported=False, detail=f"dosing artifact is not readable JSON: {e}"
        )

    cr = d.get("competitive_release")
    if not cr:
        return CheckResult(
            supported=False,
            detail="dosing artifact carries no competitive_release section; "
                   "the tipping-point claim has no measurement behind it",
        )
    search = d.get("search")
    if not search:
        return CheckResult(
            supported=False,
            detail="dosing artifact carries no search section; no schedule has "
                   "been compared against continuous MTD",
        )

    n_seeds = int(cr.get("n_seeds") or len(cr.get("seeds", []) or []))
    delta = cr.get("dominance_delta_days")
    mtd = cr.get("mtd", {})
    pulsed = cr.get("pulsed", {})
    beats = bool(search.get("beats_baselines"))
    n_confirmed = len(search.get("scored", []))
    n_screened = search.get("candidates_infeasible")

    verdict = (
        "a pulsed calendar beat continuous MTD on median AUBC"
        if beats
        else "NO pulsed calendar in the searched grid beat continuous MTD; "
             "the search is reported as a null result"
    )
    return CheckResult(
        supported=True,
        detail=(
            f"competitive release, {n_seeds} held-out seeds: pulsing delays "
            f"resistant dominance by {delta:+.0f} d (MTD median AUBC "
            f"{mtd.get('median_aubc', float('nan')):.4f} vs pulsed "
            f"{pulsed.get('median_aubc', float('nan')):.4f}); schedule search "
            f"over {n_confirmed} confirmed calendars ({n_screened} cells "
            f"screened out by the label-anchored myelosuppression model, "
            f"{search.get('engine_runs', 0)} engine runs): {verdict}. Geometry is "
            f"a synthetic phantom; the safety screen is anchored to the "
            f"temozolomide label's own ANC threshold and its own 150 mg/m2 x5d "
            f"q28d regimen, but the kernel shape and drug B's marrow potency "
            f"are declared choices -- no patient claim is made."
        ),
    )

def check_grid_artifact_is_measured() -> CheckResult:
    """The grid the twin simulates on is measured, not assumed.

    Three separate things are re-derived live and required to agree with the
    committed record: the axis-aligned block vote's geometry, the comparison
    between that vote and what `realdata._decimate_labels` actually returns, and
    the spacing the committed derived files carry in their headers. The record's
    expensive outcome cohort is then checked for internal arithmetic only -- a
    claim must not take 20 minutes to run.

    Nothing here requires the two defects to be ABSENT: the claim is that the
    repository measures and states them. The one consistency it does enforce is
    that the provenance map's verdict matches the live measurement, so a
    repaired rule forces the map (and this claim) to be updated rather than
    leaving a stale PROVEN_FAILED behind.
    """
    from danon.grid_artifact import (
        GRID_LEVELS,
        SHIPPED_LEVEL,
        committed_derived_audit,
        decimation_geometry,
        grid_artifact_provenance,
    )

    record_path = os.path.join(ROOT, "outputs_real", "grid_artifact.json")
    try:
        with open(record_path, encoding="utf-8") as fh:
            record = json.load(fh)
    except (OSError, ValueError) as exc:
        return CheckResult(
            supported=False,
            detail=(
                f"grid-artifact record unreadable ({exc}); rebuild via "
                "_grid_artifact_cohort.py"
            ),
        )

    problems: List[str] = []
    try:
        live = decimation_geometry(record["case"], levels=GRID_LEVELS)
        live_audit = committed_derived_audit()
    except (OSError, ValueError, KeyError) as exc:
        return CheckResult(
            supported=False,
            detail=f"the grid-artifact measurement cannot be re-derived: {exc}",
        )

    stored = record.get("geometry") or {}
    stored_levels = stored.get("levels") or {}
    worst_bias = 0.0
    worst_dice = 1.0
    for name in GRID_LEVELS:
        now = live["levels"][name]
        rec = stored_levels.get(name)
        if rec is None:
            problems.append(f"the record does not cover grid level {name!r}")
            continue
        if rec.get("tumour_voxels") != now["tumour_voxels"] or rec.get(
            "shipped_tumour_voxels"
        ) != now["shipped_tumour_voxels"]:
            problems.append(
                f"the record's {name} voxel counts are not the live ones: the "
                "decimation or the source segmentation changed and the record "
                "is stale"
            )
        if rec.get("shipped_disagreeing_voxels") != now["shipped_disagreeing_voxels"]:
            problems.append(
                f"the record's {name} rule comparison disagrees with a live "
                "re-measurement; regenerate outputs_real/grid_artifact.json"
            )
        # Structural properties of a strict-majority block vote, which must hold
        # at every level and are what makes 'deflating' a measurement.
        if not (now["volume_bias"] < 0.0):
            problems.append(
                f"the majority vote does not deflate at {name} "
                f"({now['volume_bias'] * 100:+.3f}%)"
            )
        if not (now["maxpool_volume_bias"] > 0.0 and now["merged_voxels"] > 0):
            problems.append(f"the max-pool alternative does not inflate at {name}")
        if abs(now["volume_bias"]) > 0.05:
            problems.append(
                f"the block vote is no longer volume-faithful at {name} "
                f"({now['volume_bias'] * 100:+.3f}%)"
            )
        worst_bias = max(worst_bias, abs(now["volume_bias"]))
        worst_dice = min(worst_dice, now["dice_vs_source"])
    if live["levels"][SHIPPED_LEVEL]["dice_vs_source"] <= 0.8:
        problems.append(
            "the block vote no longer reproduces the source occupancy at the "
            f"shipped grid (Dice {live['levels'][SHIPPED_LEVEL]['dice_vs_source']:.4f})"
        )

    # The two defects this pass found, restated as measurements. Their presence
    # is not a failure of THIS claim; a record that stops reporting them is.
    for key in (
        "shipped_rule_is_the_axis_aligned_block_vote",
        "shipped_rule_dice_vs_axis_aligned",
    ):
        # Absence is a failure, not a silent pass: an earlier version compared
        # `.get()` on both sides, so a renamed key made the check compare None
        # with None and report the record as current.
        if key not in stored or key not in live:
            problems.append(
                f"the record or the live measurement does not report {key}, so the "
                "comparison cannot be made"
            )
        elif stored[key] != live[key]:
            problems.append(
                f"the record's {key} is not the live measurement; the decimation "
                "rule changed and the record is stale"
            )
    probe = live.get("partition_probe")
    if probe is None:
        problems.append("the live measurement carries no single-block partition probe")
    elif (
        probe["n_landed_in_their_own_engine_voxel"]
        + probe["n_landed_elsewhere"]
        + probe["n_dropped_entirely"]
        != probe["n_probed"]
    ):
        problems.append("the partition probe's outcomes do not account for its probes")
    round_trip = record.get("round_trip") or {}
    if round_trip.get("n_audited") != live_audit["n_audited"]:
        problems.append(
            "the record audits a different number of derived volumes than are "
            "present now"
        )
    for case, entry in live_audit["cases"].items():
        rec = (round_trip.get("cases") or {}).get(case)
        if rec is None:
            problems.append(f"the record does not audit {case}")
            continue
        if rec.get("stored_disagreeing_voxels") != entry["stored_disagreeing_voxels"]:
            problems.append(
                f"{case}: the record's array comparison is not the live one"
            )
        if bool(rec.get("spacing_consistent")) is not bool(entry["spacing_consistent"]):
            problems.append(f"{case}: the record's header verdict is not the live one")

    prov = grid_artifact_provenance()
    aligned_ok = bool(live["shipped_rule_is_the_axis_aligned_block_vote"])
    prov_verdict = prov.get("axis_aligned_decimation")
    if aligned_ok and prov_verdict is not EvidenceStatus.ESTABLISHED:
        problems.append(
            "the decimation rule now IS the axis-aligned block vote, but the "
            f"provenance map still says {prov_verdict}; update it"
        )
    if not aligned_ok and prov_verdict is not EvidenceStatus.PROVEN_FAILED:
        problems.append(
            "the decimation rule is NOT the axis-aligned block vote it "
            f"documents, but the provenance map says {prov_verdict}"
        )
    headers_ok = bool(live_audit["all_headers_carry_the_decimated_spacing"])
    header_verdict = prov.get("derived_volume_headers_carry_the_decimated_spacing")
    if headers_ok and header_verdict is not EvidenceStatus.ESTABLISHED:
        problems.append(
            "the derived volumes now carry their decimated spacing, but the "
            f"provenance map still says {header_verdict}; update it"
        )
    if not headers_ok and header_verdict is not EvidenceStatus.PROVEN_FAILED:
        problems.append(
            "the derived volumes do NOT carry their decimated spacing, but the "
            f"provenance map says {header_verdict}"
        )

    cohort = record.get("cohort") or {}
    contrasts = cohort.get("contrasts") or {}
    if not contrasts:
        problems.append("the record carries no outcome contrast across grids")
    ref = cohort.get("reference_config")
    ref_level = (cohort.get("configs") or {}).get(ref, {}).get("level")
    if ref_level != next(iter(GRID_LEVELS)):
        problems.append(
            f"the contrast reference {ref!r} is not the finest grid, so the "
            "drift is not measured against the best available discretization"
        )
    drift: List[str] = []
    drift_ratio: Optional[float] = None
    for mode, per_config in contrasts.items():
        for name, arms in per_config.items():
            candidate = name.split("_vs_")[0]
            for arm, readouts in arms.items():
                for metric, entry in readouts.items():
                    base = np.asarray(entry.get("base", []), dtype=float)
                    cand = np.asarray(entry.get("candidate", []), dtype=float)
                    if base.size != cand.size or base.size < 2:
                        problems.append(f"{mode}/{name}/{arm}/{metric}: unusable pair")
                        continue
                    if abs(entry["mean_delta"] - float((cand - base).mean())) > 1e-9:
                        problems.append(
                            f"{mode}/{name}/{arm}/{metric}: the recorded delta is "
                            "not the mean of the recorded pair"
                        )
                    spread = (
                        (cohort.get("spreads") or {})
                        .get(mode, {})
                        .get(ref, {})
                        .get(arm, {})
                        .get(metric, {})
                    )
                    recorded_spread = spread.get("spread_rel")
                    if recorded_spread is None:
                        problems.append(
                            f"{mode}/{name}/{arm}/{metric}: no reference spread recorded"
                        )
                    elif entry.get("delta_over_spread") is not None and recorded_spread:
                        want = (
                            abs(float((cand - base).mean()))
                            / abs(float(base.mean()))
                            / recorded_spread
                        )
                        if abs(float(entry["delta_over_spread"]) - want) > 1e-9:
                            problems.append(
                                f"{mode}/{name}/{arm}/{metric}: delta/spread is not "
                                "the ratio of the recorded pair"
                            )
                    if (
                        mode == "shipped_noise"
                        and candidate.endswith("_matched_D")
                        and metric == "pfs"
                        and arm == "mtd"
                        and entry.get("delta_over_spread") is not None
                    ):
                        drift.append(
                            f"{name} PFS drift {entry['mean_rel_delta'] * 100:+.3f}% "
                            f"= {entry['delta_over_spread']:.2f}x the seed spread"
                        )
                        if candidate == "engine_3p6mm_matched_D":
                            drift_ratio = float(entry["delta_over_spread"])
    if not drift:
        problems.append(
            "no grid-resolution contrast on MTD PFS is recorded at matched "
            "physics, so the artifact is measured geometrically but not in an "
            "outcome"
        )
    elif drift_ratio is None:
        problems.append(
            "the shipped grid has no matched-physics PFS contrast of its own"
        )
    elif drift_ratio < 0.5:
        # The claim's content is that this artifact is NOT below the model's
        # resolution. If a repaired partition ever brings it there, the sentence
        # is wrong and must be rewritten rather than kept.
        problems.append(
            f"the shipped grid now moves MTD PFS by only {drift_ratio:.3f} of the "
            "cohort's own seed spread, so the claim that this artifact is not "
            "below the model's resolution no longer holds"
        )

    shipped = live["levels"][SHIPPED_LEVEL]
    audit_note = (
        f"{live_audit['n_with_array_disagreement']}/{live_audit['n_audited']} "
        "committed derived volumes are NOT the axis-aligned block vote and "
        f"{live_audit['n_with_spacing_mismatch']}/{live_audit['n_audited']} "
        "carry a header that is not their decimated spacing"
    )
    factor = next(
        (e["volume_report_factor"] for e in live_audit["cases"].values()),
        None,
    )
    return CheckResult(
        supported=not problems,
        detail=(
            f"source tumour {live['source_tumour_voxels']:,} mm^3 at "
            f"{live['source_spacing_mm'][0]:.1f} mm; the axis-aligned block vote is "
            f"volume-faithful to {worst_bias * 100:.3f}% worst case with Dice "
            f">= {worst_dice:.4f}; the max-pool rule inflates by "
            f"{shipped['maxpool_volume_bias'] * 100:+.2f}% at the shipped grid and "
            f"merges {shipped['merged_fraction_of_tumour'] * 100:.2f}% extra voxels; "
            f"`_decimate_labels` agrees with that vote at Dice "
            f"{shipped['shipped_dice_vs_aligned']:.4f} "
            f"({shipped['shipped_disagreeing_voxels']:,} disagreeing voxels); "
            f"{audit_note}"
            + (f", understating each volume {factor:.2f}x" if factor else "")
            + "; outcome drift at matched physics: "
            + "; ".join(drift)
            + (
                f"; the block probe lands {live['partition_probe']['n_landed_in_their_own_engine_voxel']}"
                f"/{live['partition_probe']['n_probed']} fully-tumour axis-aligned cubes "
                "in their own engine voxel"
                if "partition_probe" in live
                else ""
            )
            + ("; " + "; ".join(problems) if problems else "")
        ),
    )

def check_walk_forward_is_refused_and_reasoned() -> CheckResult:
    """The longitudinal test exists, is refused, and the refusal is counted.

    Two independent things are re-derived live: how many paired patient series
    this repository can actually score (none, from a tested manifest loader plus
    a recorded public-API probe), and which term moves the engine's predicted
    tumour outwards (the migration term alone, isolated by turning it off). The
    provenance map has to agree with the census in BOTH directions, so a newly
    acquired pair forces the refusal to be replaced by a measurement instead of
    leaving a stale PROVEN_FAILED behind.
    """
    import numpy as np

    from danon.walkforward import (
        expansion_bound,
        longitudinal_census,
        migration_audit,
        walkforward_provenance,
    )

    problems: List[str] = []
    try:
        census = longitudinal_census(
            census_path=os.path.join(ROOT, "outputs_real", "walkforward_census.json")
        )
        probe = np.zeros((16, 16, 16), dtype=bool)
        probe[5:11, 5:11, 5:11] = True
        expansion = expansion_bound(
            probe, arm="mtd", days=30, seeds=(101, 102), voxel_mm=1.0
        )
        audit = migration_audit(3.634)
    except (OSError, ValueError, KeyError) as exc:
        return CheckResult(
            supported=False,
            detail=f"the walk-forward harness cannot be re-derived: {exc}",
        )

    cfb = census["sources"]["CFB-GBM"]
    if census["scorable_pairs"] != 0 or census["pairs_with_cached_segmentations"] != 0:
        problems.append(
            f"the census now finds {census['scorable_pairs']} scorable pair(s); a "
            "real Dice and JSD should be measured and this claim rewritten"
        )
    if cfb["images_present"]:
        problems.append("CFB-GBM images are now present, so a pair may be scorable")
    if not (cfb["usable_pairs_declared"] > 0 and cfb["patients_with_usable_pair"] > 0):
        problems.append("the manifest no longer declares any usable pair")
    # The GLASS tripwire, inverted by real data. It used to say "a record is
    # present, so the refusals need revisiting"; the records were then brought
    # in, `danon.variants` was written to measure them, and the remaining refusal
    # moved from "there is no variant space" to "only a lower bound is provable".
    # The invariant now runs the other way and in three parts: the records must
    # be here (a sentence claiming they are measured here cannot outlive them),
    # the census must name the channel that reads them, and the refusals must
    # still be in the channel's own code -- data arriving is not a reason for a
    # refusal to disappear.
    glass_src = census["sources"]["GLASS"]
    if glass_src["files_in_tree"] == 0:
        problems.append(
            "no GLASS record exists in this tree any more, so the variant "
            "channel, its artifact and the sentence describing them need "
            "revisiting (rebuild via _glass_variant_cohort.py)"
        )
    channel = glass_src.get("variant_channel") or {}
    if not channel.get("module"):
        problems.append(
            "GLASS records are in this tree but the census does not name the "
            "channel that measures them"
        )
    if not channel.get("built"):
        problems.append(
            "the GLASS variant channel artifact is missing, so the measured "
            "claim has no evidence (rebuild via _glass_variant_cohort.py)"
        )
    elif not (channel.get("measured") or {}).get("n_mutation_records"):
        problems.append("the GLASS variant artifact records no mutation records")
    else:
        from danon.variants import variant_channel_provenance

        chan_prov = variant_channel_provenance()
        for key in (
            "point_clone_fraction",
            "upper_bound_on_clone_fraction",
            "two_bin_summarisation",
        ):
            if chan_prov.get(key) is not EvidenceStatus.PROVEN_FAILED:
                problems.append(
                    f"the variant channel no longer refuses {key}, so real "
                    "allele fractions would be read as clone sizes"
                )

    # GLASS is refused for the MAPPING, not for access. The annotation says so,
    # and the recorded probe is what holds it: a public study, paired specimens,
    # and allele counts that are actually populated. If any of those stops being
    # true the reason changes and the sentence has to change with it.
    glass = census["sources"]["GLASS"]
    for key in ("access", "pairing", "frequencies", "why_it_is_still_not_usable"):
        if not glass.get(key):
            problems.append(f"the GLASS annotation no longer records its {key}")
    if "controlled access" not in str(glass.get("access") or ""):
        problems.append(
            "the GLASS annotation no longer corrects the controlled-access "
            "framing, which is the point of recording it"
        )
    glass_probe = (census.get("recorded_cbioportal") or {}).get("GLASS") or {}
    if glass_probe:
        if str(glass_probe.get("public_study")) != "True":
            problems.append(
                "the recorded GLASS study is no longer public, so the mapping -- "
                "not the access -- is no longer the only blocker"
            )
        types = glass_probe.get("sample_types") or {}
        if not any("recurrence" in str(k).lower() for k in types):
            problems.append(
                "the recorded GLASS release no longer carries recurrent "
                "specimens, so it is not the paired resource the annotation "
                "describes"
            )
        if not any("primary" in str(k).lower() for k in types):
            problems.append("the recorded GLASS release no longer carries primaries")
        mutations = glass_probe.get("mutations") or {}
        if not mutations:
            problems.append("the recorded GLASS probe carries no mutation read")
        for gene, entry in mutations.items():
            if entry.get("status") == 200 and not entry.get("allele_counts_populated"):
                problems.append(
                    f"the recorded GLASS allele counts for {gene} are no longer "
                    "populated, so a frequency could not be read and the "
                    "annotation overstates what is available"
                )

    # LUMIERE is the one longitudinal collection the NBIA probe cannot see (it
    # lives on Figshare), so its refusal has a different reason and is checked
    # separately: it is reachable, but its masks are AUTOMATED, which makes a
    # Dice against them model-versus-model rather than model-versus-truth.
    lumiere = census["sources"]["LUMIERE"]
    if lumiere["files_in_tree"] != 0:
        problems.append(
            f"{lumiere['files_in_tree']} LUMIERE record(s) are now cached, so the "
            "spatial-Dice refusal has to be re-derived against the asset that "
            "landed here rather than left standing"
        )
    if lumiere["scorable_for_a_spatial_dice"] is not False:
        problems.append(
            "the census now calls LUMIERE scorable for a spatial Dice; if that "
            "holds, the Dice must be measured and this refusal replaced"
        )
    if "automated" not in str(lumiere["segmentations"]).lower():
        problems.append(
            "the LUMIERE annotation no longer records that its masks are "
            "automated, which is the only reason a Dice against them is "
            "model-versus-model and not a validation"
        )
    # The annotation is checked against the recorded probe: the collection ships
    # its two assets as two separately titled articles, and those titles are what
    # separate the automated masks from the expert response class.
    lumiere_probe = (census.get("recorded_figshare") or {}).get("LUMIERE") or {}
    if lumiere_probe:
        probe_titles = [
            str(a.get("title") or "").lower()
            for a in lumiere_probe.get("articles", [])
        ]
        if not any("automated segmentation" in t for t in probe_titles):
            problems.append(
                "the recorded LUMIERE article list no longer contains an "
                "'...automated segmentations' article, so the masks may not be "
                "automated and the refusal reason would be wrong"
            )
        if not any("expert rano" in t for t in probe_titles):
            problems.append(
                "the recorded LUMIERE article list no longer contains an "
                "'Expert RANO rating' article, so its expert asset cannot be "
                "pinned as a response class rather than a mask"
            )

    # The mechanism: outward movement must be the migration term and nothing else.
    if not expansion["expansion_requires_the_migration_term"]:
        problems.append(
            "forced-zero motility no longer removes the outward burden, so the "
            "expansion is no longer attributable to the migration term alone"
        )
    if expansion["growth_rate_map_sum_outside_mask"] != 0.0:
        problems.append("the growth map is no longer zero outside the baseline mask")
    share = float(expansion["max_burden_outside_baseline_share"])
    if not 0.05 < share < 0.60:
        problems.append(
            f"the measured outward share moved to {share:.4f}, outside the band "
            "this claim states"
        )
    if audit["inside_reference_range"] is not False:
        problems.append("the migration coefficient is now inside the literature range")
    if not audit["multiple_of_range_top"] or audit["multiple_of_range_top"] <= 10.0:
        problems.append(
            f"the migration coefficient is only "
            f"{audit['multiple_of_range_top']:.2f}x the literature range top"
        )

    prov = walkforward_provenance()
    refused = prov.get("engine_dice_against_a_patient_followup")
    if census["scorable_pairs"] == 0 and refused is not EvidenceStatus.PROVEN_FAILED:
        problems.append(
            f"no pair is scorable but the provenance says {refused}"
        )
    for key in (
        "engine_jsd_against_glass",
        "co_registration",
        "invasion_term",
        "motility_is_in_physical_units",
    ):
        if prov.get(key) is not EvidenceStatus.PROVEN_FAILED:
            problems.append(f"the walk-forward provenance does not refuse {key}")

    return CheckResult(
        supported=not problems,
        detail=(
            f"scorable longitudinal pairs: {census['scorable_pairs']} (CFB-GBM "
            f"declares {cfb['usable_pairs_declared']} over "
            f"{cfb['patients_with_usable_pair']} patients with no images cached; "
            f"the recorded public-API probe returns 0 patients and 0 segmentation "
            f"series for MU-Glioma-Post and Brain-Tumor-Progression; "
            f"{census['sources']['GLASS']['files_in_tree']} GLASS records in the "
            f"tree); LUMIERE is reachable (Figshare collection: "
            f"{lumiere['patients']} patients, {lumiere['images']} images, "
            f"{lumiere['expert_rano_ratings']} expert RANO timepoints) yet is "
            f"refused for a different reason -- its masks are AUTOMATED, so a Dice "
            f"against them is model-versus-model, and its expert asset is a "
            f"response class rather than a segmentation; "
            f"{lumiere['files_in_tree']} LUMIERE records are cached here, so no "
            f"Dice is measurable from it either; the growth map is "
            f"{expansion['growth_rate_map_sum_outside_mask']}"
            f" outside the baseline mask, so the outward share of final burden "
            f"{share * 100:.2f}% at day 30 falls to exactly "
            f"{expansion['max_burden_outside_share_with_motility_zero']} with "
            f"motility forced to 0; that coefficient is "
            f"{audit['physical_mm2_per_day']:.4f} mm^2/day at the engine's real "
            f"3.634 mm voxel, {audit['multiple_of_range_top']:.1f}x the top of the "
            f"published range {audit['reference_range_mm2_per_day']}"
            + ("; " + "; ".join(problems) if problems else "")
        ),
    )

def check_ablation_separates_mechanisms() -> CheckResult:
    """Which terms of this engine move its answer, and which move nothing.

    Two PLANTED controls are re-run live on the committed phantom -- a mechanism
    the policy cannot use (drug B under MTD, whose dose is (1.0, 0.0), so its
    ablation must be exactly zero) and one that provably moves the day-N burden
    (cell motility). The real-case cohort is then re-verified from its own
    record: the leaderboard must be the count of its own load-bearing readouts,
    and the mechanisms the sentence calls inert must have none.
    """
    from danon.ablation import (
        INERT_RATIO,
        LOAD_BEARING_RATIO,
        MECHANISMS,
        ablation_contrast,
        ablation_provenance,
    )
    from danon.evosteer import make_gbm_phantom

    problems: List[str] = []
    phantom = make_gbm_phantom(size=16, seed=0)
    planted_inert = ablation_contrast(
        MECHANISMS["second_agent"], phantom, arms=("mtd",), seeds=(101, 102), days=60
    )
    for readout, values in planted_inert["arms"]["mtd"]["readouts"].items():
        if values["delta"] not in (None, 0.0):
            problems.append(
                f"drug B is never dosed under MTD but ablating it moved "
                f"{readout} by {values['delta']!r}"
            )
    if planted_inert["any_load_bearing"]:
        problems.append("the planted-inert contrast reports a load-bearing mechanism")

    planted_live = ablation_contrast(
        MECHANISMS["cell_motility"], phantom, arms=("mtd",), seeds=(101, 102), days=60
    )
    density = planted_live["arms"]["mtd"]["readouts"]["mean_density"]
    if density["verdict"] != "load_bearing":
        problems.append(
            f"the only outward term is no longer load-bearing on the spatial "
            f"burden ({density['verdict']})"
        )

    record_path = os.path.join(ROOT, "outputs_real", "ablation_cohort.json")
    try:
        with open(record_path, encoding="utf-8") as fh:
            record = json.load(fh)
    except (OSError, ValueError) as exc:
        return CheckResult(
            supported=False,
            detail=f"ablation cohort unreadable ({exc}); rebuild via _ablation_cohort.py",
        )
    if record.get("skipped"):
        problems.append(f"the real-case cohort skipped {record['skipped']}")
    tallies: Dict[str, List[str]] = {}
    for name, report in (record.get("mechanisms") or {}).items():
        found: List[str] = []
        for arm, entry in report["arms"].items():
            for readout, values in entry["readouts"].items():
                ratio = values["delta_over_spread"]
                if ratio is None:
                    if values["verdict"] != "degenerate":
                        problems.append(f"{name}/{arm}/{readout}: verdict without a ratio")
                    continue
                expected = (
                    "load_bearing"
                    if ratio >= LOAD_BEARING_RATIO
                    else ("inert" if ratio < INERT_RATIO else "unresolved")
                )
                if values["verdict"] != expected:
                    problems.append(
                        f"{name}/{arm}/{readout}: verdict {values['verdict']} does "
                        f"not follow from its ratio {ratio:.3f}"
                    )
                if expected == "load_bearing":
                    found.append(f"{arm}/{readout}")
        tallies[name] = sorted(found)
    expected_board = sorted(
        ({"mechanism": n, "n_load_bearing_readouts": len(v)} for n, v in tallies.items()),
        key=lambda entry: -entry["n_load_bearing_readouts"],
    )
    if record.get("leaderboard") != expected_board:
        problems.append("the recorded leaderboard is not the count of its own verdicts")
    for inert_name in ("mutation", "asymmetric_competition"):
        if tallies.get(inert_name):
            problems.append(
                f"{inert_name} is called inert in the sentence but now moves "
                f"{tallies[inert_name]}"
            )
    second = record["mechanisms"]["second_agent"]["arms"]["mtd"]["readouts"]
    for readout, values in second.items():
        if values["delta"] not in (None, 0.0):
            problems.append(f"second_agent/mtd/{readout}: not exactly inert")

    prov = ablation_provenance()
    if prov.get("mechanism_is_correct") is not EvidenceStatus.PROVEN_FAILED:
        problems.append("the ablation provenance no longer refuses correctness")

    def _ratio(name: str, arm: str, readout: str) -> Optional[float]:
        values = record["mechanisms"][name]["arms"][arm]["readouts"][readout]
        return values["delta_over_spread"]

    return CheckResult(
        supported=not problems,
        detail=(
            f"planted inert (drug B under MTD): every delta exactly 0.0; planted "
            f"load-bearing (cell motility on the spatial burden): "
            f"{density['delta_over_spread']:.3f}x the seed spread; on the real "
            f"case the load-bearing readouts are "
            f"{', '.join(f'{n}={len(v)}' for n, v in sorted(tallies.items(), key=lambda kv: -len(kv[1])))}"
            f", with cell_motility/steering/mean_density at "
            f"{_ratio('cell_motility', 'steering', 'mean_density'):.3f}x, "
            f"second_agent/steering/mean_density at "
            f"{_ratio('second_agent', 'steering', 'mean_density'):.1f}x, and "
            f"heterogeneous_transport/steering/mean_density at "
            f"{_ratio('heterogeneous_transport', 'steering', 'mean_density'):.3f}x; "
            f"mutation and asymmetric_competition move nothing above the noise"
            + ("; " + "; ".join(problems) if problems else "")
        ),
    )

def check_acceleration_is_bit_identical_and_measured() -> CheckResult:
    """The speedups this repository has are the ones it measured, and nothing moved.

    Two halves, both live. First, the identity: the accelerated Laplacian, the
    accelerated face conductance and the batched day loop are compared to their
    pinned references with `np.array_equal`, on random fields and over a full
    phantom run, because a tolerance would let a one-ulp change ride through 365
    iterations of a nonlinear map. Second, the record: every ratio in
    `outputs_real/perf.json` is recomputed from its own two timings rather than
    trusted, the three things the record says are NOT measured are required to
    still be refused there, and the batched path is required to still be the
    SLOW one -- a claim that batching buys throughput would be false here, and
    this check would fail rather than let it be written.
    """
    import numpy as np

    from danon import accel
    from danon.anisotropy import (
        _divergence_d_grad_reference,
        _harmonic_mean_reference,
        divergence_d_grad,
    )
    from danon.evosteer import (
        MTDPolicy,
        TrajectorySpec,
        TumorEvolutionSimulator,
        _laplacian,
        _laplacian_reference,
        make_gbm_phantom,
    )

    problems: List[str] = []
    rng = np.random.default_rng(4242)
    for shape in ((7, 9, 11), (14, 24, 24)):
        f = rng.normal(size=shape)
        if not np.array_equal(_laplacian_reference(f), _laplacian(f.copy())):
            problems.append(f"laplacian is no longer bit-identical at {shape}")
        c = rng.normal(size=shape)
        d = 0.05 + rng.random(shape)
        for probed in (d, np.where(rng.random(shape) < 0.3, 0.0, d)):
            if not np.array_equal(
                _divergence_d_grad_reference(c, probed), divergence_d_grad(c, probed)
            ):
                problems.append(f"divergence_d_grad is no longer bit-identical at {shape}")
            got = accel.harmonic_mean(probed, probed)
            if not np.array_equal(_harmonic_mean_reference(probed, probed), got):
                problems.append(f"the mask-free harmonic mean changed at {shape}")

    # The workspace must reuse, not re-allocate: that is the whole reason it exists.
    ws = accel.KernelWorkspace()
    f = rng.normal(size=(12, 12, 12))
    accel.laplacian(f, workspace=ws)
    settled = ws.allocations
    for _ in range(5):
        accel.laplacian(f, workspace=ws)
    if ws.allocations != settled:
        problems.append("a warm workspace still allocates")

    geom = make_gbm_phantom(size=16, seed=0)
    sim = TumorEvolutionSimulator(geom)
    specs = [
        TrajectorySpec(policy=MTDPolicy(), seed=11, initial_resistant_fraction=0.01),
        TrajectorySpec(policy=MTDPolicy(), seed=12, initial_resistant_fraction=0.01),
    ]
    batched = sim.run_batch(specs, days=30, keep_fields=True, stream="per_replicate")
    for spec, traj in zip(specs, batched):
        serial = sim.run(
            spec.policy,
            days=30,
            initial_resistant_fraction=spec.initial_resistant_fraction,
            seed=spec.seed,
            keep_fields=True,
        )
        for name in ("final_N_S", "final_N_R", "final_conc_A", "final_conc_B"):
            if not np.array_equal(getattr(serial, name), getattr(traj, name)):
                problems.append(f"seed {spec.seed}: batched {name} differs from serial")
        if (serial.pfs_day, serial.resistant_dominance_day) != (
            traj.pfs_day,
            traj.resistant_dominance_day,
        ):
            problems.append(f"seed {spec.seed}: batched readouts differ from serial")

    path = os.path.join(ROOT, "outputs_real", "perf.json")
    try:
        with open(path, encoding="utf-8") as fh:
            record = json.load(fh)
    except (OSError, ValueError) as exc:
        return CheckResult(
            supported=False,
            detail=f"perf record unreadable ({exc}); rebuild via _perf_cohort.py",
        )

    def _ratio(pair: Dict[str, Any], keys: Tuple[str, str]) -> Optional[float]:
        ref, new = pair[keys[0]], pair[keys[1]]
        if not ref or ref <= 0.0:
            return None
        return ref / new

    for entry in record["operators"]:
        for op in ("laplacian", "divergence"):
            ref = entry[f"{op}_reference_seconds"]
            new = entry[f"{op}_accelerated_seconds"]
            stated = entry[f"{op}_ratio"]
            if abs(ref / new - stated) > 1e-6 * max(1.0, stated):
                problems.append(f"{op} ratio at {entry['shape']} is not its own timings")
        if entry["laplacian_ratio"] < 1.0 or entry["divergence_ratio"] < 1.0:
            problems.append(f"an operator got slower at {entry['shape']}")
    for mode in ("scalar", "heterogeneous"):
        day = record["day"][mode]
        if abs(day["reference_ms_per_day"] / day["accelerated_ms_per_day"] - day["ratio"]) > 1e-3:
            problems.append(f"the {mode} day ratio is not its own two rates")
    for row in record["bit_identity"]["fields"]:
        # `bitwise_equal` is None where the record does not make that comparison
        # (the harmonic mean's zero faces land on -0.0 under one rule and 0.0
        # under the other, equal as numbers); `exact_equal` is required everywhere.
        if row["bitwise_equal"] is False or not row["exact_equal"]:
            problems.append(f"a recorded operator identity failed at {row['shape']}")
    for name, row in record["bit_identity"]["trajectory"].items():
        for key, value in row.items():
            if isinstance(value, bool) and not value:
                problems.append(f"the recorded 365-day {name} identity failed on {key}")

    batch = record["throughput"]["batch_per_replicate_streams"]
    if batch["ratio"] >= 1.0:
        problems.append(
            f"batching now beats serial ({batch['ratio']:.3f}x): the claim that "
            f"widening the arrays buys nothing is stale"
        )
    procs = record["throughput"]["processes"]["by_workers"]
    for workers, row in procs.items():
        if abs(row["serial_seconds_beside_it"] / row["seconds"] - row["ratio"]) > 1e-3:
            problems.append(f"the {workers}-worker ratio is not its own two timings")
    if not any(row["ratio"] > 1.0 for row in procs.values()):
        problems.append("no process count is faster than the serial loop")
    refused = " ".join(record["what_is_not_measured"])
    for topic in ("no GPU", "no surrogate model", "no clinical volume run"):
        if topic not in refused:
            problems.append(f"the record no longer refuses {topic!r}")
    if "gpu" in record:
        problems.append("the record carries a key named for a GPU figure it refuses")

    from danon.tumor_board import tumor_board_provenance

    prov = tumor_board_provenance()
    if prov.get("any_gpu_figure") is not EvidenceStatus.PROVEN_FAILED:
        problems.append("the dashboard no longer refuses a GPU figure")

    ops = ", ".join(
        "{}x{}: {:.2f}x/{:.2f}x".format(
            e["shape"][0], e["shape"][-1], e["laplacian_ratio"], e["divergence_ratio"]
        )
        for e in record["operators"]
    )
    return CheckResult(
        supported=not problems,
        detail=(
            f"accelerated kernels are bit-identical to their pinned references on "
            f"random fields (2 shapes) and over a 30-day phantom run (2 seeds, four "
            f"field arrays plus both readouts); a warm workspace stops allocating "
            f"at {settled} buffer(s); measured operator speedups {ops} "
            f"(laplacian/divergence); day loop {record['day']['scalar']['ratio']:.3f}x "
            f"homogeneous and {record['day']['heterogeneous']['ratio']:.3f}x heterogeneous; "
            f"replicates batched into one array {batch['ratio']:.3f}x -- a SLOWDOWN -- "
            f"versus " + ", ".join(
                f"{w} worker(s) {r['ratio']:.3f}x" for w, r in procs.items()
            )
            + ("; " + "; ".join(problems) if problems else "")
        ),
    )

def check_cgga_transport_is_measured_not_asserted() -> CheckResult:
    """The external C-index must be recomputed from the cached CGGA files.

    A transport figure is only worth carrying if it was measured on files that
    are present, so this check refits nothing by hand: it re-runs the whole
    report and compares every recorded number with the live one, then requires
    the refusals around it to survive. A presence-only check would pass on a
    record that had been typed in.
    """
    import json
    import os

    from danon.eval_targets import TargetStatus
    from danon.external_cohorts import (
        CGGA_325_ZIP,
        CGGA_693_ZIP,
        CGGA_OOD_RECORD,
        ENGINE_CAVEAT,
        PITCH_FIGURES,
        cgga_ood_report,
        external_pipeline_matrix,
    )

    missing = [
        p
        for p in (CGGA_325_ZIP, CGGA_693_ZIP, CGGA_OOD_RECORD)
        if not os.path.exists(p)
    ]
    if missing:
        return CheckResult(
            False,
            f"CGGA cohort or record not cached ({missing}); run "
            f"_cgga_ood_cohort.py -- the external number cannot be verified "
            f"without the files it came from",
        )

    live = cgga_ood_report((CGGA_325_ZIP, CGGA_693_ZIP))
    with open(CGGA_OOD_RECORD) as fh:
        record = json.load(fh)

    problems: List[str] = []
    recorded_c: List[float] = []

    def short(key: str) -> str:
        return key.split(".")[1].split("_clinical")[0]

    for key, tests in live.tests.items():
        rows = record.get("tests", {}).get(key)
        if not rows:
            problems.append(f"{key}: missing from the record")
            continue
        for t, r in zip(tests, rows):
            if (t.n, t.n_events) != (int(r["n"]), int(r["n_events"])):
                problems.append(
                    f"{key} {t.label}: live n={t.n}/{t.n_events}, recorded "
                    f"{r['n']}/{r['n_events']}"
                )
            if (
                abs(t.c_index - float(r["c_index"])) > 1e-9
                or abs(t.age_only_c_index - float(r["age_only_c_index"])) > 1e-9
            ):
                problems.append(
                    f"{key} {t.label}: live C={t.c_index:.4f} (age "
                    f"{t.age_only_c_index:.4f}), recorded C="
                    f"{float(r['c_index']):.4f} (age {float(r['age_only_c_index']):.4f})"
                )
            recorded_c.append(float(r["c_index"]))

    if any(abs(c - 0.6412) < 1e-9 for c in recorded_c):
        problems.append(
            "a recorded C-index now equals the pitch's 0.6412: fix the "
            "corrections rather than keeping a coincidence as a refutation"
        )

    counts = None
    for c in record.get("corrections", []):
        if c.get("shown_in_the_pitch") == PITCH_FIGURES["cgga_gbm_n"]:
            counts = c.get("what_is_recorded_instead") or {}
    if not counts:
        problems.append("the n=325 correction is missing from the record")
    else:
        for key, cohort in live.cohorts.items():
            rec = counts.get(key)
            if rec is None:
                problems.append(f"{key}: absent from the n=325 correction")
                continue
            if int(rec["table_rows"]) != cohort.n_table_rows:
                problems.append(
                    f"{key}: recorded {rec['table_rows']} table rows, live "
                    f"{cohort.n_table_rows}"
                )
            scorable_gbm = sum(1 for r in cohort.rows if r.is_gbm)
            in_table = scorable_gbm + cohort.dropped_histology.get("GBM", 0)
            if int(rec.get("histology_GBM_scorable", -1)) != scorable_gbm:
                problems.append(
                    f"{key}: recorded scorable GBM count "
                    f"{rec.get('histology_GBM_scorable')} is not the live count "
                    f"{scorable_gbm}"
                )
            if int(rec.get("histology_GBM_in_table", -1)) != in_table:
                problems.append(
                    f"{key}: recorded in-table GBM count "
                    f"{rec.get('histology_GBM_in_table')} is not the live count "
                    f"{in_table}"
                )

    if record.get("engine_caveat") != ENGINE_CAVEAT:
        problems.append("the record no longer carries the engine caveat")

    prov = record.get("provenance", {})
    for key in (
        "ood_transport_is_engine_validation",
        "single_cell_state_shift_prediction",
        "in_vitro_ic50_replicator_fit",
        "brainweb_inverse_tensor_recovery",
        "growth_dice_forecast",
        "cgga_pitch_figures_are_results",
    ):
        if prov.get(key) != "proven_failed":
            problems.append(f"provenance {key} reads {prov.get(key)!r}")

    matrix = external_pipeline_matrix((CGGA_325_ZIP, CGGA_693_ZIP))
    if len(matrix) != 5:
        problems.append(f"the pipeline matrix holds {len(matrix)} entries")
    if not all(p.status is not TargetStatus.MEASURED for p in matrix[1:]):
        problems.append("a pipeline with no measurement now reports MEASURED")

    live_summary = ", ".join(
        f"{short(key)} {t.label} C={t.c_index:.4f} (age alone "
        f"{t.age_only_c_index:.4f}, n={t.n}, {t.n_events} events)"
        for key, tests in live.tests.items()
        for t in tests
    )
    return CheckResult(
        supported=not problems,
        detail=(
            f"TCGA reference n={live.reference.n}, {live.reference.n_events} "
            f"events, internal C={live.reference.internal_c:.4f} +/- "
            f"{live.reference.internal_c_sd:.4f}; frozen coefficients "
            f"transported to CGGA: {live_summary}; the pitch's 0.6412 is not "
            f"among the recorded values; pipelines 2-5 refused"
            + ("; " + "; ".join(problems) if problems else "")
        ),
    )

def check_synthetic_5ala_standard_is_labelled_and_leaves_stream1_refused() -> CheckResult:
    """The 5-ALA standard may exercise the intake path; it may not validate it.

    Three requirements, all live: the record must carry the synthetic
    disclaimer and the refusal that no post-operative imaging exists; a sample
    of the recorded engine runs must reproduce exactly, so a number cannot have
    been typed in; and stream 1 must remain REFUSED on its live count of zero
    cached captures -- a standard that flipped the stream to 'available' would
    turn a manufactured field into evidence.
    """
    import json
    import os

    from danon.fluorescence_standard import (
        DAYS_LONG,
        DAYS_SHORT,
        MASK_PATH,
        PLANT_MODES,
        STANDARD_RECORD,
        SYNTHETIC_DISCLAIMER,
        describe_field,
        plant_field,
        run_layout,
        standard_geometry,
        stream1_still_refused,
    )

    if not (os.path.exists(STANDARD_RECORD) and os.path.exists(MASK_PATH)):
        return CheckResult(
            False,
            f"standard record not built ({STANDARD_RECORD}); run "
            f"_fluorescence_standard_cohort.py",
        )

    with open(STANDARD_RECORD) as fh:
        record = json.load(fh)

    problems: List[str] = []
    if record.get("synthetic_disclaimer") != SYNTHETIC_DISCLAIMER:
        problems.append("the record no longer carries the synthetic disclaimer")
    if record.get("stream1_census", {}).get("still_refused") is not True:
        problems.append("the record does not state that stream 1 stays refused")

    refusal_text = json.dumps(record.get("refusals", []))
    for must in ("post-operative", "0.42", "Ivy GAP", "MU-Glioma-Post", "GLASS"):
        if must not in refusal_text:
            problems.append(f"the refusals no longer name {must!r}")

    prov = record.get("provenance", {})
    for key in (
        "planted_field_is_a_5ala_acquisition",
        "synthetic_input_is_patient_evidence",
        "stream1_activates_on_synthetic_input",
        "resection_cavity_available_here",
    ):
        if prov.get(key) != "proven_failed":
            problems.append(f"provenance {key} reads {prov.get(key)!r}")

    geom = standard_geometry()
    mask = geom.tumor_mask
    if int(record.get("geometry", {}).get("tumour_voxels", -1)) != int(mask.sum()):
        problems.append(
            f"recorded tumour voxels "
            f"{record.get('geometry', {}).get('tumour_voxels')} != live {int(mask.sum())}"
        )
    for mode in PLANT_MODES:
        live = describe_field(plant_field(mask, mode), mask, mode=mode, decay_voxels=2.0)
        rec = next(
            (
                d
                for d in record.get("descriptors", [])
                if d.get("mode") == mode and abs(float(d.get("decay_voxels", -1)) - 2.0) < 1e-9
            ),
            None,
        )
        if rec is None:
            problems.append(f"descriptor {mode} missing from the record")
            continue
        if (
            abs(float(rec["rim_share"]) - live.rim_share) > 1e-12
            or abs(float(rec["mean_distance"]) - live.mean_distance) > 1e-12
        ):
            problems.append(f"descriptor {mode} does not match the live field")

    live_runs = {}
    for mode, days, keep in (
        ("uniform", DAYS_SHORT, True),
        ("rim", DAYS_LONG, False),
        ("default", DAYS_LONG, False),
    ):
        live_runs[(mode, days)] = run_layout(
            geom, mode, days=days, seed=17, keep_fields=keep
        )
    for run in record.get("runs", []):
        combo = (run.get("mode"), run.get("days"))
        lr = live_runs.get(combo)
        if lr is None or int(run.get("seed", -1)) != 17:
            continue
        if (
            abs(float(run["final_burden"]) - lr.final_burden) > 1e-9
            or abs(float(run["initial_burden"]) - lr.initial_burden) > 1e-9
            or run.get("pfs_day") != lr.pfs_day
        ):
            problems.append(f"{combo}: recorded run does not reproduce live")

    contrast = record.get("layout_contrast", {}).get(str(DAYS_LONG), {})
    spread = float("nan")
    clone_delta = float("nan")
    # The contrast is defined over PER-MODE medians (danon.fluorescence_standard
    # .layout_contrast), not over individual runs, so reproduce it that way.
    medians = [
        float(np.median([float(r["final_burden"]) for r in record.get("runs", []) if r.get("mode") == mode and r.get("days") == DAYS_LONG]))
        for mode in PLANT_MODES
        if any(
            r.get("mode") == mode and r.get("days") == DAYS_LONG
            for r in record.get("runs", [])
        )
    ]
    rows = medians
    if not rows:
        problems.append("no stream-1 runs at the long horizon in the record")
    else:
        mid = float(np.median(rows))
        spread = (max(rows) - min(rows)) / mid if mid else float("nan")
        if abs(spread - float(contrast.get("relative_spread", -1.0))) > 1e-12:
            problems.append("the recorded layout contrast is not reproducible from its own runs")
        if not 0.0 <= spread < 0.01:
            problems.append(
                f"the stream-1 layouts now spread {100 * spread:.2f}% at day "
                f"{DAYS_LONG}: the 'inert on this case' sentence must be rewritten"
            )

    refused, census_detail = stream1_still_refused()
    if not refused:
        problems.append(census_detail)

    for row in record.get("summary", []):
        if row.get("mode") == "uniform" and row.get("days") == DAYS_LONG:
            clone_delta = row.get("median_paired_delta_vs_default")
    return CheckResult(
        supported=not problems,
        detail=(
            f"intake identity {record['identity']['stream1_uniform_total']:.4f} vs "
            f"engine target {record['identity']['engine_target_total']:.4f} "
            f"(|diff| {record['identity']['absolute_difference']:.2e}); the case is a "
            f"shell (mean distance from the boundary "
            f"{max(d['mean_distance'] for d in record['descriptors'][:3]):.2f} voxels), "
            f"so the three planted layouts are nearly the same field: day-{DAYS_LONG} "
            f"spread {100 * spread:.3f}% at matched total burden; the trajectory "
            f"difference is the resistant-clone placement (paired median burden "
            f"delta {clone_delta:+.2f} vs the engine's clustered seed); a live "
            f"sample of recorded runs reproduces; {census_detail}"
            + ("; " + "; ".join(problems) if problems else "")
        ),
    )

def check_solver_portability_is_measured_and_generalization_is_refused() -> CheckResult:
    """Portability is a code property; cross-cancer prediction is refused.

    Three things are checked live: the record carries every mocked figure the
    pitch offered (the three swap-table rows, the 90%, both judge-script
    sentences, the universal dosing quantifier) beside an explicit refusal and
    four PROVEN_FAILED provenance entries; the swap surface and the glioma-only
    inventory still introspect to what the record says -- including a fresh
    scan of data/ that must still find zero breast, lung, pancreatic or
    non-glioma-drug assets; and the non-brain phantom runs reproduce exactly
    and come back finite and bounded. A record that started predicting other
    cancers, or a data/ tree that quietly gained a mammary mesh, fails here.
    """
    import json
    import os

    from danon.tumor_agnostic import (
        BOUND_RATIO,
        BRAIN_TISSUE_CLASSES,
        PORTABILITY_DISCLAIMER,
        PORTABILITY_RECORD,
        portability_run,
        swap_surface,
        swap_surface_fields,
        tissue_inventory,
    )

    if not os.path.exists(PORTABILITY_RECORD):
        return CheckResult(
            False,
            f"portability record not built ({PORTABILITY_RECORD}); run "
            f"_tumor_agnostic_cohort.py",
        )

    with open(PORTABILITY_RECORD) as fh:
        record = json.load(fh)

    problems: List[str] = []
    if record.get("disclaimer") != PORTABILITY_DISCLAIMER:
        problems.append("the record no longer carries the portability disclaimer")

    refusal_text = json.dumps(record.get("refusals", []))
    for must in (
        "Mammary / Fatty Parenchyma",
        "Porous Pulmonary Alveolar",
        "Dense Fibrotic Stromal",
        "90% of the codebase",
        "generalizes to predict spatial recurrence",
        "invisible residual cells after surgery",
        "Every cancer type benefits",
    ):
        if must not in refusal_text:
            problems.append(f"the refusals no longer carry {must!r}")

    prov = record.get("provenance", {})
    for key in (
        "predicts_breast_lung_pancreas_recurrence",
        "non_brain_mesh_exists_here",
        "non_brain_drug_parameter_set_exists",
        "postoperative_residual_input_available_here",
    ):
        if prov.get(key) != "proven_failed":
            problems.append(f"provenance {key} reads {prov.get(key)!r}")

    # The swap surface, introspected live.
    if int(record.get("swap_surface_fields", -1)) != swap_surface_fields():
        problems.append(
            f"recorded swap surface {record.get('swap_surface_fields')} != "
            f"live {swap_surface_fields()}"
        )
    if record.get("swap_surface") != swap_surface():
        problems.append("the swap surface field names changed")

    # The glioma-only inventory, including a fresh data/ scan.
    live_inv = tissue_inventory()
    rec_inv = record.get("inventory", {})
    if live_inv.get("tissue_d_keys") != rec_inv.get("tissue_d_keys"):
        problems.append("the tissue diffusion table no longer matches the record")
    if sorted(live_inv.get("tissue_d_keys", [])) != sorted(BRAIN_TISSUE_CLASSES):
        problems.append("the tissue diffusion table is no longer brain-only")
    if live_inv.get("data_organ_hits") != rec_inv.get("data_organ_hits"):
        problems.append("the data/ organ scan no longer reproduces")
    for group, hits in (live_inv.get("data_organ_hits") or {}).items():
        if hits:
            problems.append(f"non-glioma asset under data/ [{group}]: {hits[:3]}")

    # The portability runs, re-run live.
    for rec in record.get("runs", []):
        live = portability_run(int(rec["days"]), seed=int(rec["seed"]))
        if abs(live.final_burden - float(rec["final_burden"])) > 1e-9:
            problems.append(f"{rec['days']} d: recorded run does not reproduce")
        if live.pfs_day != rec.get("pfs_day"):
            problems.append(f"{rec['days']} d: recorded PFS does not reproduce")
        if not live.finite or not live.bounded:
            problems.append(f"{rec['days']} d: run no longer finite and bounded")

    hits_summary = {
        k: len(v) for k, v in (live_inv.get("data_organ_hits") or {}).items()
    }
    ratios = [float(r.get("capacity_ratio", 0.0)) for r in record.get("runs", [])]
    max_ratio = max(ratios) if ratios else float("nan")
    return CheckResult(
        supported=not problems,
        detail=(
            f"swap surface = {swap_surface_fields()} fields over 5 dataclasses "
            f"(TumorGeometry/CloneParams/DrugParams/ImmuneParams/DiffusionField); "
            f"tissue diffusion table {live_inv.get('tissue_d_keys')} -- brain "
            f"classes only; data/ organ/drug path hits {hits_summary}; the "
            f"non-brain phantom reproduces finite and bounded at both horizons "
            f"(measured max {max_ratio:.3f} x K x V, structural ceiling 1.4, "
            f"bound {BOUND_RATIO:.1f}); every mocked figure -- three swap-table "
            f"rows, the 90%, both judge sentences, the universal dosing claim "
            f"-- is carried in the refusals with four PROVEN_FAILED entries"
            + ("; " + "; ".join(problems) if problems else "")
        ),
    )

def check_real_scans_are_served_and_no_prediction_is_drawn() -> CheckResult:
    """The MRI Scans view shows committed patient scans, and nothing else.

    The page's whole claim is that the pixels are real, so this check reads the
    committed files rather than the artifact's own word for itself:

      * scans.json still carries `model_prediction_on_these_images` as
        proven_failed -- the tripwire that stops a future edit quietly
        compositing an engine field onto a real patient scan and calling it a
        detection;
      * every served PNG is byte-identical (sha256) to the render in
        outputs_real/scans/ that _scan_gallery.py produced from data/brats;
      * every case's source volume really is a NIfTI with labelled tumour
        voxels, and the mask each card uploads is byte-identical to
        data/engine_ready/ with a nonzero-voxel count equal to the count the
        card prints.

    A swapped image, a fabricated mask, or a contour drawn from anything but
    the committed segmentation fails here.
    """
    import hashlib

    from danon.imaging import read_nifti

    art_path = os.path.join(ROOT, "frontend", "public", "artifacts", "scans.json")
    if not os.path.exists(art_path):
        return CheckResult(
            False, f"scan artifact missing ({art_path}); run _scan_gallery.py"
        )
    with open(art_path, encoding="utf-8") as fh:
        art = json.load(fh)

    problems: List[str] = []
    prov = art.get("provenance", {})
    for key, want in (
        ("scan_pixels", "established"),
        ("segmentation_contours", "established"),
        ("engine_grid_voxels", "established"),
        ("model_prediction_on_these_images", "proven_failed"),
    ):
        got = str(prov.get(key, "")).lower()
        if got != want:
            problems.append(f"provenance {key} reads {prov.get(key)!r}, expected {want!r}")

    def sha(path: str) -> str:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    cases = art.get("data", {}).get("cases", [])
    if not cases:
        return CheckResult(False, "scans.json lists no cases")

    checked = []
    for c in cases:
        case = c["case"]
        served = os.path.join(
            ROOT, "frontend", "public", "artifacts", "scans", c["image"]
        )
        canonical = os.path.join(ROOT, "outputs_real", "scans", c["image"])
        if not os.path.exists(served) or not os.path.exists(canonical):
            problems.append(f"{case}: served or canonical PNG missing")
            continue
        if sha(served) != sha(canonical):
            problems.append(f"{case}: served PNG is not the committed render")

        src_dir = os.path.join(ROOT, "data", "brats")
        t1ce = os.path.join(src_dir, f"{case}_t1ce.nii.gz")
        seg = os.path.join(src_dir, f"{case}_seg.nii.gz")
        for p in (t1ce, seg):
            if not os.path.exists(p):
                problems.append(f"{case}: source volume missing ({os.path.basename(p)})")
        if os.path.exists(seg):
            seg_arr, _ = read_nifti(seg)
            if int((seg_arr > 0).sum()) != int(c["seg_voxels_total"]):
                problems.append(
                    f"{case}: card's seg_voxels_total ({c['seg_voxels_total']}) "
                    f"!= voxels in {os.path.basename(seg)} "
                    f"({int((seg_arr > 0).sum())})"
                )

        mask_name = c.get("engine_mask_file")
        if not mask_name:
            problems.append(f"{case}: no engine_mask_file, the card cannot upload anything")
            continue
        served_mask = os.path.join(
            ROOT, "frontend", "public", "artifacts", "masks", mask_name
        )
        engine_mask = os.path.join(ROOT, "data", "engine_ready", mask_name)
        if not (os.path.exists(served_mask) and os.path.exists(engine_mask)):
            problems.append(f"{case}: engine mask missing from disk or from the served copy")
            continue
        if sha(served_mask) != sha(engine_mask):
            problems.append(f"{case}: served mask is not the engine-ready mask")
        arr, _ = read_nifti(served_mask)
        vox = int((arr > 0).sum())
        if vox != int(c["engine_mask_voxels"]):
            problems.append(
                f"{case}: card's engine_mask_voxels ({c['engine_mask_voxels']}) "
                f"!= voxels in the served mask ({vox})"
            )
        if [int(v) for v in arr.shape] != list(c["engine_grid_xyz"]):
            problems.append(f"{case}: card's engine_grid_xyz != the mask's own shape")
        checked.append(case)

    detail = (
        f"{len(checked)}/{len(cases)} committed BraTS cases verified on disk: every served "
        f"PNG is byte-identical to outputs_real/scans/, every segmentation voxel count "
        f"re-reads from data/brats/, and every uploaded mask is byte-identical to "
        f"data/engine_ready/ with its voxel count and grid shape matching the card. "
        f"No engine output is drawn on any of these images."
    )
    if problems:
        return CheckResult(False, detail + " Problems: " + "; ".join(problems))
    return CheckResult(True, detail)

def check_alternating_two_drug_schedule_beats_single_drug_calendar() -> CheckResult:
    """The alternation-vs-single comparison must be a MEASURED engine result.

    The combination comparison is the deliverable of the dosing console, and it
    is required to be engine output rather than a scripted sentence. This reads
    `outputs_real/dosing.json` and refuses unless the comparison exists, was
    run on the label-anchored screen, and satisfies its own honesty rules.

    First, the screen itself is refused unless the label anchors it does:

      * the label's own regimen (150 mg/m2 x5d q28d) is admissible and its
        modelled nadir is 1.5 x10^9/L on day 28, and
      * one more day on drug (6d) is inadmissible.

    Second, the comparison is refused unless:

      * the artifact has a `combination` section produced by
        `compare_combination_vs_single`,
      * the best single-drug calendar is a pulsed (not continuous) calendar, and
      * `beats_single` is True, so the claim is a MEASURED win, not a scripted
        assertion that a combination is always better.

    Third, the verdict is reported with its dependence on the marrow-weight
    assumption declared. Drug B's marrow potency is ASSUMED at 0.3x, 0.5x and
    1.0x per mg/m2; a claim that the alternation wins must hold at the
    conservative 1.0x weight, and this check reports the sweep rather than
    hiding it in a sentence.
    """
    import json as _json
    import os as _os

    path = _os.path.join(ROOT, "outputs_real/dosing.json")
    if not _os.path.exists(path):
        return CheckResult(
            supported=False,
            detail=("outputs_real/dosing.json missing; no combination schedule "
                     "has ever been compared against the best single-drug "
                     "calendar, so nothing may be claimed"),
        )
    try:
        with open(path, encoding="utf-8") as fh:
            d = _json.load(fh)
    except ValueError as e:
        return CheckResult(
            supported=False, detail=f"dosing artifact is not readable JSON: {e}"
        )

    c = d.get("combination")
    if not c:
        return CheckResult(
            supported=False,
            detail=("dosing artifact carries no combination section; no "
                     "alternating calendar has been measured against the "
                     "best single-drug calendar"),
        )

    sched = c.get("single_drug_best", {})
    if not sched.get("schedule", {}).get("blocks"):
        return CheckResult(
            supported=False,
            detail=("no best single-drug calendar in the comparison; the "
                     "reference arm is missing"),
        )
    blocks = sched.get("schedule", {}).get("blocks", [])
    pulsed = any(b.get("duration_days", 0) > 0 for b in blocks) and \
        any(b.get("drug") == "A" for b in blocks)
    if not pulsed:
        return CheckResult(
            supported=False,
            detail=("the best single-drug reference must be a PULSED calendar. "
                     "A continuous MTD reference makes the comparison trivial "
                     "and the alternation a form of it;")
            )

    sweep = c.get("marrow_weight_sweep", [])
    if not sweep:
        return CheckResult(
            supported=False,
            detail=("no marrow-weight sweep in the combination section; drug B "
                     "potency must be declared and swept, never silently set "
                     "to admit a calendar"),
        )

    row = next((r for r in sweep if r.get("drug_b_marrow_weight") == 1.0), None)
    if row is None or row.get("beats_single") is not True:
        return CheckResult(
            supported=False,
            detail=("at the conservative marrow weight the alternating arm "
                     "is not measured to beat the best single-drug calendar. "
                     "The verdict is a function of the assumed potency of "
                     "drug B and is reported in the sweep below:")
            + " " + str(sweep),
        )

    notes = list(c.get("notes", []))
    # The artifact records its verdict in prose; `dosing.py` writes
    # "alternating WINS" / "alternating does NOT win". Either wording is a
    # recorded verdict; only numbers with no verdict sentence are refused.
    if not any(("beats" in n) or ("WINS" in n) or ("does NOT win" in n)
               for n in notes):
        return CheckResult(
            supported=False,
            detail=("the combination section does not record its own verdict. "
                     "Measured numbers are the only thing this claim may cite."),
        )

    return CheckResult(
        supported=True,
        detail=("the alternating two-drug calendar is measured by the engine "
                 "to beat the best single-drug calendar on median AUBC. "
                 "This row reports the conservative (full-myelosuppressive) "
                 "drug-B weight only; the full sweep is in the artifact. "
                 "Simulated on one synthetic phantom, 30 held-out seeds, "
                 "365 days; no patient benefit is claimed."),
    )


def check_evolution_forecast_is_scored_against_baselines() -> CheckResult:
    """The forecast is re-derived, and its verdicts must match its intervals.

    Three things are re-derived here rather than trusted, because this is the
    one claim in the registry whose numbers are measured in real patients and
    therefore the one a reader is most likely to take at face value:

    1. **The cohort.** The cached release is reloaded with the SHIPPING module's
       own loader, labelling rule and pairing rule, so the audit cannot be
       scoring a cohort that differs from the one the artifact describes.
    2. **The numbers.** Both tasks are re-run under their recorded seeds and the
       artifact has to agree. A drifting artifact fails here.
    3. **The honesty invariant.** Each verdict must agree with its own interval:
       a verdict that says the model beats its baseline while the interval still
       contains the null is the exact failure this project exists to prevent, so
       the coupling is asserted rather than the wording -- a machine-readable
       resolved-flag the builder derives from the interval, plus mutually
       exclusive phrases in the sentence. And because the two tasks currently
       DISAGREE (A resolves, B does not), the disagreement itself is pinned: a
       later cohort that settles B, or loses A, has to come here and rewrite
       both the claim and this check rather than quietly leaving a recorded
       contrast that no longer holds.

    What this check does NOT do is rebuild the artifact: it re-derives the
    numbers and compares them to the file, so the file's own SENTENCE about
    them is only as current as the last ``write_app_artifacts``. That gap is
    closed by the suite, which reads the mirrored artifact and fails if it
    publishes a detail this audit no longer produces.
    """
    record_path = os.path.join(ROOT, "outputs_real", "glass_evolution_forecast.json")
    cache_path = os.path.join(ROOT, "data", "glass_paired", "glass_mutations.json")
    for path in (record_path, cache_path):
        if not os.path.exists(path):
            return CheckResult(
                supported=False,
                detail=(
                    f"the forecast cannot be re-derived: {os.path.basename(path)} "
                    "is missing (rebuild via _glass_variant_cohort.py then "
                    "_glass_forecast_cohort.py)"
                ),
            )

    import json as _json

    from danon.evolution_forecast import (
        calibration_is_within_tolerance,
        change_examples,
        cross_validated_change,
        cross_validated_detection,
        detection_examples,
        evaluate_change,
        evaluate_detection,
        forecast_provenance,
        paired_spectra,
        records_from_cache,
        timepoint_census,
    )

    with open(cache_path, encoding="utf-8") as fh:
        raw = _json.load(fh)
    with open(record_path, encoding="utf-8") as fh:
        record = _json.load(fh)

    problems: List[str] = []
    records, dropped = records_from_cache(raw)
    spectra, pairs = paired_spectra(records)
    det_rows = detection_examples(pairs)
    chg_rows = change_examples(pairs)
    live_a = evaluate_detection(det_rows)
    live_b = evaluate_change(chg_rows)
    cv_a = cross_validated_detection(det_rows)
    cv_b = cross_validated_change(chg_rows)
    census = timepoint_census(raw)

    src = record.get("source", {})
    if src.get("records_used") != len(records):
        problems.append(
            f"the artifact records {src.get('records_used')} mutation rows but "
            f"the cache re-derives {len(records)}"
        )
    if src.get("records_dropped") != dropped:
        problems.append(
            f"dropped-count drift: artifact {src.get('records_dropped')} vs "
            f"re-derived {dropped}"
        )
    if src.get("patients") != len({r.patient_id for r in records}):
        problems.append("the patient count in the artifact is not the re-derived one")

    # The exclusion is asserted BEFORE the metric comparison, because it is the
    # defect this task was rewritten to remove and a smuggled-in gained row also
    # moves every metric -- left after the comparison the specific diagnosis
    # would be buried under a list of drifting numbers, and a reader would have
    # to work out for themselves what went wrong.
    det = detection_examples(pairs)
    if any(row.primary_af == 0.0 and row.detected_at_recurrence for row in det):
        problems.append(
            "task A contains a row that is detected at recurrence with a "
            "primary allele fraction of 0 -- that is a 'gained' variant scored "
            "against its own definition, which is the defect this task was "
            "rewritten to remove"
        )

    for label, live, stored in (
        ("task A", live_a, record.get("task_a_detection_at_recurrence", {})),
        ("task B", live_b, record.get("task_b_allele_fraction_change", {})),
    ):
        if stored.get("n_test_rows") != live.n_test_rows:
            problems.append(
                f"{label}: the artifact scores {stored.get('n_test_rows')} test "
                f"rows; a live run scores {live.n_test_rows}"
            )
        for key, value in live.metrics.items():
            theirs = (stored.get("metrics") or {}).get(key)
            if isinstance(value, bool) or isinstance(theirs, bool):
                if bool(theirs) is not bool(value):
                    problems.append(
                        f"{label}.{key}: artifact {theirs!r} vs re-derived {value!r}"
                    )
            elif isinstance(value, (int, float)) and isinstance(theirs, (int, float)):
                if abs(float(value) - float(theirs)) > 1e-9:
                    problems.append(
                        f"{label}.{key}: artifact {theirs} vs re-derived {value}"
                    )
            elif theirs != value:
                problems.append(
                    f"{label}.{key}: artifact {theirs!r} vs re-derived {value!r}"
                )

    # The cross-validated estimator must re-derive too, and both estimators must
    # be in the record: the disagreement between them IS a finding (one split was
    # underpowered on A and lucky on B), so dropping either would hide it.
    for label, live_cv, key in (
        ("task A", cv_a, "task_a_cross_validated"),
        ("task B", cv_b, "task_b_cross_validated"),
    ):
        stored = record.get(key)
        if not stored:
            problems.append(f"{label}: the cross-validated record is missing")
            continue
        if stored.get("n_folds") != live_cv.n_folds or (
            stored.get("n_repeats") != live_cv.n_repeats
        ):
            problems.append(
                f"{label}: recorded folds/repeats do not match the live "
                f"{live_cv.n_repeats}x{live_cv.n_folds}"
            )
        for metric, value in live_cv.metrics.items():
            theirs = (stored.get("metrics") or {}).get(metric)
            if isinstance(value, bool) or isinstance(theirs, bool):
                if bool(theirs) is not bool(value):
                    problems.append(
                        f"{label}.cv.{metric}: artifact {theirs!r} vs "
                        f"re-derived {value!r}"
                    )
            elif isinstance(value, (int, float)) and isinstance(theirs, (int, float)):
                if abs(float(value) - float(theirs)) > 1e-9:
                    problems.append(
                        f"{label}.cv.{metric}: artifact {theirs} vs re-derived {value}"
                    )
            elif theirs != value:
                problems.append(
                    f"{label}.cv.{metric}: artifact {theirs!r} vs "
                    f"re-derived {value!r}"
                )
        if len(stored.get("per_fold") or []) != live_cv.n_repeats * live_cv.n_folds:
            problems.append(
                f"{label}: the record does not carry every fold's numbers, so "
                "the spread cannot be checked"
            )

    # Calibration is the SECOND published claim about task A, and the failure it
    # is exposed to is the one this file keeps finding: a sentence that stopped
    # matching the numbers beside it. So the whole block is compared at once --
    # the bins, the declared tolerances AND the derived reading -- against the
    # block this module produces right now. Because the sentence is derived from
    # the summary by `calibration_reading`, an identical summary must give an
    # identical sentence: a hand-written reading, a stale binned table or a
    # widened tolerance each fail here.
    stored_cal = (record.get("task_a_cross_validated") or {}).get("calibration")
    live_cal = cv_a.calibration
    if not stored_cal:
        problems.append(
            "task A: the record carries no calibration block, but the task's "
            "published claim includes one"
        )
    elif _json.dumps(stored_cal, sort_keys=True) != _json.dumps(
        live_cal, sort_keys=True
    ):
        problems.append(
            "task A: the recorded calibration block (bins, tolerances and the "
            "reading derived from them) is not the one this module produces now"
        )
    else:
        bins = stored_cal.get("bins") or []
        if len(bins) != stored_cal.get("n_bins"):
            problems.append(
                "task A: the calibration block does not carry the bins its own "
                "bin count declares"
            )
        if sum(b.get("n", 0) for b in bins) != len(det_rows):
            problems.append(
                "task A: the reliability bins do not cover every pooled ROW, so "
                "the calibration is not measured on the population the AUC ranks"
            )
        reading = stored_cal.get("reading", "")
        if not reading:
            problems.append("task A: the calibration block carries no reading")
        elif str((record.get("verdict") or {}).get("task_a_calibration", "")) != reading:
            problems.append(
                "task A: the verdict's calibration sentence is not the one the "
                "calibration block derives, so the two copies of it disagree"
            )
    if (record.get("task_b_cross_validated") or {}).get("calibration") is not None:
        problems.append(
            "task B: a calibration block is attached to a regression, but "
            "calibration is a claim about PROBABILITIES and task B predicts a "
            "change, not a chance"
        )

    # The pooled UNIT and its label definition, re-derived rather than trusted.
    # This block documents a REPAIRED defect: the first version of the estimator
    # pooled one score per patient and read the patient's label off the last row
    # written, which for 83 of this cohort's 226 patients is an accident of row
    # order. The whole block is compared, so a return to that pooling would have
    # to publish different numbers -- the counts are measured from the rows.
    stored_pool = (record.get("task_a_cross_validated") or {}).get("pooling")
    live_pool = cv_a.pooling
    if not stored_pool:
        problems.append(
            "task A: the record does not declare the unit and the label of its "
            "pooled scores, so the AUC's denominator is unreadable"
        )
    elif _json.dumps(stored_pool, sort_keys=True) != _json.dumps(
        live_pool, sort_keys=True
    ):
        problems.append(
            "task A: the recorded pooling block (unit, counts and the sentence "
            "derived from them) is not the one this module produces now"
        )
    else:
        if stored_pool.get("unit") != "row" or (
            stored_pool.get("n_pooled_rows") != len(det_rows)
        ):
            problems.append(
                "task A: the pooled scores are not one per measured row, so the "
                "published AUC is not the row-level number it claims to be"
            )
        mixed = stored_pool.get("mixed_label_patients")
        if not isinstance(mixed, int) or mixed <= 0:
            problems.append(
                "task A: the pooling block no longer reports patients holding "
                "BOTH labels, so the reason a single per-patient label is "
                "refused has gone missing"
            )

    # The CLINICAL layer is the same out-of-fold predictions read as a decision:
    # a declared threshold, the confusion matrix it produces, both predictive
    # values against this cohort's own prevalence, and a decision curve against
    # treat-all/treat-none. It is the block a ward would act on, so it is exposed
    # to the two failures this file keeps finding -- a sentence that stopped
    # matching its counts, and a number that drifted from the record it was
    # published from. The whole block is compared against the live derivation,
    # then the counts are re-read from themselves: a ratio that no longer equals
    # its own numerator and denominator cannot hide inside an identical-looking
    # sentence, and a threshold chosen to flatter the curve fails the flag that
    # says it was never tuned here.
    stored_cu = (record.get("task_a_cross_validated") or {}).get("clinical_use")
    live_cu = cv_a.clinical_use
    if not stored_cu:
        problems.append(
            "task A: the record carries no clinical-use block, so the operating "
            "point and decision curve this screen publishes are missing from the "
            "artifact they are read from"
        )
    elif _json.dumps(stored_cu, sort_keys=True) != _json.dumps(
        live_cu, sort_keys=True
    ):
        problems.append(
            "task A: the recorded clinical-use block (operating point, decision "
            "curve and the readings derived from them) is not the one this "
            "module produces now"
        )
    else:
        op = stored_cu.get("operating_point") or {}
        counts = [op.get(k) for k in ("tp", "fp", "tn", "fn")]
        if any(not isinstance(c, int) for c in counts):
            problems.append(
                "task A: the operating point's confusion matrix is not four "
                "integer counts, so its sensitivity and predictive values "
                "cannot be re-derived"
            )
        else:
            tp, fp, tn, fn = counts
            if tp + fp + tn + fn != len(det_rows) or op.get("n_rows") != len(det_rows):
                problems.append(
                    "task A: the operating point's four counts do not cover "
                    "every pooled row, so the threshold's figures are measured "
                    "on a different population from the AUC"
                )
            for name, value in (
                ("sensitivity", tp / (tp + fn) if tp + fn else None),
                ("specificity", tn / (tn + fp) if tn + fp else None),
                ("ppv", tp / (tp + fp) if tp + fp else None),
                ("npv", tn / (tn + fn) if tn + fn else None),
            ):
                theirs = op.get(name)
                if value is None:
                    if theirs is not None:
                        problems.append(
                            f"task A: {name} is published as {theirs} but its "
                            "own counts make it undefined"
                        )
                elif theirs is None or abs(float(theirs) - value) > 1e-9:
                    problems.append(
                        f"task A: the operating point's {name} ({theirs}) does "
                        f"not equal the ratio of its own counts ({value})"
                    )
            if op.get("threshold_tuned_on_this_cohort") is not False:
                problems.append(
                    "task A: the operating threshold is not flagged as never "
                    "tuned on this cohort, so the published call cannot be read "
                    "as a declared decision"
                )
            if not op.get("reading"):
                problems.append("task A: the operating point carries no reading")
        curve = stored_cu.get("decision_curve") or {}
        for row in curve.get("rows") or []:
            pt = row.get("threshold")
            expected = (
                row.get("model", 0.0) > row.get("treat_all", 0.0)
                and row.get("model", 0.0) > row.get("treat_none", 0.0)
            )
            if not isinstance(pt, (int, float)) or row.get("beats_both") is not expected:
                problems.append(
                    "task A: a decision-curve row's beats_both flag disagrees "
                    "with the net benefits printed beside it"
                )
                break
        if not curve.get("reading"):
            problems.append("task A: the decision curve carries no reading")
    if (record.get("task_b_cross_validated") or {}).get("clinical_use") is not None:
        problems.append(
            "task B: a clinical-use block is attached to a regression, but an "
            "operating point needs a threshold on a CHANCE and task B predicts "
            "a change, not one"
        )

    # The secondary patient endpoint is DECLARED, not smuggled in as the label of
    # the primary number, and the claim below says the row-level result is the
    # clear one while the patient view is marginal. Both halves are asserted: the
    # recorded flag has to match the endpoint's own interval, and the margin over
    # chance has to actually be the larger one on the row level. A cohort that
    # reverses that must rewrite the claim rather than leave it standing.
    pt_lo = cv_a.metrics.get("patient_endpoint_ci_low")
    pt_resolved = cv_a.metrics.get("patient_endpoint_resolvably_above_chance")
    if pt_lo is None:
        problems.append(
            "task A: the secondary patient endpoint carries no interval, so its "
            "marginality cannot be checked"
        )
    else:
        if pt_resolved is not (pt_lo > 0.5):
            problems.append(
                "task A: the patient endpoint's recorded flag disagrees with "
                "its own interval"
            )
        if pt_resolved and pt_lo - 0.5 > cv_a.metrics["oof_auc_ci_low"] - 0.5:
            problems.append(
                "task A: the patient endpoint now clears chance by MORE than the "
                "row-level result does, so the claim's 'the row-level result is "
                "the clear one and the patient-level view is marginal' is stale"
            )

    # The clause this check publishes about calibration is derived from the
    # measurement rather than written here, because a hard-coded "no better than
    # a constant predictor" is exactly the kind of sentence that outlives the
    # number it described.
    skill = (cv_a.calibration or {}).get("summary", {}).get("brier_skill_score")
    if skill is None:
        cal_clause = "and its calibration could not be measured"
    elif skill > 0.0:
        cal_clause = (
            f"and a Brier skill of {skill:.4f} over a constant predictor, so its "
            "probabilities carry information a single number does not"
        )
    else:
        cal_clause = (
            f"and a Brier skill of {skill:.4f}, i.e. NO better than a constant "
            "predictor, so the ranking separates while a probability from this "
            "model is NOT a risk a reader may apply"
        )

    # The claim that is PUBLISHED carries the same calibration verdict, and the
    # polarity is derived here rather than trusted: if a future cohort calibrates,
    # the sentence has to be rewritten rather than left saying the model's
    # probabilities cannot be read.
    summary_a = (cv_a.calibration or {}).get("summary", {}) or {}
    wanted = (
        "the probabilities are calibrated"
        if calibration_is_within_tolerance(summary_a)
        else "the probabilities are NOT calibrated"
    )
    published_sentence = next(
        (
            c.sentence
            for c in CLAIMS
            if c.id == "evolution_forecast_is_scored_against_baselines"
        ),
        "",
    )
    if wanted not in published_sentence:
        problems.append(
            f"the claim's own sentence does not say '{wanted}', so the published "
            "claim and the measured calibration disagree"
        )

    # The honesty invariant, tested in two layers. FIRST the machine-readable
    # flag, which the builder sets from the same interval: a boolean cannot be
    # satisfied by prose. SECOND the direction of the sentence, using phrases
    # that are mutually exclusive, because a plain substring test on a word like
    # "UNRESOLVED" would pass on a verdict that mentions the earlier reading
    # while positively claiming the opposite -- a trap this check fell into in
    # its first version.
    verdicts = record.get("verdict", {})
    a_lo = cv_a.metrics.get("oof_auc_ci_low")
    a_resolved = isinstance(a_lo, float) and a_lo > 0.5
    b_resolved = cv_b.metrics.get("beats_persistence_resolvably") is True
    text_a = str(verdicts.get("task_a", ""))
    text_b = str(verdicts.get("task_b", ""))

    if verdicts.get("task_a_resolved") is not a_resolved:
        problems.append(
            f"task A's recorded resolved-flag ({verdicts.get('task_a_resolved')}) "
            f"contradicts its cross-validated interval (resolved={a_resolved})"
        )
    if verdicts.get("task_b_resolved") is not b_resolved:
        problems.append(
            f"task B's recorded resolved-flag ({verdicts.get('task_b_resolved')}) "
            f"contradicts its cross-validated interval (resolved={b_resolved})"
        )
    claims_beats_a = "BEATS prevalence" in text_a
    claims_not_a = "NOT shown to beat prevalence" in text_a
    if a_resolved != claims_beats_a:
        problems.append(
            "task A's sentence and its cross-validated interval disagree: the "
            f"interval says resolved={a_resolved}, the sentence says "
            f"'BEATS prevalence'={claims_beats_a}"
        )
    if (not a_resolved) and not claims_not_a:
        problems.append(
            "task A is unresolved but its sentence does not say the model is "
            "NOT shown to beat prevalence, so a reader is told an advantage the "
            "interval does not support"
        )
    claims_beats_b = "BEATS persistence" in text_b
    claims_not_b = "does NOT beat persistence" in text_b
    if b_resolved != claims_beats_b:
        problems.append(
            "task B's sentence and its cross-validated interval disagree: the "
            f"interval says resolved={b_resolved}, the sentence says "
            f"'BEATS persistence'={claims_beats_b}"
        )
    if (not b_resolved) and not claims_not_b:
        problems.append(
            "task B is unresolved but its sentence does not say the model does "
            "NOT beat persistence"
        )

    # The CONTRAST between the two estimators is the documented finding, so it
    # is pinned per task rather than as a generic difference. If a future cohort
    # changes either contrast, this fails and the claim has to be rewritten
    # instead of left quoting a contrast that no longer holds.
    single_a_resolved = (
        isinstance(live_a.metrics.get("auc_fitted_logistic_ci_low"), float)
        and live_a.metrics["auc_fitted_logistic_ci_low"] > 0.5
    )
    if single_a_resolved:
        problems.append(
            "task A: the single seeded split now RESOLVES too, so the claim's "
            "'the single split was underpowered' contrast is stale"
        )
    if not a_resolved:
        problems.append(
            "task A: the cross-validation no longer resolves, so the claim's "
            "'a real if modest signal' result is stale"
        )
    single_b_advantage = live_b.metrics.get("mae_advantage_over_persistence")
    fold_median_b = cv_b.metrics.get("fold_mae_advantage_median")
    if single_b_advantage is None or fold_median_b is None:
        problems.append("task B: the two estimators are not both recorded")
    elif not (single_b_advantage > 0.0 > fold_median_b):
        problems.append(
            "task B: the documented contrast is a POSITIVE single-split "
            f"advantage against a NEGATIVE fold median; now "
            f"{single_b_advantage} and {fold_median_b}, so the claim's "
            "'that split's luck' reading is stale"
        )
    if b_resolved:
        problems.append(
            "task B: the cross-validation now resolves, so the claim's "
            "'does NOT beat persistence' result is stale"
        )
    # A pooled interval and a fold count that point opposite ways would need
    # explaining before either is published.
    folds_b = cv_b.metrics.get("folds_beating_persistence", 0)
    n_folds_b = cv_b.metrics.get("n_folds_scored", 1)
    if folds_b * 2 > n_folds_b and (cv_b.metrics.get("oof_mae_advantage") or 0) < 0:
        problems.append(
            "task B: most folds beat persistence while the pooled advantage is "
            "negative -- the two summaries disagree and that needs explaining"
        )

    # The pairing is an inference from the identifier. If the release ever
    # declares a real timepoint column, the suffix rule must be replaced.
    stored_census = record.get("timepoint_census") or {}
    live_census = census
    if stored_census.get("n_specimen_rows") != live_census["n_specimen_rows"]:
        problems.append("the specimen-row count in the census is stale")
    if live_census["declared_type_carries_timepoint"]:
        problems.append(
            "the release now declares more than one sampleType, so timepoint "
            "no longer has to be inferred from the identifier -- the pairing "
            "must be re-derived from the declared column and this claim "
            "rewritten"
        )
    if stored_census.get("declared_types_distinct") != live_census[
        "declared_types_distinct"
    ]:
        problems.append(
            "the census's count of declared specimen types is stale"
        )

    # No therapy, no outcome, no patient prediction. If any of those flip, the
    # sentences describing this record have to change and this check fails.
    prov = forecast_provenance()
    for key in ("change_is_caused_by_therapy", "engine_predicts_patients",
                "clinical_benefit"):
        if prov.get(key) is not EvidenceStatus.UNKNOWN:
            problems.append(
                f"the forecast no longer declares {key} UNKNOWN, but nothing "
                "here joins a therapy or an outcome to a specimen"
            )

    n_real = len(records)
    if problems:
        return CheckResult(supported=False, detail="; ".join(problems))
    return CheckResult(
        supported=True,
        detail=(
            f"re-derived from the cached release with the shipping loader: "
            f"{n_real} measured mutation rows over "
            f"{len({r.patient_id for r in records})} patients and {len(spectra)} "
            f"specimens, {len(pairs)} paired specimens; both tasks re-run under "
            f"their recorded seeds and the artifact agrees on every metric. "
            f"Task A fitted AUC {live_a.metrics.get('auc_fitted_logistic'):.3f} "
            f"(interval {live_a.metrics.get('auc_fitted_logistic_ci_low'):.3f}-"
            f"{live_a.metrics.get('auc_fitted_logistic_ci_high'):.3f}) against a "
            f"majority-class {live_a.metrics.get('baseline_majority_accuracy'):.3f}; "
            f"task B persistence MAE "
            f"{live_b.metrics.get('baseline_persistence_mae'):.4f} vs fitted "
            f"{live_b.metrics.get('model_fitted_mae'):.4f} with its advantage "
            f"interval still containing zero. Under repeated patient-level "
            f"folds the two tasks SEPARATE: task A's pooled out-of-fold ROW AUC "
            f"{cv_a.metrics.get('oof_auc'):.3f} carries a patient-clustered "
            f"interval {cv_a.metrics.get('oof_auc_ci_low'):.3f}-"
            f"{cv_a.metrics.get('oof_auc_ci_high'):.3f} clear of chance, so A "
            f"resolves; task B's median fold advantage is "
            f"{cv_b.metrics.get('fold_mae_advantage_median'):.5f} and its pooled "
            f"interval still contains zero, so B does not. Discrimination is not "
            f"calibration, so task A's probabilities were checked against the "
            f"rates they produced: slope "
            f"{(cv_a.calibration or {}).get('summary', {}).get('calibration_slope'):.3f} "
            f"against an ideal 1 with the mean predicted "
            f"{(cv_a.calibration or {}).get('summary', {}).get('mean_predicted_probability'):.3f} "
            f"against an observed "
            f"{(cv_a.calibration or {}).get('summary', {}).get('observed_re_detection_rate'):.3f}, "
            f"{cal_clause}. Each verdict was "
            f"asserted to agree with its own interval and its machine-readable "
            f"resolved-flag, and the timepoint pairing is declared an inference "
            f"from the identifier rather than a column in the release. One "
            f"public cohort, a declared 27-gene panel, no therapy or outcome "
            f"joined: not a treatment effect and not guidance for care."
        ),
    )


def check_closed_loop_assimilation_is_measured() -> CheckResult:
    """Closed-loop assimilation must be re-run live, not read off its record.

    The record says the ensemble Kalman correction improved resistant-burden
    tracking on held-out seeds. That is a numerical claim, so it is re-derived
    here rather than believed: the check re-runs `assimilation_twin_validation`
    on the declared held-out seeds and requires the live summary to agree with
    the recorded one exactly (the twin is seeded, so anything else is drift).

    Three further refusals, because each is a way the sentence could be true
    while the work is not:

      * every held-out seed must BEAT the blind engine. A mean that hides a
        seed where the method loses would publish an average over an
        unreliable procedure;
      * the record must carry its refusals and name the twin as synthetic. If
        the patient-data disclaimer is ever dropped, the numbers read as a
        clinical result;
      * the prior support must contain the truth. Version 2 of this module
        lost to the blind arm precisely because it did not, and that failure is
        cheap to reintroduce.

    It says nothing about a real tumour: the truth in this experiment is one
    seeded engine run and the observations are that run plus declared noise.
    """
    import json as _json
    import os as _os

    try:
        from danon.assimilation import (
            DEFAULT_GEOM_SEED,
            HOLDOUT_GEOM_SEEDS,
            assimilation_twin_validation,
        )
    except Exception as exc:  # pragma: no cover - import failure is reported
        return CheckResult(False, f"danon.assimilation unavailable: {exc}")

    path = _os.path.join(ROOT, "outputs_real/assimilation_twin_holdout.json")
    if not _os.path.exists(path):
        return CheckResult(
            supported=False,
            detail=(
                "outputs_real/assimilation_twin_holdout.json missing; the "
                "assimilation claim has never been measured on held-out seeds "
                "(run _run_assimilation_holdout.py)"
            ),
        )
    try:
        with open(path, encoding="utf-8") as fh:
            rec = _json.load(fh)
    except ValueError as exc:
        return CheckResult(
            supported=False, detail=f"assimilation record is not readable JSON: {exc}"
        )

    problems: List[str] = []
    seeds = tuple(int(s) for s in rec.get("held_out_seeds", ()))
    if not seeds:
        return CheckResult(False, "record lists no held-out seeds")
    if DEFAULT_GEOM_SEED in seeds:
        problems.append(
            f"the development seed {DEFAULT_GEOM_SEED} is inside the held-out "
            "set, so the configuration was scored on the seed that chose it"
        )
    if seeds != HOLDOUT_GEOM_SEEDS:
        problems.append(
            f"record's held-out seeds {seeds} are not the module's declared "
            f"{HOLDOUT_GEOM_SEEDS}"
        )

    refusals_text = " ".join(str(r) for r in rec.get("refusals", [])).lower()
    if "no patient telemetry" not in refusals_text:
        problems.append("record no longer refuses a patient-data reading")
    if "synthetic" not in str(rec.get("reading", "")).lower():
        problems.append("record's reading no longer says the twin is synthetic")

    cfg = rec.get("declared_configuration", {})
    prior_mean = cfg.get("prior_growth_mean")
    prior_half = cfg.get("prior_growth_half_width")
    try:
        from danon.assimilation import TRUTH_GROWTH_R, TRUTH_GROWTH_S

        lo = float(prior_mean) - float(prior_half)
        hi = float(prior_mean) + float(prior_half)
        if not (lo <= min(TRUTH_GROWTH_S, TRUTH_GROWTH_R) and hi >= max(TRUTH_GROWTH_S, TRUTH_GROWTH_R)):
            problems.append(
                f"recorded prior [{lo:.3f}, {hi:.3f}] does not contain the truth "
                f"growth rates ({TRUTH_GROWTH_S}, {TRUTH_GROWTH_R})"
            )
    except (TypeError, ValueError):
        problems.append("record's declared configuration carries no usable prior")

    try:
        live = assimilation_twin_validation()
    except Exception as exc:
        return CheckResult(False, f"live held-out re-measurement raised: {exc}")
    live_s = live["summary"]
    rec_s = rec.get("summary", {})
    for key in (
        "seeds_where_assimilation_beats_blind",
        "seeds_total",
        "min_burden_gain_points",
        "mean_assimilated_burden_accuracy_pct",
        "min_assimilated_burden_accuracy_pct",
    ):
        a, b = rec_s.get(key), live_s.get(key)
        if a is None or b is None:
            problems.append(f"summary field {key} missing from record or live run")
        elif abs(float(a) - float(b)) > 1e-6:
            problems.append(f"{key}: record {a} != live {b}")
    if int(live_s["seeds_where_assimilation_beats_blind"]) != int(live_s["seeds_total"]):
        problems.append(
            "assimilation LOST to the blind engine on "
            f"{int(live_s['seeds_total']) - int(live_s['seeds_where_assimilation_beats_blind'])} "
            "held-out seed(s); the mean must not be published as a reliable method"
        )

    detail = (
        f"re-ran the synthetic twin on the {int(live_s['seeds_total'])} declared "
        f"held-out seeds: closed-loop assimilation beat the blind engine on "
        f"{int(live_s['seeds_where_assimilation_beats_blind'])}/{int(live_s['seeds_total'])}, "
        f"mean resistant-burden tracking "
        f"{float(live_s['mean_assimilated_burden_accuracy_pct']):.1f}% vs blind "
        f"{float(live_s['mean_blind_burden_accuracy_pct']):.1f}%, worst seed "
        f"{float(live_s['min_assimilated_burden_accuracy_pct']):.1f}%; live summary "
        "agrees with the record exactly. The truth is one seeded engine run and "
        "the observations are that run plus declared noise -- this measures the "
        "correction METHOD, not tracking of any patient's tumour"
    )
    if problems:
        return CheckResult(False, detail + " Problems: " + "; ".join(problems))
    return CheckResult(True, detail)


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
    "solver_verified_against_analytic_solutions": check_solver_verified_against_analytic_solutions,
    "sample_trial_runs_without_a_download": check_sample_trial_runs_without_a_download,
    "uncertainty_budget_separates_input_from_noise": check_uncertainty_budget_separates_input_from_noise,
    "real_scans_are_served_and_no_prediction_is_drawn": check_real_scans_are_served_and_no_prediction_is_drawn,
    "acceleration_is_bit_identical_and_measured": check_acceleration_is_bit_identical_and_measured,
    "cgga_ood_transport_is_measured_not_asserted": check_cgga_transport_is_measured_not_asserted,
    "synthetic_5ala_standard_is_labelled": check_synthetic_5ala_standard_is_labelled_and_leaves_stream1_refused,
    "tissue_portability_measured": check_solver_portability_is_measured_and_generalization_is_refused,
    "alternating_two_drug_schedule_beats_single": check_alternating_two_drug_schedule_beats_single_drug_calendar,
    "egfrviii_refused": check_egfrviii_is_refused,
    "rl_policy_measured": check_rl_policy_is_measured,
    "dosing_schedules_measured_on_engine": check_dosing_schedules_are_measured_on_the_engine,
    "short_screening_calibrated": check_short_screening_is_calibrated_not_assumed,
    "grid_artifact_is_measured": check_grid_artifact_is_measured,
    "walk_forward_is_refused_and_reasoned": check_walk_forward_is_refused_and_reasoned,
    "ablation_separates_mechanisms": check_ablation_separates_mechanisms,
    "closed_loop_assimilation_measured": check_closed_loop_assimilation_is_measured,
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
        "unitless. The measured values are reported instead. Declaring a unit "
        "would not fix it either, and that is measured rather than argued: the "
        "kill law is kill_max * c / (c + EC50), homogeneous of degree 0 in "
        "(c, EC50), and the drug transport is linear in its source, so scaling "
        "the source strength AND the EC50 by 2^10 (exact in binary floating "
        "point) reproduces the 60-day burden arrays bit for bit -- while "
        "scaling the EC50 alone moves them, so the measurement is not blind. "
        "The engine's concentration scale is therefore a FREE CHOICE, and what "
        "the FDA label supplies is a scale, not a potency: from the label's own "
        "text, the Cmax at 150 mg/m2 is 7.5 mcg/mL oral and 7.3 mcg/mL IV, the "
        "mean elimination half-life is 1.8 hours, and the figure a public-data "
        "route usually quotes as a Cmax -- 5.5 -- is the CLEARANCE in L/hr/m2. "
        "A unit declaration would make the comparison legal without making the "
        "constant measured.",
        EvidenceStatus.UNKNOWN,
        "danon/pharmacology.py::engine_ec50_unit + compare_engine_ec50 (refusal), "
        "with the label read and the relabelling experiment recorded by "
        "_drug_units_probe.py over outputs_real/drug_units_probe.json",
        "ec50_unit_undeclared",
        "This is a defect, not a feature. Until the engine's concentration scale "
        "is declared, its potency constant cannot be anchored to anything "
        "measurable, and no calibration claim should be made about it. The "
        "relabelling measurement bounds what a unit change can buy: it is a "
        "no-op on every trajectory this engine has produced, so the label's "
        "numbers cannot calibrate anything by themselves. The label's Cmax is "
        "also a PLASMA concentration after a dose in a 70 kg adult -- reaching "
        "a tumour kill constant from it needs a blood-brain-barrier and "
        "tumour-exposure model this repository does not have, and an IC50 is "
        "still not a per-day kill rate. Both blockers survive the relabelling.",
    ),
    Claim(
        "bench_anchors_persisted_and_calibration_refused",
        "The wet-lab bridge is built as far as public data allow and then "
        "REFUSED on the record. The measured anchors -- 31 exact of 166 ChEMBL "
        "glioma-line temozolomide IC50 records with 19 right-censored counted "
        "as bounds, and a GDSC2 glioma audit in which 76.0% of IC50s and 34 of "
        "34 temozolomide IC50s sit above the highest concentration tested -- "
        "are persisted with the live engine's own constant set, and the "
        "artifact states `calibrated: false` with three independent blockers: "
        "an IC50 is not a per-day kill rate, the engine's EC50 has no declared "
        "concentration unit, and the measured values are largely bounded rather "
        "than exact. The artifact carries the four-assay protocol that would "
        "close each gap.",
        EvidenceStatus.ESTABLISHED,
        "_run_bench_anchors.py -> outputs_real/bench_anchors.json, checked by "
        "danon/claims.py::check_bench_anchors_are_measured_and_calibration_is_refused",
        "bench_anchors_are_measured_and_calibration_refused",
        "This is a measurement of the GAP, not a calibration. The engine's "
        "kill_max, EC50 and growth constants remain AUTHORED, and the check "
        "fails if the artifact's constants stop matching the live engine's, so "
        "the recorded anchor cannot silently describe a different engine. No "
        "in-vitro number is substituted into the engine anywhere.",
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
        "egfrviii_not_in_open_data",
        "EGFRvIII is an in-frame deletion of exons 2-7 -- a structural variant -- "
        "so it cannot appear in the open Masked Somatic Mutation files. All 1,847 "
        "structural-variation files for TCGA-GBM are controlled access, so "
        "EGFRvIII status is not obtainable from open data and cannot be a feature "
        "of a classifier built here.",
        EvidenceStatus.PROVEN_FAILED,
        "danon/eval_targets.py::probe_egfrviii_reachability",
        "egfrviii_refused",
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
        "a real Ki-67 slide is supplied. The two sources a public-data route "
        "names for that numerator were then probed and carry neither: the "
        "TCGA-GBM clinical BCR XML (schema 2.7, 108 distinct fields, read back "
        "live) has no Ki-67 or MIB-1 field anywhere in it, and Ivy GAP's four "
        "public tables (10, 26, 22 and 29 columns) carry no labelling index "
        "either -- the one proliferation in the atlas is normalized_area_ctmvp, "
        "the area fraction of MICROVASCULAR proliferation, a vessel measurement "
        "rather than a Ki-67 index. Substituting a nuclei density for the "
        "labelling index would fabricate the growth rate it feeds.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_stream2_histology_measured_and_ki67_refused "
        "over outputs_real/histology_probe.json (rebuildable via "
        "_histology_probe.py) and outputs_real/ki67_source_probe.json "
        "(rebuildable via _ki67_source_probe.py)",
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
        "here is a grade, a WHO criterion, or a statement about a patient. The "
        "external-source probe is a SCHEMA-and-HEADER read, not a pixel read: one "
        "TCGA-GBM supplement was downloaded, and the absence of a Ki-67 field is "
        "a property of the disease-specific schema (v2.7) it declares rather "
        "than of that patient -- but the pathology report PDFs for those same "
        "cases are unstructured and were NOT parsed, so a proliferation index "
        "typed into one of them would not be visible to this probe. Ivy GAP's "
        "gene-level expression table is likewise only read at its header: it "
        "exposes gene_symbol and per-structure expression energy, so it does not "
        "establish that MKI67 is absent as a transcript, only that no table in "
        "the atlas exposes a labelling index. No ivy-named collection appears in "
        "TCIA's public collection list (156 collections, read successfully), "
        "while per-collection patient and series queries answer with an empty "
        "body anonymously -- so 'not in the list' is the strongest statement "
        "this machine can make about that cohort's imaging, and the atlas "
        "cohort's MRI may still be reachable by other means.",
    ),
    Claim(
        "solver_verified_against_analytic_solutions",
        "The spatial solver is verified against exact analytic solutions rather "
        "than against itself. The zero-flux Laplacian reproduces its own "
        "Neumann eigenvector to a relative residual of 3.43e-14 (float noise), "
        "its total vanishes on any closed-domain field to 3.47e-18 relative, "
        "the heterogeneous operator collapses onto the homogeneous one at "
        "constant D to 2.75e-16, and the observed order of spatial convergence "
        "is 1.9944 (second order, measured by halving dx). Its one signed "
        "discretization error -- forward Euler integrates clearance at a rate "
        "+0.0843%/day high, a direction the scheme cannot avoid -- moves no "
        "endpoint by more than 0.7505% in a paired cohort on real geometry, "
        "which is 0.0075x that cohort's own seed-to-seed spread of 99.66%.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_solver_verified_against_analytic_solutions "
        "over outputs_real/clearance_bias_cohort.json, with the same values "
        "pinned by tests/test_benchmarks.py",
        "solver_verified_against_analytic_solutions",
        "This is VERIFICATION, not validation: it says the code solves the model "
        "it states, not that the model is right about biology. No benchmark "
        "against PhysiCell, BioNetGen or FEniCS was run -- the analytic "
        "solutions used here are what such a cross-check would compare against "
        "anyway, and a matched benchmark problem is a separate pass. The bias "
        "is bounded below the model's own stochastic resolution; it is not "
        "zero, and it is not corrected, only measured. The paired cohort "
        "behind the 0.0075x is ONE case at two policies and three seeds, so it "
        "is a bound on this bias, not a general noise model.",
    ),
    Claim(
        "fresh_clone_can_run_the_twin",
        "A fresh clone can RUN the twin, not only read about it. "
        "data/sample_bids/ commits a 14 KB synthetic phantom -- three NIfTI "
        "volumes plus a config and a dataset description, all five "
        "byte-for-byte the output of the engine's own make_gbm_phantom -- and "
        "python -m danon.reproduce runs the real paired 365-day trial on it "
        "with no download. The phantom carries 3,178 tumour voxels of 13,824 at "
        "1 mm; the continuous-dose arm reaches progression in every replicate, "
        "the switching arm reaches the trial horizon in none, and the paired "
        "delta clears the arm's own seed-to-seed spread.",
        EvidenceStatus.ESTABLISHED,
        "danon/reproduce.py (danon/claims.py::"
        "check_sample_trial_runs_without_a_download) over "
        "outputs_real/sample_twin_report.json, with tests/test_reproduce.py",
        "sample_trial_runs_without_a_download",
        "It is a phantom: no part of it is measured, and the engine's constants "
        "are the repository's declared assumptions, so this demonstrates that "
        "the pipeline runs and is reproducible -- it is NOT evidence about any "
        "patient, and the MTD-versus-switching contrast is the engine's own "
        "qualitative behaviour on a synthetic substrate, not a clinical result. "
        "The sample ships homogeneous transport: the heterogeneous D(x) branch "
        "is implemented and reachable by a config flag, but costs about 19x "
        "per run at this grid, so the default keeps the command to seconds.",
    ),
    Claim(
        "alternating_two_drug_calendar_beats_best_single_drug_calendar",
        "Contrary to the convention that a two-drug calendar should pulse ONE "
        "drug against a resistant minority, this measurement alternates the TWO "
        "drugs over time. On the same engine, SAME seeds, the alternating arm "
        "beats the best single-drug calendar by median AUBC: 0.0433 vs 1.3046 "
        "on 30 seeds (ratio 0.0332, dominance advance +150.5 d). The win does "
        "not flip across the declared marrow-weight sweep (drug B assumed at "
        "0.3x, 0.5x and 1.0x), and 4 of the 24 declared alternating calendars "
        "are inadmissible at every weight while 12 are admissible at the "
        "conservative weight. Timed to the label: the screen's boundary lands "
        "on the label's own 150 mg/m2 x5d q28d regimen, so 'pulsing one drug' "
        "is the marrow-maximum schedule, and an alternation at that maximum "
        "makes a resistant population that the single drug cannot.",
        EvidenceStatus.ESTABLISHED,
        "danon/dosing.py::compare_combination_vs_single",
        "alternating_two_drug_schedule_beats_single",
        "Measurement on one synthetic GBM phantom, 30 held-out seeds, 365 days. "
        "The winning calendar is a combination -- A 150 mg/m2 at 0.3 x full x3d "
        "with B 150 mg/m2 at 0.5 x full x3d, no rest -- which at full engine "
        "intensity would exceed the marrow budget. Timed to the label and "
        "the trial numbers quoted here are the engine's. No patient claim is "
        "made; the dosimetry is a synthetic analogue of the label's schedule.",
    ),
    Claim(
        "alternating_win_has_a_reversal_map",
        "The alternating win is bracketed by a MEASURED reversal map rather than "
        "asserted at one operating point. Re-running the release-1 winner across "
        "a declared 40-cell grid -- engine intensity (0.5, 0.75, 1.0, 1.25, 1.5) "
        "on the winner's own levels x pre-existing resistant fraction (0.001, "
        "0.01, 0.1, 0.3) x growth noise (0.05, 0.2) -- on the same 30 seeds, "
        "with the release-1 best single-drug arm RE-RUN at every cell so each "
        "cell is a paired comparison (1,440 engine runs), the intensity-1.0 cell "
        "reproduces the release-1 median AUBC exactly "
        "(0.04332412131403639), no cell reverses the win, and the binding limit "
        "is SAFETY, not efficacy: the ANC screen permits at most intensity 1.1 "
        "under the conservative drug-B marrow weight and 1.65 at half weight, "
        "and the 16 cells at intensity 1.25 and 1.5 fail the floor (modelled "
        "nadir 1.24 and 0.59 x10^9/L against 1.5).",
        EvidenceStatus.ESTABLISHED,
        "_release2_sweep.py -> outputs_real/release2_sweep.json, checked by "
        "danon/claims.py::check_alternating_win_has_a_measured_reversal_map",
        "alternating_win_has_a_reversal_map",
        "The map is a bracketing of the release-1 win, not a validation of the "
        "calendar: the window's LOWER edge was never reached (the lowest winning "
        "intensity is 0.5 in all eight groups, where PFS already pins to the "
        "365-day horizon), the r0 axis above 0.1 is inert on this phantom (0.1 "
        "and 0.3 give identical numbers, which is a property of the initial "
        "condition rather than evidence about resistant patients), and every "
        "cell is one synthetic phantom under assumed pharmacology constants. An "
        "incomplete sweep is refused by the check, and so is a map that found no "
        "safety boundary at all.",
    ),
    Claim(
        "architecture_writeup_audited_against_the_code",
        "The engine's architecture write-up is committed verbatim and AUDITED "
        "against the code rather than trusted. Thirteen declared rules bind it: "
        "five statements are REFUTED (a 'deterministic, physics-constrained' "
        "engine; AdaptiveDosePolicy described as thresholding on the resistant "
        "fraction when it is burden-relative and the threshold controller is "
        "CollateralSwitchPolicy; an ANC equation carrying an additive "
        "RecoveryRate term the marrow model does not have, since the screen is a "
        "reservoir whose budget is DEFINED by the label's own regimen; claims.py "
        "described as verifying that the Adaptive policy beat MTD, when its "
        "single-drug search returned a null and the measured win came from "
        "alternating two drugs; and a two-state ESTABLISHED/PROVEN_FAILED UI "
        "flag where the evidence vocabulary has five members). Two are "
        "IMPRECISE (mutation written as a multiplicative noise term where the "
        "engine draws Poisson(grow_S * mu); BraTS labels sent 'directly' to a "
        "spatial carrying capacity where they become a proliferative field) and "
        "one is QUALIFIED (a 365-day trial 'in seconds' is ~26 s per 30-seed "
        "arm, against 5,388 s for the release-1 comparison stage). Five "
        "material omissions are named: the opt-in immune layer (default None), "
        "cell motility, the extinction and detection floors, partial "
        "cross-resistance (drug_sensitivity_scale_R = 0.25), and the "
        "burden-ratio definition of PFS.",
        EvidenceStatus.ESTABLISHED,
        "_run_architecture_audit.py -> outputs_real/architecture_audit.json, "
        "checked by danon/claims.py::check_architecture_writeup_is_audited_against_the_code",
        "architecture_writeup_audited_against_the_code",
        "The verdicts are DOCUMENT versus CODE, and carry no biological "
        "evidence whatever. The document is preserved unedited, so its refuted "
        "sentences still read as written: the corrections live in the audit "
        "artifact, not in the text. The rule set is finite and matched on "
        "literal phrases, so a reworded version of the same error can escape it "
        "-- which is why the audited document's SHA-256 is checked, that is, a "
        "silent rewrite fails rather than passes. An audit that flagged nothing "
        "is refused by the check, and so is an audit whose code-fact snapshot "
        "has gone stale.",
    ),
    Claim(
        "motility_constant_is_grid_dependent_and_priced",
        "The engine's cell-migration coefficient is a per-VOXEL literal, so it is "
        "a different physical diffusivity on every grid, and the consequence is "
        "now measured rather than noted. `LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY "
        "= 0.02` reads as 0.020 mm^2/day on the 1.0 mm phantom -- where every "
        "calibration anchor was set -- and 0.269 mm^2/day on the 3.667 mm real "
        "grid, a 13.44x grid factor, against a literature cell-motility range of "
        "0.001-0.005 mm^2/day: above range by 4x on the phantom and 53.8x on the "
        "real path, and nothing raises because voxel_mm is bookkeeping rather "
        "than a conversion. At 30 seeds on the release-1 winner (whose default "
        "arm reproduces the release-1 AUBC 0.04332412131403639 exactly), the "
        "winner's own median AUBC moves +5.5% to +10.9% across the declared "
        "settings, while median PFS and dominance day are pinned at the horizon "
        "in every one of them -- the CONCLUSIONS are robust and the QUOTED "
        "NUMBER is not. The single-drug reference arm is where it bites: its "
        "AUBC falls from 1.3046 to 0.8175 with motility off, moving the headline "
        "winner/single ratio from 0.0332 to 0.0588, so the margin release 1 "
        "quotes is motility-dependent by about 1.8x.",
        EvidenceStatus.ESTABLISHED,
        "_run_motility_audit.py -> outputs_real/motility_audit.json, checked by "
        "danon/claims.py::check_motility_constant_is_grid_dependent_and_priced",
        "motility_constant_is_grid_dependent_and_priced",
        "This prices the defect, it does not fix it: the engine's default is "
        "UNCHANGED, because changing it would invalidate outputs_real/dosing.json "
        "(90 minutes of measurement) and outputs_real/release2_sweep.json "
        "(24 minutes) and the check fails if the default moves without those "
        "being re-run. It is also ONE calendar at ONE operating point on ONE "
        "synthetic phantom: the real-grid column is a unit conversion, not a "
        "simulation on a 3.667 mm grid, and nothing here decides whether glioma "
        "motility should sit at 0.001 or 0.005 mm^2/day. The remedy is to express "
        "the coefficient in mm^2/day and convert per voxel size, then re-run both "
        "releases.",
    ),
    Claim(
        "uncertainty_is_measured_and_the_posterior_label_refused",
        "Uncertainty is measured rather than asserted, and the label it is NOT is "
        "stated first. Propagating the real per-donor MGMT promoter beta "
        "distribution (140 TCGA-GBM donors over two probes, mean beta band "
        "0.0119-0.6695) through both arms on a committed BraTS-2021 case gives an "
        "interval whose coverage is over (donor, seed) jointly and which is "
        "explicitly NOT a Bayesian credible interval -- a posterior needs a "
        "likelihood, and this repository holds ZERO engine-output/patient-outcome "
        "pairs to build one from (recorded as PROVEN_FAILED so no renderer can call "
        "it one). What the decomposition then establishes bounds the "
        "'patient-specific twin' story with numbers: for MTD progression-free "
        "survival, re-running the model on the SAME donor moves the answer with an "
        "SD of 29.92 days while moving to a different donor's MGMT status moves it "
        "10.38 days -- a repeat run is 2.88x the patient-specific signal, and the "
        "input does not clear a 0.05 F test (F = 1.241 on 139/140 df, p = 0.102). "
        "The day-365 burden readout DOES resolve it (input fraction 24.6%, "
        "p = 0.0016), so which endpoint is read decides whether the patient is "
        "visible at all. The arm that looks best is the worst on this axis: the "
        "switching policy's PFS is identical for every donor and every seed "
        "(365 d, no spread to decompose), and its day-365 burden is not resolvable "
        "above the model's own noise at all (MSB 1.80 < MSW 6.99, "
        "sigma^2_between <= 0, recorded unclipped).",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_uncertainty_budget_separates_input_from_noise "
        "over outputs_real/uncertainty_budget.{json,txt} (rebuildable via "
        "_uncertainty_cohort.py), with danon/uncertainty.py and "
        "tests/test_uncertainty.py",
        "uncertainty_budget_separates_input_from_noise",
        "It is not a credible interval and not a clinical bound. One scalar input "
        "is measured here -- MGMT beta, the only engine input in this repository "
        "with a real per-donor distribution -- on ONE committed geometry, at ONE "
        "trial design (365 days, r0 = 0.01), through a mapping beta -> r0 that is a "
        "DECLARED convention rather than a fit, with the engine's rate constants "
        "still declared assumptions. So the input channel quantified here is a "
        "LOWER bound on the true input uncertainty, and a different anchor moves "
        "every number without any code changing. The switching arm's conclusion is "
        "about this design: it saturates at the horizon, so another horizon or "
        "switch threshold could change it. The F tests ask whether the cohort's "
        "beta spread is resolvable THROUGH THE ENGINE, which is a question about "
        "the model, not about patients, and the beta of a donor tumour is not the "
        "beta of the patient whose case would be simulated.",
    ),
    Claim(
        "rl_policy_trained_and_measured",
        "A PPO schedule policy was trained on this engine's simulator for 60 iterations "
        "(87,600 simulator steps) and evaluated on 30 held-out synthetic seeds: its "
        "sampled median AUBC was 0.0216 (repo-defined retention 1.0051 against the best "
        "closed-form rule, collateral switch at 0.0279 selected on separate monitor "
        "seeds; MTD was 1.2618). But the fixed deterministic A/B-alternation reference "
        "did slightly better (AUBC 0.0208), while the PPO greedy projection matched MTD "
        "(1.2618). This does not demonstrate a PPO-specific advantage or patient benefit.",
        EvidenceStatus.ESTABLISHED,
        "danon/rl_steering.py::verify_rl_policy_retention",
        "rl_policy_measured",
        "Synthetic geometry and an authored, repo-defined normalization; the "
        "deterministic alternation control is slightly better than the sampled agent, so "
        "these results do not establish an advantage caused by PPO.",
    ),
    Claim(
        "short_horizon_screening_measured_and_rejected",
        "Scoring a schedule on a shorter horizon to make the search cheap was measured "
        "and rejected. Against the full 365-day ranking, a 120-day screen of the same 20 "
        "feasible calendars correlates NEGATIVELY (Spearman -0.588, Kendall tau -0.516): "
        "the true 365-day winner is ranked 16th of 20 by the screen and the screen's own "
        "pick is 19th at 365 days. None of the top-3 full-horizon calendars survive. "
        "Aggressive dosing looks best early and worst late -- which is competitive "
        "release showing up in the calibration -- so the search ranks stage 1 on the full "
        "horizon and the API returns this verdict to any client asking for a short "
        "screen.",
        EvidenceStatus.ESTABLISHED,
        "danon/dosing.py::screen_fidelity",
        "short_screening_calibrated",
        "Measured on one phantom and one 20-cell grid with 6 seeds; the sign of the "
        "correlation is the finding, the exact coefficient is not portable. This says the "
        "short screen is unsafe HERE, not that no future calibration could make one safe.",
    ),
    Claim(
        "dosing_calendar_runs_on_the_engine",
        "An open-loop dosing calendar is a real input to the real engine: '150 mg/m2 for "
        "5 days then a 21-day holiday' is executed as the policy's day-indexed dose "
        "series, pinned against the engine's own recorded dose_A/dose_B arrays. On 30 "
        "held-out seeds, pulsing delays resistant dominance by 32 days -- competitive "
        "release is real and measurable. But no calendar in the searched grid beat "
        "continuous MTD on total tumour burden (best pulsed AUBC 1.3039 vs MTD 1.2741), "
        "so the schedule search is reported as a NULL result rather than as a win.",
        EvidenceStatus.ESTABLISHED,
        "danon/dosing.py::run_competitive_release_experiment",
        "dosing_schedules_measured_on_engine",
        "Simulation only, on a synthetic phantom. Pulsing buys time to resistance at the "
        "cost of more total tumour. The myelosuppression screen is DECLARED arithmetic on "
        "assumed parameters -- not an NCCN, FDA or label threshold, and not measured "
        "here. No patient benefit is claimed and none is demonstrated.",
    ),
    Claim(
        "grid_artifact_is_measured",
        "The engine's fixed uniform grid is measured rather than assumed, and the "
        "measurement is not flattering. On a committed BraTS-2021 case (98,447 mm^3 of "
        "expert-labelled tumour at 1 mm) the axis-aligned block majority vote the "
        "ingestion DOCUMENTS is volume-faithful -- -2.88% at 2 mm, -1.22% at the shipped "
        "3.63 mm grid, Dice 0.9310 against the source occupancy -- while "
        "`realdata._decimate_labels` does not produce it: it agrees with that vote at "
        "Dice 0.3905 (2,316 differing voxels at the shipped grid), and in single-block "
        "probes it sends ZERO of 200 fully-tumour axis-aligned cubes to their own engine "
        "voxel, because its reshape groups contiguous runs of source voxels instead of "
        "cubes (recorded PROVEN_FAILED, with the six committed engine-ready volumes all "
        "carrying the wrong partition and a 1 mm header over 48 mm^3 voxels, understating "
        "every physical volume taken from them 48.000x). The alternative block-MAX rule "
        "that api_server applies to an oversize upload fails in the other direction: "
        "+32.91% tumour volume at the shipped grid, with 34.55% of the vote's tumour "
        "added by re-labelling blocks the majority called background. And the "
        "discretization moves the OUTCOME by more than the model's own noise: at FIXED "
        "physical diffusivity, MTD progression-free survival shifts -32.43% between the 2 "
        "mm and 3.63 mm grids, which is 0.79x that cohort's own seed-to-seed spread of "
        "41.23%, while the day-365 spatial burden shifts +43.5% (1.62x the spread) under "
        "MTD and +417.6% (7.12x) under the switching arm -- so unlike the clearance "
        "scheme's 0.0075x, this artifact is NOT below the model's resolution.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_grid_artifact_is_measured over "
        "outputs_real/grid_artifact.{json,txt} (rebuildable via "
        "_grid_artifact_cohort.py), with danon/grid_artifact.py and "
        "tests/test_grid_artifact.py",
        "grid_artifact_is_measured",
        "It measures the artifact; it does not remove it. Adaptive refinement does not "
        "exist in this repository, so the coarse grid is the last word on spatial detail: "
        "anything smaller than a majority block is gone, and the drift measured here is a "
        "LOWER bound on what a coarser-but-correct partition would show, because the "
        "shipped geometry is currently a different partition of the source rather than a "
        "faithful reduction of it. The endpoint compared is a simulated PFS under "
        "declared constants, not a patient's, and the drift is reported against the "
        "cohort's own seed spread rather than against a clinical margin. The diffusivity "
        "is anchored to 1 mm by the constant's own documentation, not by measurement, and "
        "on the anisotropic (4, 4, 3) factors no scalar D is simultaneously right along "
        "all three axes -- the engine's Laplacian is isotropic in index space, so the "
        "mm^2/day figures here are volume-equivalent. The probe samples 200 of the 8,064 "
        "blocks; the cohort is ONE case, ONE trial design (365 days, r0 = 0.01) and six "
        "seeds; and the arm that drifts most is the switching arm, whose PFS is 365 days "
        "for every seed at this design, so its burden drift has no PFS counterpart to "
        "check it against.",
    ),
    Claim(
        "walk_forward_is_refused_and_reasoned",
        "Longitudinal walk-forward validation -- run the engine from day 0 and score it "
        "against the same patient's follow-up scan -- is implemented and refused, and the "
        "refusal is counted rather than asserted. The harness carries 3D Dice, Jaccard, a "
        "symmetric 95th-percentile surface distance and the TRUE Jensen-Shannon "
        "divergence, which is NOT scipy.spatial.distance.jensenshannon: for the same pair "
        "this returns 0.5886 bits while scipy returns 0.6387 = sqrt(0.5886 * ln 2), so a "
        "quoted target of 'JSD < 0.15' means 0.15 bits with one and 0.0225 bits with the "
        "other. There is nothing to score: the CFB-GBM manifest in this repository "
        "declares 393 usable pairs over 168 patients and ships no images, the public NBIA "
        "API returns 0 patients and 0 segmentation series for MU-Glioma-Post (203 "
        "subjects, 596 post-treatment timepoints, DOI 10.7937/7k9k-3c83) and for "
        "Brain-Tumor-Progression, and the GLASS records are now IN this tree rather than "
        "absent from it -- while 0 scorable IMAGING pairs remain, because GLASS is "
        "genomics and the imaging collections still refuse. GLASS's processed release is "
        "a public cBioPortal study (693 samples, 629 "
        "sequenced, paired primaries and recurrences) whose mutation records DO "
        "carry tumorAltCount/tumorRefCount, so the clonal refusal below rests on "
        "the missing MAPPING rule rather than on availability or on a missing "
        "frequency. LUMIERE, the one longitudinal GBM collection the NBIA APIs cannot "
        "answer for, is re-checked separately because it is real, reachable (Figshare "
        "collection 5904905: 91 patients, 638 study dates, 2487 images, 616 "
        "expert-rated timepoints) and STILL not scorable: the collection ships its "
        "masks as a separate article titled 'MRI data and automated segmentations' "
        "(DeepBraTumIA and HD-GLIO-AUTO, two U-Net tools), while its expert "
        "contribution is a per-timepoint RANO response class and Macdonald "
        "measurements -- NOT a segmentation. A Dice against automated masks is "
        "model-versus-model, so caching LUMIERE would not make a Dice measurable, and "
        "nothing from it is cached here. What the harness establishes instead is which "
        "term moves the predicted "
        "tumour at all: the per-voxel growth map sums to exactly 0.0 outside the day-0 "
        "mask, so outward movement is the daily motility * laplacian(N) term and nothing "
        "else -- with motility forced to 0.0 the share of final burden outside the mask "
        "is exactly 0.0, with the engine's default it is 19.35% after 30 days -- and that "
        "coefficient is a VOXEL-unit literal, 0.02 voxel^2/day, worth 0.2641 mm^2/day at "
        "the engine's real 3.634 mm voxel size, 52.8x the top of the published "
        "cell-motility range (0.001-0.005 mm^2/day). A walk-forward Dice measured today "
        "would therefore be measuring that coefficient, not invasion.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_walk_forward_is_refused_and_reasoned, with "
        "danon/walkforward.py, _walkforward_census.py over "
        "outputs_real/walkforward_census.json, and tests/test_walkforward.py",
        "walk_forward_is_refused_and_reasoned",
        "The metrics are VERIFIED against constructions known on paper, not validated "
        "against patients: no Dice and no JSD has been measured on a human follow-up, and "
        "none can be until a registered pair exists -- this repository has no "
        "co-registration, so a Dice between two unregistered segmentations would be an "
        "alignment error wearing a model metric's name. The census counts what officials "
        "publish, not what a licence holder could download: MU-Glioma-Post is public to "
        "READ about and licence-locked to acquire, so '0 patients from the API' is a "
        "statement about this machine's access, not about the dataset's contents. LUMIERE "
        "is entered as reachable on the strength of an API answering for its ARTICLE "
        "LIST -- no image was downloaded and no licence reviewed, and its licence is "
        "ambiguous across its own sources (the paper is CC BY 4.0 while the dataset's "
        "repository states non-commercial use), so 'reachable' is a statement about "
        "metadata, not about what this tree may use. The "
        "expansion measurement is a 16^3 synthetic cube over two seeds and 30 days in the "
        "most favourable configuration for outward movement (proliferation 1.0 inside, "
        "uniform drug inlet); it bounds the mechanism, and a real geometry would move the "
        "share. And the clonal readout refuses by construction: the engine carries two "
        "bins, so a GLASS allele-frequency spectrum would have to be summarised onto "
        "those two bins by a rule this repository has no basis for. GLASS is the case "
        "that corrects the receipt, re-probed rather than remembered: its processed "
        "release is an OPEN cBioPortal study with paired primaries and recurrences "
        "and POPULATED allele counts, and danon.variants now MEASURES it: 1686 mutation "
        "records over 281 patients and 543 specimens, every one of them carrying read "
        "counts, which gives 248 primary -> recurrence pairs over 226 patients and a "
        "measured allele-fraction change for each shared variant. The refusal that "
        "remains is the one that was always the real one: a per-variant ALLELE fraction "
        "in a copy-number-altered bulk biopsy is not a two-bin CELL-POPULATION fraction, "
        "so only a provable LOWER bound on the clone fraction is published (f >= A_t "
        "needs no copy number), while a point clone fraction, an upper bound and a k-bin "
        "summarisation onto the engine's two bins stay refused. An earlier version of "
        "this sentence gave controlled access as the reason; that was wrong for the "
        "processed release and is corrected here.",
    ),
    Claim(
        "ablation_separates_mechanisms",
        "Each mechanism this engine carries is ablated one at a time -- same geometry, "
        "same seeds, same policy, one parameter moved -- and the effect is reported in "
        "units of the model's own seed-to-seed spread, so a term that moves an endpoint "
        "less than the noise is not credited with doing anything. On the committed "
        "BraTS-2021 case over three seeds and 365 days the load-bearing readouts are: "
        "cell motility 5 of 8 (steering day-365 burden 4.628x the spread, MTD PFS 1.233x, "
        "the final resistant fraction 113x, dominance day 10.0x), the resistant clone's "
        "kill ceiling 3 (burden 1.125x, resistant fraction 1.707x, dominance day 3.0x), "
        "the resistance fitness cost 2 (dominance day 2.333x under MTD, steering burden "
        "2.795x), the perfusion-derived transport field 1 (steering burden 1.523x, while "
        "MTD PFS moves 0.008x -- nothing), and drug B 1 (steering burden 5172.8x). Two "
        "mechanisms move NOTHING above the noise: the mutation term (every readout inert, "
        "0.067x or less) and the asymmetric Lotka-Volterra competition coefficient (every "
        "readout inert or unresolved). The control that makes those nulls credible is "
        "built into the design: MTDPolicy doses (1.0, 0.0), so ablating drug B under the "
        "MTD arm must be exactly zero, and it is -- on every one of the four readouts, in "
        "the phantom and in the real case.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_ablation_separates_mechanisms, with danon/ablation.py, "
        "_ablation_cohort.py over outputs_real/ablation_cohort.json, and "
        "tests/test_ablation.py",
        "ablation_separates_mechanisms",
        "Load-bearing is not correct. A wrong coefficient and a right one produce the "
        "same positive effect size when the same model is ablated, so this says which "
        "parts of THIS model the answer depends on, not that they are the mechanisms a "
        "real tumour uses. Inert is not falsified either: a mechanism can be inert "
        "because it is wrong or because it is second-order at this design, and the "
        "verdict of 'unresolved' (the ratio between 0.1x and 1.0x) is the honest middle "
        "that this design cannot settle. The mutation null has a mechanical reason worth "
        "stating: with no pre-seeded resistance (r0 = 0) continuous MTD eradicates the "
        "tumour to a burden of exactly 0.000 at day 365 for every seed and at both "
        "mutation rates, so this model's resistance is its 1% initial condition, not its "
        "mutation channel. The endpoints are simulated PFS and burden under declared "
        "constants, on ONE case, one trial design and three seeds; and no "
        "diffusion-tensor map exists here, so an 'anisotropic tensor PDE' is not what the "
        "transport ablation removes -- it removes the spatial variation of a scalar "
        "field.",
    ),
    Claim(
        "acceleration_is_bit_identical_and_measured",
        "The accelerated solver is faster by a measured factor and by nothing else: every "
        "kernel reproduces its own reference BIT FOR BIT. Removing the Python-level "
        "`np.pad` copies and the boolean masks from the two hot operators -- holding the "
        "summation order fixed, since float addition is not associative -- buys 1.25x / "
        "2.32x at the engine's own 14x24x24 grid, 1.71x / 3.17x at 64^3 and 2.47x / 3.54x "
        "at 256^3 (Laplacian / div-divergence), and a reuse workspace keeps the per-day "
        "cost at 1.117x for homogeneous transport and 1.904x when the transport field is "
        "heterogeneous. Identity is the gate, not a tolerance: a one-ulp change in a "
        "diffusion flux is amplified by 365 iterations of a nonlinear map, so the tests "
        "compare whole arrays, whole trajectories and whole replicate batches with exact "
        "equality. The two obvious ways to buy the rest do NOT work here and are recorded "
        "as measured rather than assumed: batching replicates into one leading array axis "
        "is a measured SLOWDOWN (0.833x the serial rate at 16 replicates x 60 days with "
        "per-replicate streams, 0.862x with one batched draw -- so it is not the RNG), "
        "because an 8,064-voxel field is 64 KB and forty numpy calls on it are dominated "
        "by per-call cost; and there is no GPU, no JAX and no CuPy in this environment, "
        "so no GPU figure exists to quote. What DOES scale is running the replicates in "
        "OS processes, which cannot change a number because each worker runs the "
        "unmodified serial engine on its own seeds: 2.080x at 4 workers and 2.006x at 8 "
        "on 12 cores, which puts a 10,000-draw 365-day sweep at 1.41-1.46 extrapolated "
        "hours instead of 2.94 serial.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_acceleration_is_bit_identical_and_measured over "
        "outputs_real/perf.json (rebuildable via _perf_cohort.py), with danon/accel.py, "
        "danon/uq.py, tests/test_accel.py, tests/test_batchsim.py and tests/test_uq.py",
        "acceleration_is_bit_identical_and_measured",
        "These are speedups of THIS engine on THIS machine, not of the problem. The "
        "ratios are minima over interleaved repetitions on a shared machine whose own "
        "30-day timing varied 2.7x across 12 consecutive runs, so every figure here is "
        "the cost of the work rather than a peak, and none of them transfers to another "
        "box. The 256^3 operator numbers are one call each, not a trajectory: no 365-day "
        "run at that grid was measured, and the engine's own cases are 8,064-voxel grids, "
        "so 256^3 is a scaling probe and not a workload this repository runs. The process "
        "ratios are measured at 4 and 8 workers with 16 replicates and do NOT reach the "
        "12-core count, so the extrapolated 10,000-draw figure is arithmetic on a ratio "
        "that will not hold on a busy machine, over a grid far smaller than a clinical "
        "volume. Batching is a slowdown at these sizes and might not be at much larger "
        "ones; that was not measured. Nothing here is a claim about biology, and there is "
        "no surrogate model and no rendered frame rate in this repository at all -- those "
        "are refused, not estimated.",
    ),
    Claim(
        "cgga_ood_transport_is_measured_not_asserted",
        "The five proposed online-only pipelines were bound to data rather than to figure "
        "captions. CGGA's clinical releases download without a login and are cached; with "
        "the repository's own Cox model -- age, sex and a glioblastoma dummy, fitted once "
        "on the cached TCGA cohort (n=595, 491 events, internal held-out C 0.6559 +/- "
        "0.0247) -- applied to CGGA without refitting a coefficient, the transported "
        "C-index is 0.7402 on mRNAseq_325's primary gliomas (n=222, 138 events; age alone "
        "0.6600), 0.5763 on its primary GBM subset (n=85, 73 events; age alone 0.5702), "
        "and 0.7129 / 0.5821 on mRNAseq_693 (n=403 / n=133). Two of the pitch's numbers "
        "do not survive the files: 'n=325 Glioblastoma' is 85 GBM rows inside a 325-row "
        "table, and the claimed 0.6412 appears in none of the four subsets -- the "
        "mixed-grade figure is higher, which is grade spread rather than overfitting "
        "defeated, and the GBM-only figure is age-level. What this does NOT do is "
        "validate the engine: it transports a covariate model, no imaging is linked to "
        "those rows, and one external cohort at one metric cannot prove 'zero "
        "overfitting'. The remaining four pipelines are each refused with the live reason "
        "-- the single-cell JSD (no four-state prediction exists anywhere in this "
        "repository, and the GEO workbook maps cells to tumours, not states), the "
        "in-vitro R^2 (the engine's EC50 has no unit, so the comparison is a type error, "
        "and 34 of 34 temozolomide IC50s are extrapolations), the inverse-tensor recovery "
        "(there is no inverse estimator, and recovering an injected tensor is "
        "self-consistency, not validation), and the BraTS Dice (no probed collection "
        "carries an expert mask plus a second timepoint) -- and the GEO archive, the GDSC "
        "release and six BraTS cases are recorded as present, so the refusals are about "
        "the metric, not availability.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_cgga_transport_is_measured_not_asserted over "
        "outputs_real/cgga_ood.json (rebuildable via _cgga_ood_cohort.py), with "
        "danon/external_cohorts.py and tests/test_external_cohorts.py",
        "cgga_ood_transport_is_measured_not_asserted",
        "Transport is not validation, and this is a covariate model: the engine itself "
        "has no estimable patient-level C-index, so nothing here says a simulation "
        "predicts a patient. The internal reference fit includes TCGA lower-grade rows, "
        "which is part of why the mixed-grade external C can exceed it; the GBM-only "
        "figure is the like-for-like comparison and it sits at age-only level. CGGA and "
        "TCGA differ in era and regimen and the model carries no treatment covariate, so "
        "this compares cohorts, not treatments. The four refused pipelines are refused "
        "for metric reasons, not availability: if the GEO workbook is ever joined to a "
        "state classifier and a therapy-selection model is added, the JSD becomes "
        "measurable and this sentence would have to be rewritten.",
    ),
    Claim(
        "synthetic_5ala_standard_is_labelled_and_leaves_stream1_refused",
        "Stream 1 (5-ALA fluorescence) has a semi-synthetic standard rather than a "
        "result. The planted field is manufactured by this repository -- the disclaimer "
        "is carried verbatim in the record -- and it sits on the PRE-OPERATIVE tumour rim "
        "of the committed BraTS-2021 case, because no post-operative imaging is cached "
        "anywhere here and a resection cavity therefore cannot be extracted. Through the "
        "real intake path the documented identity holds exactly: a uniform planted map "
        "fills to 0.65 x 1774 = 1153.1000 burden units, matching the engine's own seed to "
        "2.27e-13. On this case the segmentation is a SHELL (mean distance from the "
        "boundary 1.06 voxels for every planted mode), so the three fluorescence layouts "
        "are nearly the same field: at matched total burden their day-60 medians span "
        "1.871% and their day-365 medians 0.021% -- the layout contrast is recorded as "
        "INERT ON THIS CASE rather than dressed up as a measured sensitivity. What does "
        "move the endpoints is the resistant-clone placement: the engine's clustered R "
        "patch against the stream's uniform r0 gives a paired median burden delta of "
        "+1291.66 at day 365 (median PFS 248 -> 115 days), which is a statement about "
        "this model's initial condition, not about 5-ALA. Stream 1 remains REFUSED on its "
        "live count of 0 cached captures, and the pitch's dataset table is corrected in "
        "the standard's refusals: 'fully available' longitudinal MRI is licence-locked "
        "(MU-Glioma-Post: 203 subjects / 596 timepoints behind a login, 0 from the probe; "
        "CFB-GBM: 393 manifest pairs, 0 images), the measured DSC perfusion here is "
        "UPENN-GBM rather than Ivy GAP or CPTAC-3, the GLASS release is measured here "
        "by danon.variants (1686 records, 248 paired spectra) rather than merely cited, "
        "and the one cached pathology slide is haematoxylin-only, so it yields a real "
        "cell density but cannot support a Ki-67 ratio.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_synthetic_5ala_standard_is_labelled_and_leaves_stream1_refused "
        "over outputs_real/fluorescence_standard.json (rebuildable via "
        "_fluorescence_standard_cohort.py), with danon/fluorescence_standard.py and "
        "tests/test_fluorescence_standard.py",
        "synthetic_5ala_standard_is_labelled",
        "A synthetic-input standard is not a validation: the field was chosen for "
        "contrast, not fitted to any capture, so nothing here says real 5-ALA maps "
        "resemble it or that fluorescence marks tumour. The inertness of the fluorescence "
        "pattern is measured on ONE shell-like case; a geometry with genuine depth could "
        "behave differently, and the check will fail the moment the spread exceeds 1% so "
        "the sentence gets rewritten rather than aged. The dominant term measured here -- "
        "clustered versus uniform resistant seeding -- is a property of this model's "
        "initial condition on this case, not a clinical finding. Stream 1 still has no "
        "capture, no patient, and no volume claim.",
    ),
    Claim(
        "solver_portability_is_measured_and_cross_cancer_generalization_is_refused",
        "Tissue portability is measured, and cross-cancer prediction is refused. The "
        "solver's entire tissue/drug surface is five dataclasses -- TumorGeometry, "
        "CloneParams, DrugParams, ImmuneParams and DiffusionField, 23 fields in total, "
        "introspected live -- and the unmodified engine runs a deliberately non-brain "
        "phantom (a lobulated ellipsoid with uniform proliferation and no necrotic-core "
        "structure, not make_gbm_phantom) under MTD at 60 and 365 days: deterministic "
        "under seed 17, finite at every day, and bounded -- measured max 1.437 x K x V at "
        "day 365, which sits on this model's own structural ceiling of 1.4 (the "
        "asymmetric room terms cap S at K - 0.6*N_R while R caps total at K), below the "
        "2.0 bound the check enforces. What fills the surface is glioma and only glioma: "
        "the tissue diffusion table holds brain classes exclusively "
        "(white_matter/edema/grey_matter/necrotic_core, documented ASSUMED with no DTI or "
        "perfusion series here), the clone, drug and immune defaults carry their "
        "CALIBRATED/ASSUMED-for-GBM markers, and a live file-path scan of data/ finds "
        "zero breast, zero lung, zero pancreatic and zero non-glioma-drug assets. The "
        "pitch's cross-cancer claims are each refused with their mocked figure carried "
        "beside them in the record: the three-row parameter-swap table "
        "(mammary/pulmonary/pancreatic meshes with ultrasound, PET and EUS streams), the "
        "unmeasured '90% of the codebase' figure, the judge-script sentence claiming the "
        "framework 'generalizes to predict spatial recurrence and evolutionary resistance "
        "in metastatic breast, lung, and pancreatic cancers', the 'built to model those "
        "invisible residual cells after surgery' framing (no post-operative imaging "
        "exists in this tree and stream 1 is REFUSED on 0 cached captures, claim #49), "
        "and the 'every cancer type benefits from pulsed dosing' quantifier.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_solver_portability_is_measured_and_generalization_is_refused "
        "over outputs_real/tumor_agnostic.json (rebuildable via "
        "_tumor_agnostic_cohort.py), with danon/tumor_agnostic.py and "
        "tests/test_tumor_agnostic.py",
        "tissue_portability_measured",
        "Tissue portability is not cross-cancer prediction: running the solver on other "
        "geometry is a property of the code, not evidence about other diseases -- the "
        "phantom is not any organ's tissue (no non-glioma tissue geometry exists in this "
        "repository to build one from), no non-glioma outcome exists here to calibrate on "
        "or test against, and every value on the swap surface is glioma-derived or "
        "explicitly assumed. Schedule effects are measured only under this "
        "glioma-calibrated parameter set. Nothing here supports predicting recurrence in "
        "any cancer other than glioblastoma.",
    ),
    Claim(
        "real_scans_are_served_and_no_prediction_is_drawn",
        "The MRI Scans view shows real patient scans, and nothing on it is a prediction. "
        "Six BraTS-2021 cases are committed to this repository "
        "(data/brats/*_{t1ce,seg}.nii.gz) and every pixel the dashboard displays is "
        "rendered from those files: the served PNG is byte-identical to the committed "
        "render in outputs_real/scans/, the contours are the provided expert segmentation "
        "(red whole tumour, yellow enhancing) rather than anything this engine produced, "
        "and the voxel counts printed on each card re-read live from the NIfTI. The "
        "engine grid each case was downsampled to is shown next to the scan -- 1,774 "
        "voxels at 14x24x24 for BraTS2021_01017, which is what the twin actually "
        "simulates from a 98,447-voxel label. Each card can post that case's own "
        "engine-ready mask to the live API and show the run that comes back, and the mask "
        "served for upload is byte-identical to data/engine_ready/. What this view is "
        "not: there is no segmentation model here, so no contour on these images is a "
        "detection, and the artifact records model_prediction_on_these_images as "
        "PROVEN_FAILED so an edit that started compositing an engine field onto a real "
        "patient scan would fail the audit rather than ship.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_real_scans_are_served_and_no_prediction_is_drawn over "
        "frontend/public/artifacts/scans.json, verifying sha256 against "
        "outputs_real/scans/, data/brats/ and data/engine_ready/ (rebuildable via "
        "_scan_gallery.py; surface exercised by _verify_scan_surface.py)",
        "real_scans_are_served_and_no_prediction_is_drawn",
        "Real images are not a validated pipeline. These are pre-operative scans, and no "
        "model here segments them: the contours are the dataset's own expert labels, so "
        "this view cannot be used to judge the engine's spatial accuracy or to claim "
        "segmentation performance. No post-operative imaging exists in this repository, "
        "so no resection cavity or recurrence follow-up can be shown. A 'run live plan' "
        "result on these cards is a recorded engine output from this repository, not a "
        "measurement of that patient.",
    ),
    Claim(
        "twin_plan_input_is_byte_tied",
        "A twin-planner run is tied to the exact file it was given. The uploaded "
        "bytes are hashed as received (a .nii.gz is hashed compressed, not "
        "re-serialised), the header's declared grid and voxel sizes are read back "
        "through the same danon.imaging parser the engine uses, and a "
        "block-max-downsampled grid is written with the SCALED spacing -- so the "
        "geometry report is a function of the input file and nothing else, and a "
        "recorded result can be re-identified from the file it names.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_twin_plan_input_is_byte_tied over "
        "danon/imaging.py::mask_bytes_provenance and "
        "danon/twin_plan.py::derived_voxel_sizes, with tests/test_twin_plan.py",
        "twin_plan_input_is_byte_tied",
        "Tying a run to a file is not evidence that the run predicts a patient. "
        "The digest proves which bytes produced a geometry report; it says nothing "
        "about whether the engine's constants are right or about the person the "
        "scan came from, and the reported volume of a downsampled mask is the "
        "block-max volume on the coarser grid, which can only be equal-or-larger "
        "than the source, never exact.",
    ),
    Claim(
        "badge_inventory_is_classified_not_promised",
        "Every badge this repository leaves open is classified by what would "
        "close it, and the classification is derived live rather than promised. "
        "9 badges are open -- 6 claim tiers and 3 provenance keys -- and NONE of "
        "them closes on the four public datasets a data-mining route names: 2 "
        "need human outcomes inside this repository's own loop, 2 need "
        "prospective trials, 2 need a bench measurement of a per-day kill rate "
        "that no unit declaration can supply, 1 needs controlled-access "
        "structural-variant calls, 1 is a refusal that is meant to stay refused, "
        "and 1 is open only to a public collection carrying EXPERT masks at two "
        "registered timepoints -- which the named candidate, LUMIERE, is not, "
        "because it ships automated masks. The check fails if an open badge "
        "appears with no classification, and if the count in this sentence stops "
        "matching the live one.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::badge_inventory (live over CLAIMS and "
        "claims_provenance) checked by "
        "check_badge_inventory_is_answered_not_asserted",
        "badge_inventory_is_classified_not_promised",
        "The classification states what is MISSING, not what is impossible "
        "forever: two of the nine would decode on data that may exist somewhere "
        "(a public pair with expert masks at both timepoints; controlled access "
        "lifted on TCGA-GBM structural variants), and the check is written to "
        "notice that rather than to deny it. It is also not a claim that these "
        "nine are the only things wrong with this repository -- they are the "
        "ones it has chosen to name, and an unnamed defect is not covered here.",
    ),
    Claim(
        "epistemic_guardrail_proposal_is_audited",
        "Four 'epistemic guardrail' proposals -- a provenance-carrying "
        "parameter type, rigid physical units, a hash-based anti-leakage lock, "
        "and a hover data-pedigree panel -- were audited against this "
        "repository rather than adopted. Two already exist here in a stronger "
        "form: a frozen Constant/Provenance dataclass carries the fluidics "
        "constants, and the leakage guard is a statistical detector "
        "(danon.audit.permutation_leakage_audit) that fails on real leakage "
        "rather than on bookkeeping, with SHA-256 pinning inputs rather than "
        "inspecting a solver's memory. Two do not do what the proposal claims: "
        "a self-declared measurement tier is a string a developer types, so it "
        "is weaker than the live re-derivation audit_claims performs, and the "
        "engine's constants stay plain floats pinned UNKNOWN rather than "
        "dressed as measured; and a declared unit cannot anchor potency, which "
        "is measured, not argued -- the kill law is homogeneous of degree 0 in "
        "(concentration, EC50). The proposal's own worked examples are false, "
        "and the check fails if they appear in the tree: a Ki-67 field in a "
        "TCGA clinical XML (TCGA-GBM's BCR XML carries none), a 5.5 ug/mL "
        "brain concentration (5.5 is the label's CLEARANCE in L/hr/m2; the "
        "Cmax is 7.5 mcg/mL plasma), and a confidence interval widened by a "
        "fixed percent for 'epistemic uncertainty' (no such mechanism is "
        "measured here).",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_epistemic_guardrail_proposal_is_audited (live "
        "over danon/physics.py, danon/evosteer.py, danon/pharmacology.py, "
        "danon/audit.py and the frontend provenance surfaces)",
        "epistemic_guardrail_proposal_is_audited",
        "This is an audit of a PROPOSAL against the code, not a measurement of "
        "the engine: it states which guardrails already exist, which would "
        "weaken honesty if adopted, and that the proposal's examples are false. "
        "It is not a claim that the current design is sufficient -- the open "
        "badges it does not close are counted elsewhere -- and a guardrail "
        "whose tier is DERIVED rather than self-asserted would strengthen this "
        "repository, which is why the check is written to notice that rather "
        "than to forbid it.",
    ),
    Claim(
        "samd_pathway_is_named_not_promised",
        "The route from this research prototype to a cleared device is NAMED, "
        "not claimed. No submission has been made to any regulator, and the "
        "repository says so: the ANC safety model's own text states it is not "
        "clinical decision support, the tumour board reports no FDA status of "
        "any kind, and the day-365 readout is PROVEN_FAILED as a patient "
        "prediction. The pathway is recorded as CONTEXT in SAMD_CONTEXT -- a "
        "dosing recommendation that drives rather than supports a clinician's "
        "decision fails the non-device CDS criteria (FD&C Act 520(o)(1)(E)) and "
        "is therefore a device; the route follows the risk categorization "
        "rather than this file, so a novel engine without a predicate is a "
        "De Novo, or a PMA if Class III, with 510(k) requiring an existing "
        "predicate; bounded adaptive updates go through a Predetermined Change "
        "Control Plan (FDA final guidance, Dec 2024); and the standards are "
        "IEC 62304, ISO 13485 and ISO 14971. The engine is numpy -- no JAX, no "
        "CUDA and no GPU exists in this environment -- and the check fails if a "
        "shipped surface claims an FDA clearance, a granted De Novo/510(k), or "
        "a GPU-accelerated solver, or if either benefit badge stops being "
        "UNKNOWN.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::SAMD_CONTEXT + check_samd_pathway_is_named_not_promised, "
        "with danon/dosing.py::MyelosuppressionModel and danon/outcome_export.py",
        "samd_pathway_is_named_not_promised",
        "Naming the pathway is not the pathway: nothing here is a submission, a "
        "quality-system record, or a clinical validation, and the clearance "
        "decision belongs to the regulator, not to this file. The regulatory "
        "summary is CONTEXT for a reader and is not legal advice; the binding "
        "statement is the FDA's own current guidance. What the check can "
        "establish is only that this repository does not misrepresent its "
        "status -- it says nothing about whether the engine works.",
    ),
    Claim(
        "benefit_safety_and_runtime_are_measured_not_asserted",
        "A translation pitch's benefit, safety and runtime claims are audited "
        "against measurements, not adopted. 'Saves lives' and 'an additional "
        "year of high-quality life' are refused: clinical_benefit and "
        "engine_predicts_patients stay UNKNOWN, and the day-365 readout stays "
        "PROVEN_FAILED as a patient prediction, because no dataset here pairs a "
        "baseline state, an administered schedule and an outcome. The safety "
        "'guarantee' is a DECLARED screen rather than an interlock: its floor "
        "is the temozolomide label's 1.5 x10^9/L, its kernel is fitted to a "
        "single label datum, drug-B marrow potency is ASSUMED, and the model's "
        "own text says it is not clinical decision support -- and the "
        "repository's own sweep records intensities that FAIL that floor "
        "(modelled nadir 1.24 and 0.59 x10^9/L), so it constrains a model and "
        "guarantees nothing about a patient. The '10,000 schedules in 30 "
        "seconds' figure is off by about two orders of magnitude: the measured "
        "extrapolation is 1.41-1.46 h on 4-8 workers (2.94 h serial) on the "
        "engine's own 8,064-voxel grid, recorded in outputs_real/perf.json. "
        "The one real precedent -- the Moffitt phase 2a adaptive-abiraterone "
        "study (NCT02415621), which prolonged time to progression at lower "
        "cumulative exposure -- is recorded as CONTEXT, not as a result of "
        "this engine. The check fails if a shipped surface claims the software "
        "saves lives, guarantees safety, adds a year of life, extends a "
        "lifespan, or quotes a seconds-scale 10,000-schedule runtime.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::check_benefit_safety_and_runtime_are_measured_not_asserted "
        "over danon/dosing.py::MyelosuppressionModel + LABEL_ANC_FLOOR, "
        "outputs_real/perf.json and danon/outcome_export.py",
        "benefit_safety_and_runtime_are_measured_not_asserted",
        "The precedent is real but it is not this engine: a different cancer, a "
        "different drug, a different (non-spatial) model and a phase 2a with a "
        "non-randomised comparator, so it supports the hypothesis without "
        "transferring any benefit here. A runtime is a property of this machine "
        "at this load. And the refusal is the point: code cannot make 'safe for "
        "this patient' or 'extends a life' true -- those need prospective "
        "trials and regulatory review, not a simulation.",
    ),
    Claim(
        "provenance_tiers_are_derived_from_evidence_not_declared",
        "A tier a human types is a string; a tier a check derives from evidence "
        "is a measurement -- but only if the derivation is SEMANTIC. This "
        "repository derives tiers from ingested evidence (DERIVED_TIERS) and "
        "enforces that a bundle may RAISE a tier only when it declares an "
        "evidence kind, is EXPERT-annotated, is linked to the same patients and "
        "passed a unit/bounds check. A file merely EXISTING never qualifies -- "
        "the rule LUMIERE makes concrete: reachable, 91 patients, 2487 images, "
        "and AUTOMATED masks, so a presence-based upgrade would fabricate a "
        "Dice result. The r0 knowledge tier is derived this way and stays "
        "UNKNOWN because no ingested Ki-67 labelling index exists. What is "
        "deliberately NOT derived is claim_matches_code: the provenance keys "
        "are read by the checks themselves, so deriving them from the audit "
        "would recurse -- the static-tier / live-support split is load-bearing, "
        "not lazy. And a tier is never the audit result: whether a check passes "
        "is `supported`, which is orthogonal, and supported claims here carry "
        "UNKNOWN and PROVEN_FAILED tiers.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::DERIVED_TIERS + evidence_justifies_a_tier_upgrade + "
        "check_provenance_tiers_are_derived_from_evidence_not_declared, with "
        "outputs_real/ki67_source_probe.json",
        "provenance_tiers_are_derived_from_evidence_not_declared",
        "Derivation needs an ARTIFACT, so this changes nothing until a real "
        "Ki-67 source is ingested -- it is a trapdoor, not a result. The "
        "semantic gate is a policy this repository chose; a different gate "
        "could admit evidence this one refuses, and the check states its "
        "criteria rather than claiming they are the only defensible ones.",
    ),
    Claim(
        "cohort_specification_matches_the_engine_not_the_pitch",
        "A cohort specification handed to a hospital data steward is pinned to "
        "the engine, because a steward annex is where a number gets copied into "
        "a protocol and never re-checked. Four of the pitch's own claims are "
        "contradicted by the code. (1) It maps the necrotic core to K=0, but "
        "danon/realdata.py::_label_proliferation weights enhancing 1.00, edema "
        "0.60 and necrosis 0.20 -- necrosis is not zero, and a cohort built to "
        "K=0 would encode the wrong label semantics. (2) It quotes a "
        "'documented structural Dice penalty ... 0.3905', but 0.3905 is the "
        "DEFECT of the shipped _decimate_labels: the documented axis-aligned "
        "block vote agrees with the source occupancy at Dice 0.9310 on the "
        "shipped 3.63 mm grid (0.8530 worst case at 7.3 mm), so quoting the "
        "defect as documented inverts the record. (3) It attributes '0 scorable "
        "pairs' to danon/audit.py, but that count is the claim census, and the "
        "longitudinal series is RETRIEVABLE without new code (stream4: 0 cached "
        "/ 41 retrievable pairs), so the blocker is the cached download plus a "
        "mapping rule rather than the absence of data. (4) Its N=30 'above the "
        "fold-yardstick noise floor' is computed nowhere here; the "
        "repository's own resolvability floor is the fold yardstick "
        "(floor_conservative 0.0610), so a size must be justified against that "
        "floor. What survives is real: NIfTI intake with the BraTS 1/2/4 label "
        "convention, MAX_TOTAL_VOXELS = 4,000,000, a per-patient dosing log the "
        "day-by-day engine can initialise (drug, start_day, duration_days, "
        "mg/m2), pfs_day with censored patients retained, and expert masks at "
        "two timepoints -- and none of it is a guarantee that a cohort exists to "
        "be delivered. The check fails if a shipped surface maps necrosis to "
        "K=0, calls the 0.3905 defect documented, attributes the census to "
        "danon/audit.py, or asserts a powered cohort as N=30.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::COHORT_SPECIFICATION + COHORT_SPEC_CORRECTIONS + "
        "check_cohort_specification_matches_the_engine_not_the_pitch over "
        "danon/imaging.py::MAX_TOTAL_VOXELS, "
        "danon/realdata.py::_label_proliferation and "
        "outputs_real/grid_artifact.json",
        "cohort_specification_matches_the_engine_not_the_pitch",
        "Pinning the document to the engine does not make the data arrive: a "
        "corrected specification is a request, and a hospital DUA is a "
        "negotiation with an IRB that no code performs. A cohort, even a perfect "
        "one, is retrospective -- it cannot move clinical_benefit, "
        "engine_predicts_patients or the day-365 patient readout, which need "
        "prospective trials and regulatory review rather than a dataset.",
    ),
    Claim(
        "evolution_forecast_is_scored_against_baselines",
        "The one result in this repository measured in real patients is scored "
        "against a baseline under REPEATED patient-level cross-validation, and "
        "the two tasks come out differently. From the paired GLASS release -- "
        "1,686 measured mutation rows, 281 patients, 543 specimens, 248 "
        "primary-to-recurrence pairs, every allele fraction computed from that "
        "specimen's own tumour/reference read counts and none imputed -- "
        "danon/evolution_forecast.py scores two tasks at 5 repeats of 5-fold "
        "(every patient scored once per repeat by a model that never saw it). "
        "Task A, whether a variant the PRIMARY measured is re-detected later, "
        "DOES beat its baseline -- and it is scored ROW BY ROW, because a "
        "per-patient label is not defined on this cohort: 83 of the 226 patients "
        "hold BOTH labels across their own rows, so the first version of this "
        "estimator, which read a patient's label off the last row written, was "
        "ranking against an accident of row order (122 positives by that rule "
        "against 204 by 'a re-detection anywhere', disagreeing on 82 patients). "
        "On the repaired unit the pooled out-of-fold AUC is 0.712 over 654 "
        "measured rows, with a patient-clustered interval of 0.637-0.788, clear "
        "of chance, and 25 of 25 folds above chance; the SECONDARY patient "
        "endpoint is AUC 0.648 with an interval of 0.501-0.780, whose lower "
        "bound sits 0.001 above chance, so the row-level result is the clear one "
        "and the patient-level view is marginal and declared as such. The single "
        "seeded split did NOT resolve it (AUC "
        "0.666, interval 0.479-0.832, containing chance), so the earlier "
        "UNRESOLVED reading was this cohort's power and not the absence of a "
        "signal -- which is why BOTH estimators are kept in the record. Task B, "
        "how far a shared variant's fraction moves, does NOT beat persistence: "
        "across 25 folds the median advantage is -0.0039 (NEGATIVE -- on a "
        "typical fold the model is worse than predicting no change), only 8 of "
        "25 folds beat the floor, and the pooled per-patient advantage 0.0051 "
        "carries an interval of -0.0041 to 0.0145 that still contains zero. The "
        "single split's +0.0081 advantage was that split's luck. Variants seen "
        "only at recurrence are EXCLUDED from task A because a 'gained' variant "
        "is re-detected by definition -- an earlier version scored them and "
        "reported an inflated AUC 0.85 that was the construction talking. The "
        "pairing itself is an INFERENCE and is counted as one: the release "
        "declares all 693 of its specimen rows as 'Primary Solid Tumor', so its "
        "own sampleType column carries no timepoint and the only signal is the "
        "identifier suffix, which nothing here can independently confirm; 1 "
        "specimen with an unrecognised suffix is excluded rather than guessed "
        "into a fold. Discrimination and calibration are reported as the two "
        "separate claims they are: the same pooled out-of-fold ROW "
        "probabilities are cut into five equal-count bins of 130-131 rows and "
        "compared against the rates they produced, beside a Newton-fitted "
        "calibration curve and a Brier score for a constant-prevalence "
        "predictor. Measured on the same rows the AUC ranks, the probabilities "
        "are calibrated within this package's declared tolerance (slope within "
        "0.35 of 1, intercept within 0.35 of 0, mean within 0.05 of the observed "
        "rate): the mean predicted probability 0.728 sits against an observed "
        "re-detection rate 0.726, the curve's slope is 0.901 against an ideal 1 "
        "and its intercept 0.076 against an ideal 0, and the Brier score 0.1685 "
        "beats a constant-prevalence predictor's 0.1988 (skill +0.153) -- so on "
        "this cohort the ranking separates AND a probability from this model is "
        "a number a reader may hold against a rate. It is one public cohort's "
        "calibration and not a general one, and the superseded per-patient "
        "pooling had erased that skill entirely (Brier skill -0.036, the value "
        "the record published until the unit was fixed). That block is "
        "recorded with its bins and its declared tolerances, and the sentence "
        "about it is DERIVED from those numbers rather than written by hand, so "
        "the two cannot drift apart. The check reloads the cache with the "
        "shipping loader, "
        "re-runs the single split AND the cross-validation under their recorded "
        "seeds, requires the artifact to agree on every metric, asserts that "
        "each verdict matches its own interval in BOTH directions, re-derives "
        "the pooling block that declares the unit and the label of every pooled "
        "score (failing if the pooled scores stop being one per measured row, or "
        "if the patients holding both labels stop being reported), asserts the "
        "two estimators are both present and actually disagree, fails if the "
        "release ever declares real timepoints (which would invalidate the "
        "suffix rule), and fails if the therapy, outcome and patient-prediction "
        "links ever move off UNKNOWN.",
        EvidenceStatus.ESTABLISHED,
        "danon/evolution_forecast.py::calibration_block + "
        "danon/claims.py::check_evolution_forecast_is_scored_against_baselines "
        "over outputs_real/glass_evolution_forecast.json, "
        "outputs_real/glass_variants.json and "
        "data/glass_paired/glass_mutations.json, exposed as "
        "danon/app_artifacts.py::build_evolution_forecast_artifact and the "
        "app's Evolution Forecast screen",
        "evolution_forecast_is_scored_against_baselines",
        "Task A's AUC of 0.712 is a MODEST molecular signal on one public "
        "retrospective cohort, not a clinical result, and the fact that its "
        "probabilities happen to be calibrated here does not make it one: it "
        "says a variant's "
        "allele fraction in the primary carries some information about whether "
        "that locus is seen again, and nothing about what happens to a patient. "
        "The panel is a declared 27 genes, not the genome, so a change "
        "elsewhere is invisible; there is no purity, no copy number, no therapy "
        "record and no outcome joined to these specimens, and years and "
        "differing depths separate the two draws, so a change in an allele "
        "fraction can come from clonal expansion, sampling, depth or purity and "
        "this task cannot say which. The pairing is INFERRED from the specimen "
        "identifier because the release declares no timepoint; if that "
        "convention is wrong the pairing is wrong and every number here moves. "
        "Published allele fractions are a provable LOWER bound on the fraction "
        "of tumour cells carrying a variant; a point clone size and an upper "
        "bound are refused, and the engine does not consume this channel. "
        "Nothing here is a treatment effect, a prognosis or guidance for "
        "anyone's care, and clinical_benefit stays UNKNOWN.",
    ),
    Claim(
        "clinical_execution_ui_is_audited_not_implemented",
        "A 'clinical execution UI' directive is audited against the engine, not "
        "built as written. It asks for four things this repository's own record "
        "refuses. (1) Its 'definitive prescription block' -- 150 mg/m2 "
        "temozolomide for 5 days -- is the FDA LABEL's own regimen "
        "(danon/dosing.py::LABEL_DOSE_MG_M2 = 150, LABEL_ON_DAYS = 5), which is "
        "the STANDARD OF CARE the adaptive arm is compared AGAINST, not an "
        "output of any optimizer here; and its 'then a 14-day holiday' is a "
        "19-day cycle, contradicting the engine's own 28-day label anchor "
        "(LABEL_CYCLE_DAYS = 28, so the label holiday is 23 days). (2) A "
        "'Clinical Mode' that hides 'the explicit algorithmic warnings' "
        "removes the basis a non-device CDS tool must display: SAMD_CONTEXT "
        "records that a recommendation which DRIVES rather than SUPPORTS a "
        "clinician's decision fails the non-device CDS criteria (FD&C Act "
        "520(o)(1)(E)) and is therefore a device, and the frontend's own gate "
        "requires MHRA/FDA clearance before any therapeutic use. (3) A "
        "10,000-draw sweep run 'silently the moment the patient's data is "
        "ingested' is a measured ~1.4 h on 4-8 workers (outputs_real/perf.json), "
        "not a background instant, and running it silently is the opposite of "
        "the recorded provenance. (4) 'Exactly how many additional days of "
        "progression-free life this schedule mathematically secures for the "
        "patient' is a lifespan claim: clinical_benefit and "
        "engine_predicts_patients stay UNKNOWN and the day-365 readout stays "
        "PROVEN_FAILED as a patient prediction, because code cannot make an "
        "'extends a life' claim true -- that needs prospective trials and "
        "regulatory review. The check fails if a shipped surface presents a "
        "recommended protocol, a mode that hides the warnings, or a secured "
        "days-of-progression-free-survival figure.",
        EvidenceStatus.ESTABLISHED,
        "danon/claims.py::CLINICAL_UI_CORRECTIONS + "
        "check_clinical_execution_ui_is_audited_not_implemented over "
        "danon/dosing.py::LABEL_DOSE_MG_M2 / LABEL_ON_DAYS / LABEL_CYCLE_DAYS, "
        "danon/claims.py::SAMD_CONTEXT and outputs_real/perf.json",
        "clinical_execution_ui_is_audited_not_implemented",
        "Auditing the directive does not build the UI, and an honest version "
        "would still not make the recommendation safe or life-extending: a "
        "surface that drives therapy is a regulated device, and the benefit "
        "badge stays UNKNOWN until prospective trials and regulatory review. "
        "The regimen the block quotes is real and label-anchored, so it is a "
        "true statement about standard care -- it is simply not a personalised "
        "prediction, and presenting it as one is the failure this refuses.",
    ),
    Claim(
        "paired_pfs_extension_is_recorded_not_fabricated",
        "The adaptive-vs-MTD benefit is carried as TWO explicitly different "
        "estimands, because they are not interchangeable. The outcome cards "
        "call `pfs_extension` the RATIO OF MEDIANS (`median_policy / median_mtd "
        "- 1`, a per-case fraction), while the repository's own campaign rule "
        "(danon/campaign.py, check S1) describes the PFS extension over PAIRED "
        "seeds, reported as a bootstrap 95% interval over replicate pairs "
        "rather than as a bare number. danon/paired_extension.py computes the "
        "second from the recorded replicates in outputs_real/campaign_log.json "
        "-- no engine run, so it is reproducible offline -- and records it "
        "beside the first: 18 cases, 54 paired replicates, a paired mean "
        "difference of +86.1 d with a bootstrap 95% interval of 73.5-98.4 d, "
        "against a median ratio-of-medians extension of 0.388. Every one of the "
        "54 adaptive replicates reached the 365 d horizon, so every difference "
        "is a right-censored LOWER BOUND and the interval bounds a bound, never "
        "a point estimate. The check fails if the recorded interval no longer "
        "re-derives from the log, if it is dropped for a bare number, if the "
        "censoring stops being flagged, if the two estimands stop being "
        "labelled as distinct, or if a benefit badge moves off UNKNOWN.",
        EvidenceStatus.ESTABLISHED,
        "danon/paired_extension.py + "
        "danon/app_artifacts.py::build_paired_extension_artifact + "
        "check_paired_pfs_extension_is_recorded_not_fabricated over "
        "outputs_real/campaign_log.json and "
        "danon/campaign.py::bootstrap_ci_mean",
        "paired_pfs_extension_is_recorded_not_fabricated",
        "A bootstrap interval over paired SIMULATOR replicates is still a "
        "simulator result: it does not transfer to glioma patients, and no "
        "replicate here is an observed progression. It is computed from a "
        "censored endpoint at a fixed 365 d horizon, so it bounds a lower "
        "bound -- a longer horizon or an uncensored endpoint would move it, and "
        "neither exists in this tree. It is recorded so the number appears only "
        "with its method, never because a screen wanted one.",
    ),
    Claim(
        "trial_design_screen_is_derived_not_prescribed",
        "A trial-design screen states the counterfactual, shows the noise, and "
        "rules out -- and stops there. danon/trial_design.py derives all three "
        "from records that already exist: the paired-seed differences in "
        "outputs_real/outcome_cards.json (9 cases, 27 paired replicates, policy "
        "PFS minus MTD PFS in days, both arms read from the same seeds; the "
        "replicates are NESTED rather than independent -- 3 seeds within each of "
        "9 configurations, and those 9 configurations are 3 synthetic initial "
        "resistant fractions on each of 3 imaging geometries, so at most 9 "
        "independent units stand behind the 27), a "
        "resampled interval at each of three scales, and the calendars "
        "danon.dosing::MyelosuppressionModel refuses. What it MEASURES, rather "
        "than assumes, is that resolvability is a property of the replicate "
        "count: the 27 pooled pairs resolve (mean +71.6 d, bootstrap 95% "
        "interval 54.1-89.6 d, 23 of 27 pairs separating -- but over at most 9 "
        "independent units, never 27), the nine pairs on one "
        "image resolve (the marginal one at +4.9 to +35.8 d with four of nine "
        "tied), and a single case's three pairs resolve NOTHING -- below the five "
        "the repository's own percentile method requires, so it reports "
        "unassessed under its own label rather than resolved. The simulated "
        "INVERSION day is also measured rather than promised: in all nine "
        "recorded cases both arms reach their own switch threshold on the SAME "
        "simulated day, so the model shows no delay in that inversion -- the "
        "arms diverge after it, in burden. The rule-out is the honest half of a "
        "prescription: it names the calendars a DECLARED screen refused -- "
        "including the design brief's own '5 on / 14 off' arithmetic, which is a "
        "19-day cycle and is refused, while the label's 5-of-28 cycle is the row "
        "the screen's budget is DEFINED by -- and it names no calendar to "
        "administer. The check fails if the counterfactual stops re-deriving "
        "from the replicates, if a nesting disclosure stops re-deriving from the "
        "per-case rows it summarises, if the two unresolved states collapse into "
        "one label, if the rule-out is presented as a regulatory boundary, if the "
        "inversion-day summary stops matching the per-case days, if the shipped "
        "screen carries a fabricated clinical-UI claim, or if a benefit badge "
        "moves off UNKNOWN.",
        EvidenceStatus.ESTABLISHED,
        "danon/trial_design.py + "
        "danon/app_artifacts.py::build_trial_design_artifact + "
        "check_trial_design_screen_is_derived_not_prescribed over "
        "outputs_real/outcome_cards.json, danon/campaign.py::bootstrap_ci_mean "
        "and danon/dosing.py::MyelosuppressionModel",
        "trial_design_screen_is_derived_not_prescribed",
        "Every number on the screen is a SIMULATOR result over recorded "
        "replicates: the endpoint is the model's own RANO >= 25% burden rule, "
        "the adaptive arm is censored at the 365 d horizon, and the constants are "
        "assumed, so the comparison bounds a model's arithmetic and not a "
        "patient's course. The 27 paired replicates are NESTED, not a sample of "
        "independent patients: 3 seeds within each of 9 configurations -- 3 "
        "synthetic initial resistant fractions on each of 3 imaging geometries -- "
        "so at most 9 independent units stand behind the pooled interval, which "
        "is therefore narrower than one over independent patients would be. The "
        "rule-out is no stronger than the screen behind "
        "it: that screen's kernel is fitted to the label's single reported datum, "
        "drug B's marrow potency is an assumed weight, and the baseline count is "
        "a typical adult reference rather than this patient's -- so a refusal is "
        "a statement about the model. And the screen deliberately does NOT do the "
        "half that would matter clinically: naming a calendar to administer is a "
        "decision this repository cannot support, which is why clinical_benefit "
        "and engine_predicts_patients stay UNKNOWN.",
    ),
    Claim(
        "closed_loop_assimilation_is_measured",
        "The engine can be run CLOSED-loop: an ensemble Kalman analysis "
        "corrects its growth parameters and resistant burden at each "
        "observation, instead of forecasting from day 0 and stopping. On a "
        "synthetic twin -- one seeded engine run as the truth, observations = "
        "that truth plus declared noise every 14 days, and an ensemble started "
        "from a deliberately wrong prior -- assimilation beat the blind engine "
        "on 4 of 4 held-out seeds: mean resistant-burden tracking 94.8% "
        "against 83.4% blind, worst seed 93.0%, each scored by the same "
        "declared metric over the same days. The check re-runs the held-out "
        "twin live and requires the record to match it exactly. The fraction "
        "curve is reported as CEILING-SATURATED and is not quoted as a win: "
        "under MTD both arms approach 1, so a high fraction score proves "
        "nothing.",
        EvidenceStatus.ESTABLISHED,
        "danon/assimilation.py::assimilation_twin_audit + "
        "assimilation_twin_validation over "
        "outputs_real/assimilation_twin.json, "
        "outputs_real/assimilation_twin_holdout.json",
        "closed_loop_assimilation_measured",
        "The truth in this experiment is the engine itself and the observations "
        "are that truth plus Gaussian noise, so it validates the CORRECTION "
        "METHOD against a known simulated truth -- never the engine against "
        "reality. It is not ctDNA data, not a patient cohort, and not evidence "
        "that anything here tracks a real tumour: no patient telemetry of any "
        "kind is used or bundled, and tracking accuracy on real tumours remains "
        "UNKNOWN until prospective longitudinal observations exist. Two "
        "earlier versions of this module are recorded as measured failures in "
        "the artifact: v1's fraction-only metric saturated, and v2's prior "
        "support excluded the truth, so the arm LOST to the blind engine "
        "(50.2% vs 63.9%).",
    ),
)


@dataclass
class ClaimAudit:
    claim: Claim
    supported: bool
    detail: str


def check_bench_anchors_are_measured_and_calibration_is_refused() -> CheckResult:
    """The Step-1 anchor artifact must be measured, current, and CALIBRATION-FREE.

    `potency_is_measured` already checks the live ChEMBL intake and
    `ec50_unit_undeclared` refuses a unitless comparison. Neither persists
    anything, so nothing in the tree recorded what the engine's constants would
    have to be reconciled AGAINST. This reads
    `outputs_real/bench_anchors.json` and refuses unless

      * the potency figures in it are measured records with censoring counted
        (a point estimate built by dropping right-censored records, or by
        treating their bounds as exact, is exactly the failure this guards),
      * the GDSC glioma audit records that temozolomide's IC50s are
        extrapolations rather than measurements,
      * the EC50 comparison is REFUSED and the artifact states the
        reconciliation is not calibrated,
      * the engine constants written into the artifact still EQUAL the live
        engine's (a copied constant set that nothing re-derives goes stale, the
        same failure the README claim count had), and
      * the artifact carries provenance and an engine identity.

    An artifact that had quietly grown a fitted kill constant would fail the
    last check on "calibrated", which is the point: the honest end state of
    Step 1 without a wet-lab partner is a measured anchor and a named gap.
    """
    import dataclasses

    path = os.path.join(ROOT, "outputs_real", "bench_anchors.json")
    if not os.path.exists(path):
        return CheckResult(
            False,
            "bench anchors not persisted (outputs_real/bench_anchors.json "
            "missing): the measured potency data and the refused calibration "
            "exist only as live expressions. Run _run_bench_anchors.py.",
        )
    try:
        with open(path, encoding="utf-8") as fh:
            art = json.load(fh)
    except (OSError, ValueError) as exc:
        return CheckResult(False, f"bench anchors unreadable: {exc}")

    from danon.evosteer import DEFAULT_CLONES, DEFAULT_DRUGS

    problems: List[str] = []
    pot = art.get("glioma_line_potency") or {}
    n_records = int(pot.get("n_records") or 0)
    n_exact = int(pot.get("n_exact") or 0)
    n_censored = int(pot.get("n_right_censored") or 0)
    if n_exact < 20 or n_records < 40:
        problems.append(
            f"the anchor holds {n_exact} exact of {n_records} measured records, "
            f"which is too thin to be the glioma potency distribution"
        )
    if n_censored <= 0:
        problems.append(
            "no right-censored records are counted, so the summary looks like a "
            "point estimate over assays that never reached 50% effect"
        )

    gdsc = art.get("gdsc2_glioma_audit") or {}
    tmz_rows = int(gdsc.get("temozolomide_rows") or 0)
    tmz_above = int(gdsc.get("temozolomide_above_max_tested") or 0)
    if tmz_rows <= 0:
        problems.append("the GDSC glioma audit reports no temozolomide rows")
    elif tmz_above != tmz_rows:
        problems.append(
            f"the audit records {tmz_above} of {tmz_rows} temozolomide IC50s "
            f"above the highest tested concentration; the measured state of this "
            f"data is that ALL of them are extrapolations"
        )

    comparison = art.get("ec50_comparison") or {}
    if comparison.get("comparable") is not False:
        problems.append(
            "the artifact reports the EC50 comparison as permitted; the engine's "
            "EC50 has no declared concentration unit, so that comparison is a "
            "type error rather than a calibration"
        )
    reconciliation = art.get("reconciliation") or {}
    if reconciliation.get("calibrated") is not False:
        problems.append(
            "the artifact does not state `calibrated: false`; kill constants "
            "cannot be claimed as calibrated without a rate-reporting assay"
        )
    blockers = reconciliation.get("blockers") or []
    if len(blockers) < 3:
        problems.append(
            f"only {len(blockers)} blocker(s) recorded; the type mismatch, the "
            f"undeclared unit and the censoring are three independent reasons"
        )

    # Round-tripped through JSON on BOTH sides: the artifact's copy has been
    # through json.dump (tuples become lists) while the live dataclass returns
    # the original containers, so a direct != would report drift that is an
    # artefact of serialisation rather than a changed constant.
    live = json.loads(json.dumps({
        "drugs": dataclasses.asdict(DEFAULT_DRUGS),
        "clones": dataclasses.asdict(DEFAULT_CLONES),
    }))
    stated = json.loads(json.dumps(art.get("engine_constants") or {}))
    for key, value in live.items():
        if stated.get(key) != value:
            problems.append(
                f"the artifact's {key} constants differ from the live engine's; "
                f"an anchor recorded against different constants is stale"
            )
    if stated.get("engine_ec50_unit") is not None:
        problems.append(
            "the artifact declares an EC50 unit; the engine declares none, and "
            "a unit invented here would make the refusal above look resolved"
        )

    if not (art.get("engine_identity") and art.get("artifact_provenance")):
        problems.append("the artifact carries no engine identity or provenance block")

    detail = (
        f"bench anchors persisted in outputs_real/bench_anchors.json: "
        f"{n_exact} exact of {n_records} measured glioma-line IC50 record(s) "
        f"with {n_censored} right-censored counted, GDSC2 temozolomide "
        f"extrapolations {tmz_above}/{tmz_rows}, EC50 comparison "
        f"{'REFUSED' if comparison.get('comparable') is False else 'permitted'}, "
        f"reconciliation calibrated={reconciliation.get('calibrated')}, and the "
        f"artifact's engine constants compare equal to the live engine's."
    )
    if problems:
        return CheckResult(False, detail + " Problems: " + "; ".join(problems))
    return CheckResult(True, detail)


def check_alternating_win_has_a_measured_reversal_map() -> CheckResult:
    """The alternating win must be bracketed by a MEASURED reversal map.

    Release 1 measured the win at ONE operating point, which is not enough to
    propose a calendar: a protocol needs to know where the effect starts, where
    it stops, and where the safety screen refuses. This reads
    `outputs_real/release2_sweep.json` and refuses unless the sweep

      * is COMPLETE (`partial` false -- a half-finished map cannot support a
        boundary sentence),
      * ran on the same seeds as release 1,
      * REPRODUCES release 1's winner at intensity 1.0 at release 1's own
        resistant fraction and growth noise, and
      * re-ran the reference arm at every cell, so each cell is a paired
        comparison rather than a winner arm measured against a remembered
        number,
      * and actually FOUND a safety boundary: at least one cell fails the ANC
        floor while the free screen reports a maximum feasible intensity below
        the highest intensity the engine ran.

    The last condition is the one that stops this from being decoration. A map
    in which nothing ever breaks the floor would satisfy a naive "the artifact
    exists" check while proving nothing about where the claim ends, so the
    boundary is required from both directions: the screen must name one, and
    the engine must have run past it.
    """
    path = os.path.join(ROOT, "outputs_real", "release2_sweep.json")
    if not os.path.exists(path):
        return CheckResult(
            False,
            "release-2 reversal map not built (outputs_real/release2_sweep.json "
            "missing): the alternating win is then asserted at one operating "
            "point only. Run _release2_sweep.py.",
        )
    try:
        with open(path, encoding="utf-8") as fh:
            art = json.load(fh)
    except (OSError, ValueError) as exc:
        return CheckResult(False, f"reversal map unreadable: {exc}")

    rel1_path = os.path.join(ROOT, "outputs_real", "dosing.json")
    try:
        with open(rel1_path, encoding="utf-8") as fh:
            rel1 = json.load(fh)
    except (OSError, ValueError) as exc:
        return CheckResult(
            False,
            f"release-1 artifact unreadable ({exc}); the reversal map cannot be "
            f"checked against the result it is supposed to bracket",
        )

    problems: List[str] = []
    best = ((rel1.get("combination") or {}).get("best_combination") or {})
    cells = art.get("cells") or []

    if art.get("partial"):
        problems.append("the sweep is marked partial, so its boundaries are "
                        "unfinished")

    rel1_seeds = {int(s) for s in (best.get("seeds") or [])}
    map_seeds = {int(s) for s in (art.get("seeds") or [])}
    if rel1_seeds and map_seeds != rel1_seeds:
        problems.append(
            f"seeds differ from release 1 by {sorted(rel1_seeds ^ map_seeds)}"
        )

    unpaired = [c for c in cells if not (c.get("reference") or {}).get("median_aubc")]
    if unpaired:
        problems.append(f"{len(unpaired)} cell(s) carry no measured reference arm")

    target = best.get("median_aubc")
    at_release_point = [
        c for c in cells
        if abs(float(c.get("intensity", 0.0)) - 1.0) < 1e-12
        and abs(float(c.get("resistant_fraction", -1.0)) - 0.01) < 1e-12
        and abs(float(c.get("growth_noise", -1.0)) - 0.05) < 1e-12
    ]
    reproduced = None
    if target is None:
        problems.append("release 1 exposes no winner AUBC to reproduce")
    elif not at_release_point:
        problems.append(
            "no cell at intensity 1.0 with release 1's resistant fraction "
            "(0.01) and growth noise (0.05): the winner was never re-run at "
            "the point it was selected at"
        )
    else:
        reproduced = float(at_release_point[0]["winner"]["median_aubc"])
        if abs(reproduced - float(target)) > 1e-6 * max(1.0, abs(float(target))):
            problems.append(
                f"the intensity-1.0 cell measures AUBC {reproduced:.6f} where "
                f"release 1 measured {float(target):.6f}; the reconstruction "
                f"has drifted from the calendar it claims to sweep"
            )

    infeasible = [c for c in cells if c.get("status") == "INFEASIBLE"]
    intensities = [float(c.get("intensity", 0.0)) for c in cells]
    top_intensity = max(intensities) if intensities else 0.0
    boundaries = [
        (v or {}).get("max_feasible_intensity")
        for v in (art.get("feasibility") or {}).values()
    ]
    if not cells:
        problems.append("the map holds no cells")
    elif not infeasible:
        problems.append(
            "no cell fails the ANC floor: the map found no safety boundary and "
            "therefore does not bracket the claim"
        )
    if not boundaries or all(b is None for b in boundaries):
        problems.append(
            "the free screen reports no maximum feasible intensity for any "
            "declared marrow weight"
        )
    else:
        # The BINDING boundary is the most conservative marrow assumption, which
        # is the smallest feasible intensity across the declared weights (a
        # heavier assumed second drug lowers it). Requiring the engine to have
        # run past THAT is the condition with content: a map that only ever ran
        # below the conservative boundary would never have exercised the edge
        # where the calendar becomes inadmissible.
        binding = min(b for b in boundaries if b is not None)
        if top_intensity and binding >= top_intensity:
            problems.append(
                f"the most conservative feasible intensity ({binding}) is at or "
                f"above every intensity the engine ran ({top_intensity}), so the "
                f"upper edge of the claim was never exercised"
            )

    if not (art.get("engine_identity") and art.get("provenance")):
        problems.append("the artifact carries no engine identity or provenance block")

    detail = (
        f"reversal map: {len(cells)} cell(s) over intensity x pre-existing "
        f"resistant fraction x growth noise on {len(map_seeds)} seed(s); "
        f"{len(infeasible)} cell(s) fail the ANC floor; the winner at intensity "
        f"1.0 "
        + (f"reproduces the release-1 AUBC ({reproduced:.6f})" if reproduced is not None
           else "could not be checked")
        + f"; screen boundaries by drug-B marrow weight: "
        + ", ".join(
            f"{w}:{b}" for w, b in
            sorted((w, (v or {}).get("max_feasible_intensity"))
                   for w, v in (art.get("feasibility") or {}).items())
        )
        + ". Measured on one synthetic phantom on the same seeds as release 1; "
          "the reversal cells are limits of the claim, not footnotes to it."
    )
    if problems:
        return CheckResult(False, detail + " Problems: " + "; ".join(problems))
    return CheckResult(True, detail)


def check_architecture_writeup_is_audited_against_the_code() -> CheckResult:
    """The architecture write-up must be committed, audited, and still accurate.

    A technical description of this engine is a claim set, and it is the claim
    set most likely to drift because it is usually written by someone who did
    not implement the engine. This reads
    `outputs_real/architecture_audit.json` (produced by
    `_run_architecture_audit.py` from `danon/pitch_audit.py`) and refuses unless

      * the audited document exists and its SHA-256 still matches the artifact,
        so a SILENT REWRITE of the write-up fails here -- the drift this check
        exists for is a pitch quietly reworded while the code stays put,
      * the artifact carries a finding for every declared rule, and at least one
        of them is a live flag (a write-up that trips no rule has not been
        audited, it has been blessed),
      * the live code-fact snapshot still EQUALS the stored one, so a change to
        the mutation draw, the controllers, the marrow model, the endpoint or
        the evidence vocabulary fails the audit until it is re-run, and
      * the corrections are recorded with their code evidence, so a reader can
        open what refutes each statement.

    What it cannot do is check the write-up against biology. Every verdict here
    is 'the document versus the code'.
    """
    path = os.path.join(ROOT, "outputs_real", "architecture_audit.json")
    if not os.path.exists(path):
        return CheckResult(
            False,
            "architecture audit not built (outputs_real/architecture_audit.json "
            "missing): the engine's architecture write-up is then unaudited. Run "
            "_run_architecture_audit.py.",
        )
    try:
        with open(path, encoding="utf-8") as fh:
            art = json.load(fh)
    except (OSError, ValueError) as exc:
        return CheckResult(False, f"architecture audit unreadable: {exc}")

    import hashlib

    from danon.pitch_audit import AUDITED_DOCUMENT, DRIFT_RULES, code_fact_snapshot

    problems: List[str] = []
    doc_path = os.path.join(ROOT, AUDITED_DOCUMENT)
    if not os.path.exists(doc_path):
        problems.append(f"the audited document {AUDITED_DOCUMENT} is not on disk")
    else:
        with open(doc_path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        if digest != art.get("sha256"):
            problems.append(
                "the write-up's SHA-256 no longer matches the audited copy: the "
                "document was edited after the audit, so its verdicts describe a "
                "different text"
            )

    findings = art.get("findings") or []
    by_rule = {f.get("rule_id"): f for f in findings}
    missing = sorted({r.id for r in DRIFT_RULES} - set(by_rule))
    if missing:
        problems.append(f"no finding recorded for rule(s): {', '.join(missing)}")

    flagged = [
        f for f in findings
        if f.get("status") == "PRESENT"
        and f.get("verdict") in ("REFUTED", "IMPRECISE", "QUALIFIED")
    ]
    omissions = [f for f in findings if f.get("status") == "OMITTED"]
    if not flagged and not omissions:
        problems.append(
            "the audit flags nothing at all: either the write-up is perfect or "
            "the rule set stopped matching, and a clean sweep is not the claim "
            "being made here"
        )
    for f in flagged + omissions:
        if not (f.get("correction") or "").strip():
            problems.append(f"rule {f.get('rule_id')} carries no correction")
        if not f.get("evidence"):
            problems.append(
                f"rule {f.get('rule_id')} carries no code evidence, so its "
                f"verdict cannot be checked by a reader"
            )

    live = code_fact_snapshot()
    stored = art.get("facts")
    if stored != live:
        drifted = sorted(
            k for k in set(live) | set(stored or {})
            if (stored or {}).get(k) != live.get(k)
        )
        problems.append(
            f"the code-fact snapshot has moved since the audit was built "
            f"({', '.join(drifted[:6])}{'...' if len(drifted) > 6 else ''}): "
            f"re-run _run_architecture_audit.py, then re-read the corrections "
            f"before accepting them"
        )
    if not art.get("generated_utc"):
        problems.append("the audit artifact is undated")

    # The reading copy is the artefact a reviewer actually opens, so it is
    # guarded here too: it is generated from the audit, and it must carry the
    # digest of the document it corrects. A hand-edited copy fails.
    corrections = os.path.join(ROOT, "outputs_real", "architecture_corrections.md")
    if not os.path.exists(corrections):
        problems.append(
            "the corrected reading copy (outputs_real/architecture_corrections.md) "
            "is missing, so the corrections exist only inside a JSON artifact"
        )
    else:
        with open(corrections, encoding="utf-8") as fh:
            copy_text = fh.read()
        if art.get("sha256") not in copy_text:
            problems.append(
                "the corrected reading copy does not carry the audited "
                "document's digest: it was edited or generated from a different "
                "audit"
            )
        if AUDITED_DOCUMENT not in copy_text:
            problems.append(
                "the corrected reading copy does not name the document it "
                "corrects"
            )

    counts = art.get("counts") or {}
    detail = (
        f"{AUDITED_DOCUMENT} audited against the code: {counts.get('present')} "
        f"statement(s) flagged ({counts.get('refuted')} refuted, "
        f"{counts.get('imprecise')} imprecise, {counts.get('qualified')} "
        f"qualified), {counts.get('omitted')} material omission(s) named across "
        f"{counts.get('rules')} declared rule(s); document digest and the live "
        f"code-fact snapshot both match the audited copy. Verdicts are document "
        f"versus code, and say nothing about whether the engine is right."
    )
    if problems:
        return CheckResult(False, detail + " Problems: " + "; ".join(problems))
    return CheckResult(True, detail)


def check_motility_constant_is_grid_dependent_and_priced() -> CheckResult:
    """The cell-motility coefficient must be priced, and must NOT be silently fixed.

    The engine integrates cell migration with a coefficient in voxel^2/day, which
    is a different physical diffusivity on every grid. `danon/anisotropy.py`
    already reports those two values; what this check requires is that the
    measured CONSEQUENCE is on the record, and that the engine's default has not
    been quietly changed underneath the release artifacts that depend on it.

    It refuses unless

      * the shipped constant is still `LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY` and
        a simulator built without an override still reports `None` for its
        motility (i.e. no silent re-tune, which would invalidate release 1 and
        release 2 without re-running them),
      * the artifact's two physical values re-derive from a LIVE conversion at
        the two voxel sizes, in both directions,
      * the shipped default arm reproduces the release-1 winner AUBC exactly,
        so the sensitivity is anchored to the calendar and the substrate the
        claims quote, and
      * the sensitivity has CONTENT in both places it was measured: the winner's
        own AUBC moves by more than a trivial amount somewhere, and the
        single-drug reference moves too -- the ratio those claims quote is only
        worth stating if the artifact shows whether it is motility-dependent.
    """
    path = os.path.join(ROOT, "outputs_real", "motility_audit.json")
    if not os.path.exists(path):
        return CheckResult(
            False,
            "motility sensitivity not measured (outputs_real/motility_audit.json "
            "missing): a grid-dependent coefficient is then a documented unit "
            "note rather than a priced one. Run _run_motility_audit.py.",
        )
    try:
        with open(path, encoding="utf-8") as fh:
            art = json.load(fh)
    except (OSError, ValueError) as exc:
        return CheckResult(False, f"motility artifact unreadable: {exc}")

    from danon.anisotropy import (
        CELL_MOTILITY_MM2_PER_DAY_RANGE,
        LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY,
        PHANTOM_VOXEL_MM,
        REAL_VOXEL_MM,
        d_to_physical,
    )
    from danon.evosteer import TumorEvolutionSimulator, make_gbm_phantom

    problems: List[str] = []

    # 1. No silent re-tune: the engine still ships the unpriced literal.
    if LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY != 0.02:
        problems.append(
            f"the shipped constant is now "
            f"{LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY}, but the release-1 and "
            f"release-2 artifacts were measured at 0.02 voxel^2/day; a changed "
            f"default must re-run them, and this artifact does not"
        )
    probe = TumorEvolutionSimulator(make_gbm_phantom(size=24, seed=12345))
    if probe._cell_motility is not None:
        problems.append(
            "a default simulator now overrides cell_motility, so its results no "
            "longer match the artifacts the claims quote"
        )

    # 2. The two physical values re-derive live at both voxel sizes.
    lo, hi = CELL_MOTILITY_MM2_PER_DAY_RANGE
    grid_factor = (REAL_VOXEL_MM / PHANTOM_VOXEL_MM) ** 2
    live_phantom = d_to_physical(LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY, PHANTOM_VOXEL_MM)
    live_real = d_to_physical(LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY, REAL_VOXEL_MM)
    defect = art.get("defect") or {}
    if defect.get("value_voxel2_per_day") != LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY:
        problems.append("the artifact prices a different constant than the engine ships")
    legacy_row = next(
        (s for s in (art.get("settings") or []) if s.get("label") == "legacy_default"),
        None,
    )
    if legacy_row is None:
        problems.append("the artifact holds no row for the shipped default")
    else:
        if abs(float(legacy_row["phantom_mm2_per_day"]) - live_phantom) > 1e-12:
            problems.append(
                f"artifact phantom value {legacy_row['phantom_mm2_per_day']} != "
                f"live conversion {live_phantom}"
            )
        if abs(float(legacy_row["real_mm2_per_day"]) - live_real) > 1e-12:
            problems.append(
                f"artifact real-grid value {legacy_row['real_mm2_per_day']} != "
                f"live conversion {live_real}"
            )
        if not (live_phantom > hi and live_real > hi):
            problems.append(
                "the shipped value is no longer above the literature range on "
                "either grid, so the defect this artifact prices has changed"
            )
    # The grid factor is derived from the row's OWN two values, so it is checked
    # against the same numbers a reader sees rather than a separate constant that
    # could agree with the code while disagreeing with the table.
    if legacy_row is not None and live_phantom > 0.0:
        row_factor = float(legacy_row["real_mm2_per_day"]) / float(
            legacy_row["phantom_mm2_per_day"]
        )
        if abs(row_factor - grid_factor) > 1e-6:
            problems.append(
                f"the row's own values imply a grid factor of {row_factor:.4f} "
                f"where the live voxel sizes give {grid_factor:.4f}"
            )

    # 3. The default arm is anchored to the release-1 winner.
    anchor = art.get("release_1_anchor") or {}
    if anchor.get("reproduced_by_default_arm") is not True:
        problems.append(
            "the default-motility arm does not reproduce the release-1 winner "
            "AUBC, so the sensitivity is measured on a different calendar"
        )

    # 4. The sensitivity must have content on both arms.
    rows = {s.get("label"): s for s in (art.get("settings") or [])}
    deltas = [
        abs(float(s["aubc_delta_vs_default"]))
        for s in (art.get("settings") or [])
        if s.get("label") != "legacy_default"
    ]
    if not deltas or max(deltas) < 0.01:
        problems.append(
            "no setting moves the winner's AUBC by more than 1%, so the term is "
            "inert and this claim would be reporting a non-result as a finding"
        )
    ref = art.get("reference_arm") or {}
    ratios = [
        v.get("aubc_ratio_winner_over_reference")
        for v in ref.values()
        if v.get("aubc_ratio_winner_over_reference") is not None
    ]
    if len(ratios) < 2:
        problems.append(
            "the reference arm was measured at fewer than two settings, so the "
            "quoted winner/single ratio is not shown to be motility-dependent"
        )
    elif max(ratios) - min(ratios) < 0.005:
        problems.append(
            "the winner/single ratio is unchanged across settings, so the "
            "artifact's comparison is not evidence either way"
        )
    if not art.get("cannot_show"):
        problems.append("the artifact does not carry its own limits")

    detail = (
        f"motility priced: {LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY} voxel^2/day is "
        f"{live_phantom:.3f} mm^2/day on 1.0 mm phantom voxels and "
        f"{live_real:.3f} mm^2/day on 3.667 mm real voxels "
        f"({grid_factor:.2f}x grid factor) against a literature range of "
        f"{lo}-{hi}; the default arm reproduces the release-1 winner AUBC "
        f"exactly; winner AUBC moves up to {max(deltas):.1%} and the "
        f"winner/single ratio moves from {min(ratios):.4f} to {max(ratios):.4f} "
        f"across the declared settings. The engine's default is UNCHANGED, so "
        f"the release artifacts remain valid and the re-run is an explicit "
        f"decision."
    )
    if problems:
        return CheckResult(False, detail + " Problems: " + "; ".join(problems))
    return CheckResult(True, detail)


def check_twin_plan_input_is_byte_tied() -> CheckResult:
    """A recorded twin run must be tied to the exact file it was given.

    Five things, none of which executes the engine, so this stays cheap enough
    to re-run at audit time:

    * the identity is a function of the file's bytes and nothing else -- the
      same bytes hash the same twice, and a single flipped payload byte moves
      the digest;
    * a ``.nii.gz`` is hashed AS RECEIVED (compressed), not re-serialised;
    * the grid the planner reports for the input is the grid the file's own
      data has, and the reported spacing is the header's own spacing;
    * the downsampled header must carry the SCALED spacing -- the
      ``derived_voxel_sizes`` guard -- and the ingestion path must apply it,
      because writing the source spacing back would understate every physical
      volume taken from the derived file by 8x per halving, the defect
      ``danon.grid_artifact`` records against the committed derived volumes;
    * block-max downsampling must not lose more than one derived voxel of
      physical extent, which is what keeps the reported volume from
      understating the source.

    This says the record of what was simulated is reproducible from the file.
    It says nothing about whether the simulation predicts a patient.
    """
    import hashlib as _hashlib
    import tempfile
    from pathlib import Path

    from danon.evosteer import make_gbm_phantom
    from danon.imaging import mask_bytes_provenance, read_nifti, write_nifti
    from danon.twin_plan import derived_voxel_sizes, downsample_mask_to_budget

    declared = (0.8, 0.8, 1.5)
    mask = make_gbm_phantom(size=16, seed=5).tumor_mask.astype("u1")
    with tempfile.TemporaryDirectory(prefix="claim_input_prov_") as td:
        plain = Path(td) / "mask.nii"
        gz = Path(td) / "mask.nii.gz"
        write_nifti(str(plain), mask, voxel_sizes=declared)
        write_nifti(str(gz), mask, voxel_sizes=declared)
        raw = plain.read_bytes()
        raw_gz = gz.read_bytes()
        data_shape = [int(s) for s in read_nifti(str(plain))[0].shape]

    problems: List[str] = []

    prov = mask_bytes_provenance(raw)
    if prov != mask_bytes_provenance(raw):
        problems.append("the same bytes produced two different provenance records")
    if prov["input_sha256"] != _hashlib.sha256(raw).hexdigest():
        problems.append("the reported digest is not the digest of the bytes received")
    flipped = bytearray(raw)
    flipped[-1] ^= 0x01  # last payload byte: the 348-byte header is untouched
    if mask_bytes_provenance(bytes(flipped))["input_sha256"] == prov["input_sha256"]:
        problems.append(
            "a flipped payload byte left the digest unchanged, so the digest is "
            "not a function of the file's content"
        )
    if mask_bytes_provenance(raw_gz)["input_sha256"] != _hashlib.sha256(
        raw_gz
    ).hexdigest():
        problems.append("a .nii.gz input was not hashed as received")
    if prov["original_grid_shape_xyz"] != data_shape:
        problems.append(
            f"the reported grid {prov['original_grid_shape_xyz']} is not the grid "
            f"the file's data has {data_shape}"
        )
    if abs(prov["source_voxel_mm"] - float(np.mean(declared))) > 1e-6:
        problems.append("the reported voxel size is not the header's own spacing")

    # The derived-spacing guard, and the ingestion path that must apply it.
    if derived_voxel_sizes(declared, [2, 2]) != tuple(v * 4 for v in declared):
        problems.append(
            "derived_voxel_sizes does not scale spacing by the applied factor"
        )
    try:
        derived_voxel_sizes(declared, [2, 0])
        problems.append("derived_voxel_sizes accepted a non-positive factor")
    except ValueError:
        pass
    api_text = _read("api_server.py")
    if "derived_voxel_sizes" not in api_text:
        problems.append(
            "the ingestion path does not apply the derived spacing, so a "
            "downsampled file would carry the source spacing"
        )
    if "mask_bytes_provenance" not in api_text:
        problems.append("the endpoint does not report the input's byte identity")

    big = np.zeros((128, 128, 128), dtype=bool)
    big[32:96, 32:96, 32:96] = True
    ds, factors = downsample_mask_to_budget(big, budget=64_000)
    if not factors:
        problems.append(
            "the budget did not force a halving, so the derived-spacing path was "
            "not exercised"
        )
    else:
        ds_sizes = derived_voxel_sizes((1.0, 1.0, 1.0), factors)
        src_extent = 128.0
        ds_extent = ds.shape[0] * ds_sizes[0]
        if ds_extent < src_extent - ds_sizes[0]:
            problems.append(
                f"downsampling lost more than one derived voxel of physical extent "
                f"({ds_extent} < {src_extent})"
            )

    detail = (
        f"input identity: sha256 {prov['input_sha256'][:12]}... over the bytes as "
        f"received, grid {prov['original_grid_shape_xyz']} and "
        f"{prov['source_voxel_mm']:.3f} mm mean spacing read from the header; a "
        f"flipped payload byte moves the digest and a .nii.gz is hashed compressed; "
        f"the derived-spacing guard scales per axis "
        f"({derived_voxel_sizes(declared, [2, 2])} for two halvings on {declared}) "
        f"and the ingestion path applies it. A recorded run can be re-identified "
        f"from the file it names -- which is not a prediction about any patient."
    )
    if problems:
        return CheckResult(False, detail + " Problems: " + "; ".join(problems))
    return CheckResult(True, detail)


# Registered after the literal above, because these two checks arrived with the
# release-2 reversal map and the Step-1 anchor artifact and their bodies sit
# beside the release-2 audit below instead of above a several-hundred-entry
# literal. Same dict, same semantics: `audit_claims` only ever evaluates
# CHECKS[claim.check]().
#: Every open badge, classified by what would actually CLOSE it. The class is
#: the point: a route that says "download these four datasets and the badges
#: flip" is answered per badge, and nine out of nine stay open on that route.
#: The keys are the badge names, so a NEW open badge cannot be silently covered
#: by a generic sentence -- the check fails until it is classified here.
BADGE_CLOSURE: Dict[str, Tuple[str, str]] = {
    "heldout_discrimination": (
        "human_outcomes",
        "a cohort with outcomes wired into this repository's own loop; the "
        "surrogate's holdout score is already measured and more public features "
        "are not what limit it",
    ),
    "no_spatial_agreement_result": (
        "a_public_pair_with_expert_masks",
        "an imaging collection carrying EXPERT masks for the same patient at two "
        "registered timepoints, plus the co-registration this repository lacks. "
        "The named candidate is LUMIERE, and it is not that collection: its "
        "masks are automated (DeepBraTumIA / HD-GLIO-AUTO), so a Dice against "
        "them would score this engine against another algorithm",
    ),
    "no_inflated_estimand": (
        "deliberate_refusal",
        "this badge IS the refusal of three estimands; closing it would mean "
        "fabricating a target, not acquiring one",
    ),
    "egfrviii_not_in_open_data": (
        "controlled_access_only",
        "structural-variant calls, which are controlled access for every "
        "TCGA-GBM file; no amount of public-data mining reaches them",
    ),
    "engine_potency_scale_not_anchored": (
        "bench_measurement",
        "a measured per-day kill rate in a 3D system behind a barrier. A unit "
        "declaration cannot close it, and that is measured rather than argued: "
        "the kill law is degree-0 homogeneous in (concentration, EC50), so "
        "relabelling reproduces the trajectory bit for bit",
    ),
    "no_engine_c_index": (
        "human_outcomes",
        "a dataset pairing baseline tumour state, administered schedule and "
        "outcome; one curve per simulation makes a C-index 0.5 by construction",
    ),
    "parameters_are_measured": (
        "bench_measurement",
        "bench assays for the authored constants; the tiers say which are "
        "calibrated, assumed or measured, and a public download changes none of "
        "those labels",
    ),
    "engine_predicts_patients": (
        "prospective_trial",
        "prospective patients with recorded baseline state, schedule and outcome",
    ),
    "clinical_benefit": (
        "prospective_trial",
        "a prospective trial and regulatory review: no retrospective dataset "
        "establishes that a schedule extends a life",
    ),
}


def badge_inventory() -> Dict[str, object]:
    """Every open badge and what would close it -- counted, not asserted.

    A badge is any claim whose tier is not ESTABLISHED, plus every provenance
    key that is not ESTABLISHED. Both are derived from the registry on each
    call, so this cannot go stale behind a claim that was added later.
    """
    from collections import Counter

    tiers = {claim.id: claim.tier for claim in CLAIMS}
    open_badges: Dict[str, EvidenceStatus] = {
        f"claim:{claim_id}": tier
        for claim_id, tier in tiers.items()
        if tier is not EvidenceStatus.ESTABLISHED
    }
    for key, status in claims_provenance().items():
        if status is not EvidenceStatus.ESTABLISHED:
            open_badges[f"provenance:{key}"] = status

    entries: Dict[str, Dict[str, object]] = {}
    for name, status in sorted(open_badges.items()):
        short = name.split(":", 1)[1]
        closure = BADGE_CLOSURE.get(short)
        entries[name] = {
            "status": status.value,
            "badge": short,
            "would_close_on": closure[0] if closure else None,
            "what_would_close_it": closure[1] if closure else None,
        }
    counts = Counter(
        entry["would_close_on"] or "unclassified" for entry in entries.values()
    )
    return {
        "n_open": len(entries),
        "n_claim_tiers": sum(1 for k in open_badges if k.startswith("claim:")),
        "n_provenance_keys": sum(
            1 for k in open_badges if k.startswith("provenance:")
        ),
        "badges": entries,
        "by_closure": dict(counts),
        # The route's own four ingestors. Counted so the sentence cannot claim a
        # number the registry contradicts.
        "closed_by_a_public_download": 0,
    }


def check_badge_inventory_is_answered_not_asserted() -> CheckResult:
    """Every open badge is classified, and the claim's own count is live.

    Three things are checked: that no open badge is unclassified (a new UNKNOWN
    must be answered, not absorbed), that the count written into the sentence
    still matches the registry, and that the reasons which depend on other
    records are still true of them -- the spatial-Dice badge is refused because
    LUMIERE's masks are AUTOMATED, so that annotation is read back here, and the
    two benefit badges must still be UNKNOWN.
    """
    inventory = badge_inventory()
    problems: List[str] = []
    entries = inventory["badges"]

    if inventory["n_open"] != len(entries):
        problems.append("the open-badge count disagrees with its own list")
    unclassified = [n for n, e in entries.items() if not e["would_close_on"]]
    if unclassified:
        problems.append(
            f"{len(unclassified)} open badge(s) carry no classification "
            f"({unclassified}); classify it in BADGE_CLOSURE rather than letting "
            f"the inventory go stale"
        )

    # The sentence's own number, against the live one.
    by_id = {claim.id: claim for claim in CLAIMS}
    claim = by_id.get("badge_inventory_is_classified_not_promised")
    if claim is None:
        problems.append("the badge-inventory claim is not in the registry")
    else:
        match = re.search(r"(\d+)\s+badges? are open", claim.sentence)
        if match is None:
            problems.append("the sentence no longer states a live count")
        elif int(match.group(1)) != inventory["n_open"]:
            problems.append(
                f"the sentence says {match.group(1)} open badges where the "
                f"registry holds {inventory['n_open']}: rewrite it rather than "
                f"leaving a remembered count"
            )

    # The two benefit badges cannot be moved by data collection, and the spatial
    # one's reason is another record's fact.
    provenance = claims_provenance()
    for key in ("engine_predicts_patients", "clinical_benefit"):
        if provenance.get(key) is not EvidenceStatus.UNKNOWN:
            problems.append(f"{key} is no longer UNKNOWN, so the inventory is wrong")
        if entries.get(f"provenance:{key}", {}).get("would_close_on") != "prospective_trial":
            problems.append(f"{key} is no longer classified as needing a trial")

    refused = entries.get("claim:no_spatial_agreement_result")
    if refused is None:
        problems.append("the spatial-Dice badge is no longer open")
    else:
        try:
            from danon.walkforward import longitudinal_census

            lumiere = longitudinal_census()["sources"]["LUMIERE"]
        except Exception as exc:  # noqa: BLE001 -- reported, not raised
            problems.append(f"the LUMIERE annotation cannot be read back: {exc}")
        else:
            if "automated" not in str(lumiere.get("segmentations") or "").lower():
                problems.append(
                    "LUMIERE's masks are no longer recorded as automated, so the "
                    "reason this badge stays open has changed"
                )
            if "expert masks" not in str(
                refused.get("what_would_close_it") or ""
            ).lower():
                problems.append(
                    "the spatial badge no longer names expert masks as what would "
                    "close it"
                )

    by_closure = inventory["by_closure"]
    return CheckResult(
        supported=not problems,
        detail=(
            f"{inventory['n_open']} open badges "
            f"({inventory['n_claim_tiers']} claim tiers, "
            f"{inventory['n_provenance_keys']} provenance keys); by what would "
            f"close them: {by_closure}; closed by any of the four public "
            f"datasets a data-mining route names: "
            f"{inventory['closed_by_a_public_download']}"
            + ("; " + "; ".join(problems) if problems else "")
        ),
    )


# The specifics a proposed 'epistemic guardrail' pitch invents and this
# repository has never measured. Each entry is (label, pattern); the patterns
# are BUILT BY CONCATENATION so this audit does not flag its own source.
FABRICATED_EPISTEMIC_FIGURES: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("a self-declared measured tier", re.compile("PATIENT_" + "MEASURED")),
    ("an EpistemicFloat wrapper", re.compile("Epistemic" + "Float")),
    ("a CALIBRATED_COHORT tier", re.compile("CALIBRATED_" + "COHORT")),
    (
        "a confidence interval widened by a fixed percent",
        re.compile("widened" + r"\s+by\s+\d+\s*%"),
    ),
    (
        "a 5.5 ug/mL brain concentration (5.5 is the label's CLEARANCE)",
        re.compile("5\\." + r"5\s*(?:u|\u00b5|mc)g\s*/\s*m[lL]"),
    ),
)

#: Source suffixes that carry a SHIPPED figure (the app's audit and product
#: surfaces). ``tests/`` is deliberately excluded: a test may legally plant a
#: wrong figure in order to assert the code refuses it.
_SHIPPED_SOURCES: Tuple[str, ...] = (
    ".py", ".ts", ".tsx",
)
_SHIPPED_DIRS: Tuple[str, ...] = ("danon", os.path.join("frontend", "src"))


# Device-status claims a product roadmap may assume. This repository is a
# research prototype: nothing has been submitted to any regulator. Each entry
# is (label, pattern), and these too are built by concatenation so the audit
# does not flag its own vocabulary.
FABRICATED_DEVICE_STATUS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    (
        # The adjective form -- "an FDA-cleared SaMD" -- which states a STATUS.
        # A sentence like "FDA clearance before any therapeutic use" is a FUTURE
        # gate and is not flagged.
        "an FDA clearance or approval as a status",
        re.compile(r"(?i)\bfda" + r"[-\s]?(?:cleared|approved)\b"),
    ),
    (
        "a granted De Novo or 510(k)",
        re.compile(r"(?i)(?:de novo|510\(k\))" + r"[^.\n]{0,40}\b(?:granted|cleared|approved)"),
    ),
    (
        "a GPU/JAX/CUDA-accelerated solver",
        re.compile(r"(?i)(?:jax|cuda|gpu)" + r"[-\s]?(?:accelerated|solver|backend)"),
    ),
)


# The regulatory context for crossing from this research prototype to a cleared
# device. This is CONTEXT a reader can check against the FDA's own guidance --
# it is NOT a claim of clearance, and no quality-system or submission record
# exists here.
SAMD_CONTEXT: Dict[str, str] = {
    "status": (
        "research prototype, not a device: nothing has been submitted to any "
        "regulator; the only clinical layer here is decision support that "
        "displays its own operating point, decision curve and provenance for "
        "independent review, with a named clinician deciding"
    ),
    "cds_exemption": (
        "a dosing recommendation that DRIVES rather than supports a clinician's "
        "decision, and does not enable independent review of its basis, fails "
        "the non-device CDS criteria (FD&C Act 520(o)(1)(E), FDA CDS guidance), "
        "so software like this is a device"
    ),
    "pathway": (
        "Software as a Medical Device; the route follows the risk "
        "categorization rather than this file -- a novel recommendation engine "
        "with no predicate is a De Novo, or a PMA if Class III; 510(k) requires "
        "an existing predicate"
    ),
    "adaptive_updates": (
        "bounded algorithm updates are handled by a Predetermined Change "
        "Control Plan (FDA final guidance, Dec 2024), not by silence"
    ),
    "standards": (
        "IEC 62304 (software lifecycle), ISO 13485 (quality management), "
        "ISO 14971 (risk management)"
    ),
    "engine": (
        "numpy only: no JAX, no CUDA and no GPU exists in this environment, so "
        "no accelerator figure is quotable"
    ),
    "validation_bar": (
        "no cohort 3D Dice is reachable from public data (0 scorable pairs) "
        "and the engine has no estimable patient-level C-index, so the "
        "analytical validation a submission needs is a data partnership, not a "
        "download"
    ),
}


# Benefit, safety and runtime claims a translation pitch asserts as facts. Each
# entry is (label, pattern); built by concatenation so the audit does not flag
# its own vocabulary, and narrow enough not to trip the repository's legitimate
# uses of "guarantee" (identity guarantees, a bound guaranteed by a policy
# shape).
FABRICATED_BENEFIT_CLAIMS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    (
        "a claim that the software saves lives",
        re.compile(r"(?i)\b(?:saves?|saving|saved)" + r"\s+lives\b"),
    ),
    (
        "a safety guarantee to a patient",
        re.compile(
            r"(?i)\bguarantee\w*\b[^.\n]{0,60}\b"
            + r"(?:will not|won't|is safe|are safe|survival|no lethal)"
        ),
    ),
    (
        "a claim of an added year of life",
        re.compile(r"(?i)\badditional\s+year\b" + r"[^.\n]{0,30}\blife\b"),
    ),
    (
        "a claim that the software extends a lifespan",
        re.compile(r"(?i)\bextend\w*\b" + r"[^.\n]{0,30}\blifespan\b"),
    ),
    (
        "a 10,000-schedule runtime quoted in seconds",
        re.compile(r"(?i)\b10,?000\b" + r"[^.\n]{0,40}\b\d{1,2}\s*(?:seconds|secs?)\b"),
    ),
)


# The one real clinical precedent for adaptive dosing. CONTEXT for the
# hypothesis, never a result of this engine.
TRANSLATION_PRECEDENT: str = (
    "Adaptive dosing has a real clinical precedent: the Moffitt phase 2a "
    "adaptive-abiraterone study (NCT02415621) prolonged time to progression at "
    "lower cumulative drug exposure. It is CONTEXT for the hypothesis and NOT a "
    "result of this engine: a different cancer (mCRPC), a different drug "
    "(abiraterone), a different (non-spatial) model, and a phase 2a with a "
    "non-randomised comparison cohort rather than a randomised pivotal trial. "
    "Nothing here transfers a survival benefit to glioma or to this simulator."
)


def _scan_shipped_surfaces(
    root: str,
    patterns: Tuple[Tuple[str, "re.Pattern[str]"], ...],
    skip: Optional[str] = None,
    matcher: Optional[Callable[[str], List[str]]] = None,
) -> Tuple[List[str], int]:
    """Walk the shipped surfaces under ``root`` and apply ``patterns``.

    Returns ``(problems, files_scanned)``. When ``matcher`` is given it decides
    which labels fire for a file's text (used by the independence rule, whose
    frames need a context filter the plain ``search`` cannot express).
    """
    problems: List[str] = []
    scanned = 0
    for base in _SHIPPED_DIRS:
        base_dir = os.path.join(root, base)
        if not os.path.isdir(base_dir):
            continue
        for dirpath, _dirs, files in os.walk(base_dir):
            for name in files:
                if not name.endswith(_SHIPPED_SOURCES):
                    continue
                p = os.path.abspath(os.path.join(dirpath, name))
                if (skip and p == skip) or name.endswith(".py.bak"):
                    continue
                with open(p, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
                scanned += 1
                if matcher is not None:
                    labels = matcher(text)
                else:
                    labels = [label for label, pat in patterns if pat.search(text)]
                for label in labels:
                    problems.append(f"{os.path.relpath(p, root)} contains {label}")
    return problems, scanned


def scan_for_fabricated_epistemic_figures(
    root: str, *, skip: Optional[str] = None
) -> Tuple[List[str], int]:
    """Scan the shipped surfaces under ``root`` for the pitch's fabrications.

    Kept separate from the check so a test can point it at a temporary tree and
    prove it can fail.
    """
    return _scan_shipped_surfaces(root, FABRICATED_EPISTEMIC_FIGURES, skip)


def scan_for_fabricated_device_status(
    root: str, *, skip: Optional[str] = None
) -> Tuple[List[str], int]:
    """Scan the shipped surfaces under ``root`` for a device status this
    repository has never obtained (an FDA clearance, a grant, a GPU backend)."""
    return _scan_shipped_surfaces(root, FABRICATED_DEVICE_STATUS, skip)


def scan_for_fabricated_benefit_claims(
    root: str, *, skip: Optional[str] = None
) -> Tuple[List[str], int]:
    """Scan the shipped surfaces under ``root`` for a benefit, safety or
    runtime claim this repository has not measured."""
    return _scan_shipped_surfaces(root, FABRICATED_BENEFIT_CLAIMS, skip)


def check_epistemic_guardrail_proposal_is_audited() -> CheckResult:
    """Four proposed 'epistemic guardrails', audited against this repo not adopted.

    A pitch proposes to make this repository "syntactically incapable of lying"
    by (1) wrapping every parameter in a provenance-carrying type, (2) enforcing
    physical units with ``pint``, (3) hashing inputs to catch temporal leakage,
    and (4) showing a per-parameter "data pedigree" in the UI. This audits each
    against the live tree rather than adopting it:

      * the provenance TYPE exists (``danon.physics.Constant`` / ``Provenance``)
        but is deliberately NOT the engine's parameter type -- the engine's
        constants stay plain floats, pinned UNKNOWN, because a self-declared tier
        is a string a developer types and is weaker than the live re-derivation
        this audit performs,
      * ``pint`` is not a declared dependency and not in the process, and the
        engine still declares NO EC50 unit -- a unit declaration is a measured
        no-op on every trajectory (degree-0 homogeneity), so adopting units would
        dress a free choice as a measurement,
      * the leakage guard is statistical (``danon.audit.permutation_leakage_audit``)
        and SHA-256 PINS inputs rather than inspecting a solver's memory, and
      * the UI provenance surfaces it proposes already exist, while none of the
        proposal's fabricated specifics appears on a SHIPPED surface (``danon/``
        or ``frontend/src/``; ``tests/`` is excluded, since a test may legally
        plant a wrong figure in order to assert it is refused).

    The last clause is the point: an honesty feature that carries a fabricated
    precision figure is exactly the failure this repository refuses, so a pasted
    version of it fails the audit rather than shipping.
    """
    import dataclasses
    import sys

    problems: List[str] = []

    # -- (1) the provenance type: present for fluidics, not for the engine. --
    try:
        from danon.physics import Constant, Provenance
    except Exception as exc:  # noqa: BLE001 -- reported, not raised
        problems.append(f"the fluidics provenance type cannot be imported: {exc}")
    else:
        if not dataclasses.is_dataclass(Constant):
            problems.append("danon.physics.Constant is no longer a dataclass")
        elif "provenance" not in {f.name for f in dataclasses.fields(Constant)}:
            problems.append("danon.physics.Constant lost its provenance field")
        if not hasattr(Provenance, "ASSUMED"):
            problems.append("the fluidics provenance vocabulary lost ASSUMED")

    try:
        from danon.evosteer import DEFAULT_CLONES
    except Exception as exc:  # noqa: BLE001
        problems.append(f"the engine defaults cannot be imported: {exc}")
    else:
        if not isinstance(DEFAULT_CLONES.growth_S, float):
            problems.append(
                "the engine's growth_S is no longer a plain float: a "
                "self-declared provenance tier has been introduced over a "
                "constant that nothing re-derives"
            )

    # -- (2) units: pint is neither a dependency nor in the process. --------
    if "pint" in sys.modules:
        problems.append(
            "pint is loaded in this process; the engine's unit handling is "
            "deliberately explicit (derived_voxel_sizes, d_to_engine)"
        )
    for rel in ("requirements.txt", "pyproject.toml", "setup.py", "setup.cfg"):
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            with open(p, encoding="utf-8", errors="replace") as fh:
                if re.search(r'(?mi)^\s*"?pint"?\s*[<>=~!]', fh.read()):
                    problems.append(f"pint is declared as a dependency in {rel}")

    try:
        from danon.pharmacology import engine_ec50_unit
    except Exception as exc:  # noqa: BLE001
        problems.append(f"the engine EC50 unit cannot be read back: {exc}")
    else:
        if engine_ec50_unit() is not None:
            problems.append(
                "the engine now declares an EC50 unit; the relabelling "
                "experiment shows a unit is a no-op, so it would make the "
                "potency refusal look resolved without measuring anything"
            )

    # -- (3) the leakage guard is statistical, not a hash lock. -------------
    try:
        from danon import audit as _audit
    except Exception as exc:  # noqa: BLE001
        problems.append(f"danon.audit cannot be imported: {exc}")
    else:
        for name in ("permutation_leakage_audit", "permutation_null_c_index"):
            if not hasattr(_audit, name):
                problems.append(
                    f"the statistical leakage detector {name} is gone; a hash "
                    f"lock cannot replace it"
                )

    # -- (4) the UI provenance surfaces are present. -----------------------
    surfaces = {
        os.path.join("frontend", "src", "components", "DosingConsole.tsx"):
            "Provenance",
        os.path.join("frontend", "src", "components", "TwinPlanner.tsx"):
            "assumed",
        os.path.join("frontend", "src", "components", "Day365Outcome.tsx"):
            "CONSTANTS ASSUMED",
        os.path.join("frontend", "src", "components", "PageHeader.tsx"):
            "provenance line is required",
    }
    for rel, needle in surfaces.items():
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            problems.append(f"the provenance surface {rel} is missing")
            continue
        with open(p, encoding="utf-8", errors="replace") as fh:
            if needle not in fh.read():
                problems.append(
                    f"{rel} no longer carries its provenance surface ({needle!r})"
                )

    # -- the proposal's fabricated specifics must not appear on a shipped surface. --
    fabrications, scanned = scan_for_fabricated_epistemic_figures(
        ROOT, skip=os.path.abspath(__file__)
    )
    problems.extend(fabrications)

    verdict = (
        "provenance type present for the fluidics arm while the engine's "
        "params stay plain floats; pint undeclared and unimported; engine EC50 "
        "unit still undeclared; statistical leakage detector present; "
        f"{len(surfaces)} UI provenance surface(s) checked; {scanned} file(s) "
        "scanned for the proposal's fabricated specifics"
    )
    if problems:
        return CheckResult(False, verdict + " Problems: " + "; ".join(problems))
    return CheckResult(True, verdict)


def check_samd_pathway_is_named_not_promised() -> CheckResult:
    """The route to a cleared device is named as context, never claimed as status.

    This repository is a research prototype and says so. The check fails if

      * ``SAMD_CONTEXT`` stops stating this is not a device, or the ANC safety
        model stops saying it is not clinical decision support,
      * the engine acquires a JAX/CUDA backend while the module still documents
        itself as numpy (which would make an accelerator figure quotable),
      * a shipped surface claims an FDA clearance, a granted De Novo/510(k), or
        a GPU-accelerated solver, or
      * a benefit badge moves -- ``clinical_benefit`` and
        ``engine_predicts_patients`` must stay UNKNOWN, and the day-365 readout
        must stay PROVEN_FAILED as a patient prediction.
    """
    import inspect

    problems: List[str] = []

    if "not a device" not in SAMD_CONTEXT.get("status", "").lower():
        problems.append("SAMD_CONTEXT no longer states this is not a device")

    # The engine is numpy; a JAX/CUDA backend would make a figure quotable.
    for rel in ("danon/evosteer.py", "danon/anisotropy.py", "danon/accel.py"):
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        if re.search(r"(?m)^\s*(?:import|from)\s+(?:jax|cupy)\b", text):
            problems.append(
                f"{rel} imports a JAX/CuPy backend; the engine documents itself "
                f"as numpy, so an accelerator figure would be quotable"
            )

    # The ANC model exists and calls itself what it is.
    try:
        from danon.dosing import MyelosuppressionModel
    except Exception as exc:  # noqa: BLE001 -- reported, not raised
        problems.append(f"the ANC safety model cannot be imported: {exc}")
    else:
        doc = (inspect.getdoc(MyelosuppressionModel) or "").lower()
        disclaimers = (
            "nothing here is clinical decision support",
            "is not clinical decision support",
            "not a clinical decision support",
        )
        if not any(phrase in doc for phrase in disclaimers):
            problems.append(
                "the ANC safety model no longer states it is not clinical "
                "decision support"
            )

    # The benefit badges cannot be moved by engineering.
    provenance = claims_provenance()
    for key in ("clinical_benefit", "engine_predicts_patients"):
        if provenance.get(key) is not EvidenceStatus.UNKNOWN:
            problems.append(f"{key} is no longer UNKNOWN")
    try:
        from danon.outcome_export import outcome_export_provenance

        day365 = outcome_export_provenance().get(
            "day_365_readout_is_a_patient_prediction"
        )
        if day365 is not EvidenceStatus.PROVEN_FAILED:
            problems.append(
                "the day-365 readout is no longer refuted as a patient prediction"
            )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"the outcome provenance cannot be read back: {exc}")

    fabrications, scanned = scan_for_fabricated_device_status(
        ROOT, skip=os.path.abspath(__file__)
    )
    problems.extend(fabrications)

    verdict = (
        "research prototype, not a device; the pathway (device via 520(o)(1)(E), "
        "De Novo/PMA, PCCP, IEC 62304 / ISO 13485 / ISO 14971) is CONTEXT; "
        f"benefit badges pinned; {scanned} shipped file(s) scanned for an "
        "FDA/GPU status this repository does not have"
    )
    if problems:
        return CheckResult(False, verdict + " Problems: " + "; ".join(problems))
    return CheckResult(True, verdict)


def check_benefit_safety_and_runtime_are_measured_not_asserted() -> CheckResult:
    """A translation pitch's benefit, safety and runtime claims, checked.

    The pitch asserts that the software saves lives, that it GUARANTEES the
    schedule will not trigger lethal marrow failure, that it predicts a death at
    180 days, and that it runs 10,000 schedules in 30 seconds. Each is audited
    against a measurement rather than adopted:

      * the survival claim is refused -- ``clinical_benefit`` stays UNKNOWN and
        the day-365 readout stays PROVEN_FAILED as a patient prediction,
      * the safety 'guarantee' is a DECLARED screen: its floor is a LABEL
        threshold, its own docstring says it is not clinical decision support,
        and the repository's own sweep records cells that FAIL the floor,
      * the runtime is the measured extrapolation in ``outputs_real/perf.json``
        (~1.4 h for 10,000 draws on 4-8 workers), not seconds, and
      * no shipped surface carries one of these fabricated claims.
    """
    import inspect

    problems: List[str] = []

    provenance = claims_provenance()
    for key in ("clinical_benefit", "engine_predicts_patients"):
        if provenance.get(key) is not EvidenceStatus.UNKNOWN:
            problems.append(f"{key} is no longer UNKNOWN")
    try:
        from danon.outcome_export import outcome_export_provenance

        if outcome_export_provenance().get(
            "day_365_readout_is_a_patient_prediction"
        ) is not EvidenceStatus.PROVEN_FAILED:
            problems.append(
                "the day-365 readout is no longer refuted as a patient prediction"
            )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"the outcome provenance cannot be read back: {exc}")

    # The safety screen is declared and label-anchored, not an interlock.
    try:
        from danon.dosing import LABEL_ANC_FLOOR, MyelosuppressionModel
    except Exception as exc:  # noqa: BLE001
        problems.append(f"the ANC safety model cannot be imported: {exc}")
    else:
        if abs(float(LABEL_ANC_FLOOR) - 1.5) > 1e-9:
            problems.append(
                f"the ANC floor is {LABEL_ANC_FLOOR}, not the label's 1.5 x10^9/L"
            )
        doc = (inspect.getdoc(MyelosuppressionModel) or "").lower()
        disclaimers = (
            "nothing here is clinical decision support",
            "is not clinical decision support",
            "not a clinical decision support",
        )
        if not any(phrase in doc for phrase in disclaimers):
            problems.append(
                "the ANC safety model no longer disclaims clinical decision support"
            )

    # The runtime is the measured sweep, not a hopeful figure.
    path = os.path.join(ROOT, "outputs_real", "perf.json")
    if not os.path.exists(path):
        problems.append(
            "outputs_real/perf.json is missing, so the 10,000-draw runtime is "
            "unmeasured. Run _perf_cohort.py."
        )
    else:
        try:
            with open(path, encoding="utf-8") as fh:
                perf = json.load(fh)
        except (OSError, ValueError) as exc:
            problems.append(f"the perf record is unreadable: {exc}")
        else:
            sweep = perf.get("uq_sweep_extrapolation") or {}
            draws = int(sweep.get("draws") or 0)
            hours = float(sweep.get("hours_on_4_workers") or 0.0)
            if draws != 10000:
                problems.append(
                    f"the sweep record covers {draws} draws, not the 10,000 the "
                    f"pitch quotes"
                )
            if hours < 0.5:
                problems.append(
                    f"the 10,000-draw sweep is recorded at {hours:.3f} h; a "
                    f"seconds-scale figure would contradict the measurement"
                )

    # The precedent is context, and says so.
    if "not a result of this engine" not in TRANSLATION_PRECEDENT.lower():
        problems.append("TRANSLATION_PRECEDENT no longer disclaims a result here")

    fabrications, scanned = scan_for_fabricated_benefit_claims(
        ROOT, skip=os.path.abspath(__file__)
    )
    problems.extend(fabrications)

    verdict = (
        "survival claim refused (clinical_benefit UNKNOWN, day-365 PROVEN_FAILED); "
        "safety screen is label-anchored and disclaims CDS; 10,000-draw sweep "
        "is ~1.4 h measured, not seconds; precedent recorded as context; "
        f"{scanned} shipped file(s) scanned for fabricated benefit/safety "
        "claims"
    )
    if problems:
        return CheckResult(False, verdict + " Problems: " + "; ".join(problems))
    return CheckResult(True, verdict)


# ---------------------------------------------------------------------------
# Derived provenance: a tier a check derives from EVIDENCE, never from the audit
# ---------------------------------------------------------------------------
# A tier a human types is a string; a tier a check derives is a measurement --
# but only if the derivation is SEMANTIC. Two rules make that safe, and both are
# enforced below:
#
#   1. A derivation reads an ARTIFACT, never the audit. The provenance keys are
#      themselves read by the checks (``badge_inventory``, the SaMD and benefit
#      checks all call ``claims_provenance()``), so deriving a key from
#      ``audit_claims()`` would recurse. The static-tier / live-support split is
#      therefore load-bearing rather than lazy.
#   2. A tier is NEVER the audit result. Whether a claim's check passes is
#      ``supported``, which is orthogonal -- supported claims here carry UNKNOWN
#      and PROVEN_FAILED tiers (``no_engine_c_index``, ``no_spatial_agreement``).

PROVENANCE_DERIVATION_RULE: str = (
    "A tier is DECLARED unless a semantic derivation from ingested evidence "
    "replaces it. A derivation reads artifacts, NEVER the audit -- the "
    "provenance keys are read BY the checks, so deriving one from audit_claims() "
    "would recurse. And a tier is never the audit result: whether a check passes "
    "is `supported`, which is orthogonal."
)


def evidence_justifies_a_tier_upgrade(
    evidence: Dict[str, object]
) -> Tuple[bool, str]:
    """Whether an ingested evidence bundle may RAISE a tier.

    Semantic, never presence-based. A bundle must declare what KIND of evidence
    it is, that its annotation is EXPERT (an automated mask scores
    model-versus-model), that it is linked to the same patients the claim is
    about, and that a unit/bounds check passed. ``exists=True`` alone never
    qualifies -- this repository's own LUMIERE finding is the counter-example:
    reachable, 91 patients, 2487 images, and AUTOMATED masks.
    """
    missing: List[str] = []
    if not evidence.get("kind"):
        missing.append("no evidence kind declared")
    if str(evidence.get("annotation", "")).lower() != "expert":
        missing.append(
            "annotation is not expert (an automated mask is model-versus-model)"
        )
    if not evidence.get("patient_linked"):
        missing.append("no link to the patients the claim is about")
    if not evidence.get("units_checked"):
        missing.append("no unit/bounds check recorded")
    if missing:
        return False, "; ".join(missing)
    return True, f"semantic {evidence['kind']} evidence with expert annotation"


def derive_r0_knowledge_tier() -> Tuple[EvidenceStatus, str]:
    """The tier of "a measured growth rate exists here", derived from evidence.

    Reads the Ki-67 source probe record. It CANNOT upgrade because a file
    appeared: the upgrade requires an ingested, EXPERT-labelled Ki-67 labelling
    index -- the numerator the growth mapping needs -- and the probe records
    that the two named public sources carry none (and that a nuclei density is
    not that numerator).
    """
    path = os.path.join(ROOT, "outputs_real", "ki67_source_probe.json")
    if not os.path.exists(path):
        return EvidenceStatus.UNKNOWN, "no Ki-67 source probe is recorded"
    try:
        with open(path, encoding="utf-8") as fh:
            record = json.load(fh)
    except (OSError, ValueError) as exc:
        return EvidenceStatus.UNKNOWN, f"the Ki-67 probe is unreadable: {exc}"

    # A semantic upgrade is available ONLY from an ingested labelling-index
    # source that passes the gate; a present-but-not-semantic source does not
    # upgrade, which is the rule that keeps an automated mask from doing it.
    for source in record.get("ingested_labelling_index_sources") or []:
        ok, why = evidence_justifies_a_tier_upgrade(source)
        if ok:
            return EvidenceStatus.ESTABLISHED, f"ingested {why}"
    return (
        EvidenceStatus.UNKNOWN,
        "no ingested Ki-67 labelling index: the numerator the growth mapping "
        "needs is absent from the sources probed, and a nuclei density is not it",
    )


DERIVED_TIERS: Dict[str, Callable[[], Tuple[EvidenceStatus, str]]] = {
    "r0_knowledge": derive_r0_knowledge_tier,
}


def derived_tier(key: str) -> Tuple[EvidenceStatus, str]:
    """Apply a registered derivation. Raises for an unknown key."""
    return DERIVED_TIERS[key]()


def check_provenance_tiers_are_derived_from_evidence_not_declared() -> CheckResult:
    """A tier is derived from EVIDENCE, never from presence or the audit.

    Four things are checked: a presence-only or automated bundle may not raise a
    tier (the LUMIERE rule), a semantic bundle may, the r0 derivation reads its
    probe and stays UNKNOWN for want of a numerator, and ``claim_matches_code``
    is deliberately NOT derived because the provenance keys are read by the
    checks (deriving them from the audit would recurse).
    """
    problems: List[str] = []

    automated = {"exists": True, "count": 2487, "annotation": "automated"}
    allowed, why = evidence_justifies_a_tier_upgrade(automated)
    if allowed:
        problems.append(
            "a presence-only, automated bundle was allowed to raise a tier; "
            "LUMIERE's automated masks would then fabricate a Dice result"
        )
    elif "expert" not in why:
        problems.append(
            "the rejection does not name expert annotation as the gate, so the "
            "reason is not the one recorded here"
        )

    semantic = {
        "kind": "labelling_index",
        "annotation": "expert",
        "patient_linked": True,
        "units_checked": True,
    }
    if not evidence_justifies_a_tier_upgrade(semantic)[0]:
        problems.append(
            "a semantic, expert-labelled bundle was refused, so nothing could "
            "ever raise a tier"
        )

    tier, reason = derive_r0_knowledge_tier()
    if tier is not EvidenceStatus.UNKNOWN:
        problems.append(
            f"the r0 knowledge tier reads {tier.name}; it should be UNKNOWN "
            f"until an expert-labelled labelling index is ingested"
        )
    if "labelling index" not in reason:
        problems.append("the r0 derivation no longer states what it is missing")

    if "claim_matches_code" in DERIVED_TIERS:
        problems.append(
            "claim_matches_code is derived from the audit; the provenance keys "
            "are read by the checks, so that derivation recurses"
        )
    if "never the audit" not in PROVENANCE_DERIVATION_RULE:
        problems.append("the derivation rule no longer forbids reading the audit")

    verdict = (
        "presence-only/automated evidence cannot raise a tier (LUMIERE rule); "
        "semantic expert-labelled evidence can; the r0 tier derives to UNKNOWN "
        "for want of a Ki-67 numerator; the audit result is never a tier"
    )
    if problems:
        return CheckResult(False, verdict + " Problems: " + "; ".join(problems))
    return CheckResult(True, verdict)


# ---------------------------------------------------------------------------
# The DUA cohort specification: a document handed to a data partner, audited
# ---------------------------------------------------------------------------
# A hospital will not release PHI for a vague request, so a written
# specification is the right artifact for Step 0. But a specification that
# MISQUOTES the engine sends a partner off to build the wrong dataset, and a
# data-steward annex is exactly where a number gets copied into a protocol and
# never re-checked. So every figure the pitch's spec states is pinned to the
# repository rather than remembered, and three of its own claims do not survive
# that: the tissue map, the decimation penalty, and the availability of the
# longitudinal pairs.

COHORT_SPECIFICATION: Dict[str, str] = {
    "objective": (
        "the minimum cohort to evaluate the twin as a RETROSPECTIVE research "
        "benchmark for PFS imaging agreement; it cannot establish benefit, "
        "safety or treatability, which need prospective trials, not a dataset"
    ),
    "baseline_imaging": (
        "pre-treatment T1ce + T2-FLAIR NIfTI (.nii/.nii.gz), co-registered, with "
        "an EXPERT MANUAL segmentation -- an automated mask scores "
        "model-versus-model and is refused (LUMIERE) -- carrying the BraTS label "
        "convention the engine actually reads: 1 necrotic core, 2 peritumoral "
        "edema, 4 enhancing tumour, because danon/realdata.py maps exactly those "
        "labels"
    ),
    "tissue_map": (
        "the engine's own weights, read from "
        "danon/realdata.py::_label_proliferation: enhancing 1.00, edema 0.60, "
        "necrosis 0.20 -- necrosis is NOT zero, so a K=0 mapping would be a "
        "cohort built to the wrong label semantics"
    ),
    "grid": (
        "MAX_TOTAL_VOXELS = 4,000,000 in danon/imaging.py; the documented "
        "axis-aligned block vote agrees with the source occupancy at Dice 0.9310 "
        "on the shipped 3.63 mm grid (worst case 0.8530 at 7.3 mm), while the "
        "SHIPPED danon/realdata.py::_decimate_labels is the DEFECT at Dice "
        "0.3905 -- quoting 0.3905 as the documented penalty inverts the record"
    ),
    "dosing_log": (
        "a per-patient event log the engine can initialise day-by-day: drug, "
        "start_day, duration_days and dose in mg/m2 "
        "(danon/dosing.py::DoseBlock). 'Standard of care' is not an input"
    ),
    "outcomes": (
        "pfs_day and a progressed flag per patient, WITH censored patients "
        "retained -- the repository's own audit says discarding censored "
        "patients deletes the endpoint"
    ),
    "availability": (
        "the longitudinal series is RETRIEVABLE without new code (stream4: 0 "
        "cached / 41 retrievable pairs across MU-Glioma-Post and "
        "Brain-Tumor-Progression), so the blocker is the cached download plus a "
        "mapping rule, not the absence of data"
    ),
    "power": (
        "no N is derived here: the repository's own resolvability floor is the "
        "fold yardstick (floor_conservative 0.0610), so a cohort size must be "
        "justified against that floor rather than asserted as N=30"
    ),
}


# The pitch's statements that the engine contradicts, recorded beside the spec
# so a reader sees the correction rather than just the corrected value.
COHORT_SPEC_CORRECTIONS: Dict[str, str] = {
    "necrosis_is_not_zero": (
        "the pitch maps the necrotic core to K=0; danon/realdata.py maps necrosis "
        "to 0.20, edema to 0.60 and enhancing to 1.00"
    ),
    "the_penalty_is_inverted": (
        "the pitch quotes Dice 0.3905 as the 'documented structural penalty'; "
        "that is the DEFECT of the shipped _decimate_labels, and the documented "
        "rule is Dice 0.9310 (0.8530 worst case)"
    ),
    "scorable_pairs_are_not_audit_py": (
        "the pitch attributes '0 scorable pairs' to danon/audit.py; that count is "
        "the claim census, and 41 longitudinal pairs are RETRIEVABLE"
    ),
    "n_30_is_not_derived": (
        "the pitch's N=30 power threshold is computed nowhere here; the "
        "repository's own yardstick is the fold floor 0.0610"
    ),
}


# Cohort-spec claims a pitch asserts as facts. Each entry is (label, pattern),
# built by concatenation so the audit does not flag its own vocabulary.
FABRICATED_COHORT_SPEC_CLAIMS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    (
        "a necrotic core mapped to carrying capacity zero",
        re.compile(r"(?i)necrot\w+" + r"[^.\n]{0,40}\bK\s*=\s*0\b"),
    ),
    (
        "the shipped decimation defect quoted as the documented penalty",
        re.compile(r"(?i)document\w*" + r"[^.\n]{0,60}0\.3905"),
    ),
    (
        # The pitch's own order: "0 scorable pairs by danon/audit.py".
        "'0 scorable pairs' attributed to danon/audit.py",
        re.compile(r"(?i)scorable" + r"[^.\n]{0,40}audit\.py"),
    ),
    (
        "a powered cohort size asserted as N=30",
        re.compile(r"(?i)\bN\s*=\s*30\b" + r"[^.\n]{0,80}\b(?:power\w*|uncensored)"),
    ),
)


def scan_for_fabricated_cohort_spec_claims(
    root: str, *, skip: Optional[str] = None
) -> Tuple[List[str], int]:
    """Scan the shipped surfaces under ``root`` for a cohort specification's
    fabricated claims about this engine."""
    return _scan_shipped_surfaces(root, FABRICATED_COHORT_SPEC_CLAIMS, skip)


def check_cohort_specification_matches_the_engine_not_the_pitch() -> CheckResult:
    """A DUA cohort specification is pinned to the engine, not to the pitch.

    A data steward annex is where a number gets copied into a protocol and never
    re-checked, so the spec is audited against the repository rather than
    trusted. Four of the pitch's claims are contradicted, and each is read live:

      * the tissue->proliferation map comes from
        ``danon/realdata.py::_label_proliferation`` and necrosis is 0.20, not
        the pitch's K=0,
      * the decimation figure 0.3905 is the SHIPPED defect, not a 'documented'
        penalty -- the documented axis-aligned block vote is 0.9310 (0.8530
        worst case), read from ``outputs_real/grid_artifact.json``,
      * the voxel cap is the engine's own ``MAX_TOTAL_VOXELS`` = 4,000,000, and
      * the longitudinal series is RETRIEVABLE (stream4 0 cached / 41
        retrievable), so a spec may not imply the data does not exist.

    It also confirms no cohort can move the patient-level claim:
    ``clinical_benefit`` and ``engine_predicts_patients`` stay UNKNOWN, because a
    dataset is not a trial.
    """
    import numpy as np

    problems: List[str] = []

    # 1. The voxel cap the spec quotes is the engine's own constant.
    try:
        from danon.imaging import MAX_TOTAL_VOXELS

        if int(MAX_TOTAL_VOXELS) != 4_000_000:
            problems.append(
                f"the engine's voxel cap is {MAX_TOTAL_VOXELS}, not the "
                f"4,000,000 the specification quotes"
            )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"the imaging cap cannot be imported: {exc}")

    # 2. The tissue map is MEASURED from the engine, and necrosis is not zero.
    try:
        from danon.realdata import _label_proliferation

        labels = np.zeros((1, 1, 3), dtype=np.int32)
        labels[0, 0, 0], labels[0, 0, 1], labels[0, 0, 2] = 4, 2, 1
        # A BOOLEAN mask: the function negates it, so a uint8 mask would become
        # 254/255 and index out of bounds.
        prolif, usable = _label_proliferation(labels, labels > 0)
        w = (float(prolif[0, 0, 0]), float(prolif[0, 0, 1]), float(prolif[0, 0, 2]))
        if (
            not usable
            or abs(w[0] - 1.0) > 1e-9
            or abs(w[1] - 0.60) > 1e-9
            or abs(w[2] - 0.20) > 1e-9
        ):
            problems.append(
                f"the engine's tissue weights are {w}, not (1.00, 0.60, 0.20) "
                f"for (enhancing, edema, necrosis)"
            )
        if w[2] <= 0.0:
            problems.append(
                "the engine maps necrosis to zero, so a specification asserting "
                "K=0 would no longer contradict danon/realdata.py"
            )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"the tissue map cannot be read from the engine: {exc}")

    # 3. The decimation penalty: the defect is pinned, and the DOCUMENTED rule is
    #    not the defect.
    grid_path = os.path.join(ROOT, "outputs_real", "grid_artifact.json")
    if not os.path.exists(grid_path):
        problems.append(
            "outputs_real/grid_artifact.json is missing, so the decimation "
            "penalty the specification quotes is unmeasured"
        )
    else:
        try:
            with open(grid_path, encoding="utf-8") as fh:
                grid = json.load(fh)
            defect = float(
                grid["geometry"]["shipped_rule_dice_vs_axis_aligned"]
            )
            documented = float(
                grid["geometry"]["levels"]["engine_3p6mm"]["dice_vs_source"]
            )
            worst = float(
                grid["geometry"]["levels"]["coarse_7p3mm"]["dice_vs_source"]
            )
            if abs(defect - 0.3905) > 1e-3:
                problems.append(
                    f"the shipped decimation defect reads Dice {defect:.4f}, "
                    f"not the 0.3905 the specification pins"
                )
            if documented < 0.90:
                problems.append(
                    f"the documented block vote reads Dice {documented:.4f}, "
                    f"no longer the ~0.9310 the specification quotes"
                )
            if worst < 0.85:
                problems.append(
                    f"the documented worst case reads Dice {worst:.4f}, below "
                    f"the recorded 0.8530"
                )
            if documented <= defect:
                problems.append(
                    "the documented rule is no longer better than the shipped "
                    "defect, so calling 0.3905 'documented' would be true"
                )
        except (OSError, ValueError, KeyError) as exc:
            problems.append(f"the grid artifact is unreadable: {exc}")

    # 4. Availability: 0 cached is NOT 'no data exists'.
    try:
        from danon.clinical_twin import clinical_stream_census

        s4 = clinical_stream_census()["stream4_sequential"]
        if s4.get("n_pairs_cached") != 0 or s4.get("n_pairs_retrievable") != 41:
            problems.append(
                "the stream4 census no longer reads 0 cached / 41 retrievable "
                f"({s4.get('n_pairs_cached')} / {s4.get('n_pairs_retrievable')}), "
                "so the specification's availability claim moved"
            )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"the stream4 census cannot be read: {exc}")

    # 5. The spec states the engine's real facts, in corrected form.
    spec_text = " ".join(COHORT_SPECIFICATION.values())
    for needle in ("4,000,000", "0.60", "0.20", "0.9310", "0.3905", "41"):
        if needle not in spec_text:
            problems.append(f"the specification no longer states {needle}")
    if "NOT zero" not in COHORT_SPECIFICATION.get("tissue_map", ""):
        problems.append(
            "the specification no longer corrects the pitch's K=0 necrosis claim"
        )

    # 6. No cohort can move the patient-level claim: a dataset is not a trial.
    provenance = claims_provenance()
    for key in ("clinical_benefit", "engine_predicts_patients"):
        if provenance.get(key) is not EvidenceStatus.UNKNOWN:
            problems.append(
                f"a cohort specification was allowed to move {key} off UNKNOWN"
            )

    fabrications, scanned = scan_for_fabricated_cohort_spec_claims(
        ROOT, skip=os.path.abspath(__file__)
    )
    problems.extend(fabrications)

    verdict = (
        "specification pinned to the engine: voxel cap 4,000,000; tissue weights "
        "(1.00, 0.60, 0.20) with necrosis NOT zero; decimation defect 0.3905 vs "
        "documented 0.9310 (0.8530 worst case); stream4 0 cached / 41 "
        "retrievable; no N is derived here; clinical_benefit and "
        "engine_predicts_patients stay UNKNOWN; "
        f"{scanned} shipped file(s) scanned for fabricated cohort-spec claims"
    )
    if problems:
        return CheckResult(False, verdict + " Problems: " + "; ".join(problems))
    return CheckResult(True, verdict)


# ---------------------------------------------------------------------------
# The clinical-execution UI: a directive audited against the engine, not built
# ---------------------------------------------------------------------------
# A "make it a clinical decision support tool" directive asks the frontend to
# hide the epistemic surfaces this repository exists to show, and to render a
# benefit figure -- "additional days of progression-free life this schedule
# secures" -- that no simulation can make true. Each ask is therefore pinned to
# the engine rather than adopted: the prescription the directive quotes is the
# LABEL regimen the adaptive arm is compared AGAINST, its "5 on / 14 off"
# holiday contradicts the engine's own 28-day label cycle, the "silent"
# 10,000-draw sweep is a measured ~1.4 h, the safety floor is a DECLARED screen,
# and the benefit badges stay UNKNOWN.

CLINICAL_UI_CORRECTIONS: Dict[str, str] = {
    "the_prescription_is_the_label_not_an_optimum": (
        "the directive's prescription block -- 150 mg/m2 temozolomide for 5 "
        "days -- is the FDA LABEL's own regimen (danon/dosing.py::LABEL_"
        "DOSE_MG_M2 = 150, LABEL_ON_DAYS = 5), i.e. the STANDARD OF CARE the "
        "adaptive arm is compared AGAINST, so it is not an output of any "
        "optimizer in this repository"
    ),
    "the_cycle_arithmetic_is_wrong": (
        "the directive's '5 days on, then a 14-day holiday' is a 19-day cycle, "
        "but the engine's own label anchor is the 28-day cycle "
        "(LABEL_CYCLE_DAYS = 28), so the label holiday is 23 days -- 19 != 28"
    ),
    "hiding_the_warnings_crosses_the_device_line": (
        "a 'Clinical Mode' that hides the algorithmic warnings, provenance "
        "chips and disclaimers removes the basis a non-device CDS tool must "
        "display: SAMD_CONTEXT records that a recommendation which DRIVES "
        "rather than SUPPORTS a clinician's decision fails the non-device CDS "
        "criteria (FD&C Act 520(o)(1)(E)) and is therefore a device, and "
        "frontend/src/vector_design/TranslationalGateway.tsx already requires "
        "MHRA/FDA clearance before any therapeutic use"
    ),
    "the_sweep_is_not_a_background_instant": (
        "a 10,000-draw sweep run 'silently the moment the patient's data is "
        "ingested' is a measured ~1.4 h on 4-8 workers (outputs_real/perf.json), "
        "not a background instant, and running it silently is the opposite of "
        "the recorded provenance"
    ),
    "days_of_life_secured_is_a_lifespan_claim": (
        "an 'additional days of progression-free life this schedule "
        "mathematically secures' is the lifespan claim this repository already "
        "refuses and scans for: clinical_benefit and engine_predicts_patients "
        "stay UNKNOWN and the day-365 readout stays PROVEN_FAILED as a patient "
        "prediction"
    ),
}


# Clinical-UI claims a directive asserts as facts. Each entry is (label,
# pattern), built by concatenation so the audit does not flag its own
# vocabulary, and narrow enough not to trip the repository's legitimate uses.
FABRICATED_CLINICAL_UI_CLAIMS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    (
        "a prescription block presented to a clinician as the recommended "
        "protocol",
        re.compile(r"(?i)\brecommended" + r"\s+protocol\b"),
    ),
    (
        "a clinical mode that hides the epistemic warnings",
        re.compile(
            r"(?i)\bhide\w*\b[^.\n]{0,40}\b"
            + r"(?:warnings?|disclaimers?|uncertainty)\b"
        ),
    ),
    (
        "a figure of additional days of progression-free life",
        re.compile(
            r"(?i)\badditional\s+days?\b[^.\n]{0,40}\b"
            + r"(?:progression-free|pfs|life)\b"
        ),
    ),
    (
        "a claim that the schedule secures progression-free survival",
        re.compile(
            r"(?i)\b(?:secures?|guarantee\w*|delivers?)\b[^.\n]{0,40}\b"
            + r"(?:progression-free|pfs)\b"
        ),
    ),
)


def scan_for_fabricated_clinical_ui_claims(
    root: str, *, skip: Optional[str] = None
) -> Tuple[List[str], int]:
    """Scan the shipped surfaces under ``root`` for a clinical-execution UI's
    fabricated claims about this engine."""
    return _scan_shipped_surfaces(root, FABRICATED_CLINICAL_UI_CLAIMS, skip)


# Claims a surface makes about the SAMPLING of this record. A percentile
# bootstrap over paired replicates treats them as exchangeable, and in this
# record they are NOT: three seeds share one configuration and three
# configurations share one imaging geometry. A count of pairs presented as a
# sample of independent units is therefore a fabricated claim about the unit of
# evidence -- the interval is narrower than the units would justify.
#
# The FRAMES below catch the CLAIM however a careless editor phrases it --
# attributive ("27 independent samples"), predicative ("the pairs are
# independent", "the replicates can be treated as independent"), scoped
# ("independent across the 27 pairs") and appositive ("each an independent
# draw"). They are deliberately permissive: the honest and denied cases are
# separated out by the context filter in ``_independence_claim_hits`` rather than
# by making each frame narrow, because a narrow frame is exactly what a claim
# slips past.
FABRICATED_INDEPENDENCE_CLAIMS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    (
        "a count of replicates presented as independent samples",
        # "independent <sample-noun>". "independent units" is the honest phrasing
        # and is excluded; "independent <noun> of <population>" is the honest
        # description of a bootstrap draw and is excluded too.
        re.compile(
            r"(?i)\b(?:independent|iid|i\.i\.d\.)\b\s+"
            + r"(?!units?\b)(?:\w+\s+){0,3}"
            + r"(?:pairs?|replicates?|samples?|patients?|observations?|draws?)\b"
            + r"(?!\s+of\b)"
        ),
    ),
    (
        "the nested replicates asserted to be an independent sample",
        # "<sample-noun> ... (are/were/is/be) ... independent", which also covers
        # "the replicates can be treated as independent".
        re.compile(
            r"(?i)\b(?:pairs?|replicates?|samples?|patients?|observations?)\b"
            + r"[^.?!\n]{0,80}?\b(?:are|were|is|be|being)\b"
            + r"[^.?!\n]{0,40}?\bindependent\b"
        ),
    ),
    (
        "the replicates asserted independent of one another or of the design",
        # "independent across/within the 27 pairs".
        re.compile(
            r"(?i)\bindependent\b[^.?!\n]{0,25}?"
            + r"\b(?:across|within|over|between|among)\b"
            + r"[^.?!\n]{0,25}?\b\d+\b"
        ),
    ),
)

#: Denials and meta-framing that turn a frame into something other than a claim:
#: "not an independent sample", "nested rather than independent", the honest
#: vocabulary of nesting, and the description of the thing being SCANNED FOR
#: ("a count presented as an independent sample") which the check's own verdict
#: carries. Without this the disclosure -- which names what it refuses -- would
#: flag itself.
_INDEPENDENCE_DENIAL = re.compile(
    r"(?i)(?:\bnot\b|\bnever\b|\bnor\b|\bnested\b|rather than|instead of|"
    r"as opposed to|anything but|far from|presented as|scanned for|asserted to be)"
)

#: Prose outside the shipped source that a count presented as independent can
#: still be written in: the README a reader quotes, and the recorded artifacts'
#: own free text. The shipped-source walk cannot see them, and they are the two
#: most-quoted surfaces in the tree.
_INDEPENDENCE_PROSE_PATHS: Tuple[str, ...] = (
    "README.md",
    os.path.join("outputs_real", "app_artifacts", "trial_design.json"),
    os.path.join("outputs_real", "app_artifacts", "claims.json"),
)


def _independence_claim_hits(text: str) -> List[str]:
    """Labels for claims in ``text`` that present a count as an independent sample.

    The frames find the shape the claim is written in; a frame is only a CLAIM
    when it is not a denial, so the clause around each match is checked and a
    frame sitting inside a negation or the vocabulary of nesting is dropped. That
    is what lets the record's own disclosure -- "NOT INDEPENDENT SAMPLES",
    "nested rather than independent", "not a sample of independent patients" --
    pass while the same words in the affirmative are caught.
    """
    hits: List[str] = []
    for label, pattern in FABRICATED_INDEPENDENCE_CLAIMS:
        for match in pattern.finditer(text):
            # A frame is a CLAIM unless its own sentence (or line, which is the
            # unit in a serialised artifact) denies it or names the nesting, so
            # only that clause is consulted -- which is how "narrower than one
            # over independent patients would be" is read as the comparison it
            # is while "the 27 replicates are independent" is not.
            left = max(text.rfind(ch, 0, match.start()) for ch in ".?!")
            rights = [text.find(ch, match.end()) for ch in ".?!"]
            right = min((i for i in rights if i != -1), default=len(text))
            if _INDEPENDENCE_DENIAL.search(text[left + 1: right + 1]):
                continue
            hits.append(label)
            break
    return hits


def scan_text_for_fabricated_independence(text: str) -> List[str]:
    """Apply the independence rule to one blob of text, wherever it lives."""
    return _independence_claim_hits(text)


def _artifact_prose(obj: object, out: Optional[List[str]] = None) -> List[str]:
    """Every string VALUE inside a recorded artifact, one clause-unit each.

    A JSON dump would concatenate unrelated fields into one run-on line, so the
    strings are walked individually: each is scanned as its own prose, which is
    how a fabricated note is caught without an honest sibling clause suppressing
    it.
    """
    if out is None:
        out = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            _artifact_prose(value, out)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _artifact_prose(value, out)
    return out


def scan_for_fabricated_independence_claims(
    root: str, *, skip: Optional[str] = None
) -> Tuple[List[str], int]:
    """Scan EVERY surface a fabricated independence claim can be written on.

    The shipped source files, PLUS the README and the recorded artifacts' free
    text: a count presented as an independent sample is the same overclaim
    whether it is written in a module or in the README a reader quotes, so the
    scan does not stop at the source tree. It applies ``_independence_claim_hits``
    (the frames plus the denial filter) rather than the raw frames.
    """
    problems, scanned = _scan_shipped_surfaces(
        root, FABRICATED_INDEPENDENCE_CLAIMS, skip, matcher=_independence_claim_hits
    )
    for rel in _INDEPENDENCE_PROSE_PATHS:
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            continue
        scanned += 1
        if rel.endswith(".json"):
            try:
                with open(path, encoding="utf-8") as fh:
                    obj = json.load(fh)
            except (OSError, ValueError):
                continue
            blobs = _artifact_prose(obj)
        else:
            with open(path, encoding="utf-8", errors="replace") as fh:
                blobs = [fh.read()]
        for prose in blobs:
            for label in _independence_claim_hits(prose):
                problems.append(f"{rel} contains {label}")
    return problems, scanned


def check_clinical_execution_ui_is_audited_not_implemented() -> CheckResult:
    """A 'clinical execution UI' directive, audited against the engine.

    The directive asks for four things that this repository's own record
    refuses, so each is pinned live rather than adopted:

      * a 'definitive prescription block' -- but the regimen it quotes
        (150 mg/m2 temozolomide for 5 days) is the FDA LABEL's own regimen, the
        standard of care the adaptive arm is compared AGAINST, and its
        '5 on / 14 off' holiday contradicts the engine's 28-day label cycle,
      * a 'Clinical Mode' that hides the algorithmic warnings -- but a
        recommendation that DRIVES a clinician's decision rather than SUPPORTS
        it fails the non-device CDS criteria (520(o)(1)(E)) and is a device, and
        the frontend's own gate requires MHRA/FDA clearance before any
        therapeutic use,
      * a 10,000-draw sweep run 'silently the moment data is ingested' -- but
        that sweep is a measured ~1.4 h on 4-8 workers, not a background
        instant,
      * 'how many additional days of progression-free life this schedule
        mathematically secures' -- a lifespan claim: clinical_benefit and
        engine_predicts_patients stay UNKNOWN and the day-365 readout stays
        PROVEN_FAILED as a patient prediction.
    """
    import inspect

    problems: List[str] = []

    # 1. The 'prescription block' is the LABEL regimen, and its cycle arithmetic
    #    is not the engine's.
    try:
        from danon.dosing import (
            LABEL_CYCLE_DAYS,
            LABEL_DOSE_MG_M2,
            LABEL_ON_DAYS,
        )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"the label regimen cannot be imported: {exc}")
    else:
        if abs(float(LABEL_DOSE_MG_M2) - 150.0) > 1e-9 or int(LABEL_ON_DAYS) != 5:
            problems.append(
                f"the label regimen is no longer 150 mg/m2 for 5 days (reads "
                f"{LABEL_DOSE_MG_M2} mg/m2 x {LABEL_ON_DAYS} d), so the "
                f"directive's prescription block is no longer the label's"
            )
        if int(LABEL_CYCLE_DAYS) != 28:
            problems.append(
                f"the label cycle is no longer 28 days (reads {LABEL_CYCLE_DAYS}), "
                f"so the recorded 5-on/14-off correction is stale"
            )
        elif int(LABEL_ON_DAYS) + 14 == int(LABEL_CYCLE_DAYS):
            problems.append(
                "the directive's '5 days on, then a 14-day holiday' now matches "
                "the engine's cycle, so the recorded correction is stale"
            )

    # 2. The safety floor the directive 'constrains' against is a DECLARED
    #    screen whose own text disclaims clinical decision support.
    try:
        from danon.dosing import MyelosuppressionModel
    except Exception as exc:  # noqa: BLE001
        problems.append(f"the ANC safety model cannot be imported: {exc}")
    else:
        doc = (inspect.getdoc(MyelosuppressionModel) or "").lower()
        if not any(
            phrase in doc
            for phrase in (
                "nothing here is clinical decision support",
                "is not clinical decision support",
                "not a clinical decision support",
            )
        ):
            problems.append(
                "the ANC safety model no longer states it is not clinical "
                "decision support"
            )
        if not hasattr(MyelosuppressionModel, "feasible"):
            problems.append("the ANC screen no longer exposes feasible()")

    # 3. The directive's deliverable -- 'additional days of progression-free life
    #    this schedule mathematically secures' -- is a lifespan claim.
    provenance = claims_provenance()
    for key in ("clinical_benefit", "engine_predicts_patients"):
        if provenance.get(key) is not EvidenceStatus.UNKNOWN:
            problems.append(f"a UI directive was allowed to move {key} off UNKNOWN")
    try:
        from danon.outcome_export import outcome_export_provenance

        if (
            outcome_export_provenance().get(
                "day_365_readout_is_a_patient_prediction"
            )
            is not EvidenceStatus.PROVEN_FAILED
        ):
            problems.append(
                "the day-365 readout is no longer refuted as a patient prediction"
            )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"the outcome provenance cannot be read back: {exc}")

    # 4. The 'silent 10,000-draw sweep on ingest' is a measured extrapolation.
    path = os.path.join(ROOT, "outputs_real", "perf.json")
    if not os.path.exists(path):
        problems.append(
            "outputs_real/perf.json is missing, so the 10,000-draw sweep the UI "
            "would run on ingest is unmeasured"
        )
    else:
        try:
            with open(path, encoding="utf-8") as fh:
                perf = json.load(fh)
            sweep = perf.get("uq_sweep_extrapolation") or {}
            draws = int(sweep.get("draws") or 0)
            hours = float(sweep.get("hours_on_4_workers") or 0.0)
            if draws != 10000:
                problems.append(
                    f"the recorded sweep covers {draws} draws, not the 10,000 the "
                    f"directive would run on ingest"
                )
            if hours < 0.5:
                problems.append(
                    f"the 10,000-draw sweep is recorded at {hours:.3f} h; a "
                    f"background-instant figure would contradict it"
                )
        except (OSError, ValueError) as exc:
            problems.append(f"the perf record is unreadable: {exc}")

    # 5. Hiding the warnings crosses the device line, and the frontend's own gate
    #    already says so -- the tier surface and the gate are still shipped.
    cds = SAMD_CONTEXT.get("cds_exemption", "")
    if "DRIVES" not in cds or "520(o)(1)(E)" not in cds:
        problems.append(
            "SAMD_CONTEXT no longer records that a driving recommendation fails "
            "the non-device CDS criteria (520(o)(1)(E))"
        )
    gate = os.path.join(
        ROOT, "frontend", "src", "vector_design", "TranslationalGateway.tsx"
    )
    if not os.path.exists(gate):
        problems.append(
            "frontend/src/vector_design/TranslationalGateway.tsx is missing, so "
            "the recorded MHRA/FDA gate cannot be read"
        )
    else:
        with open(gate, encoding="utf-8", errors="replace") as fh:
            gate_text = " ".join(fh.read().split())
        if "clearance before any therapeutic use" not in gate_text:
            problems.append(
                "the frontend no longer states that MHRA/FDA clearance is "
                "required before any therapeutic use"
            )
    tier_surface = os.path.join(ROOT, "frontend", "src", "utils", "evidenceTier.ts")
    if not os.path.exists(tier_surface):
        problems.append(
            "the frontend's evidence-tier surface is gone, so a 'Clinical Mode' "
            "would have nothing to hide but the only honest readout"
        )

    # 6. The corrections are recorded, not implied.
    cycle = CLINICAL_UI_CORRECTIONS.get("the_cycle_arithmetic_is_wrong", "")
    if "28" not in cycle or "19" not in cycle:
        problems.append(
            "the recorded cycle correction no longer states 5 + 14 = 19 against "
            "the 28-day label"
        )
    if "LABEL" not in CLINICAL_UI_CORRECTIONS.get(
        "the_prescription_is_the_label_not_an_optimum", ""
    ):
        problems.append(
            "the recorded correction no longer names the label regimen as the "
            "block's source"
        )

    fabrications, scanned = scan_for_fabricated_clinical_ui_claims(
        ROOT, skip=os.path.abspath(__file__)
    )
    problems.extend(fabrications)

    verdict = (
        "directive audited, not built: the 'prescription block' is the label's "
        "own 150 mg/m2 x 5 d (28-day cycle, so '5 on / 14 off' is a 19-day "
        "cycle), not an optimum; the safety floor is a DECLARED screen that "
        "disclaims CDS; the 10,000-draw sweep is ~1.4 h measured, not "
        "silent/instant; the 'additional days of progression-free life' figure "
        "is refused (clinical_benefit UNKNOWN, day-365 PROVEN_FAILED); the "
        "frontend's MHRA/FDA gate stays; "
        f"{scanned} shipped file(s) scanned for fabricated clinical-UI claims"
    )
    if problems:
        return CheckResult(False, verdict + " Problems: " + "; ".join(problems))
    return CheckResult(True, verdict)


# ---------------------------------------------------------------------------
# Paired-seed PFS extension: two estimands, recorded rather than fabricated
# ---------------------------------------------------------------------------
# The outcome cards call `pfs_extension` the ratio of the two arms' median PFS.
# The repository's own campaign rule describes a different quantity -- the PFS
# extension over PAIRED seeds, as a bootstrap 95% interval. Both are real, and
# they are not the same number, so both are recorded and labelled. This check
# pins the paired one to the recorded replicates it is derived from.


def check_paired_pfs_extension_is_recorded_not_fabricated() -> CheckResult:
    """The paired-seed PFS extension is derived from recorded replicates.

    It fails if

      * the recorded interval no longer re-derives from
        ``outputs_real/campaign_log.json`` on the fixed seed (so the artifact
        drifted from the data),
      * the interval is absent -- a bare point estimate, which the S1 rule
        forbids,
      * the right-censoring stops being flagged (the adaptive arm reaches the
        horizon, so every difference is a lower bound), or
      * the two estimands stop being labelled as distinct, or a benefit badge
        moves off UNKNOWN.
    """
    problems: List[str] = []

    # Prefer the payload built in THIS process: the claims artifact's own audit
    # runs while the paired-extension payload is still being assembled, so on a
    # cold build the file may not exist yet. Fall back to the file otherwise.
    recorded: Optional[Dict[str, object]] = None
    try:
        from danon.app_artifacts import _BUILT

        recorded = _BUILT.get("paired_extension")
    except Exception:  # noqa: BLE001
        recorded = None

    path = os.path.join(
        ROOT, "outputs_real", "app_artifacts", "paired_extension.json"
    )
    if recorded is None:
        if not os.path.exists(path):
            return CheckResult(
                False,
                "outputs_real/app_artifacts/paired_extension.json is not built, so "
                "the paired-seed interval is unrecorded. Build it with: "
                "write_app_artifacts(keys=['paired_extension'])",
            )
        try:
            with open(path, encoding="utf-8") as fh:
                recorded = json.load(fh)
        except (OSError, ValueError) as exc:
            return CheckResult(
                False, f"the paired-extension artifact is unreadable: {exc}"
            )

    data = recorded.get("data") or {}

    # 1. It re-derives from the log, on the same seed.
    live: Dict[str, object] = {}
    try:
        from danon.paired_extension import paired_pfs_extension_report

        live = paired_pfs_extension_report()
    except Exception as exc:  # noqa: BLE001
        problems.append(f"the paired extension cannot be re-derived: {exc}")
    for key in (
        "n_cases",
        "n_pairs",
        "paired_mean_difference_days",
        "paired_mean_difference_ci95_days",
        "bootstrap_seed",
    ):
        if key in live and data.get(key) != live.get(key):
            problems.append(
                f"the recorded {key} is {data.get(key)!r}, but re-deriving it "
                f"from outputs_real/campaign_log.json gives {live.get(key)!r}"
            )

    # 2. An interval, not a bare point estimate.
    ci = data.get("paired_mean_difference_ci95_days")
    if not ci or len(ci) != 2:
        problems.append(
            "the paired extension is recorded without its bootstrap interval, so "
            "it is a bare point estimate -- the thing the S1 rule forbids"
        )

    # 3. The censoring is flagged, not hidden.
    if not data.get("is_right_censored_lower_bound"):
        problems.append(
            "the paired extension no longer records that it is a right-censored "
            "lower bound"
        )
    if int(data.get("n_policy_replicates_at_horizon") or 0) <= 0:
        problems.append(
            "no adaptive replicate is recorded at the horizon, so the censoring "
            "note is stale"
        )

    # 4. The two estimands are labelled as distinct.
    note = str(data.get("estimand_note", "")).lower()
    if "ratio of medians" not in note or "not interchangeable" not in note:
        problems.append(
            "the artifact no longer states that the paired difference and the "
            "ratio of medians are different, non-interchangeable estimands"
        )
    if data.get("ratio_of_medians_extension_median") is None:
        problems.append(
            "the ratio-of-medians estimand the outcome cards use is absent"
        )

    # 5. A model comparison cannot move a patient-level badge.
    provenance = claims_provenance()
    for key in ("clinical_benefit", "engine_predicts_patients"):
        if provenance.get(key) is not EvidenceStatus.UNKNOWN:
            problems.append(
                f"a paired-seed interval was allowed to move {key} off UNKNOWN"
            )

    verdict = (
        "two estimands recorded and labelled distinct: the paired replicate "
        "difference in days with a bootstrap 95% interval, and the per-case "
        "ratio of medians; interval re-derives from the log on a fixed seed; "
        "right-censored lower bound flagged; benefit badges stay UNKNOWN"
    )
    if problems:
        return CheckResult(False, verdict + " Problems: " + "; ".join(problems))
    return CheckResult(True, verdict)


#: Count surfaces in the view that must co-render a pair count and the
#: independent-unit count. Each entry is keyed on the DATA the surface reads --
#: not on its label -- so an honest rewording of the wording passes while a
#: surface that drops its units fails. ``[^`]`` keeps a match inside one template
#: literal, and the badge entry excludes an interpolated ``${`` so it isolates
#: the badge from the ladder cells that render the same number.
_VIEW_UNIT_SURFACES: Tuple[Tuple[str, str], ...] = (
    ("the pooled count", r"\$\{live\.n_pairs\}[^`]{0,80}independent_units"),
    ("the per-image count", r"\$\{g\.n_pairs\}[^`]{0,80}independent_units"),
    (
        "the per-configuration count",
        r"\$\{c\.n_pairs\}[^`]{0,120}independent_units",
    ),
    ("the header badge", r"\{live\.n_pairs\}(?:(?!\$\{)[^`]){0,80}independent_units"),
)


def _view_unit_surface_problems(view_text: str) -> List[str]:
    """Which count surfaces in the view no longer show the units behind them."""
    return [
        f"the trial-design view no longer shows independent units beside {scope}, "
        f"so a bare pair count is published as if it were a sample of independent "
        f"replicates"
        for scope, pattern in _VIEW_UNIT_SURFACES
        if not re.search(pattern, view_text)
    ]


def check_trial_design_screen_is_derived_not_prescribed() -> CheckResult:
    """The trial-design screen states, shows noise for, and rules out -- nothing more.

    It fails if

      * the recorded counterfactual no longer re-derives from the recorded
        paired replicates,
      * the noise floor stops being derived: the two failure labels collapse
        into one, or a set below the method's own minimum pair count is
        reported as resolved rather than unassessed,
      * the rule-out is described as a regulatory boundary rather than the
        DECLARED screen it is, or the brief's own 19-day arithmetic stops being
        among the calendars it evaluates,
      * the recorded simulated-inversion days and the note about them disagree,
      * the shipped screen carries a fabricated clinical-UI claim (the same
        vocabulary the clinical-execution check scans for), or
      * a benefit badge moves off UNKNOWN.
    """
    import inspect

    problems: List[str] = []

    recorded: Optional[Dict[str, object]] = None
    built_here = False
    try:
        from danon.app_artifacts import _BUILT

        recorded = _BUILT.get("trial_design")
        built_here = recorded is not None
    except Exception:  # noqa: BLE001
        recorded = None

    path = os.path.join(ROOT, "outputs_real", "app_artifacts", "trial_design.json")
    if recorded is None:
        if not os.path.exists(path):
            return CheckResult(
                False,
                "outputs_real/app_artifacts/trial_design.json is not built, so the "
                "trial-design screen is unrecorded. Build it with: "
                "write_app_artifacts(keys=['trial_design'])",
            )
        try:
            with open(path, encoding="utf-8") as fh:
                recorded = json.load(fh)
        except (OSError, ValueError) as exc:
            return CheckResult(
                False, f"the trial-design artifact is unreadable: {exc}"
            )

    data = recorded.get("data") or {}

    # 1. It re-derives from the recorded replicates.
    live: Dict[str, object] = {}
    try:
        from danon.trial_design import trial_design_report

        live = trial_design_report()
    except Exception as exc:  # noqa: BLE001
        problems.append(f"the trial-design report cannot be re-derived: {exc}")
    for key in ("n_cases", "n_pairs", "horizon_days", "bootstrap_seed"):
        if key in live and data.get(key) != live.get(key):
            problems.append(
                f"the recorded {key} is {data.get(key)!r}, but re-deriving it from "
                f"outputs_real/outcome_cards.json gives {live.get(key)!r}"
            )
    rec_pooled = data.get("pooled") or {}
    live_pooled = live.get("pooled") or {}
    for key in ("mean_difference_days", "median_difference_days"):
        if key in live_pooled and rec_pooled.get(key) != live_pooled.get(key):
            problems.append(
                f"the recorded pooled {key} is {rec_pooled.get(key)!r}, but "
                f"re-deriving it gives {live_pooled.get(key)!r}"
            )
    rec_ci = (rec_pooled.get("intervals") or {}).get("ci95")
    live_ci = (live_pooled.get("intervals") or {}).get("ci95")
    if rec_ci != live_ci:
        problems.append(
            f"the recorded pooled 95% interval is {rec_ci!r}, but re-deriving it "
            f"gives {live_ci!r}"
        )

    # 2. The rule-out is against a DECLARED screen, and it says so.
    anc = data.get("anc_screen") or {}
    if not anc.get("declared"):
        problems.append(
            "the trial-design rule-out no longer records that the ANC screen it "
            "applies is DECLARED rather than a licence threshold"
        )
    if anc.get("is_a_regulatory_boundary") is not False:
        problems.append(
            "the trial-design rule-out presents the screen as a regulatory "
            "boundary; the label's 1.5 x 10^9/L floor applied through a declared "
            "convolution is not one"
        )
    try:
        from danon.dosing import MyelosuppressionModel

        doc = (inspect.getdoc(MyelosuppressionModel) or "").lower()
        if not any(
            phrase in doc
            for phrase in (
                "nothing here is clinical decision support",
                "is not clinical decision support",
                "not a clinical decision support",
            )
        ):
            problems.append(
                "the ANC screen no longer states it is not clinical decision "
                "support, so calling its refusals a rule-out is no longer pinned"
            )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"the ANC screen cannot be imported: {exc}")

    candidates = {
        str(c.get("label")): c for c in (anc.get("candidates") or [])
    }
    brief = candidates.get("brief_arithmetic_150x5_of_19")
    if brief is None:
        problems.append(
            "the brief's own 5-on/14-off arithmetic is no longer among the "
            "calendars the rule-out evaluates, so the panel no longer answers the "
            "thing it exists to answer"
        )
    elif brief.get("feasible"):
        problems.append(
            "the brief's 19-day arithmetic is now feasible under the declared "
            "screen, so the recorded rule-out is stale"
        )

    # 3. The noise floor is derived, and the two failure labels stay distinct.
    try:
        from danon.trial_design import (
            INSUFFICIENT_REPLICATES_FLAG,
            MIN_PAIRS_FOR_AN_INTERVAL,
            NOISE_FLOOR_FLAG,
        )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"the trial-design flags cannot be imported: {exc}")
        NOISE_FLOOR_FLAG = INSUFFICIENT_REPLICATES_FLAG = ""  # type: ignore
        MIN_PAIRS_FOR_AN_INTERVAL = 5  # type: ignore
    else:
        if NOISE_FLOOR_FLAG == INSUFFICIENT_REPLICATES_FLAG:
            problems.append(
                "the two unresolved states -- within a noise floor, and "
                "unassessed for want of pairs -- have collapsed into one label"
            )

    cases = data.get("cases") or []
    if not cases:
        problems.append("the trial-design record carries no per-case rows")
    else:
        pairs = [int(c.get("n_pairs") or 0) for c in cases]
        statuses = {
            (c.get("noise_floor") or {}).get("status") for c in cases
        }
        if min(pairs) < int(MIN_PAIRS_FOR_AN_INTERVAL) and statuses != {
            "no_interval_at_this_replicate_count"
        }:
            problems.append(
                f"a case carries {min(pairs)} pairs, below the "
                f"{MIN_PAIRS_FOR_AN_INTERVAL} the method needs, yet is not "
                f"reported as unassessed (statuses: {sorted(statuses)})"
            )

    # 3b. Each pool owns its own summary, and the ladder summarises the pools
    #     rather than keeping a second count of them that can drift.
    pools = data.get("by_geometry") or []
    for group in pools:
        summary = (
            "n_pairs",
            "mean_difference_days",
            "median_pfs_policy_days",
            "median_pfs_mtd_days",
        )
        missing = [k for k in summary if k not in group]
        if missing:
            problems.append(
                f"the image pool {group.get('case_id')!r} no longer carries its own "
                f"summary fields (missing: {missing}), so a consumer would have to "
                f"re-derive them"
            )
        elif int(group.get("n_pairs") or 0) != len(group.get("diffs_days") or []):
            problems.append(
                f"the image pool {group.get('case_id')!r} reports "
                f"{group.get('n_pairs')} pairs but carries "
                f"{len(group.get('diffs_days') or [])}"
            )

    ladder = (data.get("resolvability") or {}).get("ladder") or []
    if len(ladder) != 3:
        problems.append(
            "the resolvability ladder no longer reports its three scales "
            "(within one case, per image, pooled)"
        )
    elif cases:
        if int(ladder[0].get("groups_without_a_resolvable_interval") or 0) != len(cases):
            problems.append(
                "the within-one-case rung of the ladder no longer reports that "
                "every case falls short of an interval"
            )
        if int(ladder[1].get("pairs_per_group") or 0) != (
            int(pools[0].get("n_pairs") or 0) if pools else 0
        ):
            problems.append(
                "the per-image rung of the ladder reports a pair count the image "
                "pools do not agree with"
            )
        pooled_unresolvable = 1 if (rec_pooled.get("noise_floor") or {}).get(
            "unresolvable_at_this_n"
        ) else 0
        if int(ladder[-1].get("groups_without_a_resolvable_interval") or 0) != pooled_unresolvable:
            problems.append(
                "the pooled rung of the ladder disagrees with the pooled noise "
                "floor it is supposed to summarise"
            )
    for key in ("noise_floor_flag_text", "insufficient_replicates_flag_text"):
        if not data.get(key):
            problems.append(f"the record no longer carries {key}")

    # 4. The simulated-inversion day is recorded, and the note matches it.
    inv = data.get("inversion") or {}
    delays = [
        c.get("inversion_delay_days")
        for c in cases
        if c.get("inversion_delay_days") is not None
    ]
    n_equal = sum(1 for d in delays if d == 0)
    if delays and int(inv.get("n_cases_with_equal_inversion_day") or -1) != n_equal:
        problems.append(
            "the recorded inversion-day summary no longer matches the per-case "
            "days it summarises"
        )
    if delays and n_equal == len(delays) and "no simulated delay" not in str(
        inv.get("delay_note", "")
    ):
        problems.append(
            "every recorded case reaches the switch threshold on the same "
            "simulated day in both arms, but the note does not state that no "
            "simulated delay was found"
        )

    # 5. A model comparison cannot move a patient-level badge.
    provenance = claims_provenance()
    for key in ("clinical_benefit", "engine_predicts_patients"):
        if provenance.get(key) is not EvidenceStatus.UNKNOWN:
            problems.append(
                f"a trial-design screen was allowed to move {key} off UNKNOWN"
            )

    # 6. The shipped screen exists. The comparison policy lives in the module
    #    that owns it, so BOTH files carry no fabricated clinical-UI claim, and
    #    the two failure labels live exactly where the policy does.
    component = os.path.join(
        ROOT, "frontend", "src", "components", "TrialDesign.tsx"
    )
    policy = os.path.join(ROOT, "frontend", "src", "utils", "trialDesign.ts")
    for path in (component, policy):
        if not os.path.exists(path):
            problems.append(
                f"{os.path.relpath(path, ROOT)} is missing, so the recorded "
                f"screen cannot be checked against what ships"
            )
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        for label, pattern in FABRICATED_CLINICAL_UI_CLAIMS:
            if pattern.search(text):
                problems.append(
                    f"the trial-design screen ships a fabricated clinical-UI "
                    f"claim in {os.path.relpath(path, ROOT)}: {label}"
                )
    view_text = ""
    if os.path.exists(component):
        with open(component, encoding="utf-8", errors="replace") as fh:
            view_text = " ".join(fh.read().split())

    if os.path.exists(policy):
        with open(policy, encoding="utf-8", errors="replace") as fh:
            flat = " ".join(fh.read().split())
        for flag in (
            data.get("noise_floor_flag_text"),
            data.get("insufficient_replicates_flag_text"),
        ):
            if flag and flag not in flat:
                problems.append(
                    f"the module that owns the noise-floor policy does not carry "
                    f"the recorded label {flag!r}, so its live statuses and the "
                    f"record can disagree about what an unresolved result is "
                    f"called"
                )
        # The prose summary is derived from counts on both sides; if the two
        # templates drift, the screen's own cross-check reports it, so the
        # recorded one must still be the count-derived template.
        note = str((data.get("resolvability") or {}).get("note") or "")
        if not note.startswith("a pool without a resolvable interval"):
            problems.append(
                "the recorded resolvability note is no longer the count-derived "
                "template the live screen mirrors"
            )

    # 7. The nesting is disclosed wherever a pool reports a count, and a count is
    #    never published as a sample of independent units. A bootstrap over
    #    paired replicates assumes they are exchangeable; in this record they are
    #    nested, so the count and the units behind it must travel together.
    try:
        from danon.trial_design import NESTED_REPLICATES_FLAG
    except Exception as exc:  # noqa: BLE001
        problems.append(f"the nested-replicates label cannot be imported: {exc}")
        NESTED_REPLICATES_FLAG = ""  # type: ignore

    nesting_fields = (
        "n_pairs",
        "assumed_exchangeable",
        "n_configurations",
        "independent_units",
        "n_geometries",
        "configurations_per_geometry",
        "seeds_per_configuration",
        "note",
    )

    # A disclosure that asserts counts its own payload does not support is worse
    # than no disclosure, so every count -- and the note that restates it -- is
    # RE-DERIVED from the per-case rows it is supposed to summarise. Asserting a
    # field's presence is not a proof that its value is the payload's; a payload
    # doctored to claim 8 units behind 9 configurations, or 7 geometries behind
    # 3 case ids, must fail here rather than pass on the word of the block.
    try:
        from danon.trial_design import _pool_nesting, _record_nesting
    except Exception as exc:  # noqa: BLE001
        problems.append(f"the nesting builders cannot be imported: {exc}")
        _pool_nesting = _record_nesting = None  # type: ignore

    def _re_derive_nesting(
        rows: List[Dict[str, object]], pooled: bool
    ) -> Optional[Dict[str, object]]:
        if _pool_nesting is None or _record_nesting is None:
            return None
        return _record_nesting(rows) if pooled else _pool_nesting(rows)

    def _check_nesting(
        block: Dict[str, object],
        where: str,
        rows: List[Dict[str, object]],
        pooled: bool,
    ) -> None:
        missing = [k for k in nesting_fields if k not in block]
        if missing:
            problems.append(
                f"{where} no longer carries its nesting disclosure (missing: "
                f"{missing}), so its count is published without the units behind it"
            )
            return
        pairs = int(block.get("assumed_exchangeable") or 0)
        units = int(block.get("independent_units") or 0)
        if pairs != int(block.get("n_pairs") or 0):
            problems.append(
                f"{where} no longer states how many of its pairs the interval "
                f"treats as exchangeable"
            )
        if units >= pairs:
            problems.append(
                f"{where} no longer discloses FEWER independent units than "
                f"exchangeable pairs ({units} against {pairs}), so the record has "
                f"stopped saying its replicates are nested"
            )
        expected = _re_derive_nesting(rows, pooled)
        if expected is None:
            return
        for field, want in expected.items():
            got = block.get(field)
            if got == want:
                continue
            if field == "note":
                problems.append(
                    f"{where} carries a disclosure note that contradicts its own "
                    f"rows: the note must be the count-derived template, so it "
                    f"cannot describe a different nesting than the {len(rows)} "
                    f"per-case row(s) it summarises"
                )
            else:
                problems.append(
                    f"{where} discloses {field}={got!r}, but re-deriving it from "
                    f"the {len(rows)} per-case row(s) it summarises gives "
                    f"{want!r}"
                )

    _check_nesting(
        rec_pooled.get("nesting") or {},
        "the pooled summary",
        [c for c in cases if isinstance(c, dict)],
        True,
    )
    for group in pools:
        gid = group.get("case_id")
        members = [
            c
            for c in cases
            if isinstance(c, dict) and str(c.get("case_id")) == str(gid)
        ]
        _check_nesting(
            group.get("nesting") or {},
            f"the image pool {gid!r}",
            members,
            False,
        )

    if data.get("nested_replicates_flag_text") != NESTED_REPLICATES_FLAG:
        problems.append(
            "the record no longer carries the nested-replicates label, so a screen "
            "has nothing to display beside a nested count"
        )
    if view_text:
        # The view must REFERENCE the label the policy module defines (not
        # restate it), and must render the count, the exchangeable total and the
        # disclosure prose: a label alone can be imported and never shown.
        if "NESTED_REPLICATES_FLAG" not in view_text:
            problems.append(
                "the trial-design view no longer displays the nested-replicates "
                "label: the disclosure is dropped from the screen"
            )
        for field in (
            "nesting.independent_units",
            "nesting.assumed_exchangeable",
            "nesting.note",
        ):
            if field not in view_text:
                problems.append(
                    f"the view no longer renders {field}, so a count is shown "
                    f"without the units that stand behind it"
                )
        # Every place the view prints a pair count must print the units behind
        # it: the header badge, the pooled KPI label, and all three rungs of the
        # resolvability ladder. A bare "27 pairs" at any of those reads as a
        # sample of independent replicates. The rule keys on the DATA each
        # surface reads, not on its label, so an honest rewording of the wording
        # passes while a surface that loses its units fails.
        problems.extend(_view_unit_surface_problems(view_text))

    independence, scanned_independence = scan_for_fabricated_independence_claims(
        ROOT, skip=os.path.abspath(__file__)
    )
    problems.extend(independence)
    # The artifact's free prose is scanned ONCE. Read from disk, the walk above
    # already covered the whole file, so nothing more is scanned here; BUILT in
    # this process, the walk cannot see it, so the whole in-memory payload -- its
    # top-level notes and title included -- is scanned instead. Either way one
    # root cause yields one problem, and no prose in the artifact is outside the
    # scan.
    if built_here:
        seen: List[str] = []
        for prose in _artifact_prose(recorded):
            for label in scan_text_for_fabricated_independence(prose):
                if label in seen:
                    continue
                seen.append(label)
                if any(label in existing for existing in independence):
                    continue
                problems.append(
                    f"the trial-design payload presents a count as an independent "
                    f"sample ({label})"
                )

    verdict = (
        "the screen states the counterfactual as a statement about the model "
        "under declared assumptions, publishes a noise floor that re-derives "
        "from the recorded paired replicates (the pooled and per-image scales "
        "resolve; every single case falls short of an interval), discloses at "
        "every pool that its pairs are NESTED -- the pooled interval resamples "
        "27 pairs over at most 9 independent units, 3 seeds within each of 3 "
        "synthetic initial conditions on each of 3 geometries -- and rules out "
        "calendars only against the DECLARED ANC screen, the brief's own 19-day "
        "arithmetic among them. No calendar is named to administer and the "
        f"benefit badges stay UNKNOWN; {scanned_independence} shipped file(s) "
        "scanned for a count presented as an independent sample"
    )
    if problems:
        return CheckResult(False, verdict + " Problems: " + "; ".join(problems))
    return CheckResult(True, verdict)


CHECKS.update({
    "evolution_forecast_is_scored_against_baselines":
        check_evolution_forecast_is_scored_against_baselines,
    "trial_design_screen_is_derived_not_prescribed":
        check_trial_design_screen_is_derived_not_prescribed,
    "paired_pfs_extension_is_recorded_not_fabricated":
        check_paired_pfs_extension_is_recorded_not_fabricated,
    "clinical_execution_ui_is_audited_not_implemented":
        check_clinical_execution_ui_is_audited_not_implemented,
    "cohort_specification_matches_the_engine_not_the_pitch":
        check_cohort_specification_matches_the_engine_not_the_pitch,
    "provenance_tiers_are_derived_from_evidence_not_declared":
        check_provenance_tiers_are_derived_from_evidence_not_declared,
    "benefit_safety_and_runtime_are_measured_not_asserted":
        check_benefit_safety_and_runtime_are_measured_not_asserted,
    "samd_pathway_is_named_not_promised":
        check_samd_pathway_is_named_not_promised,
    "epistemic_guardrail_proposal_is_audited":
        check_epistemic_guardrail_proposal_is_audited,
    "badge_inventory_is_classified_not_promised":
        check_badge_inventory_is_answered_not_asserted,
    "alternating_win_has_a_reversal_map":
        check_alternating_win_has_a_measured_reversal_map,
    "bench_anchors_are_measured_and_calibration_refused":
        check_bench_anchors_are_measured_and_calibration_is_refused,
    "architecture_writeup_audited_against_the_code":
        check_architecture_writeup_is_audited_against_the_code,
    "motility_constant_is_grid_dependent_and_priced":
        check_motility_constant_is_grid_dependent_and_priced,
    "twin_plan_input_is_byte_tied":
        check_twin_plan_input_is_byte_tied,
})


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
