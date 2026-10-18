"""Audit an architecture or pitch write-up against the code it describes.

WHY THIS EXISTS
---------------
`danon/claims.py` was written because a pitch drifts from its code, and always in
the same direction. Its docstring lists three drifts found in the ISEF pitch
("provenance registry (ESTABLISHED, DERIVED, PROVEN_FAILED)", "deterministic math
and physics, not a black box", "528 tests"). Those were caught by a human reading
the pitch beside the repository. This module mechanises that reading for the
document class most likely to drift next: a technical architecture description
written by someone who did not implement the engine.

WHAT IT DOES
------------
It holds a declared set of rules, each naming a statement a write-up commonly
makes, the literal phrases that signal the statement, the code that confirms or
refutes it, and the corrected sentence. It then reports, per rule:

  * ``PRESENT`` -- the document makes the statement (the rule's verdict applies:
    ``REFUTED``, ``IMPRECISE`` or ``QUALIFIED``), or
  * ``ABSENT``  -- the statement is gone, which is either a fix or an omission.

A second rule kind covers material OMISSIONS: mechanisms the engine has that a
complete description must mention. ``OMITTED`` means the document is silent about
something that changes how its own numbers read.

WHAT MAKES IT MORE THAN A GREP
------------------------------
``code_fact_snapshot()`` re-derives, from the live modules, every fact the
corrections rest on -- the mutation draw's distribution, which policy actually
thresholds on resistance, the absence of an additive recovery term in the
marrow model, the endpoint's definition, the evidence vocabulary's size. The
audit artifact stores that snapshot, and `danon/claims.py` refuses unless a fresh
snapshot still equals it. So the corrections cannot go stale silently: change the
engine and the audit fails until it is re-run, exactly like the README's claim
count.

WHAT IT CANNOT DO
-----------------
It checks statements against the engine, not the engine against biology. A
statement this module marks as confirmed is confirmed *as a description of the
code*, which says nothing about whether the model is right. And its rule set is
declared and finite: a novel drift that no rule names is not reported.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple  # noqa: F401

from danon.cure import EvidenceStatus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The document this module was built to audit. Declared here rather than passed
#: around so the audit and its registry check cannot disagree about the subject.
AUDITED_DOCUMENT = "ARCHITECTURE_EVOTWIN_10X.md"

DRIFT = "drift"
OMISSION = "omission"


@dataclass(frozen=True)
class DriftRule:
    """One statement class a write-up tends to get wrong, bound to the code."""

    id: str
    kind: str
    #: Case-insensitive literal phrases; any match means the document makes the
    #: statement (drift rule) or covers the mechanism (omission rule).
    needles: Tuple[str, ...]
    #: For drift rules: how the code judges the statement.
    verdict: str = ""
    #: What the document claims, and what the code supports instead.
    claim: str = ""
    correction: str = ""
    #: `path::symbol` anchors a reader can open.
    evidence: Tuple[str, ...] = ()


#: The declared rule set. Deliberately finite and explicit: a rule is an
#: allegation with a citation, and the citation is what makes it checkable.
DRIFT_RULES: Tuple[DriftRule, ...] = (
    DriftRule(
        id="deterministic_engine",
        kind=DRIFT,
        needles=("deterministic, physics-constrained",),
        verdict="REFUTED",
        claim="the engine is a deterministic, physics-constrained simulator",
        correction=(
            "the engine is STOCHASTIC by construction, which is why every "
            "headline number is a 30-seed median rather than a run: the S->R "
            "mutation is a Poisson draw and growth carries multiplicative "
            "noise. 'Deterministic math and physics, not a black box' is one of "
            "the three pitch drifts danon/claims.py was written to catch."
        ),
        evidence=(
            "danon/evosteer.py::TumorEvolutionSimulator.run (rng.poisson mutation draw)",
            "danon/evosteer.py::run(growth_noise=0.05)",
            "danon/claims.py::module docstring, third listed drift",
        ),
    ),
    DriftRule(
        id="adaptive_switches_on_resistance",
        kind=DRIFT,
        needles=("Monitors the global Resistant Fraction",),
        verdict="REFUTED",
        claim="AdaptiveDosePolicy monitors the resistant fraction and switches or holidays on a threshold",
        correction=(
            "AdaptiveDosePolicy is BURDEN-relative, not resistance-thresholded: "
            "it doses at maximum while total burden exceeds its own initial "
            "value and withdraws entirely when the tumour is pushed back to "
            "baseline. The resistance-threshold controller is a different "
            "policy, CollateralSwitchPolicy, with hysteresis "
            "(theta_on=0.15, theta_off=0.05). The write-up fuses two "
            "controllers into one description."
        ),
        evidence=(
            "danon/evosteer.py::AdaptiveDosePolicy.dose",
            "danon/evosteer.py::CollateralSwitchPolicy (theta_on/theta_off)",
        ),
    ),
    DriftRule(
        id="anc_has_additive_recovery_term",
        kind=DRIFT,
        needles=(r"+ \text{RecoveryRate}",),
        verdict="REFUTED",
        claim="ANC(t) = ANC0 - integral(Dose * lognormal) + RecoveryRate",
        correction=(
            "there is no additive recovery term. The screen is a marrow "
            "RESERVOIR: the dose series is convolved with a unit-area "
            "log-normal kernel, and ANC = baseline * max(0, 1 - "
            "(1 - floor/baseline) * S / budget). Recovery is the kernel's tail, "
            "not a separate rate. The budget itself is DEFINED by the label's "
            "own 150 mg/m2 x5d q28d regimen, and the kernel is fitted to one "
            "label datum -- the median day of neutrophil nadir -- which the "
            "model declares about itself."
        ),
        evidence=(
            "danon/dosing.py::MyelosuppressionModel (budget, kernel, ANC from reservoir)",
            "danon/dosing.py::MyelosuppressionModel.kernel  (fitted_to: the label's nadir day)",
            "outputs_real/dosing.json::myelosuppression_model",
        ),
    ),
    DriftRule(
        id="mutation_is_multiplicative_noise",
        kind=DRIFT,
        needles=(r"\mu \cdot S \cdot (1 + \text{Noise})",),
        verdict="IMPRECISE",
        claim="Mutation = mu * S * (1 + Noise)",
        correction=(
            "the mutation term is a POISSON event count on realised growth, "
            "new_R = Poisson(grow_S * mu), where grow_S has already absorbed "
            "the noise factor and the competition room term. Poisson events "
            "with a mean, not a multiplicative perturbation of a rate, and the "
            "difference is visible at small populations."
        ),
        evidence=("danon/evosteer.py::run (new_R = rng.poisson(np.clip(grow_S, 0, None) * mu))",),
    ),
    DriftRule(
        id="segmentation_labels_are_carrying_capacity",
        kind=DRIFT,
        needles=("directly to the spatial carrying capacity",),
        verdict="IMPRECISE",
        claim="BraTS segmentation labels map directly to a spatial K(x)",
        correction=(
            "the labels become a PROLIFERATIVE field (after decimation and a "
            "bounding-box crop), not a carrying capacity: K is the scalar "
            "CloneParams.carrying_capacity. What the labels drive is where "
            "growth is suppressed and where tumour can sit, and the module "
            "reports its own provenance rather than pretending the mapping is "
            "a measurement."
        ),
        evidence=(
            "danon/realdata.py::_label_proliferation",
            "danon/realdata.py::prepare_brats_case",
            "danon/evosteer.py::CloneParams.carrying_capacity",
        ),
    ),
    DriftRule(
        id="trial_in_seconds_unqualified",
        kind=DRIFT,
        needles=("to simulate a 365-day clinical trial in seconds",),
        verdict="QUALIFIED",
        claim="a 365-day clinical trial is simulated in seconds",
        correction=(
            "per engine RUN, yes, and measured: ~3.2 s for one 24^3 / 365-day "
            "seed single-threaded and ~0.88 s across the measured 4-worker "
            "candidate pool (3.63x). A 30-seed ARM is therefore ~26 s, and the "
            "release-1 alternating-vs-single stage -- 46 arms x 30 seeds -- "
            "took 5,388 s of wall clock. 'In seconds' without the seed count is "
            "the difference between one trajectory and a trial."
        ),
        evidence=(
            "outputs_real/dosing.json::wall_seconds",
            "outputs_real/release2_sweep.json::wall_seconds",
        ),
    ),
    DriftRule(
        id="claims_py_verifies_adaptive_beats_mtd",
        kind=DRIFT,
        needles=("It mathematically verifies that the Adaptive policy actually generated a higher",),
        verdict="REFUTED",
        claim="claims.py verifies that the Adaptive policy produced a higher PFS than MTD",
        correction=(
            "claims.py runs a declared registry of claims, each bound to a live "
            "check that reads the repository; it does not compare two policies' "
            "PFS. The policy comparison that exists is the competitive-release "
            "experiment (continuous MTD vs a label-shaped pulsed calendar), and "
            "release 1's single-drug search returned a NULL -- no pulsed "
            "calendar beat the best baseline. The measured win came from "
            "ALTERNATING two drugs, which is a different experiment again."
        ),
        evidence=(
            "danon/claims.py::audit_claims",
            "danon/dosing.py::run_competitive_release_experiment",
            "outputs_real/dosing.json::search.notes (the null result)",
        ),
    ),
    DriftRule(
        id="binary_evidence_flags",
        kind=DRIFT,
        needles=("it outputs the `ESTABLISHED` flag for the UI",),
        verdict="REFUTED",
        claim="the UI receives ESTABLISHED or PROVEN_FAILED",
        correction=(
            "the evidence vocabulary has five members -- ESTABLISHED, "
            "EMERGING_HUMAN, ANIMAL_ONLY, PROVEN_FAILED, UNKNOWN -- and the app "
            "artifact carries a per-claim row with its own verdict and detail, "
            "so a failing check is a visible row rather than a dropped one. A "
            "two-state description cannot represent 'measured in animals, not "
            "in humans', which is where most of this repository's honesty lives."
        ),
        evidence=(
            "danon/cure.py::EvidenceStatus",
            "danon/app_artifacts.py::build_claims_artifact",
            "frontend/public/artifacts/claims.json",
        ),
    ),
    DriftRule(
        id="omits_immune_layer",
        kind=OMISSION,
        needles=("ImmuneParams", "T-cell", "checkpoint"),
        verdict="OMISSION",
        claim="a complete description would name the immune clearance layer",
        correction=(
            "the engine carries an opt-in resistance-blind, density-gated "
            "immune kill (ImmuneParams.kill_max, neoantigen_factor), defaulting "
            "to None in both run and run_batch -- i.e. OFF in every number the "
            "write-up quotes. danon/evotwin.py is where it is switched on, and "
            "the module's own docstring states the base model has no immune "
            "compartment."
        ),
        evidence=(
            "danon/evosteer.py::run(immune: Optional[ImmuneParams] = None)",
            "danon/evosteer.py::ImmuneParams",
            "danon/evotwin.py::immune trait wiring",
        ),
    ),
    DriftRule(
        id="omits_cell_motility",
        kind=OMISSION,
        needles=("motility", "migration"),
        verdict="OMISSION",
        claim="a complete description would mention cell migration",
        correction=(
            "the day step applies a slow cell-migration Laplacian to both "
            "clones with a coefficient named in physical units -- the same "
            "coefficient is 0.02 mm^2/day on the phantom and 0.27 mm^2/day on "
            "the real 3.67 mm geometry, against a literature motility range of "
            "0.001-0.005 mm^2/day. Omitting it hides a term that is out of "
            "range on the real-geometry path."
        ),
        evidence=(
            "danon/evosteer.py::run (motility Laplacian on N_S and N_R)",
            "danon/anisotropy.py::LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY",
        ),
    ),
    DriftRule(
        id="omits_extinction_and_detection_floors",
        kind=OMISSION,
        needles=("extinction", "MIN_VIABLE", "detection floor", "detectable"),
        verdict="OMISSION",
        claim="a complete description would name the extinction and detection floors",
        correction=(
            "two floors decide the reported numbers: a sub-half-cell voxel is "
            "set to zero (continuous densities have no extinction and every "
            "nonzero population regrows), and the resistant fraction is gated "
            "on a detection burden so that an ERADICATED tumour cannot read as "
            "100% resistant and trigger a policy switch. The second is why the "
            "engine's own claim text says a dominance day is a model event."
        ),
        evidence=(
            "danon/evosteer.py::MIN_VIABLE_DENSITY (extinction floor)",
            "danon/evosteer.py::DETECTION_BURDEN_DENSITY (R-fraction gating)",
        ),
    ),
    DriftRule(
        id="omits_partial_cross_resistance",
        kind=OMISSION,
        needles=("cross-resistance", "drug_sensitivity_scale_R", "partial resistance"),
        verdict="OMISSION",
        claim="a complete description would state that resistance is partial",
        correction=(
            "resistance is not binary: drug A's kill of the resistant clone is "
            "scaled by CloneParams.drug_sensitivity_scale_R = 0.25, so A still "
            "prunes R in well-perfused regions. That single scale is what makes "
            "vasculature geometry, rather than the clone label, decide where R "
            "survives -- and it is invisible in a write-up that quotes only "
            "kill_max_S and kill_max_R."
        ),
        evidence=(
            "danon/evosteer.py::CloneParams.drug_sensitivity_scale_R",
            "danon/evosteer.py::run (kill_R_A scaling)",
        ),
    ),
    DriftRule(
        id="omits_endpoint_definition",
        kind=OMISSION,
        needles=("RANO", "1.25x", "+25%", "progression fraction"),
        verdict="OMISSION",
        claim="a complete description would define how progression is decided",
        correction=(
            "'PFS' is a BURDEN-RATIO endpoint: progression is the first day "
            "total burden exceeds (1 + RANO_PROGRESSION_FRACTION) * initial, "
            "with RANO_PROGRESSION_FRACTION = 0.25, censored at the horizon. It "
            "is neither an imaging endpoint nor a survival endpoint, and the "
            "qualifier 'RANO' in the constant's name is a nod to a clinical "
            "growth threshold, not a claim to measure RANO."
        ),
        evidence=(
            "danon/evosteer.py::RANO_PROGRESSION_FRACTION",
            "danon/evosteer.py::run (pfs_threshold)",
            "danon/evosteer.py::SimulationTrajectory.pfs_value (censored)",
        ),
    ),
)


# ---------------------------------------------------------------------------
# The live code-side facts the corrections rest on
# ---------------------------------------------------------------------------


def _read_source(relpath: str) -> str:
    with open(os.path.join(ROOT, relpath), encoding="utf-8") as fh:
        return fh.read()


def _source_contains(relpath: str, needle: str) -> bool:
    return needle in _read_source(relpath)


def _raises(fn: Callable[[], Any]) -> bool:
    try:
        fn()
    except Exception:  # noqa: BLE001 -- any refusal counts as a refusal
        return True
    return False


def _jsonable(value: Any) -> Any:
    """Normalise through JSON so a stored snapshot compares equal on reload."""
    return json.loads(json.dumps(value))


def code_fact_snapshot() -> Dict[str, Any]:
    """Re-derive, from the live modules, every fact the corrections rest on.

    Each key is a sentence in a correction turned into something executable. If
    an engine change makes a correction false, the snapshot moves and the
    registry check fails until the audit is re-run.
    """
    from danon import dosing as D
    from danon import evosteer as E

    clones = dataclasses.asdict(E.DEFAULT_CLONES)
    drugs = dataclasses.asdict(E.DEFAULT_DRUGS)
    myelo = D.MyelosuppressionModel()
    myelo_fields = {f.name for f in dataclasses.fields(D.MyelosuppressionModel)}
    run_sig = inspect.signature(E.TumorEvolutionSimulator.run)
    batch_sig = inspect.signature(E.TumorEvolutionSimulator.run_batch)

    adaptive = E.AdaptiveDosePolicy()
    collateral = E.CollateralSwitchPolicy()

    def state(burden: float, burden_initial: float, r_frac: float) -> Any:
        return E.PolicyState(
            day=1, burden=burden, burden_initial=burden_initial,
            resistant_fraction=r_frac,
        )

    facts: Dict[str, Any] = {
        # --- mutation is a Poisson event count, not a noise multiply ---------
        "mutation_rate": clones["mutation_rate"],
        "mutation_draw_is_poisson_on_realized_growth": _source_contains(
            "danon/evosteer.py", "new_R = rng.poisson(np.clip(grow_S"
        ),
        # --- the two policies are different controllers ---------------------
        "adaptive_dose_at_baseline_burden": list(adaptive.dose(state(10.0, 10.0, 0.9))),
        "adaptive_dose_above_baseline_burden": list(adaptive.dose(state(11.0, 10.0, 0.01))),
        "collateral_switch_theta_on": collateral.theta_on,
        "collateral_switch_theta_off": collateral.theta_off,
        "collateral_switch_dose_at_high_resistance": list(
            collateral.dose(state(1.0, 1.0, 0.5))
        ),
        "collateral_switch_rejects_inverted_band": _raises(
            lambda: E.CollateralSwitchPolicy(theta_on=0.05, theta_off=0.15)
        ),
        # --- the marrow screen is a reservoir with no additive recovery -----
        "myelo_has_additive_recovery_field": any("recovery" in f for f in myelo_fields),
        "myelo_fields": sorted(myelo_fields),
        "myelo_budget_mg_m2": myelo.suppression_budget(),
        # The label anchors the screen to the label's OWN regimen: this is the
        # fact behind the correction's claim that the budget is defined rather
        # than chosen.
        "myelo_anchor_checks": _jsonable(myelo.anchor_checks(365)),
        "myelo_kernel_log_mu": myelo.kernel_log_mu,
        "myelo_kernel_log_sd": myelo.kernel_log_sd,
        "myelo_kernel_fitted_to_one_label_datum": (
            "median day of neutrophil nadir" in (D.MyelosuppressionModel.__doc__ or "")
        ),
        "myelo_status": myelo.to_dict().get("status"),
        "myelo_measured_in_this_repository": myelo.to_dict().get(
            "measured_in_this_repository"
        ),
        "myelo_anc_floor": myelo.anc_floor,
        # --- the endpoint is a burden ratio ---------------------------------
        "rano_progression_fraction": E.RANO_PROGRESSION_FRACTION,
        "pfs_threshold_is_burden_ratio": _source_contains(
            "danon/evosteer.py", "pfs_threshold = (1.0 + RANO_PROGRESSION_FRACTION) * b_init"
        ),
        # --- stochasticity and the opt-in immune layer ----------------------
        "growth_noise_default": run_sig.parameters["growth_noise"].default,
        "batch_growth_noise_default": batch_sig.parameters["growth_noise"].default,
        "immune_off_by_default": run_sig.parameters["immune"].default is None,
        "immune_off_by_default_in_batch": batch_sig.parameters["immune"].default is None,
        "source_states_no_immune_compartment": "No immune compartment"
        in _read_source("danon/evosteer.py"),
        # --- floors, motility and partial resistance ------------------------
        "extinction_floor_present": _source_contains("danon/evosteer.py", "MIN_VIABLE_DENSITY"),
        "detection_gating_present": _source_contains(
            "danon/evosteer.py", "if (burden > 0.0 and detectable) else 0.0"
        ),
        "motility_term_present": _source_contains("danon/evosteer.py", "motility * _laplacian"),
        "drug_sensitivity_scale_R": clones["drug_sensitivity_scale_R"],
        # --- the evidence vocabulary is not binary --------------------------
        "evidence_status_members": sorted(m.name for m in EvidenceStatus),
        "evidence_status_member_count": len(list(EvidenceStatus)),
        "has_derived_member": "DERIVED" in {m.name for m in EvidenceStatus},
        # --- the registry the write-up misdescribes -------------------------
        "default_clones": clones,
        "default_drugs": drugs,
    }

    # Read lazily: this module is imported BY claims.py, so the registry import
    # stays inside the function. The snapshot records the registry's STRUCTURE,
    # not its size: adding a claim does not change any correction here, and a
    # fact that churns for unrelated reasons would train readers to re-run the
    # audit without reading it.
    from danon.claims import CHECKS, CLAIMS

    facts["every_claim_binds_a_declared_check"] = all(
        c.check in CHECKS for c in CLAIMS
    )
    facts["registry_is_not_empty"] = len(CLAIMS) > 0
    return _jsonable(facts)


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    rule_id: str
    kind: str
    status: str          # PRESENT | ABSENT | OMITTED | COVERED
    verdict: str         # REFUTED | IMPRECISE | QUALIFIED | OMISSION | ""
    matched: Optional[str]
    snippet: str
    claim: str
    correction: str
    evidence: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ArchitectureAudit:
    path: str
    sha256: str
    n_words: int
    findings: Tuple[Finding, ...]
    facts: Dict[str, Any] = field(default_factory=dict)

    @property
    def present(self) -> List[Finding]:
        return [f for f in self.findings if f.status == "PRESENT"]

    @property
    def omitted(self) -> List[Finding]:
        return [f for f in self.findings if f.status == "OMITTED"]

    def describe(self) -> str:
        return (
            f"{self.path}: {len(self.present)} statement(s) bound to the code "
            f"({', '.join(sorted({f.verdict for f in self.present})) or 'none'}), "
            f"{len(self.omitted)} material omission(s) named, "
            f"{sum(1 for f in self.findings if f.status in ('ABSENT', 'COVERED'))} "
            f"rule(s) not triggered"
        )

    def to_dict(self) -> Dict[str, Any]:
        # Normalised through JSON so this dict equals the one reloaded from the
        # artifact: `dataclasses.asdict` keeps tuples, `json.dump` turns them into
        # lists, and a comparison across the two would report a difference that
        # is an artefact of serialisation rather than a changed finding.
        return _jsonable({
            "path": self.path,
            "sha256": self.sha256,
            "n_words": self.n_words,
            "findings": [f.to_dict() for f in self.findings],
            "facts": self.facts,
            "counts": {
                "present": len(self.present),
                "omitted": len(self.omitted),
                "rules": len(self.findings),
                "refuted": sum(1 for f in self.findings if f.verdict == "REFUTED"),
                "imprecise": sum(1 for f in self.findings if f.verdict == "IMPRECISE"),
                "qualified": sum(1 for f in self.findings if f.verdict == "QUALIFIED"),
            },
            "what_this_cannot_show": what_the_audit_cannot_show(),
        })


def _find_snippet(text: str, needle: str) -> str:
    """The sentence around a match, so a finding is readable without the file."""
    low = text.lower()
    idx = low.find(needle.lower())
    if idx < 0:
        return ""
    start = max(text.rfind("\n", 0, idx), text.rfind(". ", 0, idx)) + 1
    end = text.find("\n", idx)
    return re.sub(r"\s+", " ", text[start:end if end > 0 else idx + 200]).strip()[:400]


def audit_text(text: str, path: str = "<text>") -> ArchitectureAudit:
    findings: List[Finding] = []
    for rule in DRIFT_RULES:
        matched = next(
            (n for n in rule.needles if n.lower() in text.lower()), None
        )
        if rule.kind == DRIFT:
            status = "PRESENT" if matched else "ABSENT"
            verdict = rule.verdict if matched else ""
        else:
            status = "COVERED" if matched else "OMITTED"
            verdict = "" if matched else rule.verdict
        findings.append(Finding(
            rule_id=rule.id,
            kind=rule.kind,
            status=status,
            verdict=verdict,
            matched=matched,
            snippet=_find_snippet(text, matched) if matched else "",
            claim=rule.claim,
            correction=rule.correction,
            evidence=rule.evidence,
        ))
    return ArchitectureAudit(
        path=path,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        n_words=len(text.split()),
        findings=tuple(findings),
        facts=code_fact_snapshot(),
    )


def audit_document(path: str = AUDITED_DOCUMENT) -> ArchitectureAudit:
    """Audit a document on disk, relative to the repository root."""
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    with open(full, encoding="utf-8") as fh:
        text = fh.read()
    return audit_text(text, path=os.path.relpath(full, ROOT).replace(os.sep, "/"))


#: Where each rule applies in the audited write-up, so the corrected reading
#: copy can be ordered the way the document reads instead of the way the rule
#: list happens to be written.
RULE_SECTIONS: Dict[str, str] = {
    "deterministic_engine": "Closing summary",
    "adaptive_switches_on_resistance": "Layer 5 -- simulation loop & policies",
    "claims_py_verifies_adaptive_beats_mtd": "The Audit Pipeline",
    "binary_evidence_flags": "The Audit Pipeline",
    "anc_has_additive_recovery_term": "Layer 4 -- safety screen",
    "mutation_is_multiplicative_noise": "Layer 2 -- core mathematics",
    "segmentation_labels_are_carrying_capacity": "Layer 1 -- geometry",
    "trial_in_seconds_unqualified": "Preamble",
    "omits_immune_layer": "Omissions",
    "omits_cell_motility": "Omissions",
    "omits_extinction_and_detection_floors": "Omissions",
    "omits_partial_cross_resistance": "Omissions",
    "omits_endpoint_definition": "Omissions",
}

SECTION_ORDER: Tuple[str, ...] = (
    "Preamble",
    "Layer 1 -- geometry",
    "Layer 2 -- core mathematics",
    "Layer 3 -- drug delivery",
    "Layer 4 -- safety screen",
    "Layer 5 -- simulation loop & policies",
    "The Audit Pipeline",
    "Closing summary",
    "Omissions",
)


def render_corrected_copy(audit: ArchitectureAudit) -> str:
    """The corrected READING COPY of the audited write-up.

    The submitted document is preserved unedited; this is the version a reviewer
    or a sponsor should read, assembled from the audit so the two cannot
    disagree. Every correction is a sentence about this repository's own source,
    and every claim carries the anchors that let a reader open it.
    """
    flagged = [f for f in audit.findings if f.status in ("PRESENT", "OMITTED")]
    by_section: Dict[str, List[Finding]] = {}
    for f in flagged:
        by_section.setdefault(RULE_SECTIONS.get(f.rule_id, "Other"), []).append(f)

    lines: List[str] = [
        "# EvoTwin-10X architecture -- CORRECTED READING COPY (generated)",
        "",
        "Do not edit this file. It is generated by `_run_architecture_audit.py` "
        "from `outputs_real/architecture_audit.json`, so it cannot drift from "
        "the audit it reports.",
        "",
        f"- audited document: `{audit.path}`",
        f"- audited-document-sha256: `{audit.sha256}`",
        f"- rules triggered: {len(flagged)} of {len(audit.findings)} declared",
        "",
        "The submitted write-up is preserved UNEDITED in the file named above; "
        "this copy restates each flagged statement beside the code that judges "
        "it. Nothing here is a new measurement -- every correction is a "
        "sentence about this repository's own source, and every verdict is "
        "document-versus-code, which says nothing about whether the engine is "
        "right about biology.",
        "",
        "| section | rule | status | verdict |",
        "|---|---|---|---|",
    ]
    for section in SECTION_ORDER:
        for f in by_section.get(section, []):
            lines.append(
                f"| {section} | `{f.rule_id}` | {f.status} | {f.verdict or '-'} |"
            )
    lines.append("")

    for section in SECTION_ORDER:
        findings = by_section.get(section)
        if not findings:
            continue
        lines.append(f"## {section}")
        lines.append("")
        for f in findings:
            badge = f.verdict or "OMISSION"
            lines.append(f"### `{f.rule_id}` -- {badge} ({f.status})")
            lines.append("")
            if f.status == "PRESENT":
                lines.append(f"**As submitted:** {f.snippet}")
            else:
                lines.append(
                    "**As submitted:** not mentioned, and its absence changes how "
                    "the write-up's own numbers read."
                )
                lines.append("")
                lines.append(f"**What the write-up says instead:** {f.claim}")
            lines.append("")
            lines.append(f"**Corrected statement:** {f.correction}")
            lines.append("")
            lines.append("**Open in the repository:**")
            for anchor in f.evidence:
                lines.append(f"- `{anchor}`")
            lines.append("")

    facts = audit.facts
    lines += [
        "## Appendix: the facts the corrections were re-derived from",
        "",
        "Re-derived live at audit time, and compared against the stored snapshot "
        "by the registry check -- so a change to any of these fails the audit "
        "until it is re-run.",
        "",
        "```json",
        json.dumps(facts, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def render_corrections(audit: ArchitectureAudit) -> str:
    """The corrected version of every statement the audit flagged."""
    lines: List[str] = []
    for f in audit.findings:
        # Only flagged findings get a correction: PRESENT (the document makes a
        # statement the code judges) and OMITTED (the document is silent about a
        # mechanism that changes how its own numbers read).
        if f.status not in ("PRESENT", "OMITTED"):
            continue
        head = f"[{f.verdict or 'OMISSION'}] {f.rule_id} ({f.status})"
        lines.append(head)
        if f.snippet:
            lines.append(f"  document: {f.snippet}")
        lines.append(f"  code    : {f.correction}")
        for anchor in f.evidence:
            lines.append(f"  evidence: {anchor}")
        lines.append("")
    return "\n".join(lines)


def pitch_audit_provenance() -> Dict[str, EvidenceStatus]:
    """What each sort of output from this module is, in the repo's vocabulary."""
    return {
        "engine_is_stochastic": EvidenceStatus.ESTABLISHED,
        "adaptive_policy_is_burden_based": EvidenceStatus.ESTABLISHED,
        "anc_model_has_no_additive_recovery": EvidenceStatus.ESTABLISHED,
        "document_statement_verdicts": EvidenceStatus.ESTABLISHED,
        "any_claim_about_biology_from_this_audit": EvidenceStatus.UNKNOWN,
    }


def what_the_audit_cannot_show() -> List[str]:
    """Limits that travel with every finding."""
    return [
        "The verdicts are about the DOCUMENT versus the CODE, not about the "
        "model versus biology. A statement marked confirmed is a correct "
        "description of this engine, which is silent on whether the engine is "
        "right.",
        "The rule set is declared and finite. A drift no rule names -- a wrong "
        "constant, an invented citation, a swapped sign -- is not reported, and "
        "absence of a finding is not evidence of accuracy.",
        "A rule is matched on literal phrases, so a reworded version of the "
        "same wrong statement can slip past it; the document digest recorded "
        "here is what makes a silent rewrite visible, not the matching itself.",
    ]
