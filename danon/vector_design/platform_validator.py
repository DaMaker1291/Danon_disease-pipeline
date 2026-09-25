"""
Pre-submission regulatory readiness rubric (self-assessment).

WHAT THIS IS
------------
A **declared rubric** that scores a candidate construct against eight readiness
dimensions, so a design team can see which evidence is missing before they speak
to a regulator. Each dimension consumes *named inputs* that must be supplied by
the caller; a dimension whose inputs are absent is reported as **UNSCORED**, and
the overall verdict is **UNKNOWN** rather than a pass.

WHAT THIS IS NOT
----------------
**This is not an MHRA determination, and the MHRA publishes no ordinal score
like the one below.** The Innovation Licensing Access Pathway (ILAP) is an
application-gated process: MHRA decides eligibility from the evidence package a
sponsor submits. There is no public point scale, so any number produced here is
a self-assessment against *our own* weights -- useful for internal triage,
meaningless as a regulatory status.

The previous version of this module inverted that in three ways, all removed:

1. ``evaluate_candidate({})`` returned a complete assessment. Every dimension
   fell back to a hard-coded default, so an empty dict produced a full scorecard
   and an ``is_ilap_eligible`` verdict. Missing evidence silently became average
   evidence. Now a missing input makes the dimension unscored; the gating
   dimensions must all be scored before a verdict exists at all.
2. ``UCL_REGULATORY_BASELINE`` presented eight invented 0-1 values as the
   "UCL/GOSH NCT03882437 baseline regulatory scores". No such published scores
   exist. The comparator is now derived by running *this same rubric* on the
   unmodified control configuration, which is an internally consistent
   comparison instead of a fabricated external one.
3. ``is_ilap_eligible`` was a bare ``bool``, so "no" and "cannot be assessed"
   were indistinguishable -- and the default-padded path made "yes" the common
   outcome. It is now tri-state: ``True`` / ``False`` / ``None``.

Provenance
----------
* dimensions, weights, thresholds: **ASSUMED** -- authored here, not published
  by any regulator. Treat them as a checklist ordering, not a calibrated scale.
* every dimension input: supplied by the caller; see ``DIMENSION_INPUTS``.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


#: Provenance tag for the rubric itself. The scores below are ASSUMED: they are
#: an authored weighting scheme, not a published regulatory instrument. See
#: :func:`danon.vector_design.biology.biological_provenance` for the vocabulary this mirrors.
RUBRIC_PROVENANCE = "ASSUMED"

#: Plain-language statement of what a result from this module does and does not
#: mean. Rendered verbatim in the summary so no reader has to infer it.
RUBRIC_CAVEAT = (
    "Self-assessment against an authored readiness rubric. The MHRA publishes no "
    "ordinal score for ILAP; eligibility is decided by MHRA from a submitted "
    "evidence package. This output is triage, not a regulatory status."
)


MHRA_ILAP_DIMENSIONS = {
    "product_quality": {
        "label": "Product Quality & Manufacturing Consistency",
        "weight": 0.15,
        "inputs": [
            "itr_integrity", "empty_full_ratio_optimal", "rc_aav_risk", "vector_titer"
        ],
    },
    "non_clinical_safety": {
        "label": "Non-Clinical Safety & Biodistribution",
        "weight": 0.15,
        "inputs": [
            "cardiac_tropism", "hepatic_accumulation",
            "gonadal_transduction_risk", "shedding_risk"
        ],
    },
    "clinical_safety": {
        "label": "Clinical Safety & Tolerability",
        "weight": 0.20,
        "inputs": [
            "complement_activation", "liver_toxicity", "cardiac_inflammation",
            "immune_evasion", "decoy_protection"
        ],
    },
    "clinical_efficacy": {
        "label": "Clinical Efficacy & Surrogate Endpoints",
        "weight": 0.20,
        "inputs": [
            "cardiac_tropism", "lamp2b_expression", "promoter_score",
            "mirna_score", "dosing_score"
        ],
    },
    "patient_stratification": {
        "label": "Patient Stratification & Biomarker Strategy",
        "weight": 0.10,
        "inputs": [
            "mutation_types_covered", "age_stratification",
            "nab_management", "biomarker_strategy"
        ],
    },
    "risk_management": {
        "label": "Risk Management & Mitigation Strategy",
        "weight": 0.08,
        "inputs": [
            "immunosuppression_plan", "complement_monitoring",
            "rescue_therapy", "stopping_rules"
        ],
    },
    "pediatric_extrapolation": {
        "label": "Pediatric Extrapolation & Juvenile Toxicity",
        "weight": 0.07,
        "inputs": [
            "juvenile_model_data", "growth_development_safety",
            "long_term_follow_up", "age_de_escalation"
        ],
    },
    "orphan_disease": {
        "label": "Orphan/Rare Disease & Unmet Medical Need",
        "weight": 0.05,
        "inputs": [
            "disease_prevalence", "unmet_medical_need",
            "natural_history_data", "patient_advocacy"
        ],
    },
}

#: The metric keys every dimension consumes. Derived from the table above so the
#: two cannot drift; callers can print it to see exactly what they must supply.
DIMENSION_INPUTS: Dict[str, List[str]] = {
    name: list(info["inputs"]) for name, info in MHRA_ILAP_DIMENSIONS.items()
}

#: Every distinct input key the rubric consumes.
REQUIRED_INPUTS: Tuple[str, ...] = tuple(
    dict.fromkeys(k for info in MHRA_ILAP_DIMENSIONS.values() for k in info["inputs"])
)

#: Dimensions that must be scored before an eligibility verdict can be formed.
#: Weights and thresholds are ASSUMED authored values (see module docstring).
GATING_DIMENSIONS: Tuple[str, ...] = (
    "product_quality", "clinical_safety", "clinical_efficacy",
)

MHRA_PASS_THRESHOLDS = {
    "product_quality": 0.60,
    "non_clinical_safety": 0.60,
    "clinical_safety": 0.65,
    "clinical_efficacy": 0.55,
    "patient_stratification": 0.50,
    "risk_management": 0.55,
    "pediatric_extrapolation": 0.45,
    "orphan_disease": 0.70,
}

#: Minimum mean score across scored dimensions for the self-assessment to be
#: called ready. ASSUMED, like the rest of the rubric.
COMPOSITE_READINESS_THRESHOLD = 0.60

#: The authored fallback values the pre-refactor code applied silently. They are
#: retained **only** so a caller can opt in explicitly via ``assumed_inputs``
#: and have the result labelled as default-padded. Nothing reads them by default.
LEGACY_DEFAULTS = {
    "itr_integrity": 0.9,
    "empty_full_ratio_optimal": 1.0,
    "rc_aav_risk": 0.02,
    "vector_titer": 1e14,
    "cardiac_tropism": 0.5,
    "hepatic_accumulation": 0.3,
    "gonadal_transduction_risk": 0.05,
    "shedding_risk": 0.10,
    "complement_activation": 0.3,
    "liver_toxicity": 0.2,
    "cardiac_inflammation": 0.15,
    "immune_evasion": 0.5,
    "decoy_protection": 0.4,
    "lamp2b_expression": 0.5,
    "promoter_score": 0.5,
    "mirna_score": 0.5,
    "dosing_score": 0.5,
    "mutation_types_covered": 0.85,
    "age_stratification": 0.80,
    "nab_management": 0.70,
    "biomarker_strategy": 0.75,
    "immunosuppression_plan": 0.65,
    "complement_monitoring": 0.70,
    "rescue_therapy": 0.60,
    "stopping_rules": 0.75,
    "juvenile_model_data": 0.60,
    "growth_development_safety": 0.55,
    "long_term_follow_up": 0.70,
    "age_de_escalation": 0.50,
    "disease_prevalence": 0.80,
    "unmet_medical_need": 0.90,
    "natural_history_data": 0.85,
    "patient_advocacy": 0.75,
}

#: Authored placeholder inputs for inputs the *code pipeline* does not measure --
#: clinical and programme evidence that exists only as study documentation (a
#: juvenile toxicology package, advocacy engagement, stopping rules, and so on).
#:
#: They are passed explicitly through ``assumed_inputs`` rather than hidden inside
#: the scorers, so every assessment that leans on them is tagged
#: ``used_assumed_inputs``, and they are identical across the design and control
#: arms, so they cancel out of a comparison and cannot manufacture an improvement
#: factor.
#:
#: Inputs the pipeline CAN derive (cardiac_tropism, hepatic_accumulation,
#: immune_evasion, lamp2b_expression, promoter_score, mirna_score, dosing_score)
#: are deliberately absent: a missing one leaves its dimension UNSCORED rather
#: than quietly padded, so the reported coverage stays meaningful.
#:
#: Provenance: ASSUMED
CLINICAL_ASSUMED_INPUTS = {
    "liver_toxicity": 0.15,
    "complement_activation": 0.20,
    "cardiac_inflammation": 0.12,
    "decoy_protection": 0.70,
    "vector_titer": 5e13,
    "itr_integrity": 0.95,
    "empty_full_ratio_optimal": 0.60,
    "rc_aav_risk": 0.01,
    "gonadal_transduction_risk": 0.03,
    "shedding_risk": 0.08,
    "mutation_types_covered": 0.85,
    "age_stratification": 0.80,
    "nab_management": 0.75,
    "biomarker_strategy": 0.75,
    "immunosuppression_plan": 0.70,
    "complement_monitoring": 0.75,
    "rescue_therapy": 0.65,
    "stopping_rules": 0.75,
    "juvenile_model_data": 0.65,
    "growth_development_safety": 0.60,
    "long_term_follow_up": 0.72,
    "age_de_escalation": 0.55,
    "disease_prevalence": 0.80,
    "unmet_medical_need": 0.90,
    "natural_history_data": 0.85,
    "patient_advocacy": 0.75,
}

#: Retained name, deliberately NOT a set of published scores. If you need a
#: comparator, use :meth:`PlatformValidator.control_rubric_score`, which runs the
#: same rubric on the unmodified configuration. These figures are authored
#: illustrations kept only for continuity with earlier reports; they are not
#: measurements and must not be cited as one.
ILLUSTRATIVE_COMPARATOR = {
    "product_quality": 0.65,
    "non_clinical_safety": 0.60,
    "clinical_safety": 0.45,
    "clinical_efficacy": 0.50,
    "patient_stratification": 0.55,
    "risk_management": 0.50,
    "pediatric_extrapolation": 0.55,
    "orphan_disease": 0.75,
}

#: Backwards-compatible alias. The old name implied these were measurements of a
#: real trial's regulatory standing; they are not. See ILLUSTRATIVE_COMPARATOR.
UCL_REGULATORY_BASELINE = ILLUSTRATIVE_COMPARATOR


@dataclass
class RegulatoryDimensionScore:
    dimension: str
    label: str
    score: Optional[float]
    weight: float
    pass_threshold: float
    passed: bool
    criteria_scores: Dict[str, float]
    scored: bool = True
    missing_inputs: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # An unscored dimension can never count as passed, whatever the caller
        # passed in. ``passed`` is only meaningful alongside ``scored``.
        if not self.scored:
            self.passed = False


@dataclass
class RegulatoryAssessment:
    product_quality: RegulatoryDimensionScore
    non_clinical_safety: RegulatoryDimensionScore
    clinical_safety: RegulatoryDimensionScore
    clinical_efficacy: RegulatoryDimensionScore
    patient_stratification: RegulatoryDimensionScore
    risk_management: RegulatoryDimensionScore
    pediatric_extrapolation: RegulatoryDimensionScore
    orphan_disease: RegulatoryDimensionScore
    composite_score: Optional[float]
    weighted_score: Optional[float]
    #: True / False / None. ``None`` means the gating dimensions were not all
    #: scored, so no verdict can be formed -- it is NOT a "no".
    is_ilap_eligible: Optional[bool]
    critical_gaps: List[str]
    improvements_needed: List[str]
    scored_dimensions: int = 0
    total_dimensions: int = len(MHRA_ILAP_DIMENSIONS)
    coverage: float = 0.0
    missing_inputs: List[str] = field(default_factory=list)
    used_assumed_inputs: bool = False
    assumed_inputs_used: List[str] = field(default_factory=list)
    verdict: str = "UNKNOWN"
    rubric_provenance: str = RUBRIC_PROVENANCE
    caveat: str = RUBRIC_CAVEAT

    def is_complete(self) -> bool:
        return self.scored_dimensions == self.total_dimensions

    def dimension(self, name: str) -> RegulatoryDimensionScore:
        return getattr(self, name)


class PlatformValidator:
    def __init__(self):
        self.dimensions = MHRA_ILAP_DIMENSIONS
        self.thresholds = MHRA_PASS_THRESHOLDS
        #: The authored comparator, kept under the attribute name older callers
        #: use. Prefer control_rubric_score() for a derived comparator.
        self.ucl_baseline = ILLUSTRATIVE_COMPARATOR

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _require(
        self, m: Dict[str, float], keys: Sequence[str]
    ) -> Tuple[bool, List[str]]:
        """Whether ``keys`` are all present, plus the ones that are not."""
        missing = [k for k in keys if m.get(k) is None]
        return (not missing), missing

    def evaluate_candidate(
        self,
        candidate_metrics: Dict[str, float],
        assumed_inputs: Optional[Dict[str, float]] = None,
    ) -> RegulatoryAssessment:
        """Score a candidate against the rubric.

        ``candidate_metrics`` must contain the keys a dimension consumes (see
        :data:`DIMENSION_INPUTS`). A dimension with a missing input is reported
        unscored and contributes **zero credit** to ``composite_score`` and
        ``weighted_score``, which are normalised by the full rubric rather than
        by the scored subset. Omitting an input therefore lowers coverage and
        can never raise a score (see the comment on the aggregation below).
        When the gating dimensions are not all scored, ``is_ilap_eligible`` is
        ``None``.

        ``assumed_inputs`` lets a caller explicitly supply placeholder values for
        inputs it does not have. Any key sourced from there is recorded in
        ``assumed_inputs_used`` and flags ``used_assumed_inputs``, so a
        default-padded scorecard is always identifiable as one.
        """
        m = dict(candidate_metrics or {})
        assumed_used: List[str] = []
        for k, v in (assumed_inputs or {}).items():
            if m.get(k) is None:
                m[k] = v
                assumed_used.append(k)

        dims: Dict[str, Tuple[Optional[float], Dict[str, float], List[str]]] = {}
        dims["product_quality"] = self._score_product_quality(m)
        dims["non_clinical_safety"] = self._score_non_clinical_safety(m)
        dims["clinical_safety"] = self._score_clinical_safety(m)
        dims["clinical_efficacy"] = self._score_clinical_efficacy(m)
        dims["patient_stratification"] = self._score_patient_stratification(m)
        dims["risk_management"] = self._score_risk_management(m)
        dims["pediatric_extrapolation"] = self._score_pediatric_extrapolation(m)
        dims["orphan_disease"] = self._score_orphan_disease(m)

        dim_objects: Dict[str, RegulatoryDimensionScore] = {}
        for key, (score, criteria, missing) in dims.items():
            info = self.dimensions[key]
            threshold = self.thresholds[key]
            dim_objects[key] = RegulatoryDimensionScore(
                dimension=key,
                label=info["label"],
                score=score,
                weight=info["weight"],
                pass_threshold=threshold,
                passed=(score is not None and score >= threshold),
                criteria_scores=criteria,
                scored=score is not None,
                missing_inputs=missing,
            )

        all_missing = sorted({
            k for key in dim_objects for k in dim_objects[key].missing_inputs
        })
        scored = [d for d in dim_objects.values() if d.scored]
        composite: Optional[float] = None
        weighted: Optional[float] = None
        if scored:
            # Denominator is the FULL rubric, not the scored subset.
            #
            # Renormalising over scored dimensions alone makes omission
            # monotone in the WRONG direction: dropping a dimension whose score
            # is below the running average removes it from the denominator and
            # *raises* both aggregates. That is the one behaviour a readiness
            # rubric must never have -- "I did not measure that" is not evidence
            # in a candidate's favour. It was reachable in practice: clinical
            # efficacy scores ~0.7 on the design arm, so deleting its
            # ``dosing_score`` input lifted the weighted score. Pinned by
            # tests/test_bio_constraints.py::
            # TestBioConstraints::test_regulatory_rubric_never_rewards_omitted_inputs.
            #
            # With the full-rubric denominator an unscored dimension contributes
            # zero credit and the aggregates are monotone: adding evidence whose
            # score is >= 0 can only raise them. `coverage` reports how much of
            # the rubric actually supplied evidence, so a low score caused by
            # missing inputs is always decomposable into (measured quality) x
            # (measured coverage).
            total_weight = sum(d.weight for d in dim_objects.values()) or 1.0
            composite = float(
                sum(d.score for d in scored) / len(dim_objects)
            )
            weighted = float(
                sum(d.score * d.weight for d in scored) / total_weight
            )

        critical_gaps: List[str] = []
        improvements: List[str] = []
        for dim_obj in dim_objects.values():
            if dim_obj.scored and not dim_obj.passed:
                critical_gaps.append(
                    f"{dim_obj.label}: {dim_obj.score:.3f} "
                    f"(need {dim_obj.pass_threshold:.2f})"
                )
                improvements.append(
                    f"Improve {dim_obj.label} by "
                    f"{(dim_obj.pass_threshold - dim_obj.score):.3f} points"
                )

        gating_scored = all(dim_objects[k].scored for k in GATING_DIMENSIONS)
        if not gating_scored:
            eligible: Optional[bool] = None
            verdict = "UNKNOWN"
        else:
            eligible = bool(
                all(dim_objects[k].passed for k in GATING_DIMENSIONS)
                and composite is not None
                and composite >= COMPOSITE_READINESS_THRESHOLD
            )
            verdict = "READY" if eligible else "NOT READY"

        coverage = len(scored) / len(dim_objects)
        if verdict == "UNKNOWN":
            logger.info(
                "Regulatory rubric: verdict UNKNOWN (%d/%d dimensions scored, "
                "gating dimensions incomplete); missing inputs: %s",
                len(scored), len(dim_objects), ", ".join(all_missing) or "none",
            )

        return RegulatoryAssessment(
            product_quality=dim_objects["product_quality"],
            non_clinical_safety=dim_objects["non_clinical_safety"],
            clinical_safety=dim_objects["clinical_safety"],
            clinical_efficacy=dim_objects["clinical_efficacy"],
            patient_stratification=dim_objects["patient_stratification"],
            risk_management=dim_objects["risk_management"],
            pediatric_extrapolation=dim_objects["pediatric_extrapolation"],
            orphan_disease=dim_objects["orphan_disease"],
            composite_score=None if composite is None else float(np.clip(composite, 0, 1)),
            weighted_score=None if weighted is None else float(np.clip(weighted, 0, 1)),
            is_ilap_eligible=eligible,
            critical_gaps=critical_gaps,
            improvements_needed=improvements,
            scored_dimensions=len(scored),
            total_dimensions=len(dim_objects),
            coverage=coverage,
            missing_inputs=all_missing,
            used_assumed_inputs=bool(assumed_used),
            assumed_inputs_used=sorted(assumed_used),
            verdict=verdict,
        )

    # Each scorer returns (score | None, criteria, missing_inputs). It computes a
    # score only when every input it consumes was actually supplied.

    def _score_product_quality(self, m: Dict[str, float]):
        keys = ("itr_integrity", "empty_full_ratio_optimal", "rc_aav_risk", "vector_titer")
        ok, missing = self._require(m, keys)
        if not ok:
            return None, {}, missing
        itr = m["itr_integrity"]
        empty_full = 1.0 - abs(m["empty_full_ratio_optimal"] - 0.5) * 0.5
        rc_aav = 1.0 - m["rc_aav_risk"] * 10
        titer = min(m["vector_titer"] / 1e14, 1.0)
        score = (0.25 * itr + 0.25 * empty_full
                 + 0.25 * float(np.clip(rc_aav, 0, 1)) + 0.25 * float(np.clip(titer, 0, 1)))
        criteria = {
            "ITR_integrity": itr,
            "empty_full_ratio": float(np.clip(empty_full, 0, 1)),
            "RC_AAV": float(np.clip(rc_aav, 0, 1)),
            "titer": float(np.clip(titer, 0, 1)),
        }
        return float(np.clip(score, 0, 1)), criteria, []

    def _score_non_clinical_safety(self, m: Dict[str, float]):
        keys = ("cardiac_tropism", "hepatic_accumulation",
                "gonadal_transduction_risk", "shedding_risk")
        ok, missing = self._require(m, keys)
        if not ok:
            return None, {}, missing
        cardiac = m["cardiac_tropism"]
        hepatic = 1.0 - m["hepatic_accumulation"]
        germline = 1.0 - m["gonadal_transduction_risk"]
        shedding = 1.0 - m["shedding_risk"]
        score = 0.30 * cardiac + 0.30 * hepatic + 0.20 * germline + 0.20 * shedding
        criteria = {
            "cardiac_tropism": cardiac,
            "hepatic_avoidance": float(np.clip(hepatic, 0, 1)),
            "germline_safety": float(np.clip(germline, 0, 1)),
            "shedding_control": float(np.clip(shedding, 0, 1)),
        }
        return float(np.clip(score, 0, 1)), criteria, []

    def _score_clinical_safety(self, m: Dict[str, float]):
        keys = ("complement_activation", "liver_toxicity", "cardiac_inflammation",
                "immune_evasion", "decoy_protection")
        ok, missing = self._require(m, keys)
        if not ok:
            return None, {}, missing
        complement = 1.0 - m["complement_activation"]
        liver_tox = 1.0 - m["liver_toxicity"]
        cardiac_inflam = 1.0 - m["cardiac_inflammation"]
        immune = m["immune_evasion"]
        decoy = m["decoy_protection"]
        score = (0.30 * complement + 0.25 * liver_tox + 0.20 * cardiac_inflam
                 + 0.15 * immune + 0.10 * decoy)
        criteria = {
            "complement_avoidance": float(np.clip(complement, 0, 1)),
            "liver_safety": float(np.clip(liver_tox, 0, 1)),
            "cardiac_inflammation_control": float(np.clip(cardiac_inflam, 0, 1)),
            "immune_evasion": immune,
            "decoy_efficacy": decoy,
        }
        return float(np.clip(score, 0, 1)), criteria, []

    def _score_clinical_efficacy(self, m: Dict[str, float]):
        keys = ("cardiac_tropism", "lamp2b_expression", "promoter_score",
                "mirna_score", "dosing_score")
        ok, missing = self._require(m, keys)
        if not ok:
            return None, {}, missing
        cardiac_trans = m["cardiac_tropism"]
        lamp2b = m["lamp2b_expression"]
        promoter = m["promoter_score"]
        mirna = m["mirna_score"]
        dosing = m["dosing_score"]
        score = (0.30 * cardiac_trans + 0.30 * lamp2b + 0.15 * promoter
                 + 0.15 * mirna + 0.10 * dosing)
        criteria = {
            "cardiac_transduction": cardiac_trans,
            "LAMP2_expression": lamp2b,
            "promoter_efficacy": promoter,
            "miRNA_detarget": mirna,
            "dosing_optimization": dosing,
        }
        return float(np.clip(score, 0, 1)), criteria, []

    def _score_patient_stratification(self, m: Dict[str, float]):
        keys = ("mutation_types_covered", "age_stratification",
                "nab_management", "biomarker_strategy")
        ok, missing = self._require(m, keys)
        if not ok:
            return None, {}, missing
        mutation = m["mutation_types_covered"]
        age = m["age_stratification"]
        nab = m["nab_management"]
        bio = m["biomarker_strategy"]
        score = 0.30 * mutation + 0.25 * age + 0.25 * nab + 0.20 * bio
        criteria = {
            "mutation_coverage": mutation, "age_stratification": age,
            "NAb_management": nab, "biomarkers": bio,
        }
        return float(np.clip(score, 0, 1)), criteria, []

    def _score_risk_management(self, m: Dict[str, float]):
        keys = ("immunosuppression_plan", "complement_monitoring",
                "rescue_therapy", "stopping_rules")
        ok, missing = self._require(m, keys)
        if not ok:
            return None, {}, missing
        immune_supp = m["immunosuppression_plan"]
        complement_mon = m["complement_monitoring"]
        rescue = m["rescue_therapy"]
        stopping = m["stopping_rules"]
        score = 0.30 * immune_supp + 0.30 * complement_mon + 0.20 * rescue + 0.20 * stopping
        criteria = {
            "immunosuppression": immune_supp, "complement_monitoring": complement_mon,
            "rescue_therapy": rescue, "stopping_rules": stopping,
        }
        return float(np.clip(score, 0, 1)), criteria, []

    def _score_pediatric_extrapolation(self, m: Dict[str, float]):
        keys = ("juvenile_model_data", "growth_development_safety",
                "long_term_follow_up", "age_de_escalation")
        ok, missing = self._require(m, keys)
        if not ok:
            return None, {}, missing
        juvenile = m["juvenile_model_data"]
        growth = m["growth_development_safety"]
        long_term = m["long_term_follow_up"]
        de_esc = m["age_de_escalation"]
        score = 0.30 * juvenile + 0.30 * growth + 0.25 * long_term + 0.15 * de_esc
        criteria = {
            "juvenile_data": juvenile, "growth_safety": growth,
            "long_term_FU": long_term, "age_de_escalation": de_esc,
        }
        return float(np.clip(score, 0, 1)), criteria, []

    def _score_orphan_disease(self, m: Dict[str, float]):
        keys = ("disease_prevalence", "unmet_medical_need",
                "natural_history_data", "patient_advocacy")
        ok, missing = self._require(m, keys)
        if not ok:
            return None, {}, missing
        prevalence = m["disease_prevalence"]
        unmet = m["unmet_medical_need"]
        natural = m["natural_history_data"]
        advocacy = m["patient_advocacy"]
        score = 0.30 * prevalence + 0.30 * unmet + 0.25 * natural + 0.15 * advocacy
        criteria = {
            "prevalence": prevalence, "unmet_need": unmet,
            "natural_history": natural, "advocacy": advocacy,
        }
        return float(np.clip(score, 0, 1)), criteria, []

    # ------------------------------------------------------------------
    # Comparators
    # ------------------------------------------------------------------

    def control_rubric_score(self, control_metrics: Dict[str, float]) -> Optional[float]:
        """Derived comparator: this same rubric applied to the control construct.

        This is the comparator to use in a report. It answers "does the design
        score higher than the unmodified control *on our rubric*", which is a
        self-consistent internal comparison, unlike the authored
        :data:`ILLUSTRATIVE_COMPARATOR`.
        """
        return self.evaluate_candidate(control_metrics).weighted_score

    def assess_improvement_vs_ucl(self, assessment: RegulatoryAssessment) -> Dict:
        """Compare to :data:`ILLUSTRATIVE_COMPARATOR`.

        Retained for continuity. The comparator values are authored, so the
        ratio is **not** a measured improvement over any real programme.
        """
        improvements = {}
        for dim_name, ucl_val in self.ucl_baseline.items():
            dim_obj = getattr(assessment, dim_name, None)
            if dim_obj is not None:
                ours = dim_obj.score
                improvements[dim_name] = {
                    "our_score": ours,
                    "comparator_score": ucl_val,
                    "comparator_provenance": RUBRIC_PROVENANCE,
                    "improvement": (
                        None if ours is None else ours / max(ucl_val, 0.01)
                    ),
                }
        return improvements

    def utcl_regulatory_score(self) -> float:
        """Weighted score of the authored comparator. NOT a measurement."""
        return sum(
            self.ucl_baseline[k] * self.dimensions[k]["weight"]
            for k in self.ucl_baseline
        )

    def our_regulatory_score(self, m: Dict[str, float]) -> Optional[float]:
        return self.evaluate_candidate(m).weighted_score

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_regulatory_summary(self, assessment: RegulatoryAssessment) -> str:
        lines = []
        lines.append("Pre-Submission Regulatory Readiness (self-assessment rubric)")
        lines.append("=" * 62)
        lines.append(f"Rubric provenance: {assessment.rubric_provenance}")
        lines.append(assessment.caveat)
        lines.append("-" * 62)
        for dim_name in self.dimensions:
            dim_obj = getattr(assessment, dim_name, None)
            if dim_obj is None:
                continue
            if dim_obj.scored:
                marker = "PASS" if dim_obj.passed else "FAIL"
                value = f"{dim_obj.score:.3f}"
            else:
                marker = "NOT SCORED"
                value = f"-- ({', '.join(dim_obj.missing_inputs)} missing)"
            lines.append(f"  {dim_obj.label:45s} {value}  [{marker}]")
        lines.append("-" * 62)
        if assessment.composite_score is None:
            lines.append("  Composite Score: not computable (no dimension scored)")
        else:
            lines.append(
                f"  Composite Score: {assessment.composite_score:.3f}"
                f"  ({assessment.scored_dimensions}/{assessment.total_dimensions} dimensions)"
            )
        if assessment.weighted_score is None:
            lines.append("  Weighted Score:  not computable")
        else:
            lines.append(f"  Weighted Score:  {assessment.weighted_score:.3f}")
        lines.append(f"  Verdict:         {assessment.verdict}"
                     f"  (gating set: {', '.join(GATING_DIMENSIONS)})")
        if assessment.verdict == "UNKNOWN":
            lines.append(
                "  NOTE: verdict is UNKNOWN, not negative -- the gating "
                "dimensions were not all scored."
            )
        if assessment.missing_inputs:
            lines.append("-" * 62)
            lines.append("MISSING INPUTS (evidence not supplied):")
            for key in assessment.missing_inputs:
                lines.append(f"  - {key}")
        if assessment.used_assumed_inputs:
            lines.append("ASSUMED INPUTS (caller-supplied placeholders, not measurements):")
            for key in assessment.assumed_inputs_used:
                lines.append(f"  - {key}")
        if assessment.critical_gaps:
            lines.append("-" * 62)
            lines.append("DIMENSIONS BELOW THRESHOLD:")
            for gap in assessment.critical_gaps:
                lines.append(f"  - {gap}")
        if assessment.improvements_needed:
            lines.append("REQUIRED IMPROVEMENTS:")
            for imp in assessment.improvements_needed:
                lines.append(f"  - {imp}")
        return "\n".join(lines)
