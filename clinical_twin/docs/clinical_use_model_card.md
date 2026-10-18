---
_applies_to_licenses:
  - 'observational-registry-twin'
---

# Clinical-use model card — Enhanced Clinical Twin

## Intended use (what this system is FOR)

Decision **support** for oncology care teams, layered on a sealed, research-only data
core. The core produces *labelled research evidence*. This layer converts evidence
into an auditable, human-approved clinical action at the bedside or tumor board.

### Intended users
- Attending oncologist / tumor board chair (final decision authority)
- Clinical pharmacologist (dosing guidance)
- Molecular tumor board (variant-matched therapy ranking)
- Research coordinator (evidence audit trail)

### Approved clinical actions (one tier each)
| Tier | Action | Human sign-off required |
| ---- | ------ | ------------------------ |
| 1 | Vigilance plan (repeat imaging / labs interval) | Attending |
| 2 | Dose adjustment guidance (within protocol limits) | Attending + clinical pharmacologist |
| 3 | Variant-matched therapy **ranking** (never auto-prescribe) | Molecular tumor board |

## NOT for use (hard guardrails)

1. **No autonomous action.** Every tier-1/2/3 action carries an `override_reason`
   field; withholding or overriding is always permitted and logged.
2. **No diagnosis.** Outputs are associations on observational data, never
   diagnostic certainty.
3. **No data leak into research core.** Clinical ids, notes, and free-text must
   never reach `seal_clinical_fields()` upstream; the seal is one-way.
4. **No extrapolation beyond registry envelope.** If dose/variant/timepoint falls
   outside the registries' observed range, the layer returns `outside_envelope`
   and suppresses automated guidance.
5. **No survival or response claims** without an external-validation artifact.

## Evidence hierarchy (how a claim is graded)

| Grade | Source | Example |
| ----- | ------ | ------- |
| A | Randomized trial (in-registry matched) | Phase III regimen arm |
| B | Prospective single-arm / external cohort | CGGA external validation |
| C | Retrospective observational (this twin's core) | Longitudinal registry association |
| D | Preclinical / in-silico only | Physics-informed forecast, twin delta |

Every surfaced claim carries: `claim_id`, `evidence_grade`, `source_refs`,
`model_version`, and `confidence_interval`. Cards without all five are not rendered.

## Thresholds & hyperparameters (edit ONLY with re-validation)

| Parameter | Default | Range | Re-validates against |
| --------- | ------- | ----- | --------------------- |
| `net_benefit_min` | 0.02 | 0.0–0.2 | Decision-curve replay |
| `calibration_lost_margin` | 0.05 | 0.0–0.2 | Reliability diagram replay |
| `cap_miss_threshold` | 0.10 | 0.05–0.30 | Safety-cap ablation |
| `default_tier` | 1 | 1–3 | Escalation policy review |

## Model provenance

- Sealed research core (unchanged): `models/`, `twin/`, registries
- Clinical logic added: `clinical/` package only
- Twin version declared at runtime via `EnhancedTwin.model_version`
- Validation status: **internal hold-out only** — external validation pending