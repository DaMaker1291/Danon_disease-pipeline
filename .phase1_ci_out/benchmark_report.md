# EvoTwin Phase 1 — change-point detection benchmark

RESEARCH-ONLY. Synthetic data, no clinical content.

## Aggregate

- mean sensitivity 1.000 | mean specificity 0.833 | mean precision 0.950 | false alarms/1000d 0.46 | mean detection delay 4.0 d | calibration 0.667 | 2 flagged / 0 missed / 1 false alarms over 24 cases

## Per-seed runs

| seed | sensitivity | specificity | precision | FAR/1000d | delay | insuff. |
|---|---|---|---|---|---|---|
| 11 | 1.000 | 1.000 | 1.000 | 0.00 | 5.0 | 0 |
| 23 | 1.000 | 0.667 | 0.900 | 0.91 | 3.0 | 0 |

## False alarms (named, not averaged away)

- seed 23: synthetic-case-010

## Limitations (verbatim from the record)

- Synthetic only: the generator is exp-growth in radius with a rate change, not a tumour model; sensitivity on this data says nothing about real imaging or liquid-biopsy series.
- Single change per case: no multiple-change, no gradual-drift, and no regression-to-baseline scenarios are in Phase 1.
- Call-level, not event-level metrics: precision counts flagged CASES, and a case can only flag once in this version.
- The detector knows the generative family (exp-growth, log-linear); a model violation is out of scope and would need a new baseline.
- Calibration is scored only for confirmed calls with a planted day; suspected-only calls lower sensitivity but are not delay-scored.
- No clinical claim of any kind: the report describes a controlled synthetic experiment; no patient benefit is implied or measured.
