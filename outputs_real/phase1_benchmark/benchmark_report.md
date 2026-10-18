# EvoTwin Phase 1 — change-point detection benchmark

RESEARCH-ONLY. Synthetic data, no clinical content.

## Aggregate

- mean sensitivity 0.978 | mean specificity 0.800 | mean precision 0.939 | false alarms/1000d 0.55 | mean detection delay 7.5 d | calibration 0.749 | 3 flagged / 1 missed / 3 false alarms over 60 cases

## Per-seed runs

| seed | sensitivity | specificity | precision | FAR/1000d | delay | insuff. |
|---|---|---|---|---|---|---|
| 11 | 0.933 | 0.800 | 0.933 | 0.55 | 5.5 | 0 |
| 23 | 1.000 | 0.600 | 0.882 | 1.10 | 8.0 | 0 |
| 37 | 1.000 | 1.000 | 1.000 | 0.00 | 9.0 | 0 |

## Missed changes (named, not averaged away)

- seed 11: synthetic-case-009

## False alarms (named, not averaged away)

- seed 11: synthetic-case-019
- seed 23: synthetic-case-015
- seed 23: synthetic-case-017

## Limitations (verbatim from the record)

- Synthetic only: the generator is exp-growth in radius with a rate change, not a tumour model; sensitivity on this data says nothing about real imaging or liquid-biopsy series.
- Single change per case: no multiple-change, no gradual-drift, and no regression-to-baseline scenarios are in Phase 1.
- Call-level, not event-level metrics: precision counts flagged CASES, and a case can only flag once in this version.
- The detector knows the generative family (exp-growth, log-linear); a model violation is out of scope and would need a new baseline.
- Calibration is scored only for confirmed calls with a planted day; suspected-only calls lower sensitivity but are not delay-scored.
- No clinical claim of any kind: the report describes a controlled synthetic experiment; no patient benefit is implied or measured.
