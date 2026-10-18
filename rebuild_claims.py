#!/usr/bin/env python3
"""Rebuild danon/claims.py from a clean git HEAD copy, then insert the new
release-2 alternating-two-drug claim right after the last existing claim
(before the holdout_emulator_090 claim), keeping the CLAIMS tuple in order
and removing every duplicated block (ClaimAudit, audit_claims,
render_claims_markdown, claims_provenance, what_these_claims_cannot_establish).

The clean reference lives next to this script:
    C:\\Users\\supro\\Downloads\\life-clean\\claims_orig.py
(the output of `git show HEAD:danon/claims.py`, 3040 lines, 43 Claim()
entries, PARSES OK).  The corrupted working copy had two of every helper
plus 62 Claim() entries; the extra 19 duplicate Claim() tuples are the break.
"""

import ast
import sys

SRC = "C:/Users/supro/Downloads/life-clean/danon/claims.py"
REF = "C:/Users/supro/Downloads/life-clean/claims_orig.py"

# ---------------------------------------------------------------------------
# Step 1: confirm the clean reference
# ---------------------------------------------------------------------------
ref_src = open(REF, encoding="utf-8").read()
ast.parse(ref_src, filename=REF)  # must parse
ref_claims = ref_src.count("    Claim(")
print(f"clean reference: {len(ref_src.splitlines())} lines, "
      f"{ref_claims} Claim() entries")

# ---------------------------------------------------------------------------
# Step 2: build a clean file from the reference, inserting the new claim
# ---------------------------------------------------------------------------
new_claim = '''    Claim(
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
    ),'''

ref_lines = ref_src.splitlines(keepends=True)
out = []
inserted = False
claim_count = 0
in_claims = False

for line in ref_lines:
    stripped = line.strip()
    if stripped.startswith("CLAIMS: Tuple"):
        in_claims = True
        out.append(line)
        continue
    if in_claims:
        if stripped.startswith("Claim("):
            claim_count += 1
        # Insert the new claim immediately after the last existing claim
        # (after the holdout_emulator_090 tuple closes, before the next
        # "    Claim(" for the stream4 claims).
        if not inserted and claim_count == ref_claims:
            out.append(new_claim + "\n")
            inserted = True
        out.append(line)
    else:
        out.append(line)

clean_src = "".join(out)
assert inserted, "failed to insert the new claim"
print(f"inserted new claim after claim #{claim_count} "
      f"(total {ref_claims} existing -> {ref_claims + 1})")

# ---------------------------------------------------------------------------
# Step 3: sanity-check with ast
# ---------------------------------------------------------------------------
try:
    ast.parse(clean_src, filename=SRC)
    print("ast.parse: PARSES OK")
except SyntaxError as exc:
    print(f"SYNTAX ERROR at line {exc.lineno}: {exc.msg}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 4: write it back
# ---------------------------------------------------------------------------
with open(SRC, "w", encoding="utf-8") as fh:
    fh.write(clean_src)
print(f"wrote {SRC} ({len(clean_src.splitlines())} lines)")

# ---------------------------------------------------------------------------
# Step 5: report new counts
# ---------------------------------------------------------------------------
print("--- new file summary ---")
print(f"Claim() entries: {clean_src.count('    Claim(')}")
print(f"class ClaimAudit: {clean_src.count('class ClaimAudit')}")
print(f"def audit_claims: {clean_src.count('def audit_claims')}")
print(f"def render_claims_markdown: {clean_src.count('def render_claims_markdown')}")
print(f"def claims_provenance: {clean_src.count('def claims_provenance')}")
print(f"def what_these_claims_cannot_establish: "
      f"{clean_src.count('def what_these_claims_cannot_establish')}")
for cid in (
    "alternating_two_drug_calendar_beats_best_single_drug_calendar",
    "holdout_emulator_090",
):
    found = f'"{cid}"' in clean_src
    print(f"claim {cid}: {'present' if found else 'MISSING'}")
