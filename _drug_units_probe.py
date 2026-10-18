"""Record what the FDA label's concentration scale really is, and whether changing it matters.

A common "use public data" route for the potency gap says: the Temodar label
publishes the human plasma kinetics (a Cmax in ug/mL), so convert the engine's
drug PDEs to those units and anchor the EC50 to a clinical molarity. Two
separable things are checked here and recorded:

  * WHAT THE LABEL ACTUALLY SAYS. The sentences carrying the concentration
    scale are extracted from the label text and stored verbatim, with the URL,
    so a remembered number cannot stand in for a read one. (The number a route
    usually quotes as a Cmax is, on the label, the CLEARANCE.)
  * WHETHER THE RELABELLING CHANGES ANYTHING. The engine's kill law is
    `kill_max * c / (c + EC50)`, homogeneous of degree 0 in (c, EC50), and its
    transport is linear in the source. So scaling the source strength AND the
    EC50 by the same factor is an exact relabelling of concentration units, and
    the trajectory should be BIT-IDENTICAL. The scale used is a power of two,
    because scaling by 2^k is exact in binary floating point -- any difference
    would then be the model's, not the arithmetic's.

The control matters as much as the finding: scaling the EC50 ALONE must move the
trajectory, or "identical" would be a statement about a measurement that cannot
see anything.

Run:  python _drug_units_probe.py            # reads the label, then measures
      python _drug_units_probe.py --offline  # skips the label, still measures
"""

import argparse
import dataclasses
import datetime
import json
import os
import re
import urllib.error
import urllib.request

OUT = os.path.join("outputs_real", "drug_units_probe.json")

#: The current Temodar label on DailyMed (capsules + injection, Merck).
LABEL_URL = (
    "https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm"
    "?setid=046a9011-3911-4d3f-a15f-fbb56d5aad56&type=display"
)

#: Sentences carrying a concentration scale, a clearance or a half-life.
LABEL_TOKENS = ("l/hr", "mcg/ml", "ug/ml", "half-life", "cmax", "clearance")

#: A power of two: scaling by it is EXACT in binary floating point, so an
#: exact relabelling must reproduce the trajectory bit for bit.
SCALE = 1024.0


def _get(url: str, timeout: int = 45, attempts: int = 3):
    headers = {"User-Agent": "danon-audit/1.0", "Accept": "*/*"}
    last = None
    for _ in range(attempts):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read()
        except Exception as exc:  # noqa: BLE001 -- the record carries failures too
            last = (None, f"{type(exc).__name__}: {exc}")
    return last


def probe_label() -> dict:
    """Extract the label's own concentration and clearance sentences."""
    entry = {"url": LABEL_URL, "sentences": []}
    status, body = _get(LABEL_URL)
    entry["status"] = status
    if not isinstance(body, bytes):
        entry["error"] = body
        return entry
    text = body.decode("utf-8", "replace")
    # The label arrives as HTML; the sentences are what matter, not the markup.
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;|&#\d+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    for raw in re.split(r"(?<=[.;])\s+", text):
        sentence = raw.strip()
        low = sentence.lower()
        if len(sentence) < 200 and any(tok in low for tok in LABEL_TOKENS):
            if sentence not in entry["sentences"]:
                entry["sentences"].append(sentence)
        if len(entry["sentences"]) >= 14:
            break
    return entry


def measure_relabelling() -> dict:
    """Scale (source_strength, EC50) together and separately, and compare."""
    import numpy as np

    from danon.evosteer import (
        DEFAULT_DRUGS,
        DEFAULT_CLONES,
        MTDPolicy,
        TumorEvolutionSimulator,
        make_gbm_phantom,
    )

    geometry = make_gbm_phantom(size=16, seed=0)
    days, r0, seed = 60, 0.01, 101
    e0, e1 = (float(x) for x in DEFAULT_DRUGS.ec50)

    def run(drugs):
        sim = TumorEvolutionSimulator(geometry, clones=DEFAULT_CLONES, drugs=drugs)
        return sim.run(
            MTDPolicy(), days=days, initial_resistant_fraction=r0, seed=seed
        )

    base = run(DEFAULT_DRUGS)
    both = run(
        dataclasses.replace(
            DEFAULT_DRUGS,
            source_strength=float(DEFAULT_DRUGS.source_strength) * SCALE,
            ec50=(e0 * SCALE, e1 * SCALE),
        )
    )
    ec50_only = run(
        dataclasses.replace(DEFAULT_DRUGS, ec50=(e0 * SCALE, e1 * SCALE))
    )

    def totals(traj):
        return {
            "final_burden": float(traj.burden_S[-1] + traj.burden_R[-1]),
            "pfs_day": traj.pfs_day,
            "toxicity": float(traj.toxicity),
        }

    # The same relabelling, on the kill law alone, over the concentrations the
    # engine's own field actually reaches.
    fractions_identical = True
    for c in (1e-3, 5.0, 0.30, 12.0):
        f_base = c / (c + e0)
        f_scaled = (c * SCALE) / (c * SCALE + e0 * SCALE)
        if abs(f_base - f_scaled) > 0.0:
            fractions_identical = False

    return {
        "days": days,
        "initial_resistant_fraction": r0,
        "seed": seed,
        "geometry": "make_gbm_phantom(size=16, seed=0), the audit's own phantom",
        "scale": SCALE,
        "scale_note": "2^10: exact in binary floating point",
        "kill_law": "kill_max * c / (c + EC50) per drug and clone (EvosteerSim.step)",
        "transported_field_is_linear_in_the_source": (
            "c += dt * (D * lap(c) + dose * source_strength * vessels - clearance * c)"
        ),
        "base": totals(base),
        "both_scaled": totals(both),
        "ec50_scaled_only": totals(ec50_only),
        "burden_arrays_bit_identical_when_both_scaled": bool(
            np.array_equal(base.burden_S, both.burden_S)
            and np.array_equal(base.burden_R, both.burden_R)
        ),
        "burden_arrays_differ_when_only_ec50_scaled": not bool(
            np.array_equal(base.burden_S, ec50_only.burden_S)
        ),
        "kill_fraction_bit_identical_under_the_scale": fractions_identical,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    record = {
        "recorded_on": datetime.datetime.now().isoformat(timespec="seconds"),
        "offline": bool(args.offline),
        "question": (
            "does expressing the engine's drug concentration in the label's own "
            "units add information, and what does the label's concentration scale "
            "actually say?"
        ),
        "label": {},
        "relabelling": {},
    }
    if not args.offline:
        record["label"] = probe_label()
    record["relabelling"] = measure_relabelling()

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)

    rel = record["relabelling"]
    print("wrote", OUT)
    if record["label"]:
        print(f"  label: status {record['label'].get('status')}, "
              f"{len(record['label'].get('sentences') or [])} concentration sentences")
    print(f"  bit-identical when both scaled: "
          f"{rel['burden_arrays_bit_identical_when_both_scaled']}")
    print(f"  differs when only EC50 scaled:  "
          f"{rel['burden_arrays_differ_when_only_ec50_scaled']}")


if __name__ == "__main__":
    main()
