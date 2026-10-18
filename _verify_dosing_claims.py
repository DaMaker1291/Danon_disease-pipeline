"""Negative controls for the dosing-claims check.

A check that cannot fail is decoration. This proves `check_dosing_schedules_are
_measured_on_the_engine` refuses in each of its three distinct ways, by feeding
it doctored artifacts.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import danon.claims as C

REAL = os.path.join(C.ROOT, "outputs_real", "dosing.json")


def _with(mutate):
    """Run the check against a mutated copy of the real artifact."""
    with open(REAL, encoding="utf-8") as fh:
        d = json.load(fh)
    mutate(d)
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh)
    # Point the check at the doctored file.
    real_path = C.ROOT
    orig_root = C.ROOT
    C.ROOT = os.path.dirname(os.path.dirname(tmp))
    try:
        target = os.path.join(C.ROOT, "outputs_real", "dosing.json")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy(tmp, target)
        return C.check_dosing_schedules_are_measured_on_the_engine()
    finally:
        C.ROOT = orig_root
        _ = real_path
        try:
            os.unlink(tmp)
        except OSError:
            pass


def main() -> int:
    failures = []

    # The real artifact must pass.
    ok = C.check_dosing_schedules_are_measured_on_the_engine()
    print(f"  {'PASS' if ok.supported else 'FAIL'}  real artifact is supported")
    if not ok.supported:
        failures.append("real artifact rejected: " + ok.detail)

    # 1. missing artifact
    backup = REAL + ".bak"
    shutil.copy(REAL, backup)
    try:
        os.unlink(REAL)
        r = C.check_dosing_schedules_are_measured_on_the_engine()
        good = (not r.supported) and "missing" in r.detail
        print(f"  {'PASS' if good else 'FAIL'}  missing artifact is refused -- {r.detail[:70]}")
        if not good:
            failures.append("missing artifact not refused")
    finally:
        shutil.copy(backup, REAL)
        os.unlink(backup)

    # 2. missing competitive_release section
    r = _with(lambda d: d.pop("competitive_release", None))
    good = (not r.supported) and "competitive_release" in r.detail
    print(f"  {'PASS' if good else 'FAIL'}  missing CR section is refused -- {r.detail[:70]}")
    if not good:
        failures.append("missing CR section not refused")

    # 3. missing search section
    r = _with(lambda d: d.pop("search", None))
    good = (not r.supported) and "search" in r.detail
    print(f"  {'PASS' if good else 'FAIL'}  missing search section is refused -- {r.detail[:70]}")
    if not good:
        failures.append("missing search section not refused")

    # 4. unreadable JSON
    r = _with(lambda d: None)  # sanity: unchanged still passes
    good = r.supported
    print(f"  {'PASS' if good else 'FAIL'}  unchanged copy still passes")
    if not good:
        failures.append("unchanged copy rejected")

    # 5. THE IMPORTANT ONE: a doctored artifact claiming a win must not be
    #    reported as a win. The check reports whatever the artifact says, so
    #    this asserts the SENTENCE follows the artifact rather than a default.
    def fake_win(d):
        d["search"]["beats_baselines"] = True

    r = _with(fake_win)
    good = r.supported and "beat continuous MTD" in r.detail
    print(f"  {'PASS' if good else 'FAIL'}  detail follows the artifact's own verdict -- "
          f"{'reports a win' if 'beat continuous' in r.detail else 'still reports a null'}")
    if not good:
        failures.append("check verdict not derived from the artifact")

    print()
    if failures:
        for f in failures:
            print("FAILED:", f)
        return 1
    print("all negative controls passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())