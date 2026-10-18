"""Audit `ARCHITECTURE_EVOTWIN_10X.md` against the code and persist the result.

Writes `outputs_real/architecture_audit.json`: every finding (statement, verdict,
correction, code evidence), the live code-fact snapshot the corrections rest on,
and the audited document's SHA-256. `danon/claims.py` refuses unless a fresh
snapshot still equals the stored one, so a change to the engine fails the audit's
claim until this runner is invoked again.

USAGE
-----
python _run_architecture_audit.py
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from danon.pitch_audit import (  # noqa: E402
    AUDITED_DOCUMENT,
    audit_document,
    pitch_audit_provenance,
    render_corrected_copy,
    render_corrections,
    what_the_audit_cannot_show,
)

OUT = os.path.join(ROOT, "outputs_real", "architecture_audit.json")
OUT_CORRECTIONS = os.path.join(ROOT, "outputs_real", "architecture_corrections.md")


def main() -> int:
    started = time.time()
    print("=" * 78)
    print(f"ARCHITECTURE AUDIT: {AUDITED_DOCUMENT}")
    print("=" * 78, flush=True)

    audit = audit_document(AUDITED_DOCUMENT)
    print(f"{audit.describe()}\n", flush=True)

    print(f"{'rule':42s} {'status':8s} {'verdict':11s} matched")
    print("-" * 78)
    for f in audit.findings:
        print(f"{f.rule_id:42s} {f.status:8s} {f.verdict or '-':11s} "
              f"{(f.matched or '')[:22]}")

    print("\nCORRECTIONS\n" + "-" * 78)
    print(render_corrections(audit))

    payload = audit.to_dict()
    payload["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload["what_this_is"] = (
        "An audit of the submitted EvoTwin-10X architecture write-up against the "
        "code it describes. The document is preserved verbatim; every refuted or "
        "qualified statement is carried here with the code that judges it."
    )
    payload["provenance"] = {
        k: v.value for k, v in pitch_audit_provenance().items()
    }
    payload["cannot_show"] = what_the_audit_cannot_show()
    payload["wall_seconds"] = round(time.time() - started, 2)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, OUT)

    # The reading copy is regenerated from the same audit object, never hand-
    # edited, so the corrected prose and the findings cannot disagree.
    with open(OUT_CORRECTIONS, "w", encoding="utf-8") as fh:
        fh.write(render_corrected_copy(audit))

    print(f"\nwrote {OUT}", flush=True)
    print(f"wrote {OUT_CORRECTIONS}", flush=True)
    print(f"  document sha256 {audit.sha256[:16]}...  ({audit.n_words} words). "
          f"The submitted write-up is unedited on disk; the corrected reading "
          f"copy carries the same digest, which the registry check compares "
          f"against the file.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
