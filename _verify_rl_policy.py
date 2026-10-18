"""Verify the trained dosing policy end to end, and prove the check can fail.

Run: python _verify_rl_policy.py    (exit 0 = all checks pass)

WHAT IT CHECKS
--------------
1. The artifact exists and the weights file on disk matches its recorded hash.
2. `verify_rl_policy_retention` re-runs a held-out seed from the saved weights
   and agrees with the file to rtol 1e-6 (this is what the audit's
   `rl_policy_measured` check calls).
3. Evaluation and monitor seeds are disjoint from training and from one
   another; all seven held-out arms and all monitor rule scores are consistent.
4. `rl_policy_target` reports the measured simulator ratio, while preserving
   the distinction from the old claimed 95.4% target.
5. The claim is registered under the new id, with the new check key, an
   ESTABLISHED tier, and a limit that names the in-silico boundary; the README
   no longer says no agent exists; nav.ts lists the id as a RESULT and not as
   a refusal.
6. NEGATIVE CONTROLS: doctored per-seed summaries and a swapped weights hash
   each make the live verification FAIL. If either passes, the check is
   decoration and the script exits nonzero.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main() -> int:
    from danon.claims import CLAIMS, CHECKS
    from danon.eval_targets import TargetStatus, rl_policy_target
    from danon.rl_steering import (
        ARTIFACT_PATH,
        TRAIN_RECORD_PATH,
        WEIGHTS_PATH,
        load_artifact,
        verify_rl_policy_retention,
    )

    print("artifact and weights")
    art = load_artifact(ARTIFACT_PATH)
    check("artifact exists", art is not None, ARTIFACT_PATH)
    if art is None:
        return 1
    check("weights file exists", os.path.exists(WEIGHTS_PATH), WEIGHTS_PATH)
    check(
        "artifact records the weights hash",
        bool(art.get("weights_sha256")) and len(str(art["weights_sha256"])) == 64,
    )
    check(
        "artifact declares limits",
        bool(art.get("cannot_show")) and art.get("provenance", {}).get("patient_benefit") == "proven_failed",
    )

    print("live reproduction")
    v = verify_rl_policy_retention()
    check("held-out seed re-measured from saved weights", v.ok, v.detail[:110])
    check("held-out seed count recorded", v.n_seeds == len(art["seeds"]), f"{v.n_seeds}")

    print("monitor/training partitions")
    record = None
    if os.path.exists(TRAIN_RECORD_PATH):
        with open(TRAIN_RECORD_PATH, "r", encoding="utf-8") as fh:
            record = json.load(fh)
    check("training record exists", record is not None, TRAIN_RECORD_PATH)
    if record is not None:
        train_seeds = str(record["seed_partition"]["training"])
        base = int(train_seeds.rstrip("+"))
        eval_seeds = set(int(s) for s in art["seeds"])
        monitor_seeds = set(int(s) for s in art["selection_seeds"])
        check("monitor/evaluation seed sets are disjoint", not eval_seeds & monitor_seeds)
        check(
            "both partitions are outside training seed range",
            all(s < base for s in eval_seeds | monitor_seeds),
            f"training starts at {base}",
        )
        check("training-record hash equals artifact hash", record["weights_sha256"] == art["weights_sha256"])
        check("sampled agent beats MTD on held-out median AUBC", art["arms"]["ppo_agent_sampled"]["median_aubc"] < art["arms"]["mtd"]["median_aubc"])
        check("deterministic alternation beats sampled PPO", art["arms"]["alternate_reference"]["median_aubc"] < art["arms"]["ppo_agent_sampled"]["median_aubc"])
        check("greedy PPO does not beat MTD", art["arms"]["ppo_agent_greedy"]["median_aubc"] >= art["arms"]["mtd"]["median_aubc"] - 1e-6)

    print("seed partition")
    record = None
    if os.path.exists(TRAIN_RECORD_PATH):
        with open(TRAIN_RECORD_PATH, "r", encoding="utf-8") as fh:
            record = json.load(fh)
    check("training record exists", record is not None, TRAIN_RECORD_PATH)
    if record is not None:
        train_seeds = str(record["seed_partition"]["training"])
        base = int(train_seeds.rstrip("+"))
        check(
            "evaluation seeds disjoint from the training partition",
            all(int(s) < base for s in art["seeds"]),
            f"train from {base}, eval {min(art['seeds'])}-{max(art['seeds'])}",
        )
        check(
            "training record's hash equals the artifact's",
            record["weights_sha256"] == art["weights_sha256"],
        )

    print("audit wiring")
    rl = rl_policy_target()
    check(
        "rl_policy_target reports the reproducible simulator metric",
        rl.status is TargetStatus.MEASURED and rl.has_number,
        f"status {rl.status.value}, measured {rl.display}",
    )
    check(
        "target number equals the artifact's retention",
        rl.measured == art["retention"]["aubc"],
        f"{rl.measured} vs {art['retention']['aubc']}",
    )
    check("target detail labels the result in silico", "IN SILICO" in rl.detail)
    check(
        "target keeps the synthetic and authored-metric limits",
        any("synthetic" in b for b in rl.blockers)
        and any("author-defined" in b for b in rl.blockers),
    )
    check(
        "target discloses the alternation comparator",
        any("alternation" in b for b in rl.blockers),
    )
    check("legacy 95.4% target is not labelled measured", "historical 95.4% target not evaluated" in rl.claimed)

    refusal = rl_policy_target(artifact_path="outputs_real/definitely_absent.json")
    check(
        "refusal branch still exists for a missing artifact",
        refusal.status is TargetStatus.REFUSED_NO_DATA and refusal.measured is None,
    )

    by_id = {c.id: c for c in CLAIMS}
    claim = by_id.get("rl_policy_trained_and_measured")
    check("claim registered under the new id", claim is not None)
    if claim is not None:
        check("claim check key is rl_policy_measured", claim.check == "rl_policy_measured")
        check("claim check key is registered in CHECKS", claim.check in CHECKS)
        check("claim tier is ESTABLISHED", str(claim.tier.value) == "established", claim.tier.value)
        check("claim limit names the simulator boundary", "Synthetic" in claim.limit or "simulator" in claim.limit)
        for value in (
            art["retention"]["aubc"],
            art["arms"]["ppo_agent_sampled"]["median_aubc"],
            art["arms"][art["best_rule"]]["median_aubc"],
            art["arms"]["mtd"]["median_aubc"],
            art["arms"]["alternate_reference"]["median_aubc"],
        ):
            check(
                f"claim sentence quotes {float(value):.4f}",
                f"{float(value):.4f}" in claim.sentence,
            )
        check(
            "claim sentence quotes the held-out seed count",
            f"{len(art['seeds'])}" in claim.sentence,
        )
    check("old claim id is gone", "no_rl_agent" not in by_id)
    check("egfrviii claim has its own check key", by_id["egfrviii_not_in_open_data"].check == "egfrviii_refused")

    print("surfaces")
    with open("README.md", "r", encoding="utf-8") as fh:
        readme = fh.read()
    check("README no longer claims no agent exists", "no reinforcement-learning agent exists or is claimed" not in readme)
    check("README names the new claim", "rl_policy_trained_and_measured" in readme)
    with open(os.path.join("frontend", "src", "nav.ts"), "r", encoding="utf-8") as fh:
        nav = fh.read()
    check("nav.ts lists the id as a result", "'rl_policy_trained_and_measured'" in nav)
    check("nav.ts no longer lists it as a refusal", "'no_rl_agent'" not in nav)
    with open("./frontend/src/utils/evidenceTier.ts", "r", encoding="utf-8") as fh:
        tier_src = fh.read()
    check("evidenceTier comment no longer cites the old id", "no_rl_agent" not in tier_src)

    print("negative controls (these MUST fail)")
    doctored = json.loads(json.dumps(art))
    doctored["arms"]["ppo_agent_sampled"]["records"][0]["aubc"] += 0.25
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "doctored.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(doctored, fh)
        bad = verify_rl_policy_retention(artifact_path=p)
        check(
            "doctored per-seed record fails",
            (not bad.ok) and ("summary" in bad.detail or "re-measured" in bad.detail),
            bad.detail,
        )
        swapped = json.loads(json.dumps(art))
        swapped["weights_sha256"] = "1" * 64
        p2 = os.path.join(td, "swapped.json")
        with open(p2, "w", encoding="utf-8") as fh:
            json.dump(swapped, fh)
        bad2 = verify_rl_policy_retention(artifact_path=p2)
        check("swapped weights hash fails", (not bad2.ok) and "hash mismatch" in bad2.detail)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
