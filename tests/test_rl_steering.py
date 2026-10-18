"""Tests for the learned dosing policy: reproduction, guards, and the audit.

The load-bearing test is IDENTITY. Driven with a fixed schedule plan, the
environment must reproduce `TumorEvolutionSimulator.run` with a policy that
returns the same doses, element by element. The bridge moves dose decisions
across a thread boundary, so "the agent is trained on the real engine" is the
one statement that has to be pinned rather than asserted -- and if the bridge
ever perturbs a trajectory, every number in the RL artifact is describing a
different model than the one the rest of this repository reports.

The artifact tests do three things: re-hash the weights, re-run a held-out seed
through the engine, and pin that a TAMPERED record fails verification. A check
that cannot fail is decoration, so the failure path is tested as carefully as
the passing one. The last test ties the artifact to the claim sentence, because
a claim quoting a number the artifact does not contain is exactly the drift
`danon/claims.py` opens by warning about.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from danon.claims import CLAIMS
from danon.cure import EvidenceStatus
from danon.evosteer import BasePolicy, TumorEvolutionSimulator, make_gbm_phantom
from danon.rl_steering import (
    ARTIFACT_PATH,
    CO_DOSE_REFERENCE,
    HORIZON_DAYS,
    SCHEDULE_ACTIONS,
    WEIGHTS_PATH,
    EvoSteerEnv,
    aubc,
    load_artifact,
    observation_features,
    project_dose,
    schedule_dose,
    verify_rl_policy_retention,
)

RL_CLAIM_ID = "rl_policy_trained_and_measured"


class _FixedPlan(BasePolicy):
    """Returns a recorded list of schedule indices, in order."""

    def __init__(self, plan) -> None:
        self.plan = list(plan)
        self.i = 0

    def reset(self) -> None:
        self.i = 0

    def dose(self, state):
        action = SCHEDULE_ACTIONS[self.plan[self.i]]
        self.i += 1
        return action


# ---------------------------------------------------------------------------
# Identity: the environment IS the engine
# ---------------------------------------------------------------------------


def test_env_reproduces_the_engine_on_a_fixed_schedule_plan():
    plan = np.random.default_rng(11).integers(0, len(SCHEDULE_ACTIONS), size=HORIZON_DAYS)
    plan = [int(x) for x in plan]
    geom = make_gbm_phantom(seed=0)
    reference = TumorEvolutionSimulator(geom).run(_FixedPlan(plan), days=HORIZON_DAYS, seed=41)

    env = EvoSteerEnv()
    obs, info = env.reset(seed=41)
    assert obs.shape == (4,)
    done = False
    rewards = []
    while not done:
        obs, reward, done, _, info = env.step(plan[info["day"]])
        rewards.append(reward)
    trained = env.trajectory

    np.testing.assert_array_equal(reference.burden_S, trained.burden_S)
    np.testing.assert_array_equal(reference.burden_R, trained.burden_R)
    np.testing.assert_array_equal(reference.dose_A, trained.dose_A)
    np.testing.assert_array_equal(reference.dose_B, trained.dose_B)
    assert reference.pfs_day == trained.pfs_day
    assert reference.toxicity == pytest.approx(trained.toxicity, rel=0, abs=0)
    # 365 rewards of -(burden/b_init) == -AUBC * 365, and info reports the AUBC
    assert -sum(rewards) / HORIZON_DAYS == pytest.approx(aubc(trained), rel=1e-12)
    assert float(info["aubc"]) == pytest.approx(aubc(trained), rel=1e-12)


def test_env_is_reproducible_at_the_same_seed():
    env = EvoSteerEnv()
    series = []
    for _ in range(2):
        _, info = env.reset(seed=7)
        done = False
        while not done:
            _, _, done, _, info = env.step(0)  # constant full-dose A
        series.append(np.asarray(env.trajectory.burden_S + env.trajectory.burden_R))
    np.testing.assert_array_equal(series[0], series[1])


# ---------------------------------------------------------------------------
# Guards: the action space is exactly what the docstring says
# ---------------------------------------------------------------------------


def test_schedule_dose_rejects_out_of_range():
    assert schedule_dose(0) == (1.0, 0.0)
    assert schedule_dose(1) == (0.0, 1.0)
    assert schedule_dose(2) == (0.0, 0.0)
    for bad in (-1, 3, 99):
        with pytest.raises(ValueError):
            schedule_dose(bad)


def test_project_dose_is_matched_intensity_and_keeps_the_rule_actions():
    # the closed-form policies' daily actions are fixed points
    assert project_dose(np.array([1.0, 0.0])) == (1.0, 0.0)
    assert project_dose(np.array([0.0, 1.0])) == (0.0, 1.0)
    assert project_dose(np.array([0.0, 0.0])) == (0.0, 0.0)
    # over-intense requests are scaled to sum 1, not truncated
    a, b = project_dose(np.array([0.9, 0.9]))
    assert a + b == pytest.approx(1.0)
    assert a == pytest.approx(b)
    # negatives clip to zero
    assert project_dose(np.array([-3.0, 0.4])) == (0.0, 0.4)


def test_observation_features_are_bounded_and_ordered():
    from danon.evosteer import PolicyState

    lo = observation_features(PolicyState(day=0, burden=0.0, burden_initial=100.0, resistant_fraction=0.0), 0.0)
    hi = observation_features(
        PolicyState(day=HORIZON_DAYS, burden=1e9, burden_initial=100.0, resistant_fraction=1.0),
        1e9,
    )
    assert np.all(lo >= 0.0) and np.all(lo <= 1.0)
    assert np.all(hi >= 0.0) and np.all(hi <= 1.0)
    assert hi[0] == 1.0  # the cap, reached by an exploding trajectory
    assert hi[1] == 1.0


def test_the_co_dose_reference_is_not_a_learned_arm():
    # the confound arm is a fixed constant, recorded so a continuous-space win
    # cannot be quoted as if a scheduler had earned it
    assert CO_DOSE_REFERENCE == (0.5, 0.5)


# ---------------------------------------------------------------------------
# The artifact: re-verified live, and tamper-evident
# ---------------------------------------------------------------------------

art = load_artifact()
needs_artifact = pytest.mark.skipif(
    art is None or not os.path.exists(WEIGHTS_PATH),
    reason="train the policy first: python _train_evosteer_ppo.py",
)


@needs_artifact
def test_weights_and_stored_seed_verify_live():
    v = verify_rl_policy_retention()
    assert v.ok, v.detail
    assert v.n_seeds == len(art["seeds"])
    assert v.rechecked_seed == art["arms"]["ppo_agent_sampled"]["records"][0]["seed"]


@needs_artifact
def test_a_tampered_record_fails_verification(tmp_path):
    doctored = json.loads(json.dumps(art))
    doctored["arms"]["ppo_agent_sampled"]["records"][0]["aubc"] += 0.5
    path = tmp_path / "doctored.json"
    path.write_text(json.dumps(doctored), encoding="utf-8")
    v = verify_rl_policy_retention(artifact_path=str(path))
    assert not v.ok
    assert "summary" in v.detail or "re-measured" in v.detail


@needs_artifact
def test_a_swapped_weight_file_fails_verification(tmp_path):
    doctored = json.loads(json.dumps(art))
    doctored["weights_sha256"] = "0" * 64
    path = tmp_path / "hash.json"
    path.write_text(json.dumps(doctored), encoding="utf-8")
    v = verify_rl_policy_retention(artifact_path=str(path))
    assert not v.ok
    assert "hash mismatch" in v.detail


@needs_artifact
def test_the_claim_sentence_quotes_the_artifact():
    """The published sentence must contain the artifact's own numbers."""
    by_id = {c.id: c for c in CLAIMS}
    claim = by_id[RL_CLAIM_ID]
    assert claim.tier is EvidenceStatus.ESTABLISHED
    assert claim.check == "rl_policy_measured"
    sentence = claim.sentence
    for value in (
        art["retention"]["aubc"],
        art["arms"]["ppo_agent_sampled"]["median_aubc"],
        art["arms"][art["best_rule"]]["median_aubc"],
        art["arms"]["mtd"]["median_aubc"],
    ):
        assert f"{float(value):.4f}" in sentence, f"{value} missing from the claim sentence"
    assert str(len(art["seeds"])) in sentence
    # the training run is quoted too, and must match the record
    record_path = "outputs_real/rl_steering_train.json"
    if os.path.exists(record_path):
        with open(record_path, "r", encoding="utf-8") as fh:
            record = json.load(fh)
        assert f"{record['steps_total']:,}" in sentence


@needs_artifact
def test_the_artifact_declares_what_it_cannot_show():
    assert art["cannot_show"]
    joined = " ".join(art["cannot_show"]).lower()
    assert "synthetic" in joined
    assert art["provenance"]["patient_benefit"] == "proven_failed"
    assert art["provenance"]["retention_in_simulation"] == "established"
