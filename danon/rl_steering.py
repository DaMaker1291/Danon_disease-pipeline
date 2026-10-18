"""
RL dosing: a REAL learned policy for the twin's OWN simulator, and its audit.

WHY THIS MODULE EXISTS
----------------------
`danon.eval_targets.rl_policy_target` refused the claimed "95.4% normalised
reward retention" for two reasons: no reinforcement-learning agent, environment
or training loop existed in the repository, and the metric was self-defined
(the reward AND its normalization are both chosen by the author). This module
adds a trained simulator policy and makes the metric's definition and limits
explicit; it does not turn the old 95.4% claim into an externally validated
endpoint:

  * ENVIRONMENT: `EvoSteerEnv` drives `danon.evosteer.TumorEvolutionSimulator`
    through the engine's OWN day loop. The agent replaces the dosing policy via
    `_BridgePolicy`, so the dynamics, the diffusion substeps, the extinction
    floor and the RANO threshold are the engine's, not a re-implementation.
    IDENTITY: with a fixed action plan the environment reproduces
    `sim.run(FixedPlanPolicy(plan), ...)` exactly (pinned by tests).

  * ACTION SPACE: TWO modes, and the default is the honest one.

    'schedule' (default): one of three daily choices -- full-dose A, full-dose
    B, or a holiday. These are EXACTLY the options the closed-form policies
    choose among (MTDPolicy always picks A, AdaptiveDosePolicy picks A or
    nothing, CollateralSwitchPolicy picks A or B), at the same information
    (burden ratio and resistant fraction), so a win here measures the
    SCHEDULING decision and nothing else.

    'continuous': (dose_A, dose_B) in [0,1]^2 projected to dose_A+dose_B<=1.
    This mode is implemented and measured, but it is NOT the headline: the
    engine's toxicity proxy is a log, not a constraint, so co-dosing both
    drugs at ~0.5 each collapses the tumour to AUBC ~0.02 on the phantom --
    far below every rule -- for reasons that have nothing to do with
    learning. The artifact records that confound as a fixed reference arm
    (`co_dose_reference`) so a continuous-space result cannot be quoted as if
    a scheduler had earned it.

  * REWARD: r_t = -(burden_t / burden_initial) at the START of day t, so the
    episode return is exactly minus the AUBC (area under the normalised burden
    curve, mean over the 365 recorded days). The same AUBC is computed for
    every rule policy on the same seeds, so the training objective is the
    evaluation metric: no dense-training-proxy / sparse-reported-endpoint
    split. gamma = 1.0 in training, so the PPO return IS -AUBC.

  * NORMALISATION: the repository-defined retention ratio compares the agent
    with MTD and the best closed-form rule. The best rule is selected on a
    separate MONITOR partition (not the training or final evaluation seeds),
    then all arms are evaluated on the same held-out seeds:

        retention = (MTD_AUBC - agent_AUBC) / (MTD_AUBC - best_rule_AUBC)

    i.e. the fraction of the achievable improvement over standard care that
    the learned policy captures. 1.0 means it matched the best rule; 0.0 means
    it was no better than MTD; negative means it was worse.

WHAT THIS IS NOT
----------------
This is a recorded training run and a reproducible measurement on a SYNTHETIC
simulator whose parameter set is calibrated to an mCRPC adaptive-therapy anchor
in a different disease (`danon.evosteer` CALIBRATION). The deterministic
alternating A/B reference slightly outperforms sampled PPO on the held-out set,
and the greedy PPO projection matches MTD. The results therefore do not show a
PPO-specific advantage or patient benefit. `what_rl_cannot_show()` returns the
limits that travel with the artifact.

WEIGHTS AND ARTIFACTS
---------------------
Weights:   models/ppo_evosteer_policy.pt   (torch state_dict + feature spec)
Training:  outputs_real/rl_steering_train.json
Measured:  outputs_real/rl_steering.json   (read by the audit's live check)

The audit does not trust the JSON: `verify_rl_policy_retention` checks the
weights and training-record hashes, seed partition, every arm's records and
summary statistics, comparator selection and retention arithmetic, then
re-runs one held-out seed in both sampled and greedy modes. It is an auditable
spot reproduction and consistency check, not a full re-run of all 30 seeds.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import gymnasium as _gymnasium

    _GymBase = _gymnasium.Env
except Exception:  # pragma: no cover - gymnasium is optional for inference
    _gymnasium = None
    _GymBase = object

from danon.cure import EvidenceStatus
from danon.evosteer import (
    DEFAULT_CLONES,
    DEFAULT_DRUGS,
    AdaptiveDosePolicy,
    BasePolicy,
    CloneParams,
    CollateralSwitchPolicy,
    DrugParams,
    MTDPolicy,
    PolicyState,
    SimulationTrajectory,
    TumorEvolutionSimulator,
    TumorGeometry,
    make_gbm_phantom,
)

#: Where the trained weights live. Recorded here because the artifact's
#: integrity check hashes THIS file, not a copy.
WEIGHTS_PATH = "models/ppo_evosteer_policy.pt"

#: The measured evaluation, written by `write_rl_artifact`.
ARTIFACT_PATH = "outputs_real/rl_steering.json"

#: The training record (episodes, seeds, monitor curve, wall time).
TRAIN_RECORD_PATH = "outputs_real/rl_steering_train.json"

#: Episode length: the engine's own 365-day horizon.
HORIZON_DAYS = 365

#: Observation = [burden/b_init, R fraction, day/horizon, toxicity/tox_ref].
OBS_DIM = 4
ACTION_DIM = 2

#: The three daily schedule choices, in index order: A, B, holiday.
SCHEDULE_ACTIONS: Tuple[Tuple[float, float], ...] = (
    (1.0, 0.0),
    (0.0, 1.0),
    (0.0, 0.0),
)

#: The daily dose a `co_dose_reference` arm takes -- see the ACTION SPACE note.
CO_DOSE_REFERENCE = (0.5, 0.5)

#: Observation clipping: the burden ratio is capped so an exploding trajectory
#: cannot hand the policy an unbounded input. 12x initial burden is above every
#: run recorded here (MTD on the phantom reaches ~2.3x) but finite.
BURDEN_RATIO_CAP = 12.0

#: MLP width for the trained policy. Small on purpose: the state is 4 numbers.
HIDDEN = 64

#: Minimum monitor-partition AUBC advantage required before a rule becomes the
#: denominator for the descriptive (repository-defined) retention ratio.
MIN_MATERIAL_IMPROVEMENT_AUBC = 0.01

#: Seeds are partitioned, and the partition is recorded, so a seed used to tune
#: anything can never reappear as a held-out evaluation seed.
TRAIN_SEED_BASE = 1000
MONITOR_SEED_BASE = 900
EVAL_SEEDS = tuple(range(30))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Features and the action projection
# ---------------------------------------------------------------------------


def observation_features(
    state: PolicyState, toxicity: float, horizon: int = HORIZON_DAYS
) -> np.ndarray:
    """The agent's observation: exactly what a closed-form policy can see.

    burden/b_init and the resistant fraction after the engine's detection
    floor are the two numbers every rule in `danon.evosteer` acts on; day and
    cumulative toxicity are the agent's own history (it chose every dose that
    produced the toxicity), so no hidden simulator state is exposed.
    """
    b_init = float(state.burden_initial) if state.burden_initial > 0.0 else 1.0
    ratio = float(state.burden) / b_init
    tox_ref = float(horizon) * max(DEFAULT_DRUGS.toxicity_weight)
    return np.asarray(
        [
            min(max(ratio, 0.0), BURDEN_RATIO_CAP) / BURDEN_RATIO_CAP,
            float(state.resistant_fraction),
            float(state.day) / float(horizon),
            min(max(toxicity / tox_ref, 0.0), 1.0),
        ],
        dtype=np.float32,
    )


def schedule_dose(action: int) -> Tuple[float, float]:
    """Map a discrete schedule index to (dose_A, dose_B). Fails loudly."""
    idx = int(action)
    if not 0 <= idx < len(SCHEDULE_ACTIONS):
        raise ValueError(
            f"schedule action {action!r} is not one of {tuple(range(len(SCHEDULE_ACTIONS)))} "
            f"({SCHEDULE_ACTIONS})"
        )
    return SCHEDULE_ACTIONS[idx]


def project_dose(action: np.ndarray) -> Tuple[float, float]:
    """Clip to [0,1]^2 and scale down to dose_A+dose_B <= 1 (matched intensity).

    Scaling rather than truncating keeps the direction of the requested action
    and never discards the second drug's request entirely; the rule policies
    (1.0, 0.0) and (0.0, 1.0) are fixed points of this map.
    """
    a = np.clip(np.asarray(action, dtype=float).reshape(ACTION_DIM), 0.0, 1.0)
    total = float(a.sum())
    if total > 1.0:
        a = a / total
    return float(a[0]), float(a[1])


# ---------------------------------------------------------------------------
# The bridge policy: the engine drives it, the environment answers
# ---------------------------------------------------------------------------


class _BridgePolicy(BasePolicy):
    """A policy whose dose decisions arrive from another thread.

    The engine calls `dose(state)` once per simulated day, from its own loop.
    This object publishes that state on `obs_q` and BLOCKS until the
    environment puts an (already projected) action on `act_q`. The simulator
    code is untouched, so the trajectory is the engine's own.
    """

    kind = MTDPolicy.kind

    def __init__(self, obs_q: "queue.Queue", act_q: "queue.Queue") -> None:
        self._obs_q = obs_q
        self._act_q = act_q

    def reset(self) -> None:  # the engine resets policies per run
        return None

    def dose(self, state: PolicyState) -> Tuple[float, float]:
        self._obs_q.put(state)
        action = self._act_q.get(timeout=600.0)
        return float(action[0]), float(action[1])

    def describe(self) -> str:
        return "bridge(agent)"


class EvoSteerEnv(_GymBase):
    """Gymnasium-shaped environment over the real engine (day-by-day stepping).

    API: `reset(seed)` -> (obs, info); `step(action)` -> (obs, reward,
    terminated, truncated, info). Gymnasium is imported lazily ONLY to attach
    `observation_space`/`action_space` metadata (it is not in
    `requirements.txt`); training does not need it.

    `action_mode='schedule'` (default) takes an integer index into
    `SCHEDULE_ACTIONS`; `action_mode='continuous'` takes (dose_A, dose_B) and
    projects it. The two modes are separate because they are not comparable --
    see the ACTION SPACE note in the module docstring.

    The engine runs in a worker thread because `TumorEvolutionSimulator.run`
    is a whole-trajectory call: the bridge policy turns it into a per-day
    loop without duplicating one line of engine arithmetic.
    """

    def __init__(
        self,
        *,
        days: int = HORIZON_DAYS,
        geometry: Optional[TumorGeometry] = None,
        geometry_seed: int = 0,
        clones: CloneParams = DEFAULT_CLONES,
        drugs: DrugParams = DEFAULT_DRUGS,
        initial_resistant_fraction: float = 0.01,
        growth_noise: float = 0.05,
        action_mode: str = "schedule",
    ) -> None:
        super().__init__()
        if days < 2:
            raise ValueError("need at least 2 days")
        if action_mode not in ("schedule", "continuous"):
            raise ValueError(
                f"action_mode must be 'schedule' or 'continuous', got {action_mode!r}"
            )
        self.action_mode = action_mode
        self.days = int(days)
        self.geometry = geometry if geometry is not None else make_gbm_phantom(seed=geometry_seed)
        self.geometry_seed = int(geometry_seed)
        self.sim = TumorEvolutionSimulator(self.geometry, clones=clones, drugs=drugs)
        self.initial_resistant_fraction = float(initial_resistant_fraction)
        self.growth_noise = float(growth_noise)
        self._geometry = geometry
        self.trajectory: Optional[SimulationTrajectory] = None
        self._thread: Optional[threading.Thread] = None
        self._obs_q: Optional["queue.Queue"] = None
        self._act_q: Optional["queue.Queue"] = None
        self._pending: Optional[PolicyState] = None
        self._day = -1
        self._toxicity = 0.0
        self._b_init = 1.0
        if _gymnasium is not None:
            self.observation_space = _gymnasium.spaces.Box(
                low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
            )
            self.action_space = (
                _gymnasium.spaces.Discrete(len(SCHEDULE_ACTIONS))
                if self.action_mode == "schedule"
                else _gymnasium.spaces.Box(
                    low=0.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
                )
            )
            self.metadata = {}
        else:  # pragma: no cover - gymnasium absent
            self.observation_space = None
            self.action_space = None

    # -- worker -------------------------------------------------------------

    def _run(self, seed: int) -> None:
        policy = _BridgePolicy(self._obs_q, self._act_q)  # type: ignore[arg-type]
        try:
            self.trajectory = self.sim.run(
                policy,
                days=self.days,
                initial_resistant_fraction=self.initial_resistant_fraction,
                seed=int(seed),
                growth_noise=self.growth_noise,
            )
        finally:
            self._obs_q.put(None)  # type: ignore[union-attr]

    def _next_state(self) -> Optional[PolicyState]:
        assert self._obs_q is not None
        st = self._obs_q.get(timeout=600.0)
        return st

    # -- gymnasium-shaped API ----------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if _gymnasium is not None:
            super().reset(seed=seed)
        if seed is None:
            seed = 0
        self.trajectory = None
        self._toxicity = 0.0
        self._obs_q = queue.Queue()
        self._act_q = queue.Queue()
        self._thread = threading.Thread(target=self._run, args=(int(seed),), daemon=True)
        self._thread.start()
        st = self._next_state()
        if st is None:  # pragma: no cover - a zero-day run cannot happen
            raise RuntimeError("engine finished before the first dose decision")
        self._pending = st
        self._day = 0
        self._b_init = float(st.burden_initial) if st.burden_initial > 0 else 1.0
        return observation_features(st, self._toxicity, self.days), self._info()

    def step(self, action):
        if self._pending is None:
            raise RuntimeError("step() before reset()")
        dose_a, dose_b = (
            schedule_dose(action)
            if self.action_mode == "schedule"
            else project_dose(action)
        )
        assert self._act_q is not None
        self._act_q.put((dose_a, dose_b))
        self._toxicity += (
            dose_a * self.sim.drugs.toxicity_weight[0]
            + dose_b * self.sim.drugs.toxicity_weight[1]
        )
        # Reward charges the day the action was TAKEN ON (the state the agent
        # saw), not the next one. Summed over the 365 steps that is exactly
        # -365 * AUBC, day-0 term included, which is why the training return
        # and the reported metric are the same functional rather than merely
        # close; it is computed before the next state arrives so that the last
        # day is charged as well.
        reward = -float(self._pending.burden) / self._b_init
        st = self._next_state()
        terminated = st is None
        if terminated:
            if self._thread is not None:
                self._thread.join(timeout=600.0)
            info = self._info(closed=True)
            return np.zeros(OBS_DIM, dtype=np.float32), reward, True, False, info
        self._pending = st
        self._day += 1
        return observation_features(st, self._toxicity, self.days), reward, False, False, self._info()

    def _info(self, closed: bool = False) -> Dict[str, object]:
        info: Dict[str, object] = {
            "day": self._day,
            "toxicity": self._toxicity,
            "policy": "bridge(agent)",
        }
        if closed and self.trajectory is not None:
            traj = self.trajectory
            info["aubc"] = aubc(traj)
            info["final_burden"] = float(traj.burden_S[-1] + traj.burden_R[-1])
            info["pfs_day"] = traj.pfs_day
            info["toxicity"] = float(traj.toxicity)
        return info


# ---------------------------------------------------------------------------
# Metrics, shared by training and evaluation
# ---------------------------------------------------------------------------


def aubc(traj: SimulationTrajectory) -> float:
    """Area under the normalised burden curve: mean(burden_t)/burden_0.

    burden_0 is the engine's own recorded day-0 burden, so this is the exact
    quantity the training return integrates (gamma = 1.0).
    """
    series = np.asarray(traj.burden_S, dtype=float) + np.asarray(traj.burden_R, dtype=float)
    b0 = float(series[0]) if series.size and series[0] > 0 else 1.0
    return float(np.mean(series) / b0)


@dataclass(frozen=True)
class SeedResult:
    seed: int
    aubc: float
    final_burden_ratio: float
    pfs_days: float
    progressed: bool
    toxicity: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "seed": self.seed,
            "aubc": round(self.aubc, 6),
            "final_burden_ratio": round(self.final_burden_ratio, 6),
            "pfs_days": round(self.pfs_days, 3),
            "progressed": bool(self.progressed),
            "toxicity": round(self.toxicity, 3),
        }


def run_seed(
    sim: TumorEvolutionSimulator,
    policy: BasePolicy,
    *,
    seed: int,
    days: int = HORIZON_DAYS,
    initial_resistant_fraction: float = 0.01,
    growth_noise: float = 0.05,
) -> SeedResult:
    """One deterministic replicate -> the per-seed record the audit re-checks."""
    traj = sim.run(
        policy,
        days=days,
        initial_resistant_fraction=initial_resistant_fraction,
        seed=int(seed),
        growth_noise=growth_noise,
    )
    b0 = float(traj.burden_S[0] + traj.burden_R[0])
    final = float(traj.burden_S[-1] + traj.burden_R[-1])
    return SeedResult(
        seed=int(seed),
        aubc=aubc(traj),
        final_burden_ratio=final / b0 if b0 > 0 else float("nan"),
        pfs_days=float(traj.pfs_value(days)),
        progressed=bool(traj.progressed),
        toxicity=float(traj.toxicity),
    )


def _median(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def summarize(records: Sequence[SeedResult]) -> Dict[str, object]:
    if not records:
        raise ValueError("summarize needs at least one seed record")
    return {
        "n_seeds": len(records),
        "median_aubc": round(_median([r.aubc for r in records]), 6),
        "mean_aubc": round(float(np.mean([r.aubc for r in records])), 6),
        "median_final_burden_ratio": round(
            _median([r.final_burden_ratio for r in records]), 6
        ),
        "median_pfs_days": round(_median([r.pfs_days for r in records]), 3),
        "frac_progressed": round(float(np.mean([r.progressed for r in records])), 4),
        "median_toxicity": round(_median([r.toxicity for r in records]), 3),
        "records": [r.as_dict() for r in records],
    }


# ---------------------------------------------------------------------------
# The trained policy
# ---------------------------------------------------------------------------


class TrainedRLPolicy(BasePolicy):
    """A learned dosing policy loaded from `models/ppo_evosteer_policy.pt`.

    TWO EVALUATION MODES, because the trained policy is a distribution and the
    two forms behave differently (measured, and recorded in the artifact):

      * mode='greedy': act on the argmax of the action logits. This is the
        deterministic state-feedback rule, i.e. the class the closed-form
        policies live in.
      * mode='sample': draw the day's action from softmax(logits) with a
        generator seeded from the replicate seed. This is the policy PPO
        actually optimised, and it is reproducible: the same seed draws the
        same actions, which is what lets the audit re-measure it live.

    'continuous' checkpoints act on the MLP's mean in both modes (projected).
    torch runs on one thread so repeated evaluations are identical, and
    `reset()` clears the cumulative-toxicity feature, which -- besides the
    sampling generator -- is the policy's only state.
    """

    kind = MTDPolicy.kind

    def __init__(
        self,
        weights_path: str = WEIGHTS_PATH,
        mode: str = "greedy",
        seed: int = 0,
    ) -> None:
        import torch  # lazy: the audit must be able to report "torch absent"

        self.weights_path = weights_path
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"trained weights not found at {weights_path}")
        torch.set_num_threads(1)
        blob = torch.load(weights_path, map_location="cpu", weights_only=False)
        if str(blob.get("algo", "")).lower() != "ppo":
            raise ValueError(
                f"weights at {weights_path} are not tagged as a PPO checkpoint; refusing to call them trained PPO weights"
            )
        self.action_mode = str(blob.get("action_mode", "schedule"))
        n_out = (
            len(SCHEDULE_ACTIONS) if self.action_mode == "schedule" else ACTION_DIM
        )
        if int(blob.get("obs_dim", -1)) != OBS_DIM or int(blob.get("n_out", -1)) != n_out:
            raise ValueError(
                f"weights at {weights_path} declare obs/n_out dims "
                f"{blob.get('obs_dim')}/{blob.get('n_out')}, expected "
                f"{OBS_DIM}/{n_out} for action_mode={self.action_mode}: refusing to guess"
            )
        hidden = int(blob.get("hidden", HIDDEN))
        net = torch.nn.Sequential(
            torch.nn.Linear(OBS_DIM, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, n_out),
        )
        net.load_state_dict(blob["state_dict"])
        net.eval()
        if mode not in ("greedy", "sample"):
            raise ValueError(f"mode must be 'greedy' or 'sample', got {mode!r}")
        self.mode = mode
        self._torch = torch
        self._net = net
        self._hidden = hidden
        self._toxicity = 0.0
        self._sha = _sha256(weights_path)
        self._seed = int(seed)
        self._gen = torch.Generator().manual_seed(self._seed) if mode == "sample" else None

    @property
    def weights_sha256(self) -> str:
        return self._sha

    def reset(self) -> None:
        self._toxicity = 0.0
        if self.mode == "sample":
            # per-replicate reseed: the same replicate seed always draws the
            # same action sequence, so the audit can re-measure it exactly
            self._gen = self._torch.Generator().manual_seed(self._seed)

    def dose(self, state: PolicyState) -> Tuple[float, float]:
        obs = observation_features(state, self._toxicity)
        with self._torch.no_grad():
            out = self._net(self._torch.from_numpy(obs).unsqueeze(0))[0]
            if self.action_mode == "schedule":
                if self.mode == "sample":
                    probs = self._torch.softmax(out, dim=-1)
                    idx = int(self._torch.multinomial(probs, 1, generator=self._gen)[0])
                else:
                    idx = int(self._torch.argmax(out))
                dose_a, dose_b = schedule_dose(idx)
            else:
                dose_a, dose_b = project_dose(out.numpy())
        self._toxicity += (
            dose_a * DEFAULT_DRUGS.toxicity_weight[0]
            + dose_b * DEFAULT_DRUGS.toxicity_weight[1]
        )
        return dose_a, dose_b

    def describe(self) -> str:
        return (
            f"trained(ppo/{self.action_mode}/{self.mode}, "
            f"{os.path.basename(self.weights_path)}, sha8={self._sha[:8]})"
        )


# ---------------------------------------------------------------------------
# Evaluation: the agent against the three closed-form rules, same seeds
# ---------------------------------------------------------------------------

#: The closed-form arms. MTD is the standard-care anchor; the other two are the
#: repository's own published policies, and the "best rule" is picked on the
#: MONITOR seeds only (see `evaluate_all`).
RULE_POLICIES: Dict[str, type] = {
    "mtd": MTDPolicy,
    "adaptive_dose": AdaptiveDosePolicy,
    "collateral_switch": CollateralSwitchPolicy,
}


class _FixedDosePolicy(BasePolicy):
    """A constant dose every day -- only used to exhibit the co-dosing confound."""

    kind = MTDPolicy.kind

    def __init__(self, dose=(CO_DOSE_REFERENCE[0], CO_DOSE_REFERENCE[1])) -> None:
        self._dose = (float(dose[0]), float(dose[1]))

    def dose(self, state: PolicyState) -> Tuple[float, float]:
        return self._dose

    def describe(self) -> str:
        return f"constant({self._dose[0]:.2f}, {self._dose[1]:.2f})"


class _AlternatingPolicy(BasePolicy):
    """Drug A on even days, drug B on odd days: DETERMINISTIC alternation.

    This arm exists to separate two explanations of the learned policy's
    behaviour. If deterministic alternation also reaches the model's floor,
    then the floor is reachable without randomness and the learned policy's
    greedy collapse is an optimisation failure. If it does not, the model
    itself pays for RANDOM timing, and 'the agent's edge is randomization' is a
    property of the model rather than of the training run.
    """

    kind = MTDPolicy.kind

    def dose(self, state: PolicyState) -> Tuple[float, float]:
        return (1.0, 0.0) if int(state.day) % 2 == 0 else (0.0, 1.0)

    def describe(self) -> str:
        return "alternate(A/B daily)"


#: Fixed reference arms, NOT learned and NOT comparators: they exist to make the
#: action-space confound measurable rather than arguable.
REFERENCE_POLICIES: Dict[str, type] = {
    "co_dose_reference": lambda: _FixedDosePolicy(CO_DOSE_REFERENCE),
    "alternate_reference": _AlternatingPolicy,
}


def evaluate_all(
    *,
    seeds: Sequence[int] = EVAL_SEEDS,
    weights_path: str = WEIGHTS_PATH,
    days: int = HORIZON_DAYS,
    geometry_seed: int = 0,
    initial_resistant_fraction: float = 0.01,
    growth_noise: float = 0.05,
    selection_seeds: Sequence[int] = tuple(range(MONITOR_SEED_BASE, MONITOR_SEED_BASE + 10)),
    progress: bool = False,
) -> Dict[str, object]:
    """Evaluate the trained agent and every rule policy on identical seeds.

    The best rule is selected on `selection_seeds` (the monitor partition,
    which the agent never trained on) so that no comparator is chosen on the
    evaluation seeds.
    """
    geom = make_gbm_phantom(seed=geometry_seed)
    sim = TumorEvolutionSimulator(geom)

    if not seeds or len(set(int(s) for s in seeds)) != len(seeds):
        raise ValueError("evaluation seeds must be non-empty and unique")
    if not selection_seeds or len(set(int(s) for s in selection_seeds)) != len(selection_seeds):
        raise ValueError("selection seeds must be non-empty and unique")
    if set(int(s) for s in seeds).intersection(int(s) for s in selection_seeds):
        raise ValueError("selection and evaluation seed partitions must be disjoint")

    def _eval(policy: BasePolicy, seed_list: Sequence[int]) -> List[SeedResult]:
        out: List[SeedResult] = []
        for s in seed_list:
            out.append(
                run_seed(
                    sim,
                    policy,
                    seed=s,
                    days=days,
                    initial_resistant_fraction=initial_resistant_fraction,
                    growth_noise=growth_noise,
                )
            )
            if progress:
                print(f"  {policy.describe()} seed {s}", flush=True)
        return out

    arms: Dict[str, object] = {}
    # selection pass (monitor partition) for the rule arms only
    selection_scores: Dict[str, float] = {}
    selection_records: Dict[str, List[Dict[str, object]]] = {}
    for name, cls in RULE_POLICIES.items():
        recs = _eval(cls(), selection_seeds)
        selection_scores[name] = _median([r.aubc for r in recs])
        selection_records[name] = [r.as_dict() for r in recs]
    # A rule that beats MTD by less than 0.01 AUBC (less than one percent of
    # the initial burden integrated over a 365-day horizon) is practically a
    # tie for this model. Do not divide by a tiny accidental gap and call it
    # 'retention': in that case there is no material improvement to retain.
    min_material_gap = MIN_MATERIAL_IMPROVEMENT_AUBC
    mtd_selection = selection_scores["mtd"]
    material_rules = {
        k: v for k, v in selection_scores.items()
        if k != "mtd" and (mtd_selection - v) >= min_material_gap
    }
    best_rule = (
        min(material_rules, key=lambda k: (material_rules[k], k))
        if material_rules
        else "mtd"
    )

    # evaluation pass (held-out seeds): agent + every rule arm
    # BOTH forms of the trained policy. The sampled form is the distribution
    # PPO optimised; the greedy form is the deterministic state-feedback rule
    # the closed-form arms also are. They can differ a lot -- the first run of
    # `_train_evosteer_ppo.py` sampled to AUBC ~0.021 and projected to ~1.24 --
    # and the difference is itself a recorded result, not a footnoot.
    agent_arms: Dict[str, Dict[str, object]] = {}
    for mode in ("greedy", "sample"):
        recs: List[SeedResult] = []
        for s in seeds:
            policy = TrainedRLPolicy(weights_path, mode=mode, seed=int(s))
            recs.append(
                run_seed(
                    sim,
                    policy,
                    seed=s,
                    days=days,
                    initial_resistant_fraction=initial_resistant_fraction,
                    growth_noise=growth_noise,
                )
            )
            if progress:
                print(f"  trained/{mode} seed {s}", flush=True)
        agent_arms[mode] = {
            **summarize(recs),
            "weights_sha256": TrainedRLPolicy(weights_path, mode=mode).weights_sha256,
            "mode": mode,
            "action_mode": TrainedRLPolicy(weights_path, mode=mode).action_mode,
        }
    arms["ppo_agent_greedy"] = agent_arms["greedy"]
    arms["ppo_agent_sampled"] = agent_arms["sample"]
    for name, cls in RULE_POLICIES.items():
        arms[name] = summarize(_eval(cls(), seeds))
    for name, factory in REFERENCE_POLICIES.items():
        arms[name] = summarize(_eval(factory(), seeds))

    mtd = arms["mtd"]["median_aubc"]  # type: ignore[index]
    best_rule_aubc = arms[best_rule]["median_aubc"]  # type: ignore[index]
    denom = float(mtd) - float(best_rule_aubc)

    def _retention(arm: str) -> Optional[float]:
        if denom <= 0.0:
            return None
        aubc = float(arms[arm]["median_aubc"])  # type: ignore[index]
        return (float(mtd) - aubc) / denom

    retention = _retention("ppo_agent_sampled")
    retention_greedy = _retention("ppo_agent_greedy")

    mtd_final = arms["mtd"]["median_final_burden_ratio"]  # type: ignore[index]
    agent_final = arms["ppo_agent_sampled"]["median_final_burden_ratio"]  # type: ignore[index]
    best_final = arms[best_rule]["median_final_burden_ratio"]  # type: ignore[index]
    denom_f = float(mtd_final) - float(best_final)
    retention_final = (
        (float(mtd_final) - float(agent_final)) / denom_f if denom_f > 0.0 else None
    )

    return {
        "seeds": [int(s) for s in seeds],
        "selection_seeds": [int(s) for s in selection_seeds],
        "days": days,
        "initial_resistant_fraction": float(initial_resistant_fraction),
        "growth_noise": float(growth_noise),
        "geometry": {
            "kind": "gbm_phantom",
            "seed": geometry_seed,
            "grid": list(geom.tumor_mask.shape),
            "tumor_voxels": int(geom.tumor_mask.sum()),
        },
        "arms": arms,
        "best_rule": best_rule,
        "best_rule_selected_on": {
            "partition": "monitor",
            "median_aubc": {k: round(v, 6) for k, v in selection_scores.items()},
            "records": selection_records,
            "minimum_material_improvement_aubc": min_material_gap,
            "selection_rule": (
                "choose the best non-MTD rule only if its median AUBC improves "
                "over MTD by at least 0.01; otherwise set best_rule=MTD and the "
                "retention denominator is zero (not estimable)"
            ),
        },
        "co_dose_confound": {
            "arm": "co_dose_reference",
            "dose": list(CO_DOSE_REFERENCE),
            "median_aubc": arms["co_dose_reference"]["median_aubc"],  # type: ignore[index]
            "role": (
                "a fixed (not learned) daily co-dose. If it beats every rule and "
                "the agent, the advantage belongs to the action space, not to "
                "learning -- the engine's toxicity proxy is a log, not a constraint"
            ),
        },
        "alternation_reference": {
            "arm": "alternate_reference",
            "median_aubc": arms["alternate_reference"]["median_aubc"],  # type: ignore[index]
            "role": (
        "fixed, deterministic A/B alternation reference, included to test "
        "whether a learned policy is necessary for the simulator outcome; "
        "it is not a trained or adaptive policy"
            ),
        },
        "retention": {
            "definition": (
                "(MTD_AUBC - agent_AUBC) / (MTD_AUBC - best_rule_AUBC); "
                "1.0 = matched the best closed-form rule, 0.0 = no better than MTD"
            ),
            "headline_mode": "sample",
            "aubc": None if retention is None else round(retention, 4),
            "aubc_greedy": None if retention_greedy is None else round(retention_greedy, 4),
            "final_burden": None if retention_final is None else round(retention_final, 6),
        },
        "determinism_gap": {
            "sampled_median_aubc": arms["ppo_agent_sampled"]["median_aubc"],  # type: ignore[index]
            "greedy_median_aubc": arms["ppo_agent_greedy"]["median_aubc"],  # type: ignore[index]
            "note": (
                "the trained object is a POLICY DISTRIBUTION. Its sampled form is "
                "what PPO optimised; projecting it to its own argmax changes the "
                "number, and that gap is the honest boundary of 'the agent learned "
                "a scheduler'"
            ),
        },
    }


def load_artifact(path: str = ARTIFACT_PATH) -> Optional[Dict[str, object]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_rl_artifact(
    *,
    seeds: Sequence[int] = EVAL_SEEDS,
    weights_path: str = WEIGHTS_PATH,
    path: str = ARTIFACT_PATH,
    progress: bool = False,
) -> Dict[str, object]:
    """Measure, write the JSON, and return it. The only writer of the artifact."""
    payload = evaluate_all(seeds=seeds, weights_path=weights_path, progress=progress)
    payload.update(
        {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "danon/rl_steering.py::evaluate_all",
            "weights_path": weights_path,
            "weights_sha256": _sha256(weights_path),
            "provenance": {k: v.value for k, v in rl_steering_provenance().items()},
            "cannot_show": what_rl_cannot_show(),
        }
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    return payload


# ---------------------------------------------------------------------------
# The audit-facing live check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetentionVerification:
    ok: bool
    detail: str
    n_seeds: int = 0
    retention_aubc: Optional[float] = None
    rechecked_seed: Optional[int] = None


def verify_rl_policy_retention(
    *, artifact_path: str = ARTIFACT_PATH, weights_path: str = WEIGHTS_PATH
) -> RetentionVerification:
    """Spot-reproduce and consistency-check the recorded held-out evaluation.

    Checks the checkpoint and training-record hashes, the disjoint train /
    monitor / evaluation seed partitions, all seven arms' per-seed records and
    summaries, monitor-only rule selection, and sampled/greedy retention
    arithmetic. It then re-runs the first held-out seed from the weights in
    both policy modes. This catches a swapped checkpoint, an internally
    inconsistent or doctored summary, and a mismatched first episode; it does
    not re-run all 30 seeds, prove scientific generalization, or validate the
    author-defined metric clinically.
    """
    from danon.evosteer import TumorEvolutionSimulator, make_gbm_phantom

    try:
        art = load_artifact(artifact_path)
    except Exception as exc:
        return RetentionVerification(False, f"could not read artifact: {exc}")
    if art is None:
        return RetentionVerification(False, f"no artifact at {artifact_path}")
    if not isinstance(art, dict):
        return RetentionVerification(False, "artifact root must be a JSON object")
    if not os.path.exists(weights_path):
        return RetentionVerification(False, f"no weights at {weights_path}")
    sha = _sha256(weights_path)
    if sha != art.get("weights_sha256"):
        return RetentionVerification(
            False,
            f"weights hash mismatch: file {sha[:12]} vs artifact "
            f"{str(art.get('weights_sha256'))[:12]}",
        )

    try:
        with open(TRAIN_RECORD_PATH, "r", encoding="utf-8") as fh:
            training = json.load(fh)
    except (OSError, ValueError) as exc:
        return RetentionVerification(False, f"training record missing or invalid: {exc}")
    if training.get("weights_sha256") != sha:
        return RetentionVerification(
            False,
            "training-record weights hash does not match the checkpoint on disk",
        )

    try:
        eval_seeds = [int(s) for s in art["seeds"]]
        selection_seeds = [int(s) for s in art["selection_seeds"]]
        record_partition = training["seed_partition"]
        record_eval_seeds = [int(s) for s in record_partition["evaluation"]]
        monitor_seeds = [int(s) for s in record_partition["monitor"]]
        iterations = int(training["iterations"])
        n_envs = int(training["envs"])
        days = int(art["days"])
        initial_resistant_fraction = float(art.get("initial_resistant_fraction", 0.01))
        growth_noise = float(art.get("growth_noise", 0.05))
        geometry_seed = int((art.get("geometry") or {}).get("seed", 0))
    except (KeyError, TypeError, ValueError) as exc:
        return RetentionVerification(False, f"artifact/training metadata incomplete: {exc}")

    if len(eval_seeds) < 20 or len(set(eval_seeds)) != len(eval_seeds):
        return RetentionVerification(False, "evaluation seeds must be unique and contain at least 20 replicates")
    if eval_seeds != record_eval_seeds:
        return RetentionVerification(False, "artifact evaluation seeds differ from the training record")
    if not selection_seeds or len(set(selection_seeds)) != len(selection_seeds):
        return RetentionVerification(False, "monitor/selection seeds must be non-empty and unique")
    if selection_seeds != monitor_seeds:
        return RetentionVerification(False, "rule-selection seeds differ from the training record's monitor partition")
    if len(selection_seeds) > 10:
        return RetentionVerification(False, "monitor partition exceeds the supported 10-seed budget")
    try:
        train_base = int(str(record_partition["training"]).rstrip("+"))
    except (KeyError, TypeError, ValueError) as exc:
        return RetentionVerification(False, f"training seed base invalid: {exc}")
    train_seeds = {
        train_base + iteration * n_envs + env_index
        for iteration in range(1, iterations + 1)
        for env_index in range(n_envs)
    }
    if train_seeds.intersection(eval_seeds) or train_seeds.intersection(selection_seeds):
        return RetentionVerification(False, "training seeds overlap monitor or held-out evaluation seeds")
    if set(eval_seeds).intersection(selection_seeds):
        return RetentionVerification(False, "monitor and held-out evaluation seeds overlap")
    if iterations <= 0 or n_envs <= 0:
        return RetentionVerification(False, "training record has invalid iteration/environment counts")
    if int(training.get("steps_total", -1)) != iterations * n_envs * HORIZON_DAYS:
        return RetentionVerification(False, "training step count does not match iterations x envs x horizon")
    if not str(training.get("algorithm", "")).startswith("ppo "):
        return RetentionVerification(False, "training record does not identify the PPO algorithm")
    if int(training.get("horizon_days", -1)) != HORIZON_DAYS:
        return RetentionVerification(False, "training record horizon does not match the environment horizon")
    if days != HORIZON_DAYS:
        return RetentionVerification(False, f"unsupported evaluation horizon {days}; expected {HORIZON_DAYS}")
    if initial_resistant_fraction != 0.01 or growth_noise != 0.05:
        return RetentionVerification(False, "artifact evaluation parameters differ from the recorded default protocol")
    geometry = art.get("geometry") or {}
    geom = make_gbm_phantom(seed=geometry_seed)
    if geometry.get("kind") != "gbm_phantom" or list(geom.tumor_mask.shape) != list(geometry.get("grid", [])):
        return RetentionVerification(False, "artifact geometry does not match the supported GBM phantom")
    if int(geometry.get("tumor_voxels", -1)) != int(geom.tumor_mask.sum()):
        return RetentionVerification(False, "artifact phantom tumor-voxel count does not match the live geometry")

    arms = art.get("arms") or {}
    arm_names = {
        "ppo_agent_sampled", "ppo_agent_greedy", *RULE_POLICIES.keys(), *REFERENCE_POLICIES.keys()
    }
    if not isinstance(arms, dict) or set(arms) != arm_names:
        return RetentionVerification(False, f"artifact arm set mismatch: {sorted(arms) if isinstance(arms, dict) else type(arms).__name__}")
    summary_fields = {
        "median_aubc": ("aubc", 6),
        "mean_aubc": ("aubc", 6),
        "median_final_burden_ratio": ("final_burden_ratio", 6),
        "median_pfs_days": ("pfs_days", 3),
        "frac_progressed": ("progressed", 4),
        "median_toxicity": ("toxicity", 3),
    }
    for arm_name in sorted(arm_names):
        arm = arms[arm_name]
        if not isinstance(arm, dict):
            return RetentionVerification(False, f"{arm_name} arm must be a JSON object")
        records = arm.get("records") or []
        if int(arm.get("n_seeds", -1)) != len(eval_seeds) or len(records) != len(eval_seeds):
            return RetentionVerification(False, f"{arm_name} record count does not match held-out seeds")
        if [int(row.get("seed", -1)) for row in records] != eval_seeds:
            return RetentionVerification(False, f"{arm_name} per-seed order/ids differ from held-out seeds")
        try:
            for row in records:
                if not all(np.isfinite(float(row[k])) for k in ("aubc", "final_burden_ratio", "pfs_days", "toxicity")):
                    return RetentionVerification(False, f"{arm_name} has non-finite per-seed values")
                if not isinstance(row.get("progressed"), bool):
                    return RetentionVerification(False, f"{arm_name} has invalid progression flags")
            for field_name, (record_key, digits) in summary_fields.items():
                values = [float(row[record_key]) for row in records]
                if record_key == "progressed":
                    actual = round(float(np.mean(values)), digits)
                elif field_name == "mean_aubc":
                    actual = round(float(np.mean(values)), digits)
                else:
                    actual = round(float(np.median(values)), digits)
                stored = float(arm[field_name])
                if not np.isclose(actual, stored, rtol=0, atol=1.1e-6):
                    return RetentionVerification(
                        False,
                        f"{arm_name} summary {field_name}={stored} does not match "
                        f"its per-seed records ({actual})",
                    )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            return RetentionVerification(False, f"{arm_name} summary/records invalid: {exc}")

    for mode in ("sampled", "greedy"):
        arm = arms[f"ppo_agent_{mode}"]
        if arm.get("mode") != ("sample" if mode == "sampled" else "greedy"):
            return RetentionVerification(False, f"PPO {mode} arm has the wrong policy mode")
        if arm.get("weights_sha256") != sha:
            return RetentionVerification(False, f"PPO {mode} arm checkpoint hash mismatch")
    sampled_recs = arms["ppo_agent_sampled"]["records"]
    greedy_recs = arms["ppo_agent_greedy"]["records"]
    if not sampled_recs or not greedy_recs:
        return RetentionVerification(False, "artifact must carry per-seed records for sampled and greedy PPO")

    selection = art.get("best_rule_selected_on") or {}
    if selection.get("partition") != "monitor":
        return RetentionVerification(False, "best closed-form rule was not selected on the monitor partition")
    try:
        selection_scores = {
            str(k): float(v)
            for k, v in selection["median_aubc"].items()
        }
        if set(selection_scores) != set(RULE_POLICIES):
            return RetentionVerification(False, "monitor-selection scores do not cover exactly the closed-form rules")
        selection_records = selection.get("records")
        if selection_records is not None:
            if not isinstance(selection_records, dict) or set(selection_records) != set(RULE_POLICIES):
                return RetentionVerification(False, "monitor records do not cover exactly the closed-form rules")
            for rule_name, rows in selection_records.items():
                if [int(row.get("seed", -1)) for row in rows] != selection_seeds:
                    return RetentionVerification(False, f"{rule_name} monitor records differ from the declared monitor seeds")
                measured_score = float(np.median([float(row["aubc"]) for row in rows]))
                if not np.isclose(
                    measured_score,
                    selection_scores[rule_name],
                    rtol=0,
                    atol=1.1e-6,
                ):
                    return RetentionVerification(False, f"{rule_name} monitor AUBC does not match its per-seed records")
        min_gap = float(
            selection.get(
                "minimum_material_improvement_aubc",
                MIN_MATERIAL_IMPROVEMENT_AUBC,
            )
        )

        if not np.isclose(min_gap, MIN_MATERIAL_IMPROVEMENT_AUBC, rtol=0, atol=1e-12):
            return RetentionVerification(False, "artifact uses an unexpected material-improvement threshold")
        mtd_selection = selection_scores["mtd"]
        eligible = {
            name: value for name, value in selection_scores.items()
            if name != "mtd" and mtd_selection - value >= min_gap
        }
        selected = min(eligible, key=lambda k: (eligible[k], k)) if eligible else "mtd"
        if art.get("best_rule") != selected:
            return RetentionVerification(False, f"selected best_rule {art.get('best_rule')} disagrees with monitor scores ({selected})")
        # New artifacts also carry per-seed monitor records and must reproduce
        # each score from those records. Older artifacts from the same run may
        # omit those records; the monitor medians remain recorded provenance,
        # while held-out arms are checked below for internal consistency and a
        # sampled/greedy episode is re-run live. Avoid re-running all 30 monitor
        # simulations inside every claims audit.
    except (KeyError, TypeError, ValueError) as exc:
        return RetentionVerification(False, f"monitor-selection record invalid: {exc}")

    mtd_aubc = float(arms["mtd"]["median_aubc"])
    best_rule_aubc = float(arms[selected]["median_aubc"])
    denom = mtd_aubc - best_rule_aubc
    retention = None
    greedy_retention = None
    if denom > 0:
        retention = round((mtd_aubc - float(arms["ppo_agent_sampled"]["median_aubc"])) / denom, 4)
        greedy_retention = round((mtd_aubc - float(arms["ppo_agent_greedy"]["median_aubc"])) / denom, 4)
    stored_ret = (art.get("retention") or {})
    if stored_ret.get("aubc") != retention or stored_ret.get("aubc_greedy") != greedy_retention:
        return RetentionVerification(False, "stored sampled/greedy retention does not match the artifact's AUBC arms")
    final_denom = float(arms["mtd"]["median_final_burden_ratio"]) - float(arms[selected]["median_final_burden_ratio"])
    final_retention = None
    if final_denom > 0:
        final_retention = round(
            (float(arms["mtd"]["median_final_burden_ratio"]) - float(arms["ppo_agent_sampled"]["median_final_burden_ratio"])) / final_denom,
            6,
        )
    if stored_ret.get("final_burden") != final_retention:
        return RetentionVerification(False, "stored final-burden retention does not match its arm summaries")
    if (art.get("determinism_gap") or {}).get("sampled_median_aubc") != arms["ppo_agent_sampled"]["median_aubc"] or (art.get("determinism_gap") or {}).get("greedy_median_aubc") != arms["ppo_agent_greedy"]["median_aubc"]:
        return RetentionVerification(False, "determinism-gap summaries do not match sampled/greedy arms")
    if (art.get("alternation_reference") or {}).get("median_aubc") != arms["alternate_reference"]["median_aubc"]:
        return RetentionVerification(False, "alternation-reference summary disagrees with its per-seed arm")
    if (art.get("co_dose_confound") or {}).get("median_aubc") != arms["co_dose_reference"]["median_aubc"]:
        return RetentionVerification(False, "co-dose reference summary disagrees with its per-seed arm")

    try:
        TrainedRLPolicy(weights_path)
    except ImportError as exc:
        return RetentionVerification(False, f"torch unavailable: {exc}")
    except Exception as exc:
        return RetentionVerification(False, f"weights cannot be loaded: {exc}")
    stored = sampled_recs[0]
    seed = int(stored["seed"])
    agent = TrainedRLPolicy(weights_path, mode="sample", seed=seed)
    days = int(art.get("days", HORIZON_DAYS))
    grid = ((art.get("geometry") or {}).get("grid")) or [24, 24, 24]
    geom = make_gbm_phantom(seed=int((art.get("geometry") or {}).get("seed", 0)))
    if list(geom.tumor_mask.shape) != list(grid):
        return RetentionVerification(
            False,
            f"geometry mismatch: artifact says {grid}, phantom is "
            f"{list(geom.tumor_mask.shape)}",
        )
    got = run_seed(
        TumorEvolutionSimulator(geom),
        agent,
        seed=seed,
        days=days,
        initial_resistant_fraction=initial_resistant_fraction,
        growth_noise=growth_noise,
    )
    stored_precision = {"aubc": 6, "final_burden_ratio": 6, "pfs_days": 3, "toxicity": 3}
    for key, mine in (
        ("aubc", got.aubc),
        ("final_burden_ratio", got.final_burden_ratio),
        ("pfs_days", got.pfs_days),
        ("toxicity", got.toxicity),
    ):
        want = float(stored[key])
        if round(float(mine), stored_precision[key]) != want:
            return RetentionVerification(
                False,
                f"seed {seed} {key} re-measured {mine!r} != stored {want!r}",
                rechecked_seed=seed,
            )
    # Re-run the same held-out seed through the greedy projection too. The
    # determinism gap is a reported result, so verifying only the sampled
    # number would leave half the measurement unaudited.
    greedy_stored = greedy_recs[0]
    greedy_policy = TrainedRLPolicy(weights_path, mode="greedy", seed=seed)
    greedy_got = run_seed(
        TumorEvolutionSimulator(geom),
        greedy_policy,
        seed=seed,
        days=days,
        initial_resistant_fraction=initial_resistant_fraction,
        growth_noise=growth_noise,
    )
    for key, mine in (
        ("aubc", greedy_got.aubc),
        ("final_burden_ratio", greedy_got.final_burden_ratio),
        ("pfs_days", greedy_got.pfs_days),
        ("toxicity", greedy_got.toxicity),
    ):
        want = float(greedy_stored[key])
        if round(float(mine), stored_precision[key]) != want:
            return RetentionVerification(
                False,
                f"greedy seed {seed} {key} re-measured {mine!r} != stored {want!r}",
                rechecked_seed=seed,
            )
    # both medians must agree with their own records
    for arm_name, arm_recs in (
        ("ppo_agent_sampled", sampled_recs),
        ("ppo_agent_greedy", greedy_recs),
    ):
        med = float(np.median([float(r["aubc"]) for r in arm_recs]))
        stored_med = float(arms[arm_name]["median_aubc"])
        # Records are rounded individually before serialization, while the
        # summary is rounded from the underlying full-precision trajectories.
        if not np.isclose(med, stored_med, rtol=0, atol=1.1e-6):
            return RetentionVerification(
                False,
                f"stored median AUBC {stored_med} for {arm_name} does not match "
                f"its own {len(arm_recs)} records ({med})",
            )

    retention = (art.get("retention") or {}).get("aubc")
    n = len(sampled_recs)
    return RetentionVerification(
        True,
        f"seed {seed} reproduced from {os.path.basename(weights_path)} "
        f"(sha8={sha[:8]}); {n} held-out seeds; retention vs best rule "
        f"({art.get('best_rule')}) = {retention}",
        n_seeds=n,
        retention_aubc=None if retention is None else float(retention),
        rechecked_seed=seed,
    )


# ---------------------------------------------------------------------------
# Provenance and limits
# ---------------------------------------------------------------------------


def rl_steering_provenance() -> Dict[str, EvidenceStatus]:
    return {
        "environment": EvidenceStatus.ESTABLISHED,
        "training_run": EvidenceStatus.ESTABLISHED,
        "retention_in_simulation": EvidenceStatus.ESTABLISHED,
        "patient_benefit": EvidenceStatus.PROVEN_FAILED,
    }


def what_rl_cannot_show() -> List[str]:
    return [
        "that the learned scheduler's edge is a scheduling edge: the trained "
        "object is a policy DISTRIBUTION, its sampled performance is what PPO "
        "optimised, and projecting it to its own argmax changes the number -- "
        "the recorded determinism gap is the boundary of that sentence, not a "
        "footnote to it",
        "that a fixed co-dose schedule is a therapy: it is a reference arm "
        "that exists to expose the action-space confound, nothing more",
        "that the trained policy is needed for the simulator outcome: the "
        "fixed deterministic A/B alternation is a comparator, and on the "
        "recorded held-out set it slightly outperforms sampled PPO",
        "that this policy is better than standard care in patients: the "
        "simulator's parameters are calibrated to an mCRPC adaptive-therapy "
        "anchor in a different disease, and every trajectory is synthetic",
        "that the learned policy is needed for the effect: the held-out "
        "collateral-switch and deterministic-alternation controls both beat "
        "sampled PPO on median AUBC",
        "that the reward is anything but authored: it is fixed here as the "
        "negative normalised burden curve so that the training objective and "
        "the reported metric are the same functional, but choosing burden "
        "rather than, say, dose-days is still a modelling choice",
        "that retention > 1.0 generalises past this geometry, these clone "
        "parameters and this mutation rate: the evaluation is one phantom, "
        "thirty seeds and a fixed parameter set",
    ]


def format_retention_report(path: str = ARTIFACT_PATH) -> str:
    art = load_artifact(path)
    if art is None:
        return f"no measured RL artifact at {path}"
    arms = art["arms"]
    lines = [
        f"RL dosing-policy evaluation ({art['days']} d, {len(art['seeds'])} held-out seeds):",
    ]
    for name in (
        "ppo_agent_sampled",
        "ppo_agent_greedy",
        "mtd",
        "adaptive_dose",
        "collateral_switch",
        "co_dose_reference",
        "alternate_reference",
    ):
        a = arms[name]
        lines.append(
            f"  {name:<20} median AUBC {a['median_aubc']:.4f}  "
            f"final burden x{a['median_final_burden_ratio']:.4f}  "
            f"PFS {a['median_pfs_days']:.0f} d  tox {a['median_toxicity']:.0f}"
        )
    ret = art["retention"]
    lines.append(
        f"  best rule on monitor seeds: {art['best_rule']}; retention (AUBC, "
        f"sampled) = {ret['aubc']}; greedy projection = {ret['aubc_greedy']}"
    )
    lines.append(            f"deterministic A/B alternation reference AUBC = "

        f"{arms['alternate_reference']['median_aubc']:.4f}; "
        f"PPO-specific advantage: not shown"
    )
    lines.append(
        "  synthetic simulator only; reward and normalization are "
        "repository-defined, not clinical evidence"
    )
    gap = art.get("determinism_gap") or {}
    if gap:
        lines.append(
            f"  determinism gap: sampled {gap.get('sampled_median_aubc')} vs "
            f"greedy {gap.get('greedy_median_aubc')}"
        )
    return "\n".join(lines)
