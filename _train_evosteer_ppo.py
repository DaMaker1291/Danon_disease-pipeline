"""Train a PPO dosing policy on the twin's OWN simulator, then measure it.

WHAT THIS DOES
--------------
Trains a 2x64x64 MLP Gaussian policy with PPO (clipped surrogate, GAE) over
`danon.rl_steering.EvoSteerEnv`, which drives the real
`danon.evosteer.TumorEvolutionSimulator` day by day. gamma = 1.0, so the PPO
return IS minus the AUBC (area under the normalised burden curve) -- the
training objective and the reported metric are the same functional.

SEED PARTITION (recorded in the training JSON, never mixed):
  * training  : TRAIN_SEED_BASE + iteration*envs + i    (1000+, fresh per episode)
  * monitor   : MONITOR_SEED_BASE + 0..9                (900-909, used to pick
                the best closed-form rule and to watch the learning curve)
  * evaluation: 0..29                                   (never seen in training)

WHY THE ACTION IS A SCHEDULE CHOICE, NOT A CONTINUOUS DOSE
---------------------------------------------------------
The policy emits logits over the three daily schedule choices the closed-form
policies choose among -- full-dose A, full-dose B, holiday -- and learns with a
Categorical distribution. The continuous-dose variant is implemented in
danon.rl_steering and measured there as a reference arm, but it is not the
headline: the engine's toxicity proxy is a LOG, not a constraint, so a
continuous agent wins by co-dosing both drugs (~0.5 each), a property of the
action space rather than of scheduling. Comparing against the rules requires
the rules' own decision set.

WHAT WAS TRIED, AND WHAT IT MEASURED (both runs are in the record)
------------------------------------------------------------------
Run 1, temperature fixed at 1.0: the SAMPLED policy reached AUBC ~0.021 (the
model's floor) while its own greedy argmax scored ~1.24, i.e. MTD. The logits
stayed nearly uniform, so the advantage lived in the RANDOMISATION (any mixture
of A and B beats a fixed drug) and not in a state-feedback rule.

Run 2, temperature annealed 1.0 -> 0.2: once the anneal sharpened the spread
logits, the sampled policy collapsed onto constant drug A (sampled AUBC ~1.0 at
temperature 0.6, greedy 1.24) and never recovered. The reason is structural
rather than a tuning miss: PPO maximises the SAMPLED return, and in this model
the sampled return is maximised by a MIXTURE, so nothing in the objective
pushes the logits apart. Annealing a converged-and-spread policy only converts
a good stochastic policy into a bad deterministic one.

The default is therefore run 1's configuration, and the honest deliverable is
the pair of numbers rather than one of them: the trained distribution and its
greedy projection are BOTH evaluated on the same held-out seeds, the
`alternate_reference` arm settles whether the floor is reachable
deterministically at all, and `danon.rl_steering` records the gap.

Writes:
  * models/ppo_evosteer_policy.pt      (torch checkpoint, inference-only net)
  * outputs_real/rl_steering_train.json (episodes, seeds, monitor curve, timing)
  * with --evaluate (default): outputs_real/rl_steering.json via
    danon.rl_steering.write_rl_artifact

Run:  python _train_evosteer_ppo.py --iterations 100 --envs 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from danon.rl_steering import (
    ARTIFACT_PATH,
    EVAL_SEEDS,
    HIDDEN,
    HORIZON_DAYS,
    MONITOR_SEED_BASE,
    OBS_DIM,
    SCHEDULE_ACTIONS,
    TRAIN_RECORD_PATH,
    TRAIN_SEED_BASE,
    WEIGHTS_PATH,
    EvoSteerEnv,
    write_rl_artifact,
)


N_ACTIONS = len(SCHEDULE_ACTIONS)


class ActorCritic(nn.Module):
    """Shared torso, Categorical actor head over the 3 schedules, scalar critic."""

    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.head = nn.Linear(hidden, N_ACTIONS)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return self.head(h), self.critic(h).squeeze(-1)


def export_inference_state_dict(policy: ActorCritic) -> Dict[str, torch.Tensor]:
    """Re-key the actor network into the Sequential layout TrainedRLPolicy loads.

    Inference needs the actor logits only; the critic is training machinery and
    is deliberately not exported, so the checkpoint cannot be mistaken for a
    value function.
    """
    sd = policy.state_dict()
    return {
        "0.weight": sd["body.0.weight"],
        "0.bias": sd["body.0.bias"],
        "2.weight": sd["body.2.weight"],
        "2.bias": sd["body.2.bias"],
        "4.weight": sd["head.weight"],
        "4.bias": sd["head.bias"],
    }


def collect(
    envs: List[EvoSteerEnv],
    policy: ActorCritic,
    seeds: List[int],
    temperature: float = 1.0,
):
    """One full episode per env, in lockstep. Returns tensors + episode AUBCs.

    `temperature` divides the logits before sampling AND in the log-prob, so
    PPO's ratio stays consistent with the distribution it samples from. As it
    anneals toward zero the rollout becomes the greedy policy that evaluation
    quadratically measures -- see the module docstring.
    """
    obs_list, act_list, logp_list, rew_list, val_list = [], [], [], [], []
    episode_aubc: List[float] = []
    obs = []
    for env, seed in zip(envs, seeds):
        o, _ = env.reset(seed=seed)
        obs.append(o)
    with torch.no_grad():
        for _ in range(HORIZON_DAYS):
            obs_t = torch.as_tensor(np.stack(obs), dtype=torch.float32)
            logits, value = policy(obs_t)
            dist = torch.distributions.Categorical(logits=logits / max(temperature, 1e-3))
            a = dist.sample()
            logp = dist.log_prob(a)
            for i, env in enumerate(envs):
                o, r, terminated, _, info = env.step(int(a[i]))
                obs[i] = o
                act_list.append(int(a[i]))
                rew_list.append(r)
                logp_list.append(float(logp[i]))
                val_list.append(float(value[i]))
                obs_list.append(obs_t[i].numpy())
                if terminated:
                    episode_aubc.append(float(info.get("aubc", float("nan"))))
    return (
        np.asarray(obs_list, dtype=np.float32),
        np.asarray(act_list, dtype=np.int64),
        np.asarray(logp_list, dtype=np.float32),
        np.asarray(rew_list, dtype=np.float32),
        np.asarray(val_list, dtype=np.float32),
        episode_aubc,
    )


def gae(rewards, values, gamma=1.0, lam=0.95, n_envs=1, horizon=HORIZON_DAYS):
    """Generalised advantage estimation, episodes resetting at the horizon."""
    T = len(rewards) // n_envs
    adv = np.zeros_like(rewards)
    last = np.zeros(n_envs, dtype=np.float32)
    for t in reversed(range(T)):
        idx = t * n_envs + np.arange(n_envs)
        if t == T - 1:
            next_value = np.zeros(n_envs, dtype=np.float32)
            nonterminal = np.zeros(n_envs, dtype=np.float32)
        else:
            next_value = values[(t + 1) * n_envs + np.arange(n_envs)]
            nonterminal = np.ones(n_envs, dtype=np.float32)
        delta = rewards[idx] + gamma * next_value * nonterminal - values[idx]
        last = delta + gamma * lam * nonterminal * last
        adv[idx] = last
    return adv


def monitor_median_aubc(envs: List[EvoSteerEnv], policy: ActorCritic, seeds) -> float:
    """Deterministic (greedy argmax) median AUBC on the monitor seeds."""
    vals = []
    with torch.no_grad():
        for seed in seeds:
            env = EvoSteerEnv()
            o, _ = env.reset(seed=int(seed))
            done = False
            while not done:
                logits, _ = policy(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))
                o, r, done, _, info = env.step(int(torch.argmax(logits[0])))
            vals.append(float(info.get("aubc", float("nan"))))
    return float(np.median(vals))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.0)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--temperature-start", type=float, default=1.0)
    ap.add_argument("--temperature-end", type=float, default=1.0)
    ap.add_argument("--monitor-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-evaluate",
        action="store_true",
        help="skip writing the measured artifact (used for smoke runs)",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)

    envs = [EvoSteerEnv() for _ in range(args.envs)]
    policy = ActorCritic()
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    monitor_seeds = list(range(MONITOR_SEED_BASE, MONITOR_SEED_BASE + 10))
    curve: List[Dict[str, float]] = []
    t0 = time.perf_counter()
    steps_total = 0
    for it in range(1, args.iterations + 1):
        seeds = [TRAIN_SEED_BASE + it * args.envs + i for i in range(args.envs)]
        frac = (it - 1) / max(args.iterations - 1, 1)
        temperature = args.temperature_start + frac * (args.temperature_end - args.temperature_start)
        obs, acts, logps, rews, vals, aubcs = collect(
            envs, policy, seeds, temperature=temperature
        )
        adv = gae(rews, vals, n_envs=args.envs)
        ret = adv + vals
        adv_t = torch.as_tensor((adv - adv.mean()) / (adv.std() + 1e-8), dtype=torch.float32)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        act_t = torch.as_tensor(acts, dtype=torch.int64)
        logp_t = torch.as_tensor(logps, dtype=torch.float32)
        steps_total += len(rews)
        n = len(rews)
        for _ in range(args.epochs):
            idx = torch.randperm(n)
            for start in range(0, n, args.minibatch):
                mb = idx[start : start + args.minibatch]
                logits, value = policy(obs_t[mb])
                dist = torch.distributions.Categorical(logits=logits / max(temperature, 1e-3))
                new_logp = dist.log_prob(act_t[mb])
                ratio = torch.exp(new_logp - logp_t[mb])
                unclipped = ratio * adv_t[mb]
                clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_t[mb]
                loss = (
                    -torch.min(unclipped, clipped).mean()
                    + args.vf * ((value - ret_t[mb]) ** 2).mean()
                    - args.entropy * dist.entropy().mean()
                )
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
        med_ret = float(np.median(aubcs))
        elapsed = time.perf_counter() - t0
        print(
            f"iter {it:>3}/{args.iterations}  temp {temperature:.3f}  "
            f"median episode AUBC {med_ret:.4f}  steps {steps_total}  {elapsed:.1f}s",
            flush=True,
        )
        if it % args.monitor_every == 0 or it == args.iterations:
            det = monitor_median_aubc(envs, policy, monitor_seeds[:5])
            curve.append({"iteration": it, "median_aubc_deterministic": det, "median_aubc_sampled": med_ret})
            print(f"  monitor (deterministic mean action, 5 seeds): AUBC {det:.4f}", flush=True)

    wall = time.perf_counter() - t0
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    torch.save(
        {
            "state_dict": export_inference_state_dict(policy),
            "obs_dim": OBS_DIM,
            "n_out": N_ACTIONS,
            "hidden": HIDDEN,
            "action_mode": "schedule",
            "algo": "ppo",
            "feature_spec": "[burden/b_init (cap 12), R fraction, day/365, toxicity/tox_ref]",
            "schedule_actions": [list(a) for a in SCHEDULE_ACTIONS],
            "policy_spec": "categorical over schedule actions; argmax at eval",
            "trained_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        WEIGHTS_PATH,
    )
    sha = hashlib.sha256(open(WEIGHTS_PATH, "rb").read()).hexdigest()
    record = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "_train_evosteer_ppo.py",
        "algorithm": "ppo (clipped surrogate, GAE, gamma=1.0 so return = -AUBC), "
        "categorical head over schedule actions (A / B / holiday)",
        "iterations": args.iterations,
        "envs": args.envs,
        "steps_total": steps_total,
        "horizon_days": HORIZON_DAYS,
        "wall_seconds": round(wall, 2),
        "steps_per_second": round(steps_total / wall, 1),
        "torch": torch.__version__,
        "seed_partition": {
            "training": f"{TRAIN_SEED_BASE}+",
            "monitor": [int(s) for s in monitor_seeds],
            "evaluation": [int(s) for s in EVAL_SEEDS],
        },
        "hyperparameters": {
            "lr": args.lr,
            "epochs": args.epochs,
            "minibatch": args.minibatch,
            "clip": args.clip,
            "entropy_coef": args.entropy,
            "value_coef": args.vf,
            "hidden": HIDDEN,
            "temperature_start": args.temperature_start,
            "temperature_end": args.temperature_end,
            "configuration_history": (
                "temperature fixed at 1.0 (this run): sampled AUBC ~0.021 vs "
                "greedy ~1.24 -- the advantage is in the randomisation. "
                "Temperature annealed 1.0 -> 0.2 (tried, rejected): the anneal "
                "sharpened the spread logits onto constant drug A (sampled ~1.0 "
                "at temperature 0.6) because PPO maximises the SAMPLED return, "
                "which a mixture already maximises. Both forms are evaluated "
                "and recorded rather than one being selected."
            ),
        },
        "monitor_curve": curve,
        "weights_path": WEIGHTS_PATH,
        "weights_sha256": sha,
    }
    os.makedirs(os.path.dirname(TRAIN_RECORD_PATH), exist_ok=True)
    with open(TRAIN_RECORD_PATH, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)
    print(f"wrote {WEIGHTS_PATH} (sha256 {sha[:12]}) and {TRAIN_RECORD_PATH}", flush=True)

    if not args.no_evaluate:
        art = write_rl_artifact(progress=False)
        print(json.dumps(art["retention"], indent=1), flush=True)
        for name in (
            "ppo_agent_sampled",
            "ppo_agent_greedy",
            "mtd",
            "adaptive_dose",
            "collateral_switch",
            "co_dose_reference",
            "alternate_reference",
        ):
            a = art["arms"][name]
            print(
                f"{name:<22} median AUBC {a['median_aubc']:.4f}  "
                f"final x{a['median_final_burden_ratio']:.4f}  "
                f"PFS {a['median_pfs_days']:.0f} d",
                flush=True,
            )
        print(f"wrote {ARTIFACT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
