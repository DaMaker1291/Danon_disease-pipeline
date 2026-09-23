"""High-ceiling predictive targets: what CAN legitimately clear 0.90.

WHY THIS MODULE EXISTS
----------------------
"Make it predict to 90%+" arrives repeatedly, and against real patient
survival it is unreachable for a structural reason the repo already
measured: age ALONE scores C = 0.653 held-out, the multivariable model
0.656, and the planted-hazard test shows the estimator recovers the true
signal to a +0.0003 shortfall -- the ceiling is the data, not the code.
So this module changes the ESTIMAND to two targets whose ceilings genuinely
sit above 0.90, builds both, and measures them rather than promising them:

TARGET A -- ENGINE EMULATOR (synthetic, real dynamics)
    The spatial simulator is the most expensive object in the repo. A
    supervised surrogate that predicts its outcomes (policy/MTD PFS,
    toxicity) from simulation CONFIGURATION is a legitimate 0.90+ target:
    the ground truth is the engine itself, measured on held-out
    configurations. Claim honesty: this validates that the engine's
    outcome surface is predictable from its inputs -- a fast emulator for
    sweeps and a regression test against unmodelled drift. It is NOT
    accuracy on patients and is labelled synthetic everywhere it renders.

TARGET B -- MGMT PROMOTER STATUS (real patients, real label)
    The repo caches published MGMT promoter calls for ~690 TCGA-GBM
    patients (danon.mgmt.MgmtCallTable, from Cell 2013/2016 cohort
    publications via cBioPortal). Predicting the methylation call from the
    baseline clinical covariates is a real-classification task on real
    patients with a real, non-leaky label. Literature (e.g. Bady 2012 and
    later methylation-expression work) reports methylation class is partly
    recoverable from expression; from age/sex alone it is NOT -- the
    expected outcome is AUC well under 0.90, and if so that is the
    reported result. The 0.90 claim on this target needs a modality the
    open cache does not carry (expression/methylation arrays per patient);
    the verdict states that blocker with the live count when it lands low.

Both targets reuse the repo's own machinery: TargetVerdict for rendering,
k-fold out-of-fold scoring with bootstrap CIs as in eval_targets, and
(make_gbm_phantom, TumorEvolutionSimulator) for the engine truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.eval_targets import TargetStatus, TargetVerdict, _auc, _logistic_fit
from danon.evosteer import (
    CollateralSwitchPolicy,
    MTDPolicy,
    TumorEvolutionSimulator,
    make_gbm_phantom,
)

# ---------------------------------------------------------------------------
# Target A -- emulator of the spatial engine
# ---------------------------------------------------------------------------

#: Simulation configuration: the inputs the engine actually consumes through
#: its public run path (run() takes days, r0, fill_fraction; the phantom takes
#: geometry), with the sampled ranges. cluster_sigma is NOT included: it is a
#: parameter of seed() and is not configurable through run().
FEATURE_NAMES: Tuple[str, ...] = (
    "initial_resistant_fraction",
    "fill_fraction",
    "radius_frac",
    "core_frac",
    "n_vessels",
    "rim_vessel",
)
FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "initial_resistant_fraction": (0.005, 0.08),
    "fill_fraction": (0.45, 0.80),
    "radius_frac": (0.30, 0.46),
    "core_frac": (0.45, 0.65),
    "n_vessels": (3, 8),
    "rim_vessel": (0, 1),
}

EMULATOR_CACHE = "data/synthetic_emulator_dataset.json"

#: The dataset committed by the chunked builder (48 configs x 3 replicates).
#: These are also the defaults of load_or_build_emulator_dataset so the cache
#: key MATCHES the file it ships with -- a mismatch would silently trigger a
#: full 12-minute engine rebuild on the default call path.
BUILT_N_CONFIGS = 48
BUILT_N_REPLICATES = 3
BUILT_DAYS = 365
BUILT_SEED = 11


def _sample_configs(n: int, seed: int) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []
    for _ in range(n):
        cfg = {}
        for name, (lo, hi) in FEATURE_BOUNDS.items():
            if name == "n_vessels":
                cfg[name] = float(rng.integers(int(lo), int(hi) + 1))
            elif name == "rim_vessel":
                cfg[name] = float(rng.integers(0, 2))
            else:
                cfg[name] = float(rng.uniform(lo, hi))
        rows.append(cfg)
    return rows


def _phantom_from_config(cfg: Dict[str, float], seed: int):
    return make_gbm_phantom(
        size=24,
        seed=seed,
        radius_frac=cfg["radius_frac"],
        core_frac=cfg["core_frac"],
        n_vessels=int(cfg["n_vessels"]),
        rim_vessel=bool(cfg["rim_vessel"]),
    )


@dataclass(frozen=True)
class EmulatorDataset:
    """Configurations and the engine outcomes measured on them."""

    rows: List[Dict[str, float]]
    policy_pfs: List[float]
    mtd_pfs: List[float]
    pfs_extension: List[float]
    policy_toxicity: List[float]
    mtd_toxicity: List[float]
    n_replicates: int
    days: int
    seed: int

    @property
    def n(self) -> int:
        return len(self.rows)

    def matrix(self) -> np.ndarray:
        return np.array(
            [[row[name] for name in FEATURE_NAMES] for row in self.rows],
            dtype=float,
        )


def build_emulator_dataset(
    n_configs: int = BUILT_N_CONFIGS,
    n_replicates: int = BUILT_N_REPLICATES,
    days: int = BUILT_DAYS,
    seed: int = BUILT_SEED,
) -> EmulatorDataset:
    """Measure the engine on n_configs sampled configurations.

    Each configuration is run n_replicates times under the steering policy
    and the MTD control (same seeds per arm); the per-config outcome is the
    MEDIAN across replicates, so the target the emulator learns is the
    engine's central behaviour, not one stochastic draw.
    """
    if n_configs < 2:
        raise ValueError("n_configs must be >= 2")
    if not 1 <= n_replicates <= 8:
        raise ValueError("n_replicates must be in [1, 8]")
    if not 30 <= days <= 365:
        raise ValueError("days must be in [30, 365]")

    cfgs = _sample_configs(n_configs, seed)
    policy_pfs: List[float] = []
    mtd_pfs: List[float] = []
    ext: List[float] = []
    pol_tox: List[float] = []
    mtd_tox: List[float] = []
    for i, cfg in enumerate(cfgs):
        geom = _phantom_from_config(cfg, seed=seed + 1000 + i)
        sim = TumorEvolutionSimulator(geom)
        p_pfs, m_pfs, p_tox, m_tox = [], [], [], []
        for rep in range(n_replicates):
            s = seed + 10_000 + i * 100 + rep
            tr_p = sim.run(
                CollateralSwitchPolicy(), days=days,
                initial_resistant_fraction=cfg["initial_resistant_fraction"],
                fill_fraction=cfg["fill_fraction"], seed=s,
            )
            tr_m = sim.run(
                MTDPolicy(), days=days,
                initial_resistant_fraction=cfg["initial_resistant_fraction"],
                fill_fraction=cfg["fill_fraction"], seed=s,
            )
            p_pfs.append(tr_p.pfs_value(days))
            m_pfs.append(tr_m.pfs_value(days))
            p_tox.append(float(tr_p.toxicity))
            m_tox.append(float(tr_m.toxicity))
        policy_pfs.append(float(np.median(p_pfs)))
        mtd_pfs.append(float(np.median(m_pfs)))
        pol_tox.append(float(np.median(p_tox)))
        mtd_tox.append(float(np.median(m_tox)))
        denom = m_pfs if np.median(m_pfs) > 0 else [1.0]
        ext.append(float((np.median(p_pfs) - np.median(m_pfs)) / np.median(denom)))

    return EmulatorDataset(
        rows=cfgs, policy_pfs=policy_pfs, mtd_pfs=mtd_pfs, pfs_extension=ext,
        policy_toxicity=pol_tox, mtd_toxicity=mtd_tox,
        n_replicates=n_replicates, days=days, seed=seed,
    )


def load_or_build_emulator_dataset(
    n_configs: int = BUILT_N_CONFIGS,
    n_replicates: int = BUILT_N_REPLICATES,
    days: int = BUILT_DAYS,
    seed: int = BUILT_SEED,
) -> EmulatorDataset:
    """Cache wrapper: the dataset is deterministic given (n, reps, days, seed)."""
    import os

    key = dict(n_configs=n_configs, n_replicates=n_replicates, days=days, seed=seed)
    if os.path.exists(EMULATOR_CACHE):
        with open(EMULATOR_CACHE) as fh:
            cached = json.load(fh)
        if cached.get("key") == key:
            d = cached["dataset"]
            return EmulatorDataset(
                rows=d["rows"], policy_pfs=d["policy_pfs"], mtd_pfs=d["mtd_pfs"],
                pfs_extension=d["pfs_extension"], policy_toxicity=d["policy_toxicity"],
                mtd_toxicity=d["mtd_toxicity"], n_replicates=d["n_replicates"],
                days=d["days"], seed=d["seed"],
            )
    ds = build_emulator_dataset(n_configs, n_replicates, days, seed)
    os.makedirs(os.path.dirname(EMULATOR_CACHE), exist_ok=True)
    with open(EMULATOR_CACHE, "w") as fh:
        json.dump({"key": key, "dataset": {
            "rows": ds.rows, "policy_pfs": ds.policy_pfs, "mtd_pfs": ds.mtd_pfs,
            "pfs_extension": ds.pfs_extension, "policy_toxicity": ds.policy_toxicity,
            "mtd_toxicity": ds.mtd_toxicity, "n_replicates": ds.n_replicates,
            "days": ds.days, "seed": ds.seed,
        }}, fh)
    return ds


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge with intercept unpenalised; X must NOT include a constant yet."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    pen = np.eye(Z1.shape[1]) * lam
    pen[0, 0] = 0.0
    return np.linalg.solve(Z1.T @ Z1 + pen, Z1.T @ y), mu, sd


def _ridge_predict(beta, mu, sd, X: np.ndarray) -> np.ndarray:
    Z = (X - mu) / sd
    return np.column_stack([np.ones(len(Z)), Z]) @ beta


def _oof_r2(X: np.ndarray, y: np.ndarray, n_folds: int, seed: int) -> Tuple[float, int]:
    """Out-of-fold R^2 of ridge (lambda by within-train LOO) on held-out folds."""
    rng = np.random.RandomState(seed)
    oof = np.full(len(y), np.nan)
    lams = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    for fold in np.array_split(rng.permutation(len(y)), n_folds):
        test = np.zeros(len(y), dtype=bool)
        test[fold] = True
        train = ~test
        if train.sum() < 5 or test.sum() == 0:
            continue
        # lambda chosen by leave-one-out INSIDE the training fold only.
        best = (float("inf"), 1.0)
        tr_idx = np.where(train)[0]
        for lam in lams:
            errs = []
            for j in tr_idx:
                sub = tr_idx[tr_idx != j]
                b, mu, sd = _ridge_fit(X[sub], y[sub], lam)
                pred = _ridge_predict(b, mu, sd, X[j : j + 1])[0]
                errs.append((pred - y[j]) ** 2)
            loo = float(np.mean(errs))
            if loo < best[0]:
                best = (loo, lam)
        b, mu, sd = _ridge_fit(X[train], y[train], best[1])
        oof[test] = _ridge_predict(b, mu, sd, X[test])
    scored = np.isfinite(oof)
    if scored.sum() < 2:
        return float("nan"), int(scored.sum())
    ss_res = float(np.sum((y[scored] - oof[scored]) ** 2))
    ss_tot = float(np.sum((y[scored] - y[scored].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, int(scored.sum())


@dataclass(frozen=True)
class EmulatorResult:
    verdict: TargetVerdict
    r2_policy_pfs: float
    r2_mtd_pfs: float
    r2_extension: float
    r2_policy_toxicity: float
    n: int


def emulator_target(
    dataset: Optional[EmulatorDataset] = None,
    *,
    n_configs: int = 60,
    n_replicates: int = 4,
    days: int = 365,
    seed: int = 11,
    n_folds: int = 5,
) -> EmulatorResult:
    """Fit the surrogate on held-out configurations; the engine is ground truth."""
    ds = dataset if dataset is not None else load_or_build_emulator_dataset(
        n_configs=n_configs, n_replicates=n_replicates, days=days, seed=seed
    )
    X = ds.matrix()
    r2_p, n_used = _oof_r2(X, np.array(ds.policy_pfs), n_folds, seed)
    r2_m, _ = _oof_r2(X, np.array(ds.mtd_pfs), n_folds, seed)
    r2_e, _ = _oof_r2(X, np.array(ds.pfs_extension), n_folds, seed)
    r2_t, _ = _oof_r2(X, np.array(ds.policy_toxicity), n_folds, seed)

    blockers: List[str] = [
        "synthetic ground truth: the engine's own outcomes on sampled "
        "configurations, NOT patient data -- may never be presented as a "
        "patient-prediction accuracy",
        f"n={ds.n} configurations at {ds.n_replicates} paired replicates of "
        f"{ds.days} days each (engine median across replicates is the target)",
    ]
    verdict = TargetVerdict(
        name="Engine emulator (configuration -> PFS / toxicity)",
        claimed="R^2 >= 0.90 on held-out configurations",
        metric="out-of-fold R^2 against the simulator",
        status=TargetStatus.MEASURED if np.isfinite(r2_p) else TargetStatus.REFUSED_NO_DATA,
        measured=float(r2_p) if np.isfinite(r2_p) else None,
        n=n_used,
        detail=(
            f"held-out R^2 across {n_used} configurations: policy PFS "
            f"{r2_p:.4f}, MTD PFS {r2_m:.4f}, PFS extension {r2_e:.4f}, "
            f"toxicity {r2_t:.4f}. The target the engine was built for is "
            f"predictable from its inputs to this accuracy by a ridge "
            f"surrogate (lambda selected by within-train LOO)"
        ),
        blockers=blockers,
    )
    return EmulatorResult(
        verdict=verdict, r2_policy_pfs=r2_p, r2_mtd_pfs=r2_m,
        r2_extension=r2_e, r2_policy_toxicity=r2_t, n=ds.n,
    )


# ---------------------------------------------------------------------------
# Target B -- MGMT promoter status on real patients
# ---------------------------------------------------------------------------


def _load_call_table(path: str):
    from danon.mgmt import MgmtCallTable

    with open(path) as fh:
        raw = json.load(fh)
    sources = tuple(tuple(s) for s in raw.get("sources", []))
    return MgmtCallTable(
        sources=sources,
        calls=raw.get("calls", {}),
        vocabulary=raw.get("vocabulary", {}),
        note=raw.get("note", ""),
    )


@dataclass(frozen=True)
class MgmtStatusResult:
    verdict: TargetVerdict
    auc: float
    auc_ci: Tuple[float, float]
    n_patients: int
    n_methylated: int
    best_source_agreement: Optional[float]


def mgmt_status_target(
    *,
    calls_path: str = "data/gdc_clinical/cbioportal_mgmt_calls.json",
    clinical_path: str = "data/gdc_clinical/tcga_gbm_cases.json",
    n_folds: int = 5,
    seed: int = 0,
    n_bootstrap: int = 400,
) -> MgmtStatusResult:
    """Out-of-fold AUC for the published MGMT call from baseline covariates.

    The label is the PUBLISHED binary call (never recomputed, never coerced
    from an unknown token); features are age, sex and glioblastoma diagnosis
    joined from the cached GDC clinical export. When both publications
    recorded the same patient, the stricter rule is used: keep the patient
    only when the sources AGREE, so a contradictory label never enters.
    """
    from danon.predictive import load_cohort_rows

    calls = _load_call_table(calls_path)
    rows, _ = load_cohort_rows(clinical_path, endpoint="os")
    index = {r.submitter_id: r for r in rows}

    keep: List[str] = []
    y: List[int] = []
    disagree = agree = 0
    for sid in calls.patient_ids:
        row = calls.calls.get(sid, {})
        vals = {v.strip().lower() for v in row.values() if v}
        binarised = {
            1 if v in {"methylated"} else 0 if v in {"unmethylated"} else None
            for v in vals
        }
        binarised.discard(None)
        if len(binarised) == 1:
            keep.append(sid)
            y.append(binarised.pop())
            agree += 1
        elif len(binarised) > 1:
            disagree += 1  # sources contradict: excluded, counted
    y_arr = np.array(y, dtype=int)

    joined = [s for s in keep if s in index]
    X = np.array(
        [[index[s].age_years, index[s].male, index[s].is_glioblastoma] for s in joined],
        dtype=float,
    )
    y_join = np.array([y[keep.index(s)] for s in joined], dtype=int)

    rng = np.random.RandomState(seed)
    oof = np.full(len(y_join), np.nan)
    for fold in np.array_split(rng.permutation(len(y_join)), n_folds):
        test = np.zeros(len(y_join), dtype=bool)
        test[fold] = True
        train = ~test
        if y_join[train].sum() < 3 or y_join[test].sum() < 1:
            continue
        beta = _logistic_fit(X[train], y_join[train])
        oof[test] = np.column_stack([np.ones(test.sum()), X[test]]) @ beta
    scored = np.isfinite(oof)
    auc = _auc(oof[scored], y_join[scored])

    boot: List[float] = []
    idx = np.where(scored)[0]
    for _ in range(n_bootstrap):
        take = rng.choice(idx, size=len(idx), replace=True)
        if y_join[take].sum() in (0, len(take)):
            continue
        boot.append(_auc(oof[take], y_join[take]))
    ci = (
        (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        if boot else (float("nan"), float("nan"))
    )

    if np.isnan(auc):
        verdict = TargetVerdict(
            name="MGMT promoter status (real patients)",
            claimed="AUC >= 0.90",
            metric="out-of-fold ROC-AUC",
            status=TargetStatus.REFUSED_NO_LABEL,
            measured=None,
            n=len(joined),
            detail="scoring fold(s) empty after covariate join; no number invented",
        )
        return MgmtStatusResult(verdict, float("nan"), ci, len(joined), int(y_join.sum()), None)

    blockers: List[str] = [
        "the 0.90 band on this target needs a modality the open cache does "
        "not carry (per-patient expression or methylation-array features); "
        "from age/sex/diagnosis alone the literature ceiling is well below it",
        "the label is the PUBLISHED binary call, not a beta value; the two "
        "publications disagree for some patients and those are excluded",
    ]
    verdict = TargetVerdict(
        name="MGMT promoter status (real patients)",
        claimed="AUC >= 0.90",
        metric="out-of-fold ROC-AUC",
        status=TargetStatus.MEASURED,
        measured=float(auc),
        n=int(scored.sum()),
        detail=(
            f"published calls joined to covariates for {int(scored.sum())} "
            f"patients ({int(y_join[scored].sum())} methylated); sources "
            f"agreed for {agree} and contradicted for {disagree} (excluded). "
            f"OOF AUC {auc:.4f}, 95% bootstrap CI {ci[0]:.4f}-{ci[1]:.4f}. "
            + (
                "the CI includes 0.5: indistinguishable from chance"
                if ci[0] <= 0.5 <= ci[1]
                else "above chance, below the claimed band"
                if auc < 0.90
                else "meets the claimed band"
            )
        ),
        blockers=blockers,
    )
    return MgmtStatusResult(
        verdict=verdict, auc=float(auc), auc_ci=ci,
        n_patients=int(scored.sum()), n_methylated=int(y_join[scored].sum()),
        best_source_agreement=agree / (agree + disagree) if (agree + disagree) else None,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


HOLDOUT_PATH = "data/synthetic_emulator_holdout.json"

#: The feature subset selected by greedy forward selection on the TRAINING
#: half only (48 configs), then LOCKED before the holdout was built. The
#: one-shot holdout measurement is valid only with this exact subset.
LOCKED_FEATURES: Tuple[str, ...] = (
    "sanctuary_frac_d_gt_6",
    "core_sanctuary_frac",
    "log_tumour_voxels",
    "sanctuary_frac_d_gt_4",
    "median_dist_to_vessel",
    "sanctuary_frac_d_gt_8",
)
LOCKED_LAMBDA = 1.0


@dataclass(frozen=True)
class HoldoutMeasurement:
    """The one-shot evaluation of the locked pipeline on untouched configs."""

    r2: float
    ci: Tuple[float, float]
    n_train: int
    n_test: int
    target: str

    @property
    def meets_claim(self) -> bool:
        return bool(np.isfinite(self.r2) and self.r2 >= 0.90)


def holdout_measurement(
    n_bootstrap: int = 2000, seed: int = 0
) -> HoldoutMeasurement:
    """Re-measure the locked pipeline on the holdout, from the cached files.

    One ridge fit on the training half (lambda locked at 1.0), one prediction
    of the holdout half, one R^2 -- plus a residual bootstrap CI. Deterministic
    given the two dataset files; no engine re-runs.
    """
    from danon.emulator_features import feature_matrix

    with open(EMULATOR_CACHE) as fh:
        d1 = json.load(fh)["dataset"]
    with open(HOLDOUT_PATH) as fh:
        d2 = json.load(fh)
    cfgs = _sample_configs(96, BUILT_SEED)
    geoms = [_phantom_from_config(c, seed=BUILT_SEED + 1000 + i) for i, c in enumerate(cfgs)]
    from danon.emulator_features import FEATURE_ORDER

    X = feature_matrix(geoms)[:, [FEATURE_ORDER.index(n) for n in LOCKED_FEATURES]]
    y = np.array(d1["final_burden"] + d2["final_burden"], dtype=float)
    Xtr, Xte, ytr, yte = X[:48], X[48:], y[:48], y[48:]
    b, mu, sd = _ridge_fit(Xtr, ytr, LOCKED_LAMBDA)
    pred = _ridge_predict(b, mu, sd, Xte)
    resid = yte - pred
    ssr = float(np.sum(resid**2))
    sst = float(np.sum((yte - yte.mean()) ** 2))
    r2 = 1.0 - ssr / sst if sst > 0 else float("nan")
    rng = np.random.RandomState(seed)
    boots: List[float] = []
    for _ in range(n_bootstrap):
        take = rng.randint(0, len(yte), len(yte))
        r = resid[take]
        yy = yte[take]
        st = float(np.sum((yy - yy.mean()) ** 2))
        if st > 0:
            boots.append(1.0 - float(np.sum(r**2)) / st)
    ci = (
        (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        if boots else (float("nan"), float("nan"))
    )
    return HoldoutMeasurement(
        r2=r2, ci=ci, n_train=len(ytr), n_test=len(yte), target="final_burden"
    )


def format_high_ceiling_report(emu: EmulatorResult, mgmt: MgmtStatusResult) -> str:
    lines = []
    bar = "=" * 74
    lines.append(bar)
    lines.append("HIGH-CEILING TARGETS: WHAT CAN AND CANNOT CLEAR 0.90 -- MEASURED")
    lines.append(bar)
    lines.append("")
    lines.append("Context that is not negotiable: real-patient OS C-index ceiling.")
    lines.append("  age alone 0.653 held-out; multivariable 0.656; the planted-")
    lines.append("  hazard test bounds estimator loss at +0.0003. The 0.90 band")
    lines.append("  on patient survival is a data ceiling, so the estimand moves.")
    lines.append("")
    for v in (emu.verdict, mgmt.verdict):
        lines.append(f"[{v.status.value.upper()}] {v.describe()}")
        for b in v.blockers:
            lines.append(f"    - {b}")
        lines.append("")
    lines.append(
        "Tier-4 synthetic recovery (R^2 = 0.999951) and the emulator R^2 are "
        "the same CLASS of result: properties of the machinery. They must "
        "never be averaged with, added to, or presented beside the patient "
        "numbers as if they raised them."
    )
    return "\n".join(lines)
