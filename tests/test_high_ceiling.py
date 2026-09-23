"""Tests for the high-ceiling predictive targets.

Pins: the one-shot holdout measurement of the emulator claim (R^2 >= 0.90 on
untouched configurations with the feature subset locked before the holdout
existed), the exact-match city-block distance transform, the real-patient
MGMT classifier reported at its measured value (near chance, CI reported),
and the bound claims that keep both numbers from going stale.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from danon.emulator_features import (
    FEATURE_ORDER,
    _city_block_dist_to_mask,
    feature_matrix,
    geometry_features,
)
from danon.eval_targets import TargetStatus
from danon.high_ceiling import (
    BUILT_DAYS,
    BUILT_N_CONFIGS,
    BUILT_N_REPLICATES,
    BUILT_SEED,
    EmulatorDataset,
    _oof_r2,
    _phantom_from_config,
    _ridge_fit,
    _ridge_predict,
    _sample_configs,
    emulator_target,
    mgmt_status_target,
)

TRAIN_PATH = "data/synthetic_emulator_dataset.json"
HOLD_PATH = "data/synthetic_emulator_holdout.json"

needs_datasets = pytest.mark.skipif(
    not (os.path.exists(TRAIN_PATH) and os.path.exists(HOLD_PATH)),
    reason="emulator datasets not built (12-min engine sweep); run the chunk builders",
)


class TestDistanceTransform:
    def test_matches_bfs_exactly(self):
        from collections import deque

        cfg = _sample_configs(3, 11)[1]
        geom = _phantom_from_config(cfg, seed=1012)
        dist = _city_block_dist_to_mask(geom.tumor_mask, geom.vessels > 0)
        mask = geom.tumor_mask
        vess = geom.vessels > 0
        INF = 10**9
        ref = np.full(mask.shape, INF)
        q = deque()
        for v in np.argwhere(vess):
            ref[tuple(v)] = 0
            q.append(tuple(v))
        while q:
            x, y, z = q.popleft()
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1] and 0 <= nz < mask.shape[2]:
                    if ref[nx, ny, nz] > ref[x, y, z] + 1:
                        ref[nx, ny, nz] = ref[x, y, z] + 1
                        q.append((nx, ny, nz))
        assert int(np.abs(dist[mask] - ref[mask]).max()) == 0

    def test_zero_at_source(self):
        cfg = _sample_configs(2, 11)[0]
        geom = _phantom_from_config(cfg, seed=1011)
        dist = _city_block_dist_to_mask(geom.tumor_mask, geom.vessels > 0)
        assert float(dist[geom.vessels > 0].min()) == 0.0


class TestPhysicsFeatures:
    def test_feature_count_and_finiteness(self):
        cfgs = _sample_configs(6, 11)
        geoms = [_phantom_from_config(c, seed=1100 + i) for i, c in enumerate(cfgs)]
        X = feature_matrix(geoms)
        assert X.shape == (6, len(FEATURE_ORDER))
        assert np.isfinite(X).all()

    def test_geometry_features_keys_match_order(self):
        cfg = _sample_configs(2, 11)[0]
        geom = _phantom_from_config(cfg, seed=1011)
        f = geometry_features(geom)
        assert set(f) == set(FEATURE_ORDER)

    def test_sanctuary_features_respond_to_vessels(self):
        """More vessels -> less far-from-vessel volume (directional sanity)."""
        few = _phantom_from_config(_sample_configs(2, 11)[0], seed=1011)
        import dataclasses

        many = dataclasses.replace(few, vessels=np.where(few.tumor_mask, 1.0, 0.0))
        f_few = geometry_features(few)
        f_many = geometry_features(many)
        assert f_many["sanctuary_frac_d_gt_6"] <= f_few["sanctuary_frac_d_gt_6"]


class TestEmulatorClaim:
    @needs_datasets
    def test_holdout_r2_meets_claim(self):
        """The bound claim, re-measured: one-shot holdout R^2 >= 0.90."""
        locked = (
            "sanctuary_frac_d_gt_6", "core_sanctuary_frac", "log_tumour_voxels",
            "sanctuary_frac_d_gt_4", "median_dist_to_vessel", "sanctuary_frac_d_gt_8",
        )
        cols = [FEATURE_ORDER.index(n) for n in locked]
        cfgs = _sample_configs(96, 11)
        geoms = [_phantom_from_config(c, seed=11 + 1000 + i) for i, c in enumerate(cfgs)]
        X = feature_matrix(geoms)[:, cols]
        with open(TRAIN_PATH) as fh:
            d1 = json.load(fh)["dataset"]
        with open(HOLD_PATH) as fh:
            d2 = json.load(fh)
        y = np.array(d1["final_burden"] + d2["final_burden"], dtype=float)
        b, mu, sd = _ridge_fit(X[:48], y[:48], 1.0)
        pred = _ridge_predict(b, mu, sd, X[48:])
        yte = y[48:]
        ssr = float(np.sum((yte - pred) ** 2))
        sst = float(np.sum((yte - yte.mean()) ** 2))
        r2 = 1.0 - ssr / sst
        assert r2 >= 0.90, f"holdout R^2 fell below the claim: {r2:.4f}"

    @needs_datasets
    def test_censored_endpoints_do_not_meet_claim(self):
        """The honest miss stays pinned: censored PFS endpoints are NOT 0.90+."""
        raw = json.load(open(TRAIN_PATH))["dataset"]
        ds = EmulatorDataset(
            rows=raw["rows"], policy_pfs=raw["policy_pfs"], mtd_pfs=raw["mtd_pfs"],
            pfs_extension=raw["pfs_extension"], policy_toxicity=raw["policy_toxicity"],
            mtd_toxicity=raw["mtd_toxicity"], n_replicates=raw["n_replicates"],
            days=raw["days"], seed=raw["seed"],
        )
        res = emulator_target(ds)
        assert res.verdict.status is TargetStatus.MEASURED
        assert res.r2_policy_pfs < 0.90  # the wall of 365-day censored values

    def test_dataset_constants_match_builder(self):
        assert (BUILT_N_CONFIGS, BUILT_N_REPLICATES, BUILT_DAYS, BUILT_SEED) == (48, 3, 365, 11)

    def test_ridge_reduces_to_least_squares_at_zero_lambda(self):
        rng = np.random.RandomState(0)
        X = rng.randn(40, 3)
        y = X @ np.array([1.0, -2.0, 0.5]) + 3.0
        b, mu, sd = _ridge_fit(X, y, 1e-12)
        pred = _ridge_predict(b, mu, sd, X)
        assert float(np.max(np.abs(pred - y))) < 1e-6

    def test_oof_r2_perfect_on_linear_truth(self):
        rng = np.random.RandomState(1)
        X = rng.randn(80, 4)
        y = X @ rng.randn(4) + 0.1 * rng.randn(80)
        r2, n = _oof_r2(X, y, 5, 0)
        assert r2 > 0.95
        assert n == 80


class TestMgmtTarget:
    def test_measured_and_reported_low(self):
        m = mgmt_status_target()
        assert m.verdict.status is TargetStatus.MEASURED
        assert m.n_patients >= 100
        assert 0.0 <= m.auc <= 1.0
        # the honest miss: near chance, CI reported, CI contains 0.5
        assert m.auc < 0.70
        assert m.auc_ci[0] <= 0.5 <= m.auc_ci[1]
        assert "indistinguishable from chance" in m.verdict.detail

    def test_verdict_carries_the_blocker(self):
        m = mgmt_status_target()
        blockers = " ".join(m.verdict.blockers)
        assert "per-patient expression or methylation-array features" in blockers


class TestBoundClaims:
    def test_holdout_claim_binds(self):
        from danon.claims import audit_claims

        res = {c.claim.id: c for c in audit_claims()}
        assert res["holdout_emulator_090"].supported
        assert "0.92" in res["holdout_emulator_090"].detail

    def test_mgmt_claim_binds(self):
        from danon.claims import audit_claims

        res = {c.claim.id: c for c in audit_claims()}
        assert res["mgmt_status_not_predictable_from_open_covariates"].supported
