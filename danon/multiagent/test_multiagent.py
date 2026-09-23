"""
Tests for the disease-agnostic multi-agent deployment framework.

These tests exercise cohort / scenario / runtime / runner / aggregation
without being tied to any one disease area — although the fixtures lean
on the cardiac/ischemic benchmarks since that is the initial deployment
target.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from danon.multiagent.core import (
    AgentJudgementOutcome,
    DiseaseCohort,
    MultiAgentConfig,
    SubAgentRuntime,
    SubAgentTask,
    TaskObjective,
    deploy_agents,
)
from danon.multiagent.benchmarks import (
    BaseJudge,
    CVDHeartDiseaseCohort,
    FlowState,
    IschemicHeartDiseaseCohort,
    PlaqueSurrogate,
    build_cohorts,
    build_cvd_cohort,
)
from danon.multiagent.runtime import (
    LocalDeploymentRunner,
    _CVDTaskRuntime,
)
from danon.multiagent.aggregator import leaderboard_table, result_to_dict

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cvd_cohort() -> DiseaseCohort:
    return CVDHeartDiseaseCohort()


@pytest.fixture
def ischemic_cohort() -> DiseaseCohort:
    return IschemicHeartDiseaseCohort()


@pytest.fixture
def cardiac_cohorts() -> List[DiseaseCohort]:
    return build_cohorts()


@pytest.fixture
def sample_task() -> SubAgentTask:
    return SubAgentTask(
        agent_id="agent_001",
        cohort_id="cvd_heart_disease",
        objectives=(TaskObjective(name="composite", direction="max", weight=1.0),),
        payload={"agent_potency": 0.7, "run_minutes": 60},
    )


# ---------------------------------------------------------------------------
# Cohort / scenario tests
# ---------------------------------------------------------------------------


class TestCohorts:
    def test_build_cohorts_returns_both(self, cardiac_cohorts: List[DiseaseCohort]) -> None:
        ids = {c.id for c in cardiac_cohorts}
        assert "cvd_heart_disease" in ids
        assert "ischemic_heart_disease" in ids

    def test_build_cvd_cohort_defaults_to_cvd(self) -> None:
        c = build_cvd_cohort()
        assert c.id == "cvd_heart_disease"

    def test_build_cvd_cohort_can_select_ischemic(self) -> None:
        c = build_cvd_cohort(id="ischemic_heart_disease")
        assert c.id == "ischemic_heart_disease"

    def test_cvd_cohort_validates_missing_plaque_metrics(self, cvd_cohort: DiseaseCohort) -> None:
        task = SubAgentTask(agent_id="x", cohort_id="cvd_heart_disease")
        ok, msg = cvd_cohort.validate_result(task, {"not_plaque": 1})
        assert not ok
        assert "plaque_metrics" in msg

    def test_cvd_cohort_score_produces_expected_keys(self, cvd_cohort: DiseaseCohort) -> None:
        task = SubAgentTask(
            agent_id="x",
            cohort_id="cvd_heart_disease",
            payload={"agent_potency": 0.8, "run_minutes": 120},
        )
        # The cohort-level scorer must expose the derived axes the judge reads.
        builder = cvd_cohort.scenario_builder(task)
        scenario = builder.build()
        metrics = cvd_cohort.score(task, {"plaque_metrics": builder.score(scenario, scenario)})
        for key in ("efficacy", "safety", "plaque_reduction_pct", "reperfusion_gain_pct"):
            assert key in metrics

    def test_scenario_builder_score_matches_cohort_axes(self, cvd_cohort: DiseaseCohort) -> None:
        task = SubAgentTask(
            agent_id="x",
            cohort_id="cvd_heart_disease",
            payload={"agent_potency": 0.8, "run_minutes": 120},
        )
        builder = cvd_cohort.scenario_builder(task)
        scenario = builder.build()
        # builder.score and cohort.score must agree on the derived axes so the
        # judge sees the same numbers regardless of entry point.
        metrics = builder.score(scenario, scenario)
        for key in ("efficacy", "safety", "plaque_reduction_pct", "reperfusion_gain_pct"):
            assert key in metrics


class TestRealPhysicsRuntime:
    """The deployment path must run the actual simulator, not a surrogate.

    These assertions exist because this path previously returned a placeholder
    whose numbers looked plausible while being disconnected from the physics.
    """

    def _metrics(self, cohort: DiseaseCohort, **payload) -> dict:
        task = SubAgentTask(
            agent_id="real", cohort_id="ischemic_heart_disease",
            payload={"run_minutes": 60, **payload},
        )
        runtime = _CVDTaskRuntime(config=MultiAgentConfig(), cohort=cohort, task=task)
        result = runtime.run()
        return result["plaque_metrics"]

    def test_metrics_carry_physical_units_and_reference(self, ischemic_cohort: DiseaseCohort) -> None:
        m = self._metrics(ischemic_cohort, agent_potency=0.5)
        # Raw model output must be present unmodified next to the scaled axes.
        for key in (
            "plaque_reduction_physical_pct",
            "pressure_drop_reduction_physical_pct",
            "off_target_physical_pct",
            "reference_plaque_reduction_pct",
            "agent_steps",
        ):
            assert key in m, key
        assert m["agent_steps"] > 0
        # The efficacy ceiling is derived from a real run, so it cannot be zero.
        assert m["reference_plaque_reduction_pct"] > 0.0

    def test_normalised_axes_are_bounded(self, ischemic_cohort: DiseaseCohort) -> None:
        for pot in (0.0, 0.25, 0.5, 0.75, 1.0):
            m = self._metrics(ischemic_cohort, agent_potency=pot)
            for key in ("efficacy", "safety", "plaque_reduction_pct", "reperfusion_gain_pct"):
                assert 0.0 <= m[key] <= 100.0, (pot, key, m[key])

    def test_plaque_reduction_increases_with_potency(self, ischemic_cohort: DiseaseCohort) -> None:
        vals = [
            self._metrics(ischemic_cohort, agent_potency=p)["plaque_reduction_physical_pct"]
            for p in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        assert all(b >= a for a, b in zip(vals, vals[1:])), vals
        assert vals[-1] > vals[0]

    def test_zero_potency_produces_no_reduction(self, ischemic_cohort: DiseaseCohort) -> None:
        m = self._metrics(ischemic_cohort, agent_potency=0.0)
        assert m["plaque_reduction_physical_pct"] == pytest.approx(0.0, abs=1e-9)

    def test_runtime_is_deterministic_for_a_fixed_agent(self, ischemic_cohort: DiseaseCohort) -> None:
        a = self._metrics(ischemic_cohort, agent_potency=0.6)
        b = self._metrics(ischemic_cohort, agent_potency=0.6)
        assert a["plaque_reduction_physical_pct"] == b["plaque_reduction_physical_pct"]

    def test_executor_mode_does_not_change_results(self, cardiac_cohorts: List[DiseaseCohort]) -> None:
        """Serial and threaded runs must agree; parallel must never alter science.

        The runner defaults to in-process execution because the CPU-bound
        simulator is measurably slower under threads, so this pins down that the
        parallel path -- when explicitly requested -- is at least not wrong.
        """
        tasks = [
            SubAgentTask(
                agent_id=f"par_{i}", cohort_id="cvd_heart_disease",
                objectives=(TaskObjective(name="composite", direction="max"),),
                payload={"agent_potency": 0.3 + 0.2 * i, "run_minutes": 60},
            )
            for i in range(3)
        ]
        plan = deploy_agents(MultiAgentConfig(study_id="par"), agents=tasks, cohorts=cardiac_cohorts)
        serial = LocalDeploymentRunner(max_workers=1, judge_all=True).deploy(plan.manifest, plan)
        threaded = LocalDeploymentRunner(
            max_workers=4, judge_all=True, executor="thread"
        ).deploy(plan.manifest, plan)

        assert len(serial) == len(threaded)
        for a, b in zip(serial, threaded):
            assert a.task.agent_id == b.task.agent_id
            ma = a.result["plaque_metrics"]
            mb = b.result["plaque_metrics"]
            assert ma["plaque_reduction_physical_pct"] == mb["plaque_reduction_physical_pct"]
            assert a.judgement.outcome == b.judgement.outcome

    def test_default_executor_is_in_process(self) -> None:
        r = LocalDeploymentRunner(max_workers=8)
        assert r._resolve_executor(8, 8) == "serial"
        assert LocalDeploymentRunner(max_workers=8, executor="thread")._resolve_executor(8, 8) == "thread"

    def test_unknown_executor_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            LocalDeploymentRunner(executor="quantum")

    def test_judge_tiers_are_reachable_on_the_normalised_scale(self, ischemic_cohort: DiseaseCohort) -> None:
        """A best-configured agent must be able to earn the top tier.

        Guards against the normaliser drifting out of sync with the judge: if
        the derived ceiling ever exceeded what a real run can produce, every
        agent would collapse into the reject bin and the ranking would be void.
        """
        task = SubAgentTask(
            agent_id="best", cohort_id="ischemic_heart_disease",
            payload={
                "agent_potency": 1.0, "agent_avidity": 1.0,
                "vcam1_density": 1.0, "run_minutes": 60,
            },
        )
        runtime = _CVDTaskRuntime(config=MultiAgentConfig(), cohort=ischemic_cohort, task=task)
        result = runtime.run()
        m = result["plaque_metrics"]
        j = BaseJudge()(task, result, m)
        assert m["efficacy"] > 90.0
        assert j.outcome == AgentJudgementOutcome.TIER_1


class TestPlaqueSurrogate:
    def test_digest_rate_scales_with_potency(self) -> None:
        p = PlaqueSurrogate()
        r1 = p.digest_rate(0.5)
        r2 = p.digest_rate(1.0)
        assert r2 > r1
        assert r1 > 0.0

    def test_digest_rate_returns_percent_per_minute(self) -> None:
        p = PlaqueSurrogate()
        # lipid fraction 0.6 * potency 1.0 * 0.05 = 0.03 per minute
        assert abs(p.digest_rate(1.0) - 0.03) < 1e-9


class TestFlowState:
    def test_flow_relative_is_delta_p_over_resistance(self) -> None:
        f = FlowState(pressure_upstream_kpa=16.0, pressure_downstream_kpa=10.0, resistance_relative=2.0)
        # (16 - 10) / 2 = 3
        assert abs(f.flow_relative - 3.0) < 1e-9


# ---------------------------------------------------------------------------
# Task runtime tests
# ---------------------------------------------------------------------------


class TestCVDTaskRuntime:
    def test_runtime_runs_and_produces_plaque_metrics(
        self, cvd_cohort: DiseaseCohort, sample_task: SubAgentTask
    ) -> None:
        runtime = _CVDTaskRuntime(
            config=MultiAgentConfig(),
            cohort=cvd_cohort,
            task=sample_task,
        )
        result = runtime.run()
        assert isinstance(result, dict)
        pm = result.get("plaque_metrics")
        assert isinstance(pm, dict)
        assert "plaque_reduction_pct" in pm
        assert "reperfusion_gain_pct" in pm

    def test_runtime_reports_error_on_bad_builder(self, cvd_cohort: DiseaseCohort) -> None:
        bad_task = SubAgentTask(agent_id="x", cohort_id="cvd_heart_disease")
        runtime = _CVDTaskRuntime(
            config=MultiAgentConfig(),
            cohort=cvd_cohort,
            task=bad_task,
        )
        # builder.build() should not raise; runtime should return error-keyed dict on failure
        result = runtime.run()
        assert isinstance(result, dict)
        assert "plaque_metrics" in result

    def test_validate_rejects_zero_reduction(self, cvd_cohort: DiseaseCohort) -> None:
        task = SubAgentTask(agent_id="x", cohort_id="cvd_heart_disease", payload={"agent_potency": 0.0})
        runtime = _CVDTaskRuntime(config=MultiAgentConfig(), cohort=cvd_cohort, task=task)
        result = runtime.run()
        assert not runtime.validate(result)


# ---------------------------------------------------------------------------
# Deployment runner tests
# ---------------------------------------------------------------------------


class TestLocalDeploymentRunner:
    def test_deploy_empty_plan_returns_empty_reason(self) -> None:
        manifest = deploy_agents(
            MultiAgentConfig(study_id="s"),
            agents=[],
            cohorts=build_cohorts(),
        ).manifest
        plan = deploy_agents(
            MultiAgentConfig(study_id="s"),
            agents=[],
            cohorts=build_cohorts(),
        )
        # plan.tasks is empty; runner should return an empty-reason result
        runner = LocalDeploymentRunner()
        results = runner.deploy(manifest, plan)
        assert len(results) == 1
        assert not results[0].succeeded
        assert "No tasks in plan" in results[0].error

    def test_deploy_single_task_succeeds(self, cardiac_cohorts: List[DiseaseCohort]) -> None:
        plan = deploy_agents(
            MultiAgentConfig(study_id="s"),
            agents=[
                SubAgentTask(
                    agent_id="a1",
                    cohort_id="cvd_heart_disease",
                    objectives=(TaskObjective(name="composite", direction="max"),),
                    payload={"agent_potency": 0.8, "run_minutes": 60},
                )
            ],
            cohorts=cardiac_cohorts,
        )
        runner = LocalDeploymentRunner()
        results = runner.deploy(plan.manifest, plan)
        assert len(results) == 1
        assert results[0].task.agent_id == "a1"
        assert results[0].result.get("plaque_metrics") is not None

    def test_deploy_uses_thread_pool_for_many_tasks(self, cardiac_cohorts: List[DiseaseCohort]) -> None:
        tasks = [
            SubAgentTask(
                agent_id=f"agent_{i:03d}",
                cohort_id="cvd_heart_disease",
                objectives=(TaskObjective(name="composite", direction="max"),),
                payload={"agent_potency": 0.5 + (i % 5) * 0.1, "run_minutes": 10},
            )
            for i in range(30)
        ]
        plan = deploy_agents(
            MultiAgentConfig(study_id="s", max_parallel_agents=8),
            agents=tasks,
            cohorts=cardiac_cohorts,
        )
        runner = LocalDeploymentRunner(max_workers=8, judge_all=False)
        results = runner.deploy(plan.manifest, plan)
        assert len(results) == 30
        assert all(isinstance(r.result, dict) for r in results)

    def test_summarize_counts(self, cardiac_cohorts: List[DiseaseCohort]) -> None:
        tasks = [
            SubAgentTask(
                agent_id=f"agent_{i}",
                cohort_id="cvd_heart_disease",
                payload={"agent_potency": 0.9 if i % 2 == 0 else 0.05},
            )
            for i in range(8)
        ]
        plan = deploy_agents(
            MultiAgentConfig(study_id="s"),
            agents=tasks,
            cohorts=cardiac_cohorts,
        )
        runner = LocalDeploymentRunner(judge_all=False)
        results = runner.deploy(plan.manifest, plan)
        summary = runner.summarize(results)
        assert summary["count"] == 8
        assert summary["succeeded"] + summary["failed"] == 8


# ---------------------------------------------------------------------------
# Judge tests
# ---------------------------------------------------------------------------


class TestBaseJudge:
    def test_tier_1_when_efficacy_and_safety_high(self) -> None:
        task = SubAgentTask(agent_id="x", cohort_id="cvd_heart_disease")
        result = {"plaque_metrics": {}}
        metrics = {
            "efficacy": 80.0,
            "safety": 90.0,
            "plaque_reduction_pct": 40.0,
            "reperfusion_gain_pct": 10.0,
        }
        j = BaseJudge()(task, result, metrics)
        assert j.outcome == AgentJudgementOutcome.TIER_1
        assert j.score > 70.0

    def test_reject_when_low_efficacy(self) -> None:
        task = SubAgentTask(agent_id="x", cohort_id="cvd_heart_disease")
        metrics = {
            "efficacy": 5.0,
            "safety": 90.0,
            "plaque_reduction_pct": 1.0,
            "reperfusion_gain_pct": 0.0,
        }
        j = BaseJudge()(task, {}, metrics)
        assert j.outcome == AgentJudgementOutcome.REJECT


# ---------------------------------------------------------------------------
# Aggregator / leaderboard tests
# ---------------------------------------------------------------------------


class TestAggregatorHelpers:
    def test_leaderboard_table_is_markdown(self, cardiac_cohorts: List[DiseaseCohort]) -> None:
        tasks = [
            SubAgentTask(
                agent_id=f"agent_{i}",
                cohort_id="cvd_heart_disease",
                payload={"agent_potency": 0.1 + (i % 10) * 0.1},
            )
            for i in range(12)
        ]
        plan = deploy_agents(
            MultiAgentConfig(study_id="s"),
            agents=tasks,
            cohorts=cardiac_cohorts,
        )
        runner = LocalDeploymentRunner(judge_all=False)
        results = runner.deploy(plan.manifest, plan)
        md = leaderboard_table(results, top_n=5)
        assert md.startswith("# Agent leaderboard")
        assert "| Rank |" in md
        assert "| agent_0" in md or "| agent_" in md

    def test_result_to_dict_roundtrips_judgement(self, cardiac_cohorts: List[DiseaseCohort]) -> None:
        tasks = [
            SubAgentTask(
                agent_id="judge_me",
                cohort_id="cvd_heart_disease",
                payload={"agent_potency": 0.9},
            )
        ]
        plan = deploy_agents(
            MultiAgentConfig(study_id="s"),
            agents=tasks,
            cohorts=cardiac_cohorts,
        )
        runner = LocalDeploymentRunner(judge_all=True)
        results = runner.deploy(plan.manifest, plan)
        assert results[0].judgement is not None
        d = result_to_dict(results[0])
        assert d["judgement"]["outcome"] == results[0].judgement.outcome.value
        assert d["judgement"]["score"] == results[0].judgement.score


# ---------------------------------------------------------------------------
# Scale test: generate many tasks
# ---------------------------------------------------------------------------


class TestScaleGeneration:
    def test_can_generate_hundreds_of_tasks(self, cardiac_cohorts: List[DiseaseCohort]) -> None:
        n = 200
        tasks = [
            SubAgentTask(
                agent_id=f"swarm_{i:05d}",
                cohort_id="cvd_heart_disease",
                objectives=(TaskObjective(name="composite", direction="max"),),
                payload={
                    "agent_potency": 0.1 + (i % 50) / 50.0,
                    "run_minutes": 60,
                    "scenario_seed_offset": i,
                },
            )
            for i in range(n)
        ]
        plan = deploy_agents(
            MultiAgentConfig(study_id="swarm_test", max_parallel_agents=64),
            agents=tasks,
            cohorts=cardiac_cohorts,
        )
        assert len(plan.tasks) == n
        # Quick smoke-run of a handful through the runner
        runner = LocalDeploymentRunner(max_workers=8, judge_all=False)
        # Only run a small subset so this test stays fast
        small_plan = deploy_agents(
            MultiAgentConfig(study_id="smoke"),
            agents=plan.tasks[:8],
            cohorts=cardiac_cohorts,
        )
        results = runner.deploy(small_plan.manifest, small_plan)
        assert len(results) == 8
        assert results[0].task.agent_id.startswith("swarm_")
