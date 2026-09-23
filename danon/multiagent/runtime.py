"""
Task runtime + local deployment runner.

A disease-agnostic runtime shell (`SubAgentRuntime`) lives in core.py.
This module adds the cardiac-specific task runtime (`_CVDTaskRuntime`),
the local threaded runner, and the default judge binding.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

from danon.multiagent.core import (
    AgentJudgement,
    DeploymentManifest,
    DeploymentPlan,
    DeploymentResult,
    DiseaseCohort,
    LocalDeploymentRunner,
    MultiAgentConfig,
    ResultAggregator,
    SubAgentRuntime,
    SubAgentSpec,
    SubAgentTask,
    TaskObjective,
    _pick_cohort,
    deploy_agents,
    TaskPriority,
)

from danon.multiagent.benchmarks import (
    BaseJudge,
    IschemicPlaqueScenarioBuilder,
    normalize_candidate_score,
)


# ---------------------------------------------------------------------------
# Per-task runtime (CVD-flavored)
# ---------------------------------------------------------------------------

class _CVDTaskRuntime(SubAgentRuntime):
    """Runs one cardiac sub-agent task through a plaque/perfusion scenario."""

    def __init__(
        self,
        *,
        config: MultiAgentConfig,
        cohort: DiseaseCohort,
        task: SubAgentTask,
        builder: Optional[IschemicPlaqueScenarioBuilder] = None,
    ) -> None:
        super().__init__(config=config, cohort=cohort, task=task)
        self.builder = builder or cohort.scenario_builder(task)
        self.set_validate(self._validate)
        self.error = ""

    def run(self) -> Dict[str, Any]:
        try:
            scenario = self.builder.build()
            # The agent's policy *is* the physical plaque ABM: each sub-agent is
            # a nanoparticle population with its own potency / avidity / VCAM-1
            # exposure, and it is scored by the same vectorised simulator that
            # the standalone swarm campaign uses. There is no separate surrogate
            # path, so a runtime result and a campaign result are comparable.
            result = self._simulate_policy(scenario)
            # Optionally merge any payload the agent returned.
            if isinstance(result, Mapping):
                merged = dict(scenario)
                merged.update(result)
                merged.setdefault("plaque_metrics", self.builder.score(scenario, merged))
                self.result = merged
                return merged
            return scenario
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            return {"error": str(exc), "plaque_metrics": {}}

    def _simulate_policy(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent's policy through the real plaque simulator.

        Metrics are read from the scenario the builder already simulated, so a
        task costs exactly one physical run rather than one per consumer.
        """
        pm = self.builder.score(scenario, scenario)
        return {
            "plaque_metrics": pm,
            "off_target_healthy_wall_adhesion_pct": pm.get(
                "off_target_healthy_wall_adhesion_pct", 0.0
            ),
            "swarm_config": scenario.get("swarm_config", {}),
            "scenario": scenario,
        }

    def _validate(self, result: Mapping[str, Any]) -> Tuple[bool, str]:
        metrics = result.get("plaque_metrics", {})
        if not isinstance(metrics, Mapping):
            return False, "plaque_metrics must be a mapping"
        reg = float(metrics.get("plaque_reduction_pct", 0.0))
        # a "successful" run has at least some measurable plaque change
        if reg <= 0.0:
            return False, f"no plaque reduction (got {reg})"
        return True, ""

    def validate(self, result: Mapping[str, Any]) -> bool:
        ok, _ = self._validate(result)
        return ok


# ---------------------------------------------------------------------------
# Local threaded deployment runner
# ---------------------------------------------------------------------------

class LocalDeploymentRunner:
    """Run a DeploymentPlan locally.

    Executor selection is evidence-driven. The shipped cardiac cohort scores
    each sub-agent with a vectorised plaque simulation whose inner loop is
    Python-level over ~200 steps of small numpy operations, and numpy releases
    the GIL only for large arrays, so threads mostly contend. Measured on 8
    tasks:

        serial          2.41 s
        threads (8)     4.33 s   (1.8x slower)
        processes (8)   7.24 s   (each worker rebuilds the derived reference
                                  cache from scratch under spawn)

    Work is therefore run in-process by default. ``max_workers`` only takes
    effect when ``executor`` explicitly requests a pool, so setting a
    parallelism number cannot silently make a run slower than not setting it.
    """

    def __init__(
        self,
        *,
        config: Optional[MultiAgentConfig] = None,
        max_workers: Optional[int] = None,
        judge_all: bool = True,
        executor: str = "auto",
    ) -> None:
        self.config = config or MultiAgentConfig()
        self.max_workers = max_workers if max_workers is not None else self.config.max_parallel_agents
        self.judge_all = judge_all
        self._judge = BaseJudge()
        if executor not in {"auto", "serial", "thread"}:
            raise ValueError(
                f"executor must be 'auto', 'serial' or 'thread', got {executor!r}"
            )
        self.executor_mode = executor

    def _resolve_executor(self, worker_count: int, n_tasks: int) -> str:
        """Pick an execution mode, saying so when an explicit request is ignored."""
        if self.executor_mode != "auto":
            return self.executor_mode
        if worker_count > 1 and n_tasks > 1:
            logger.info(
                "LocalDeploymentRunner: running %d tasks in-process. The built-in "
                "cohort simulator is CPU-bound and measured slower under both "
                "threads and processes; pass executor='thread' to force a pool.",
                n_tasks,
            )
        return "serial"

    def deploy(
        self, manifest: DeploymentManifest, plan: DeploymentPlan
    ) -> List[DeploymentResult]:
        results: List[DeploymentResult] = []
        if not plan.tasks:
            empty_task = SubAgentTask(agent_id="empty", cohort_id="empty")
            results.append(
                DeploymentResult(
                    task=empty_task,
                    runtime=SubAgentRuntime(
                        config=manifest.spec,
                        cohort=manifest.cohorts[0] if manifest.cohorts else DiseaseCohort(
                            id="empty", name="empty", description="",
                        ),
                        task=empty_task,
                    ),
                    result={},
                    succeeded=False,
                    error="No tasks in plan",
                )
            )
            return results

        start_all = time.time()
        worker_count = min(
            max(1, self.max_workers),
            len(plan.tasks),
            manifest.spec.max_parallel_agents,
        )
        mode = self._resolve_executor(worker_count, len(plan.tasks))
        if mode == "serial" or worker_count <= 1 or len(plan.tasks) <= 1:
            for task in plan.tasks:
                results.append(self._execute_one(task, manifest, start_all))
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = [pool.submit(self._execute_one, task, manifest, start_all) for task in plan.tasks]
                for future in futures:
                    results.append(future.result())
        return results

    def _execute_one(
        self,
        task: SubAgentTask,
        manifest: DeploymentManifest,
        start_all: float,
    ) -> DeploymentResult:
        cohort = _pick_cohort(task.cohort_id, manifest.cohorts)
        builder = cohort.scenario_builder(task)
        runtime = _CVDTaskRuntime(
            config=manifest.spec,
            cohort=cohort,
            task=task,
            builder=builder,
        )
        try:
            result = runtime.run()
            judgement: Optional[AgentJudgement] = None
            if self.judge_all:
                # Score against the scenario the runtime already simulated. Calling
                # builder.build() here would re-run the simulator for no reason.
                scenario = runtime.result if isinstance(runtime.result, Mapping) else result
                metrics = runtime.builder.score(scenario, result)
                judgement = self._judge(task, result, metrics)
            return DeploymentResult(
                task=task,
                runtime=runtime,
                result=result,
                judgement=judgement,
                elapsed_seconds=time.time() - start_all,
                succeeded=runtime.validate(result),
                error=runtime.error,
            )
        except Exception as exc:  # noqa: BLE001
            return DeploymentResult(
                task=task,
                runtime=runtime,
                result={},
                elapsed_seconds=time.time() - start_all,
                succeeded=False,
                error=str(exc),
            )

    def summarize(self, results: Sequence[DeploymentResult]) -> Dict[str, Any]:
        agg = ResultAggregator.aggregate(results)
        return {
            "count": agg.count,
            "succeeded": agg.succeeded,
            "failed": agg.count - agg.succeeded,
            "total_elapsed_seconds": agg.total_elapsed_seconds,
            "top_scores": [
                {"score": s, "agent_id": aid} for s, aid in agg.top_scores
            ],
            "summary": agg.summary,
        }
