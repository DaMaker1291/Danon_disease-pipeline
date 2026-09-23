"""
Multi-agent deployment framework.

Disease-agnostic: the same Agent/Task/Judge runtime can be pointed at
CVD, ischemic heart disease, oncology, neurodegeneration, etc. by swapping
the cohort and benchmark factories. Cardiac-specific helpers live in
`benchmarks.py` so this file stays generic.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Type, Union

import importlib
import sys
import threading
import warnings
from enum import Enum


# ---------------------------------------------------------------------------
# Outcomes / judgement
# ---------------------------------------------------------------------------

class AgentJudgementOutcome(str, Enum):
    NOVEL = "novel"
    SAFE = "safe"
    REPRODUCIBLE = "reproducible"
    HIGH_EFFICACY = "high_efficacy"
    WEAK = "weak"
    REJECT = "reject"
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    HIGH_REPRODUCIBILITY = "high_reproducibility"
    LOW_REPRODUCIBILITY = "low_reproducibility"


class LocalDeploymentRunner:
    """Stub base so callers can depend on the name before the real impl."""
    def __init__(self, **kwargs: Any) -> None:
        pass

    def deploy(
        self, manifest: DeploymentManifest, plan: DeploymentPlan
    ) -> List[DeploymentResult]:
        # Back-compat shim: delegate to the runtime module's real runner.
        try:
            from danon.multiagent.runtime import LocalDeploymentRunner as _Runner
        except Exception:
            raise NotImplementedError from None
        runner = _Runner()
        return runner.deploy(manifest, plan)



@dataclass(frozen=True)
class AgentJudgement:
    """How a judge scores a completed sub-agent result."""

    outcome: AgentJudgementOutcome
    score: float
    rationale: str
    metrics_snapshot: Dict[str, float] = field(default_factory=dict)

    def dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "score": self.score,
            "rationale": self.rationale,
            "metrics": dict(self.metrics_snapshot),
        }


# ---------------------------------------------------------------------------
# Tasks & objectives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskObjective:
    """One measurable goal a sub-agent is asked to optimize."""

    name: str
    direction: str  # "max" or "min"
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.direction not in ("max", "min"):
            raise ValueError(f"direction must be 'max' or 'min', got {self.direction!r}")


@dataclass(frozen=True)
class SubAgentTask:
    """A concrete problem instance handed to a single deployed agent."""

    agent_id: str
    cohort_id: str
    objectives: Tuple[TaskObjective, ...] = field(default_factory=tuple)
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def primary_objective(self) -> Optional[TaskObjective]:
        return self.objectives[0] if self.objectives else None


# ---------------------------------------------------------------------------
# Disease cohort (abstract)
# ---------------------------------------------------------------------------

class DiseaseCohort:
    """Abstract base for a disease-focused cohort.

    Each cohort owns (1) how to build benchmark scenarios for its disease
    area and (2) a mechanism to score/validate an agent result in-domain.
    """

    id: str
    name: str
    description: str

    def __init__(self, *, id: str, name: str, description: str = "") -> None:
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)

    # -- benchmark interface -------------------------------------------------
    def scenario_builder(self, task: SubAgentTask) -> "ScenarioBuilder":
        """Return a builder that can generate a concrete simulation scenario."""
        raise NotImplementedError

    def validate_result(self, task: SubAgentTask, result: Mapping[str, Any]) -> Tuple[bool, str]:
        """Domain-specific sanity check on a completed result."""
        return True, ""

    # -- scoring interface ---------------------------------------------------
    def score(self, task: SubAgentTask, result: Mapping[str, Any]) -> Dict[str, float]:
        """Return domain metrics dict; keys should be comparable across cohorts."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Agent spec & deployment plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubAgentSpec:
    """Static description of an agent type you want to deploy at scale."""

    id: str
    cohort_id: str
    label: str = ""
    template_task: Optional[SubAgentTask] = None
    max_parallel_agents: int = 1
    fail_fast: bool = False


@dataclass(frozen=True)
class DeploymentManifest:
    """Top-level deployment descriptor."""

    study_id: str
    spec: MultiAgentConfig
    cohorts: List[DiseaseCohort]

    def cohort_by_id(self, cohort_id: str) -> DiseaseCohort:
        for c in self.cohorts:
            if c.id == cohort_id:
                return c
        raise KeyError(f"cohort {cohort_id!r} not in manifest")


@dataclass(frozen=True)
class DeploymentPlan:
    """Resolved list of tasks for a single deployment run."""

    study_id: str
    manifest: DeploymentManifest
    tasks: List[SubAgentTask]


@dataclass
class DeploymentResult:
    """Outcome of running one sub-agent task."""

    task: SubAgentTask
    runtime: "SubAgentRuntime"
    result: Dict[str, Any]
    judgement: Optional[AgentJudgement] = None
    elapsed_seconds: float = 0.0
    succeeded: bool = False
    error: str = ""


@dataclass
class SubAgentRuntime:
    """Lifecycle + bookkeeping for one agent execution."""

    config: MultiAgentConfig
    cohort: DiseaseCohort
    task: SubAgentTask
    builder: Optional["ScenarioBuilder"] = None
    result: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    start_ts: float = field(default_factory=time.time)
    end_ts: float = 0.0
    _validate_fn: Optional[Callable[[Mapping[str, Any]], Tuple[bool, str]]] = None

    def set_validate(self, fn: Callable[[Mapping[str, Any]], Tuple[bool, str]]) -> None:
        self._validate_fn = fn

    def validate(self, result: Mapping[str, Any]) -> bool:
        if self._validate_fn:
            ok, _ = self._validate_fn(result)
            return ok
        ok, _ = self.cohort.validate_result(self.task, result)
        return ok

    @property
    def elapsed(self) -> float:
        end = self.end_ts or time.time()
        return max(0.0, end - self.start_ts)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MultiAgentConfig:
    """Run-level knobs for a deployment."""

    study_id: str = "study"
    max_parallel_agents: int = 1
    agent_timeout_seconds: float = 120.0
    fail_fast: bool = False
    checkpoint_dir: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scenario builder (abstract)
# ---------------------------------------------------------------------------

class ScenarioBuilder:
    """Abstract scenario factory; cardiac-specific subclasses live in benchmarks.py."""

    def __init__(self, *, task: SubAgentTask, cohort: DiseaseCohort) -> None:
        self.task = task
        self.cohort = cohort

    def build(self) -> Dict[str, Any]:
        raise NotImplementedError

    def score(self, scenario: Dict[str, Any], result: Mapping[str, Any]) -> Dict[str, float]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

@dataclass
class AggregatedResult:
    count: int
    succeeded: int
    total_elapsed_seconds: float
    top_scores: List[Tuple[float, str]]
    summary: Dict[str, Any] = field(default_factory=dict)


class ResultAggregator:
    """Reduce a list of DeploymentResults into a small summary."""

    @staticmethod
    def aggregate(results: Sequence[DeploymentResult], *, top_n: int = 10) -> AggregatedResult:
        count = len(results)
        succeeded = sum(1 for r in results if r.succeeded)
        elapsed = sum(r.elapsed_seconds for r in results)
        scored: List[Tuple[float, str]] = []
        for r in results:
            if r.judgement:
                scored.append((r.judgement.score, r.task.agent_id))
            elif r.result:
                # crude fallback: hash the result into a stable pseudo-score
                h = int(hashlib.sha256(repr(sorted(r.result.items())).encode()).hexdigest(), 16)
                scored.append((float(h % 100_000) / 100_000.0, r.task.agent_id))
        scored.sort(reverse=True)
        return AggregatedResult(
            count=count,
            succeeded=succeeded,
            total_elapsed_seconds=elapsed,
            top_scores=scored[:top_n],
            summary={
                "succeeded": succeeded,
                "failed": count - succeeded,
                "total_elapsed_seconds": elapsed,
            },
        )


# ---------------------------------------------------------------------------
# Agent composition helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentNode:
    """A point of composition in a bigger agent graph."""

    id: str
    runs: Union[SubAgentTask, Sequence[SubAgentTask], Callable[[], Iterator[SubAgentTask]]]
    children: Tuple["AgentNode", ...] = field(default_factory=tuple)
    aggregate: Optional[Callable[[Sequence[Any]], Any]] = None

    def iter_tasks(self) -> Iterator[SubAgentTask]:
        if callable(self.runs):
            yield from self.runs()
        elif isinstance(self.runs, SubAgentTask):
            yield self.runs
        else:
            yield from self.runs


@dataclass(frozen=True)
class AgentProgram:
    """A DAG / tree of AgentNodes compiled into a flat task list."""

    root: AgentNode
    cohort_id: str

    def expand(self) -> List[SubAgentTask]:
        tasks: List[SubAgentTask] = []
        stack: List[AgentNode] = [self.root]
        while stack:
            node = stack.pop()
            for t in node.iter_tasks():
                tasks.append(t)
            for child in reversed(node.children):
                stack.append(child)
        return tasks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TaskPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


__all__ = [
    "AgentJudgement",
    "AgentJudgementOutcome",
    "AgentNode",
    "AgentProgram",
    "AggregatedResult",
    "DeploymentManifest",
    "DeploymentPlan",
    "DeploymentResult",
    "DiseaseCohort",
    "LocalDeploymentRunner",
    "MultiAgentConfig",
    "ResultAggregator",
    "ScenarioBuilder",
    "SubAgentRuntime",
    "SubAgentSpec",
    "SubAgentTask",
    "TaskObjective",
    "TaskPriority",
    "TaskStatus",
    "deploy_agents",
]


def _pick_cohort(cohort_id: str, cohorts: Sequence[DiseaseCohort]) -> DiseaseCohort:
    """Pick a cohort by id; delegates to runtime if available."""
    for c in cohorts:
        if c.id == cohort_id:
            return c
    raise KeyError(f"cohort {cohort_id!r} not available")


def deploy_agents(
    config: MultiAgentConfig,
    agents: Sequence[SubAgentSpec],
    cohorts: Sequence[DiseaseCohort],
    *,
    study_id: Optional[str] = None,
) -> DeploymentPlan:
    """Compose a DeploymentPlan from specs + cohorts; execution is separate."""
    sid = study_id or config.study_id
    manifest = DeploymentManifest(study_id=sid, spec=config, cohorts=list(cohorts))

    # (Re)bind LocalDeploymentRunner in core's namespace so aggregator/test code
    # that imports from core gets the real runtime impl.
    try:
        from danon.multiagent.runtime import LocalDeploymentRunner as _RealRunner
    except Exception:
        _RealRunner = LocalDeploymentRunner
    core_mod = sys.modules.get(__name__)
    if core_mod is not None and hasattr(core_mod, "LocalDeploymentRunner"):
        setattr(core_mod, "LocalDeploymentRunner", _RealRunner)

    by_cohort: Dict[str, DiseaseCohort] = {c.id: c for c in cohorts}
    tasks: List[SubAgentTask] = []
    for item in agents:
        # Accept either a ready-made SubAgentTask (already a concrete problem
        # instance) or a SubAgentSpec (an agent *type* to be instantiated).
        if isinstance(item, SubAgentTask):
            if item.cohort_id not in by_cohort:
                raise KeyError(
                    f"task {item.agent_id!r} references unknown cohort {item.cohort_id!r}"
                )
            tasks.append(
                SubAgentTask(
                    agent_id=item.agent_id,
                    cohort_id=item.cohort_id,
                    objectives=item.objectives,
                    payload=dict(item.payload),
                    metadata=dict(item.metadata),
                )
            )
            continue

        spec = item
        if spec.cohort_id not in by_cohort:
            raise KeyError(f"agent {spec.id!r} references unknown cohort {spec.cohort_id!r}")
        template = spec.template_task
        if template is None:
            template = SubAgentTask(agent_id=spec.id, cohort_id=spec.cohort_id)
        # clone template but tag it with the agent id + cohort
        tasks.append(
            SubAgentTask(
                agent_id=spec.id,
                cohort_id=spec.cohort_id,
                objectives=template.objectives,
                payload=dict(template.payload),
                metadata={
                    **template.metadata,
                    "label": spec.label or spec.id,
                    "max_parallel": spec.max_parallel_agents,
                    "fail_fast": spec.fail_fast,
                },
            )
        )

    warnings.warn(
        "deploy_agents is deprecated; build a DeploymentPlan directly and run it "
        "with LocalDeploymentRunner.",
        DeprecationWarning,
        stacklevel=2,
    )
    return DeploymentPlan(study_id=sid, manifest=manifest, tasks=tasks)
