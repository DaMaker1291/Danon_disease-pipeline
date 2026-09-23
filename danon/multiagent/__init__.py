"""
Danon multi-agent deployment framework.

Disease-agnostic by design: the same Agent/Task/Judge runtime can target
CVD, ischemic heart disease, oncology, neurodegeneration, etc., by
swapping the cohort and benchmark factories.

Cardiac / ischemic-heart-disease cohorts and plaque/fluid-dynamics
benchmarks live in `danon.multiagent.benchmarks`.
"""

from danon.multiagent.core import (
    AgentJudgement,
    AgentJudgementOutcome,
    AgentNode,
    AgentProgram,
    AggregatedResult,
    DeploymentManifest,
    DeploymentPlan,
    DeploymentResult,
    DiseaseCohort,
    LocalDeploymentRunner,
    MultiAgentConfig,
    ResultAggregator,
    ScenarioBuilder,
    SubAgentRuntime,
    SubAgentSpec,
    SubAgentTask,
    TaskObjective,
    TaskPriority,
    TaskStatus,
    _pick_cohort,
    deploy_agents,
)
from danon.multiagent.benchmarks import (
    BaseJudge,
    CohortCriteria,
    CohortEndpoints,
    CVDHeartDiseaseCohort,
    IschemicHeartDiseaseCohort,
    IschemicPlaqueScenarioBuilder,
    PlaqueSurrogate,
    FlowState,
    build_cohorts,
    build_cvd_cohort,
    normalize_candidate_score,
)
from danon.multiagent.aggregator import (
    AggregatedStudyReport,
    ParetoRankedResult,
    leaderboard_table,
    result_to_dict,
)

__all__ = [
    "AgentJudgement",
    "AgentJudgementOutcome",
    "AgentNode",
    "AgentProgram",
    "AggregatedResult",
    "AggregatedStudyReport",
    "BaseJudge",
    "CohortCriteria",
    "CohortEndpoints",
    "CVDHeartDiseaseCohort",
    "DeploymentManifest",
    "DeploymentPlan",
    "DeploymentResult",
    "DiseaseCohort",
    "FlowState",
    "IschemicHeartDiseaseCohort",
    "IschemicPlaqueScenarioBuilder",
    "LocalDeploymentRunner",
    "MultiAgentConfig",
    "ParetoRankedResult",
    "PlaqueSurrogate",
    "ResultAggregator",
    "ScenarioBuilder",
    "SubAgentRuntime",
    "SubAgentSpec",
    "SubAgentTask",
    "TaskObjective",
    "TaskPriority",
    "TaskStatus",
    "build_cohorts",
    "build_cvd_cohort",
    "deploy_agents",
    "leaderboard_table",
    "normalize_candidate_score",
    "result_to_dict",
]
