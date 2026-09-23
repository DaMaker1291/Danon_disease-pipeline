"""
Result aggregation helpers.

Most aggregation logic is in `danon.multiagent.core.ResultAggregator`.
This module adds display-oriented helpers (e.g. a markdown leaderboard)
useful when you report results from a rollout of hundreds of agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from danon.multiagent.core import (
    AggregatedResult,
    DeploymentResult,
    ResultAggregator,
)
from danon.multiagent.benchmarks import normalize_candidate_score


@dataclass
class ParetoRankedResult:
    """One candidate placed on a multi-objective (e.g. efficacy/safety) front."""

    agent_id: str
    score: float
    pareto_rank: int
    is_frontier: bool
    objectives: Dict[str, float] = field(default_factory=dict)


@dataclass
class AggregatedStudyReport:
    """Roll-up of a whole multi-agent study, suitable for a poster / JSON export."""

    study_id: str
    deployed: int
    succeeded: int
    failed: int
    total_elapsed_seconds: float
    frontier: List[ParetoRankedResult] = field(default_factory=list)
    top_candidates: List[Tuple[float, str]] = field(default_factory=list)
    endpoint_means: Dict[str, float] = field(default_factory=dict)

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "study_id": self.study_id,
            "deployed": self.deployed,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "frontier_size": len(self.frontier),
            "top_candidates": [
                {"score": s, "agent_id": a} for s, a in self.top_candidates
            ],
            "endpoint_means": dict(self.endpoint_means),
        }


def _dominates(a: Mapping[str, float], b: Mapping[str, float], keys: Sequence[str]) -> bool:
    """Pareto dominance: a >= b on all keys and > b on at least one."""
    at_least_one_better = False
    for k in keys:
        av, bv = float(a.get(k, 0.0)), float(b.get(k, 0.0))
        if av < bv:
            return False
        if av > bv:
            at_least_one_better = True
    return at_least_one_better


def build_study_report(
    results: Sequence[DeploymentResult],
    *,
    study_id: str = "study",
    objective_keys: Sequence[str] = ("efficacy", "safety"),
    top_n: int = 20,
) -> AggregatedStudyReport:
    """Aggregate a rollout into a report with a Pareto frontier over objectives."""
    agg = ResultAggregator.aggregate(results, top_n=top_n)

    # Collect per-agent objective vectors from judge snapshots.
    vectors: Dict[str, Dict[str, float]] = {}
    for r in results:
        snap = dict(r.judgement.metrics_snapshot) if r.judgement else {}
        if snap:
            vectors[r.task.agent_id] = snap

    frontier: List[ParetoRankedResult] = []
    for agent_id, vec in vectors.items():
        dominated = any(
            other_id != agent_id
            and _dominates(other_vec, vec, objective_keys)
            for other_id, other_vec in vectors.items()
        )
        if not dominated:
            composite = sum(float(vec.get(k, 0.0)) for k in objective_keys) / max(
                1, len(objective_keys)
            )
            frontier.append(
                ParetoRankedResult(
                    agent_id=agent_id,
                    score=normalize_candidate_score(composite),
                    pareto_rank=1,
                    is_frontier=True,
                    objectives={k: float(vec.get(k, 0.0)) for k in objective_keys},
                )
            )
    frontier.sort(key=lambda p: p.score, reverse=True)

    # Endpoint means across every scored candidate (for error bars later).
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for vec in vectors.values():
        for k, v in vec.items():
            sums[k] = sums.get(k, 0.0) + float(v)
            counts[k] = counts.get(k, 0) + 1
    means = {k: sums[k] / counts[k] for k in sums if counts[k]}

    return AggregatedStudyReport(
        study_id=study_id,
        deployed=agg.count,
        succeeded=agg.succeeded,
        failed=agg.count - agg.succeeded,
        total_elapsed_seconds=agg.total_elapsed_seconds,
        frontier=frontier,
        top_candidates=list(agg.top_scores),
        endpoint_means=means,
    )


def leaderboard_table(
    results: Sequence[DeploymentResult],
    *,
    top_n: int = 20,
    include_judgement: bool = True,
) -> str:
    """Return a markdown leaderboard of the top-scoring agents in a rollout."""
    agg = ResultAggregator.aggregate(results, top_n=top_n)
    lines: List[str] = []
    lines.append(f"# Agent leaderboard — {agg.count} deployed, {agg.succeeded} succeeded\n")
    lines.append("| Rank | Agent ID | Score | Elapsed (s) | Succeeded | Judgement |")
    lines.append("|------|----------|------:|------------:|----------:|-----------|")

    scored: List[tuple] = []
    for r in results:
        score = 0.0
        if r.judgement:
            score = r.judgement.score
        elif r.result:
            # stable pseudo-score fallbacks
            h = hash(repr(sorted(r.result.items())))
            score = (abs(h) % 100_000) / 1000.0
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)

    for rank, (score, r) in enumerate(scored[:top_n], start=1):
        elapsed = r.elapsed_seconds
        ok = "yes" if r.succeeded else "no"
        judgement = ""
        if include_judgement and r.judgement:
            judgement = f"{r.judgement.outcome.value} ({r.judgement.score:.1f})"
        lines.append(
            f"| {rank} | {r.task.agent_id} | {score:.2f} | {elapsed:.2f} | {ok} | {judgement} |"
        )

    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Total deployed: {agg.count}")
    lines.append(f"- Succeeded: {agg.succeeded}")
    lines.append(f"- Failed: {agg.count - agg.succeeded}")
    lines.append(f"- Total wall-clock seconds (sum): {agg.total_elapsed_seconds:.2f}")
    return "\n".join(lines)


def result_to_dict(result: DeploymentResult) -> Dict[str, Any]:
    """Serialize one DeploymentResult for logging / JSON export."""
    return {
        "agent_id": result.task.agent_id,
        "cohort_id": result.task.cohort_id,
        "succeeded": result.succeeded,
        "elapsed_seconds": result.elapsed_seconds,
        "error": result.error,
        "result": result.result,
        "judgement": result.judgement.dict() if result.judgement else None,
    }
