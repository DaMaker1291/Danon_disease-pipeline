"""
Active Learner: Records computational predictions, ingests experimental results,
and refines model weights to close the compute→wet-lab feedback loop.

After each round of wet-lab validation:
  1. Record which predictions were correct/wrong
  2. Update scoring weights via Bayesian updating
  3. Regenerate candidate ranking with refined weights
  4. Suggest next-round experiments (top candidates not yet tested)
"""
import json
import os
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ExperimentalResult:
    round_number: int
    candidate_id: int
    tested_date: str
    in_vitro_cardiac_tropism: Optional[float] = None
    in_vitro_hepatic_uptake: Optional[float] = None
    in_vitro_immune_activation: Optional[float] = None
    in_vitro_lamp2b_expression: Optional[float] = None
    in_vivo_cardiac_transduction: Optional[float] = None
    in_vivo_liver_transduction: Optional[float] = None
    in_vivo_survival_days: Optional[float] = None
    in_vivo_lvmi_reduction_percent: Optional[float] = None
    toxicity_grade: Optional[int] = None
    passed_go_no_go: Optional[bool] = None
    notes: str = ""


@dataclass
class PredictionRecord:
    candidate_id: int
    round_number: int
    predicted_cardiac_tropism: float
    predicted_hepatic_avoidance: float
    predicted_immune_evasion: float
    predicted_lamp2b: float
    predicted_fitness: float
    actual: Optional[ExperimentalResult] = None
    error_cardiac: Optional[float] = None
    error_hepatic: Optional[float] = None


LEARNING_RATE_INITIAL = 0.15
WEIGHT_DECAY = 0.95


class ActiveLearner:
    def __init__(self, history_path: str = None):
        self.history_path = history_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "experimental_history.json"
        )
        self.history: List[PredictionRecord] = []
        self.bayesian_weights = {
            "cardiac_tropism": 0.30,
            "hepatic_avoidance": 0.20,
            "immune_evasion": 0.15,
            "lamp2b": 0.15,
            "structural": 0.10,
            "stability": 0.05,
            "promoter": 0.03,
            "mirna": 0.02,
        }
        self.round = 0
        self._load_history()

    def _load_history(self):
        if os.path.exists(self.history_path):
            with open(self.history_path) as f:
                raw = json.load(f)
            for r in raw:
                rec = PredictionRecord(**r)
                if rec.actual:
                    rec.actual = ExperimentalResult(**rec.actual)
                self.history.append(rec)
            self.round = max((r.round_number for r in self.history), default=0)
            logger.info("  ActiveLearner: loaded %d records (round %d)", len(self.history), self.round)

    def _save_history(self):
        raw = []
        for rec in self.history:
            d = rec.__dict__.copy()
            if d["actual"]:
                d["actual"] = d["actual"].__dict__
            raw.append(d)
        with open(self.history_path, "w") as f:
            json.dump(raw, f, indent=2, default=str)

    def record_prediction(self, candidate_id: int,
                          cardiac: float, hepatic: float,
                          immune: float, lamp2b: float,
                          fitness: float):
        rec = PredictionRecord(
            candidate_id=candidate_id,
            round_number=self.round,
            predicted_cardiac_tropism=cardiac,
            predicted_hepatic_avoidance=hepatic,
            predicted_immune_evasion=immune,
            predicted_lamp2b=lamp2b,
            predicted_fitness=fitness,
        )
        self.history.append(rec)
        return rec

    def submit_experimental_result(self, prediction_idx: int,
                                   result: ExperimentalResult):
        if 0 <= prediction_idx < len(self.history):
            rec = self.history[prediction_idx]
            rec.actual = result
            rec.error_cardiac = abs(rec.predicted_cardiac_tropism - (result.in_vitro_cardiac_tropism or 0))
            rec.error_hepatic = abs(rec.predicted_hepatic_avoidance - (1.0 - (result.in_vitro_hepatic_uptake or 0)))
            logger.info("  Feedback: C%d cardiac_err=%.3f hepatic_err=%.3f",
                         rec.candidate_id, rec.error_cardiac, rec.error_hepatic)
            self._update_weights()
            self._save_history()
            return True
        return False

    def _update_weights(self):
        completed = [r for r in self.history if r.actual is not None]
        if len(completed) < 3:
            return

        cardiac_errors = np.array([r.error_cardiac or 0.5 for r in completed])
        hepatic_errors = np.array([r.error_hepatic or 0.5 for r in completed])
        fitness_actual = np.array([
            (r.actual.in_vitro_cardiac_tropism or 0.5)
            if r.actual else 0.5
            for r in completed
        ])
        fitness_pred = np.array([r.predicted_fitness for r in completed])

        corr = float(np.clip(np.corrcoef(fitness_pred, fitness_actual)[0, 1], 0, 1))

        lr = max(LEARNING_RATE_INITIAL * (WEIGHT_DECAY ** self.round), 0.01)

        if np.mean(cardiac_errors) > 0.2:
            self.bayesian_weights["cardiac_tropism"] = max(
                self.bayesian_weights["cardiac_tropism"] - lr * 0.5, 0.05
            )
            self.bayesian_weights["hepatic_avoidance"] = min(
                self.bayesian_weights["hepatic_avoidance"] + lr * 0.3, 0.40
            )

        if np.mean(hepatic_errors) > 0.2:
            self.bayesian_weights["hepatic_avoidance"] = max(
                self.bayesian_weights["hepatic_avoidance"] - lr * 0.5, 0.05
            )

        total = sum(self.bayesian_weights.values())
        for k in self.bayesian_weights:
            self.bayesian_weights[k] /= total

        logger.info("  ActiveLearner: round=%d, n=%d, corr=%.3f, lr=%.4f",
                     self.round, len(completed), corr, lr)

    def get_refined_fitness(self, candidate) -> float:
        cardiac = getattr(candidate, "cardiac_tropism_score", 0.5)
        hepatic = getattr(candidate, "hepatic_avoidance_score", 0.5)
        immune = getattr(candidate, "immune_evasion_score", 0.5)
        lamp2b = getattr(candidate, "lamp2b_compatibility", 0.5)
        structural = getattr(candidate, "structural_score", 0.5)
        stability = getattr(candidate, "stability_score", 0.5)

        w = self.bayesian_weights
        return (
            w["cardiac_tropism"] * cardiac +
            w["hepatic_avoidance"] * hepatic +
            w["immune_evasion"] * immune +
            w["lamp2b"] * lamp2b +
            w["structural"] * structural +
            w["stability"] * stability +
            w["promoter"] * 0.5 +
            w["mirna"] * 0.5
        )

    def suggest_next_experiments(self, candidates: list, n: int = 10) -> List[tuple]:
        tested_ids = {r.candidate_id for r in self.history if r.actual is not None}
        untested = [c for c in candidates if getattr(c, "candidate_id", -1) not in tested_ids]
        untested.sort(key=lambda c: self.get_refined_fitness(c), reverse=True)
        return untested[:n]

    def experiment_protocol(self, candidate) -> Dict:
        return {
            "candidate_id": getattr(candidate, "candidate_id", 0),
            "sequence": getattr(candidate, "sequence", "")[:30] + "...",
            "predicted_fitness": round(getattr(candidate, "fitness", 0.5), 4),
            "assays": [
                {
                    "name": "Cardiomyocyte Transduction",
                    "cell_line": "hiPSC-cardiomyocytes (Fukushima or Coriell DMND line)",
                    "readout": "%GFP+ cells by flow cytometry",
                    "expected": f"{getattr(candidate, 'cardiac_tropism_score', 0.5)*100:.0f}%",
                    "duration_weeks": 3,
                    "cost_estimate_usd": 2500,
                },
                {
                    "name": "Hepatic Off-Target",
                    "cell_line": "HepG2",
                    "readout": "Vector genomes/cell by qPCR",
                    "expected": f"{(1-getattr(candidate, 'hepatic_avoidance_score', 0.5))*100:.0f}% of cardiac",
                    "duration_weeks": 2,
                    "cost_estimate_usd": 1500,
                },
                {
                    "name": "Immune Activation",
                    "cell_line": "PBMCs from 3 healthy donors",
                    "readout": "TNFα/IFNγ by ELISA",
                    "expected": f"{'Low' if getattr(candidate, 'immune_evasion_score', 0.5) > 0.5 else 'Moderate'}",
                    "duration_weeks": 2,
                    "cost_estimate_usd": 3000,
                },
                {
                    "name": "LAMP2B Protein Expression",
                    "cell_line": "LAMP2-KO HeLa + patient cardiomyocytes",
                    "readout": "LAMP2 Western blot + immunofluorescence",
                    "expected": f"{getattr(candidate, 'lamp2b_compatibility', 0.5)*100:.0f}% of normal",
                    "duration_weeks": 4,
                    "cost_estimate_usd": 4000,
                },
            ],
            "total_cost_usd": 11000,
            "timeline_weeks": 6,
            "go_no_go_criteria": "Cardiac transduction >30% AND hepatic <50% of cardiac AND LAMP2 expression >40% of normal",
        }

    def round_report(self) -> Dict:
        completed = [r for r in self.history if r.actual is not None]
        return {
            "round": self.round,
            "total_predictions": len(self.history),
            "validated": len(completed),
            "validation_rate": len(completed) / max(len(self.history), 1),
            "mean_cardiac_error": float(np.mean([r.error_cardiac or 0.5 for r in completed])) if completed else 0,
            "mean_hepatic_error": float(np.mean([r.error_hepatic or 0.5 for r in completed])) if completed else 0,
            "bayesian_weights": self.bayesian_weights,
        }
