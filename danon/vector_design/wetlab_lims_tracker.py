import os, json, logging, numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

ORDER_STATUS = ("designed", "ordered", "in_synthesis", "qc_pending", "qc_pass", "qc_fail", "in_assay", "completed", "archived")

ASSAY_TEMPLATES = {
    "trans_splicing_western": {
        "name": "Split-Intein Trans-Splicing Western Blot",
        "cell_line": "hiPSC-cardiomyocytes (Danon null mutation line)",
        "readout_type": "western_band_intensity",
        "unit": "fraction_full_length",
        "target": 0.70,
        "go_min": 0.50,
        "cost_usd": 4000,
        "duration_days": 21,
    },
    "lamp2b_immunofluorescence": {
        "name": "LAMP2B Lysosomal Localization IF",
        "cell_line": "hiPSC-cardiomyocytes (Danon null mutation line)",
        "readout_type": "colocalization_coefficient",
        "unit": "pearson_r",
        "target": 0.65,
        "go_min": 0.40,
        "cost_usd": 3500,
        "duration_days": 14,
    },
    "hepatic_leakage_qpcr": {
        "name": "HepG2 Off-Target qPCR",
        "cell_line": "HepG2",
        "readout_type": "vector_genomes_per_cell",
        "unit": "vg/cell",
        "target": 0.5,
        "go_min": 2.0,
        "cost_usd": 2500,
        "duration_days": 10,
    },
    "immune_activation_elisa": {
        "name": "PBMC Immune Activation ELISA",
        "cell_line": "PBMCs (3 healthy donors)",
        "readout_type": "tnfa_ifng_pg_ml",
        "unit": "pg/mL",
        "target": 50.0,
        "go_min": 200.0,
        "cost_usd": 3000,
        "duration_days": 10,
    },
    "cardiac_transduction_flow": {
        "name": "Cardiomyocyte Transduction Flow Cytometry",
        "cell_line": "hiPSC-cardiomyocytes",
        "readout_type": "percent_gfp_positive",
        "unit": "%",
        "target": 30.0,
        "go_min": 15.0,
        "cost_usd": 2500,
        "duration_days": 14,
    },
}

@dataclass
class AssayResult:
    assay_type: str
    candidate_id: int
    value: float
    passed_go: bool
    passed_target: bool
    deviation_from_prediction: float
    timestamp: str
    operator_notes: str = ""

@dataclass
class ConstructOrder:
    candidate_id: int
    vendor: str
    cost_usd: float
    estimated_delivery_days: int
    sequence_hash: str
    ordered_date: str
    status: str = "designed"
    qc_verified: bool = False
    assays: List[AssayResult] = field(default_factory=list)
    predicted_cardiac: float = 0.0
    predicted_hepatic: float = 0.0
    predicted_immune: float = 0.0
    predicted_lamp2b: float = 0.0
    predicted_fitness: float = 0.0

class WetlabLIMSTracker:
    def __init__(self, orders_path: str = None):
        self.orders_path = orders_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "wetlab_orders.json"
        )
        os.makedirs(os.path.dirname(self.orders_path), exist_ok=True)
        self.orders: Dict[int, ConstructOrder] = {}
        self.assay_history: Dict[str, List[AssayResult]] = {k: [] for k in ASSAY_TEMPLATES}
        self.go_no_go_gates: Dict[str, bool] = {}
        self._load_orders()

    def _load_orders(self):
        if os.path.exists(self.orders_path):
            with open(self.orders_path) as f:
                raw = json.load(f)
            for r in raw:
                order = ConstructOrder(**r)
                self.orders[order.candidate_id] = order
            logger.info("  LIMS: loaded %d construct orders", len(self.orders))

    def _save_orders(self):
        raw = [o.__dict__ for o in self.orders.values()]
        with open(self.orders_path, "w") as f:
            json.dump(raw, f, indent=2, default=str)

    def place_order(self, candidate_id: int, sequence: str, vendor: str = "PackGene",
                    cost_usd: float = 1493.0, predicted_cardiac: float = 0.0,
                    predicted_hepatic: float = 0.0, predicted_immune: float = 0.0,
                    predicted_lamp2b: float = 0.0, predicted_fitness: float = 0.0) -> ConstructOrder:
        order = ConstructOrder(
            candidate_id=candidate_id,
            vendor=vendor,
            cost_usd=cost_usd,
            estimated_delivery_days=21,
            sequence_hash=hash(sequence) % (2**32),
            ordered_date=datetime.now().isoformat(),
            status="ordered",
            predicted_cardiac=predicted_cardiac,
            predicted_hepatic=predicted_hepatic,
            predicted_immune=predicted_immune,
            predicted_lamp2b=predicted_lamp2b,
            predicted_fitness=predicted_fitness,
        )
        self.orders[candidate_id] = order
        self._save_orders()
        logger.info("  LIMS: order placed — C%d, $%.0f, %s", candidate_id, cost_usd, vendor)
        return order

    def update_status(self, candidate_id: int, status: str):
        if candidate_id in self.orders:
            self.orders[candidate_id].status = status
            self._save_orders()
            logger.info("  LIMS: C%d status -> %s", candidate_id, status)

    def submit_assay(self, candidate_id: int, assay_type: str, value: float,
                     operator_notes: str = "") -> AssayResult:
        template = ASSAY_TEMPLATES.get(assay_type)
        if template is None:
            raise ValueError(f"Unknown assay type: {assay_type}")

        pred_map = {
            "trans_splicing_western": self.orders[candidate_id].predicted_fitness if candidate_id in self.orders else 0.5,
            "hepatic_leakage_qpcr": 1.0 - self.orders[candidate_id].predicted_hepatic if candidate_id in self.orders else 0.5,
            "cardiac_transduction_flow": self.orders[candidate_id].predicted_cardiac * 100 if candidate_id in self.orders else 30.0,
            "immune_activation_elisa": (1.0 - self.orders[candidate_id].predicted_immune) * 200 if candidate_id in self.orders else 100.0,
            "lamp2b_immunofluorescence": self.orders[candidate_id].predicted_lamp2b if candidate_id in self.orders else 0.5,
        }
        predicted = pred_map.get(assay_type, 0.5)

        comparison = template["target"]
        go_min = template["go_min"]
        if assay_type in ("hepatic_leakage_qpcr", "immune_activation_elisa"):
            passed_go = value <= go_min
            passed_target = value <= comparison
            deviation = (value - predicted) / max(predicted, 0.01)
        else:
            passed_go = value >= go_min
            passed_target = value >= comparison
            deviation = (value - predicted) / max(predicted, 0.01)

        result = AssayResult(
            assay_type=assay_type,
            candidate_id=candidate_id,
            value=value,
            passed_go=passed_go,
            passed_target=passed_target,
            deviation_from_prediction=float(deviation),
            timestamp=datetime.now().isoformat(),
            operator_notes=operator_notes,
        )

        if candidate_id in self.orders:
            self.orders[candidate_id].assays.append(result)
        self.assay_history[assay_type].append(result)
        self._save_orders()

        status = "PASS" if passed_go else "FAIL"
        logger.info("  LIMS assay: C%d %s -> %.2f (%s, dev=%.2f)",
                     candidate_id, assay_type, value, status, deviation)
        return result

    def evaluate_candidate(self, candidate_id: int) -> Dict:
        if candidate_id not in self.orders:
            return {"status": "not_found"}
        order = self.orders[candidate_id]
        results = {}
        all_pass = True
        for a in order.assays:
            tpl = ASSAY_TEMPLATES.get(a.assay_type)
            if tpl and not a.passed_go:
                all_pass = False
            results[a.assay_type] = {
                "value": a.value, "target": tpl["target"] if tpl else None,
                "passed": a.passed_go, "deviation": a.deviation_from_prediction,
            }
        return {
            "candidate_id": candidate_id,
            "status": order.status,
            "assays_completed": len(order.assays),
            "all_go_gates_passed": all_pass,
            "results": results,
            "qc_verified": order.qc_verified,
            "total_cost": sum(ASSAY_TEMPLATES[a.assay_type]["cost_usd"] for a in order.assays if a.assay_type in ASSAY_TEMPLATES),
        }

    def pipeline_dashboard(self) -> Dict:
        total = len(self.orders)
        ordered = sum(1 for o in self.orders.values() if o.status == "ordered")
        in_synthesis = sum(1 for o in self.orders.values() if o.status == "in_synthesis")
        qc_pass = sum(1 for o in self.orders.values() if o.status == "qc_pass")
        in_assay = sum(1 for o in self.orders.values() if o.status == "in_assay")
        completed = sum(1 for o in self.orders.values() if o.status == "completed")
        passed_all = sum(1 for o in self.orders.values() if self.evaluate_candidate(o.candidate_id).get("all_go_gates_passed"))
        total_spend = sum(o.cost_usd for o in self.orders.values()) + sum(
            ASSAY_TEMPLATES[a.assay_type]["cost_usd"]
            for o in self.orders.values() for a in o.assays if a.assay_type in ASSAY_TEMPLATES
        )
        return {
            "total_constructs": total, "ordered": ordered, "in_synthesis": in_synthesis,
            "qc_pass": qc_pass, "in_assay": in_assay, "completed": completed,
            "passed_go_gates": passed_all, "total_spend_usd": total_spend,
            "mean_pred_vs_actual_error": float(np.mean([
                abs(a.deviation_from_prediction) for aa in self.assay_history.values()
                for a in aa
            ])) if any(self.assay_history.values()) else 0.0,
        }

    def order_summary_for_vendor(self, vendor: str = "PackGene") -> List[Dict]:
        return [
            {"candidate_id": o.candidate_id, "cost": o.cost_usd,
             "status": o.status, "sequence_hash": o.sequence_hash}
            for o in self.orders.values() if o.vendor == vendor
        ]

    def compare_predictions_vs_actual(self) -> Dict:
        comparisons = []
        for cid, order in self.orders.items():
            for a in order.assays:
                pred_map = {
                    "trans_splicing_western": order.predicted_fitness,
                    "cardiac_transduction_flow": order.predicted_cardiac * 100,
                    "hepatic_leakage_qpcr": 1.0 - order.predicted_hepatic,
                    "immune_activation_elisa": (1.0 - order.predicted_immune) * 200,
                    "lamp2b_immunofluorescence": order.predicted_lamp2b,
                }
                predicted = pred_map.get(a.assay_type, 0.0)
                comparisons.append({
                    "candidate_id": cid, "assay": a.assay_type,
                    "predicted": predicted, "actual": a.value,
                    "error": a.value - predicted,
                    "abs_pct_error": abs(a.value - predicted) / max(abs(predicted), 0.01),
                })
        if not comparisons:
            return {"n": 0, "mean_abs_pct_error": 0.0}
        errors = [c["abs_pct_error"] for c in comparisons]
        return {
            "n": len(comparisons),
            "mean_abs_pct_error": float(np.mean(errors)),
            "median_abs_pct_error": float(np.median(errors)),
            "max_abs_pct_error": float(np.max(errors)),
        }
