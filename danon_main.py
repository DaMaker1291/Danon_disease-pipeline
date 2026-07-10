import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from danon.config import DanonConfig, danon_config
from danon.aav_generator import DanonAAVGenerator
from danon.lnp_generator import DanonLNPGenerator
from danon.tropism_filter import DanonTropismFilter
from danon.safety_engine import DanonSafetyEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("danon_pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)


class DanonPipeline:
    def __init__(self, config: DanonConfig = None):
        self.config = config or danon_config
        self.aav_gen = DanonAAVGenerator(self.config)
        self.lnp_gen = DanonLNPGenerator(self.config)
        self.tropism_filter = DanonTropismFilter(self.config)
        self.safety_engine = DanonSafetyEngine(self.config)
        self.stats = {"start_time": None, "phases": {}, "candidates": {}}

    def run(self):
        self.stats["start_time"] = datetime.now().isoformat()
        logger.info("=" * 80)
        logger.info("DANON DISEASE GENE THERAPY DISCOVERY PLATFORM")
        logger.info("Target: AAV9-LAMP2B | Vector: AAV9 | Payload: LAMP2B_Transgene")
        logger.info("Regulatory: %s", self.config.regulatory_framework)
        logger.info("=" * 80)

        phase1 = self.phase1_generate()
        phase2 = self.phase2_immune_filter(phase1)
        phase3 = self.phase3_tropism_filter(phase2)
        phase4 = self.phase4_safety_screen(phase3)
        winners = self.phase5_select_winners(phase4)

        self.stats["end_time"] = datetime.now().isoformat()
        self._save_report(winners)
        self._print_summary()
        return winners

    def phase1_generate(self) -> dict:
        logger.info("PHASE 1: Generating AAV9-LAMP2B + LNP candidates")

        aav_count = 0
        aav_candidates = []
        for batch in self.aav_gen.stream_candidates(
            self.config.aav_total_candidates, self.config.batch_size
        ):
            aav_candidates.extend(batch)
            aav_count += len(batch)
            if aav_count % 100_000 == 0:
                logger.info("AAV generated: %d / %d", aav_count, self.config.aav_total_candidates)

        lnp_count = 0
        lnp_candidates = []
        for batch in self.lnp_gen.stream_candidates(
            self.config.lnp_total_candidates, self.config.batch_size
        ):
            lnp_candidates.extend(batch)
            lnp_count += len(batch)
            if lnp_count % 100_000 == 0:
                logger.info("LNP generated: %d / %d", lnp_count, self.config.lnp_total_candidates)

        self.stats["candidates"]["phase1_aav"] = len(aav_candidates)
        self.stats["candidates"]["phase1_lnp"] = len(lnp_candidates)
        logger.info("Phase 1: %d AAV + %d LNP", len(aav_candidates), len(lnp_candidates))

        return {"aav": aav_candidates, "lnp": lnp_candidates}

    def phase2_immune_filter(self, phase1: dict) -> dict:
        logger.info("PHASE 2: Immune Evasion Filter (Gate Crash Test)")

        aav_passed = []
        for c in phase1["aav"]:
            if c.immune_evasion_score >= 0.48:
                aav_passed.append(c)

        lnp_passed = phase1["lnp"]

        self.stats["candidates"]["phase2_aav"] = len(aav_passed)
        self.stats["candidates"]["phase2_lnp"] = len(lnp_passed)

        pct = 100 * len(aav_passed) / max(len(phase1["aav"]), 1)
        logger.info("Phase 2: AAV %d -> %d (%.2f%%)",
                     len(phase1["aav"]), len(aav_passed), pct)

        return {"aav": aav_passed, "lnp": lnp_passed}

    def phase3_tropism_filter(self, phase2: dict) -> dict:
        logger.info("PHASE 3: Cardiac Tropism Filter (ZIP Code Test)")

        aav_passed = [c for c in phase2["aav"] if self.tropism_filter.passes(c)]

        lnp_passed = [c for c in phase2["lnp"]
                       if c.cardiac_delivery_score >= self.config.min_cardiac_tropism]

        self.stats["candidates"]["phase3_aav"] = len(aav_passed)
        self.stats["candidates"]["phase3_lnp"] = len(lnp_passed)

        pct = 100 * len(aav_passed) / max(len(phase2["aav"]), 1)
        logger.info("Phase 3: AAV %d -> %d (%.2f%%)",
                     len(phase2["aav"]), len(aav_passed), pct)

        return {"aav": aav_passed, "lnp": lnp_passed}

    def phase4_safety_screen(self, phase3: dict) -> dict:
        logger.info("PHASE 4: Danon Safety & Regulatory Compliance Screen")

        aav_safe = []
        for c in phase3["aav"]:
            profile = self.safety_engine.evaluate(c)
            if profile.regulatory_compliant:
                aav_safe.append(c)

        lnp_safe = []
        for c in phase3["lnp"]:
            if c.hepatic_avoidance_score >= 0.85 and c.cardiac_delivery_score >= 0.70:
                lnp_safe.append(c)

        self.stats["candidates"]["phase4_aav"] = len(aav_safe)
        self.stats["candidates"]["phase4_lnp"] = len(lnp_safe)

        logger.info("Phase 4: AAV %d -> %d safe | LNP %d -> %d safe",
                     len(phase3["aav"]), len(aav_safe),
                     len(phase3["lnp"]), len(lnp_safe))

        return {"aav": aav_safe, "lnp": lnp_safe}

    def phase5_select_winners(self, phase4: dict) -> dict:
        logger.info("PHASE 5: Selecting Top Candidates")

        top_aav = sorted(
            phase4["aav"], key=lambda c: c.fitness, reverse=True
        )[:1000]

        top_lnp = sorted(
            phase4["lnp"], key=lambda c: c.fitness, reverse=True
        )[:1000]

        self.stats["candidates"]["winners_aav"] = len(top_aav)
        self.stats["candidates"]["winners_lnp"] = len(top_lnp)

        winners = {
            "aav": [
                {
                    "candidate_id": c.candidate_id,
                    "sequence": c.sequence,
                    "mutations": len(c.mutations),
                    "fitness": round(c.fitness, 4),
                    "cardiac_tropism": round(c.cardiac_tropism_score, 4),
                    "hepatic_avoidance": round(c.hepatic_avoidance_score, 4),
                    "immune_evasion": round(c.immune_evasion_score, 4),
                    "lamp2b_compatibility": round(c.lamp2b_compatibility, 4),
                }
                for c in top_aav
            ],
            "lnp": [
                {
                    "candidate_id": c.candidate_id,
                    "ionizable_lipid": c.ionizable_lipid,
                    "peg_lipid": c.peg_lipid,
                    "helper_lipid": c.helper_lipid,
                    "ionizable_frac": round(c.ionizable_frac, 4),
                    "peg_frac": round(c.peg_frac, 4),
                    "cholesterol_frac": round(c.cholesterol_frac, 4),
                    "pka": round(c.pka, 4),
                    "fitness": round(c.fitness, 4),
                    "cardiac_delivery": round(c.cardiac_delivery_score, 4),
                    "hepatic_avoidance": round(c.hepatic_avoidance_score, 4),
                }
                for c in top_lnp
            ],
        }

        logger.info("Phase 5: %d AAV + %d LNP winners selected",
                     len(top_aav), len(top_lnp))

        return winners

    def _save_report(self, winners: dict):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        report = {
            "pipeline": "Danon Disease Gene Therapy Discovery Platform",
            "target_disease": self.config.target_disease,
            "therapeutic_payload": self.config.therapeutic_payload,
            "vector_backbone": self.config.vector_backbone,
            "regulatory_framework": self.config.regulatory_framework,
            "run_stats": self.stats,
            "top_aav_winners": winners["aav"][:10],
            "top_lnp_winners": winners["lnp"][:10],
        }
        path = os.path.join(self.config.checkpoint_dir, "danon_report.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Report saved: %s", path)

    def _print_summary(self):
        logger.info("=" * 80)
        logger.info("DANON PIPELINE SUMMARY")
        logger.info("=" * 80)
        for phase, count in self.stats.get("candidates", {}).items():
            logger.info("  %s: %s", phase, f"{count:,}" if isinstance(count, int) else count)
        logger.info("=" * 80)
        logger.info("Clinical alignment: %s", self.config.regulatory_framework)
        logger.info("Primary endpoint: %s", self.config.primary_surrogate_endpoint)
        logger.info("Regulatory target: MHRA ILAP FastTrack (UK)")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Danon Disease Gene Therapy Discovery Platform")
    parser.add_argument("--aav-candidates", type=int, default=1_000_000,
                        help="Number of AAV candidates to generate")
    parser.add_argument("--lnp-candidates", type=int, default=1_000_000,
                        help="Number of LNP candidates to generate")
    parser.add_argument("--batch-size", type=int, default=10_000,
                        help="Batch size for generation")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of workers")
    args = parser.parse_args()

    config = DanonConfig(
        aav_total_candidates=args.aav_candidates,
        lnp_total_candidates=args.lnp_candidates,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    pipeline = DanonPipeline(config)
    result = pipeline.run()

    logger.info("Danon Disease pipeline complete.")
    logger.info("Top AAV candidate: cardiac_tropism=%.4f, fitness=%.4f",
                result["aav"][0]["cardiac_tropism"] if result["aav"] else 0,
                result["aav"][0]["fitness"] if result["aav"] else 0)
    logger.info("Top LNP candidate: cardiac_delivery=%.4f, fitness=%.4f",
                result["lnp"][0]["cardiac_delivery"] if result["lnp"] else 0,
                result["lnp"][0]["fitness"] if result["lnp"] else 0)


if __name__ == "__main__":
    main()
