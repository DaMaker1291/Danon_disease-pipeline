import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from pipeline.config import PipelineConfig
from pipeline.generation.aav_generator import AAVGenerator
from pipeline.generation.lnp_generator import LNPGenerator
from pipeline.screening.filter1_immune import ImmuneEvasionFilter
from pipeline.screening.filter2_tropism import TropismFilter
from pipeline.screening.filter3_efficiency import TransductionEfficiencyFilter
from pipeline.compute.supercomputer import SupercomputerInterface
from pipeline.compute.distributed import MPIController, CheckpointManager
from pipeline.lab.robotic_synthesis import OpentronsProtocol
from pipeline.lab.barcoding import BarcodingDesigner
from pipeline.lab.sequencing import NGSAnalyzer
from pipeline.feedback.refinement import FeedbackLoop
from pipeline.data_acquisition.synthetic_generator import SyntheticDataGenerator
from pipeline.data_acquisition.real_data_loader import RealDataIntegrator
from pipeline.data_acquisition.real_screening_loader import RealScreeningDataLoader
from pipeline.training.train_loops import AAVTrainer, LNPTrainer, ImmuneTrainer, TrainingConfig
from pipeline.training.model_integration import TrainedModelScorer
from pipeline.lab.validation_bridge import CROProtocolGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)


class LongevityPipeline:
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.checkpoint = CheckpointManager(self.config.checkpoint_dir)
        self.supercomputer = SupercomputerInterface(self.config.compute)
        self.mpi = MPIController()

        self.aav_gen = AAVGenerator(self.config.generation)
        self.lnp_gen = LNPGenerator(self.config.generation)

        self.immune_filter = ImmuneEvasionFilter(self.config.filters.antibody_panel)
        self.tropism_filter = TropismFilter(
            self.config.filters.target_tissues,
            self.config.filters.avoid_tissues,
        )
        self.transduction_filter = TransductionEfficiencyFilter(
            self.config.filters.min_transduction,
            self.config.filters.max_transduction,
        )

        self.protocol_gen = OpentronsProtocol(self.config.lab)
        self.barcoding = BarcodingDesigner()
        self.ngs_analyzer = NGSAnalyzer()
        self.feedback = FeedbackLoop(
            osk_max_dox_days=self.config.lab.osk_max_dox_days
            if hasattr(self.config.lab, "osk_max_dox_days") else 56,
            osk_penalty_weight=0.3,
        )

        self.data_generator = SyntheticDataGenerator()
        self.real_data_integrator = RealDataIntegrator()
        self.real_screening_loader = RealScreeningDataLoader()
        self.cro_generator = CROProtocolGenerator()
        self.trained_scorer = TrainedModelScorer(
            self.config, checkpoint_dir=self.config.checkpoint_dir
        )

        self.run_stats = {
            "start_time": None,
            "phases": {},
            "candidates": {},
        }

    def run_full_pipeline(self):
        self.run_stats["start_time"] = datetime.now().isoformat()
        logger.info("=" * 80)
        logger.info("LONGEVITY PIPELINE - STARTING")
        logger.info("=" * 80)

        phase0 = self.phase0_prepare_training_data()
        phase1 = self.phase1_generate()
        phase2 = self.phase2_immune_filter(phase1)
        phase3 = self.phase3_tropism_filter(phase2)
        phase4 = self.phase4_transduction_filter(phase3)
        winners = self.phase5_select_winners(phase4)
        lab_output = self.phase6_lab_synthesis(winners)
        analysis = self.phase7_sequencing_analysis(lab_output)
        refined = self.phase8_feedback_refinement(analysis, winners)
        validation = self.phase9_wet_lab_validation(winners)

        self.run_stats["end_time"] = datetime.now().isoformat()
        self._save_final_report(refined)
        self._print_summary()

        return refined

    def phase0_prepare_training_data(self) -> dict:
        logger.info("PHASE 0: Preparing Training Data and Models")

        logger.info("Loading real screening data (Fit4Function AAV + LNPDB LNP)...")
        try:
            real_screening = self.real_screening_loader.load_all_real_screening_data()
            screening_paths = self.real_screening_loader.save_real_screening_data(real_screening)

            training_data = {}
            for key, path in screening_paths.items():
                training_data[key] = path
                logger.info("Real screening data: %s -> %s (%d samples)", key, path, len(real_screening[key]))

            if all(len(v) == 0 for v in real_screening.values()):
                logger.warning("No real screening data found. Using synthetic fallback...")
                training_data = self.data_generator.generate_all_synthetic()

            if "aav_tropism" not in training_data:
                logger.info("AAV tropism data missing. Generating synthetic fallback...")
                training_data["aav_tropism"] = self.data_generator.generate_aav_tropism_data()
            if "lnp_delivery" not in training_data:
                logger.info("LNP delivery data missing. Generating synthetic fallback...")
                training_data["lnp_delivery"] = self.data_generator.generate_lnp_delivery_data()
            if "immune_escape" not in training_data:
                logger.info("Immune escape data missing. Generating synthetic fallback...")
                training_data["immune_escape"] = self.data_generator.generate_immune_escape_data()
        except Exception as e:
            logger.warning("Real screening data load failed: %s. Using synthetic...", e)
            training_data = self.data_generator.generate_all_synthetic()

        logger.info("Training AAV Tropism Model...")
        train_config = TrainingConfig()
        train_config.num_epochs = 10
        train_config.batch_size = 32

        diagnostics_dir = os.path.join(self.config.checkpoint_dir, "..", "diagnostics")

        aav_trainer = AAVTrainer(train_config, diagnostics_dir=diagnostics_dir)
        aav_model, aav_history = aav_trainer.train(training_data["aav_tropism"])

        logger.info("Training LNP Delivery Model...")
        lnp_trainer = LNPTrainer(train_config, diagnostics_dir=diagnostics_dir)
        lnp_model, lnp_history = lnp_trainer.train(training_data["lnp_delivery"])

        logger.info("Training Immune Escape Model...")
        immune_trainer = ImmuneTrainer(train_config, diagnostics_dir=diagnostics_dir)
        immune_model, immune_history = immune_trainer.train(training_data["immune_escape"])

        logger.info("Loading trained models for screening...")
        self.trained_scorer.load_models()

        return {
            "training_data": training_data,
            "aav_history": aav_history,
            "lnp_history": lnp_history,
            "immune_history": immune_history,
        }

    def phase1_generate(self) -> dict:
        logger.info("PHASE 1: Generating %d billion digital candidates...",
                     self.config.generation.aav_total_candidates // 1_000_000_000)

        aav_batch_size = self.config.compute.batch_size
        aav_total = self.config.generation.aav_total_candidates // 2
        lnp_total = self.config.generation.lnp_total_candidates // 2

        aav_candidates = []
        lnp_candidates = []

        aav_count = 0
        for batch in self.aav_gen.stream_candidates(aav_total, aav_batch_size):
            aav_candidates.extend(batch)
            aav_count += len(batch)
            if aav_count % 1_000_000 == 0:
                logger.info("AAV generated: %d / %d", aav_count, aav_total)

        lnp_count = 0
        for batch in self.lnp_gen.stream_candidates(lnp_total, aav_batch_size):
            lnp_candidates.extend(batch)
            lnp_count += len(batch)
            if lnp_count % 1_000_000 == 0:
                logger.info("LNP generated: %d / %d", lnp_count, lnp_total)

        self.run_stats["candidates"]["phase1_aav"] = len(aav_candidates)
        self.run_stats["candidates"]["phase1_lnp"] = len(lnp_candidates)

        self.checkpoint.save_checkpoint("phase1", {
            "aav_count": len(aav_candidates),
            "lnp_count": len(lnp_candidates),
        })

        logger.info("Phase 1 complete: %d AAV + %d LNP = %d total",
                     len(aav_candidates), len(lnp_candidates),
                     len(aav_candidates) + len(lnp_candidates))

        return {
            "aav": aav_candidates,
            "lnp": lnp_candidates,
        }

    def phase2_immune_filter(self, phase1_output: dict) -> dict:
        logger.info("PHASE 2: Immune Evasion Filter (Gate Crash Test)")
        logger.info("Filtering %d candidates through trained immune model...",
                     len(phase1_output["aav"]))

        try:
            scored_aav = self.trained_scorer.score_immune_escape(phase1_output["aav"])
            aav_passed = [c for c in scored_aav if c.immune_evasion_score >= self.config.filters.immune_threshold]
        except Exception as e:
            logger.warning("Trained model scoring failed: %s. Using rule-based fallback.", e)
            aav_passed = [c for c in phase1_output["aav"]
                         if self.immune_filter.score(c) >= self.config.filters.immune_threshold]

        lnp_passed = phase1_output["lnp"]

        self.run_stats["candidates"]["phase2_aav"] = len(aav_passed)
        self.run_stats["candidates"]["phase2_lnp"] = len(lnp_passed)

        logger.info("Phase 2 complete: AAV %d -> %d (%.2f%%)",
                     len(phase1_output["aav"]),
                     len(aav_passed),
                     100 * len(aav_passed) / max(len(phase1_output["aav"]), 1))

        return {"aav": aav_passed, "lnp": lnp_passed}

    def phase3_tropism_filter(self, phase2_output: dict) -> dict:
        logger.info("PHASE 3: Tissue Tropism Filter (ZIP Code Test)")
        logger.info("Filtering %d AAV through trained tropism model...",
                     len(phase2_output["aav"]))

        try:
            scored_aav = self.trained_scorer.score_aav_candidates(phase2_output["aav"])
            aav_passed = [c for c in scored_aav if c.tropism_score >= self.config.filters.tropism_threshold]
        except Exception as e:
            logger.warning("Trained model scoring failed: %s. Using rule-based fallback.", e)
            aav_passed = [c for c in phase2_output["aav"]
                         if self.tropism_filter.score(c) >= self.config.filters.tropism_threshold]

        lnp_passed = []
        for c in phase2_output["lnp"]:
            score = self.tropism_filter.score(c) if hasattr(self.tropism_filter, 'score') else 0.5
            if score >= self.config.filters.tropism_threshold:
                lnp_passed.append(c)

        self.run_stats["candidates"]["phase3_aav"] = len(aav_passed)
        self.run_stats["candidates"]["phase3_lnp"] = len(lnp_passed)

        logger.info("Phase 3 complete: AAV %d -> %d (%.2f%%)",
                     len(phase2_output["aav"]),
                     len(aav_passed),
                     100 * len(aav_passed) / max(len(phase2_output["aav"]), 1))

        return {"aav": aav_passed, "lnp": lnp_passed}

    def phase4_transduction_filter(self, phase3_output: dict) -> dict:
        logger.info("PHASE 4: Transduction Efficiency Filter (Goldilocks Dial)")
        logger.info("Filtering %d AAV through transduction scoring...",
                     len(phase3_output["aav"]))

        aav_passed = []
        for c in phase3_output["aav"]:
            score = self.transduction_filter.score(c)
            if score >= self.config.filters.min_transduction:
                aav_passed.append(c)

        try:
            scored_lnp = self.trained_scorer.score_lnp_candidates(phase3_output["lnp"])
            lnp_passed = [c for c in scored_lnp if c.predicted_delivery >= self.config.filters.min_transduction]
        except Exception as e:
            logger.warning("Trained LNP model scoring failed: %s. Using rule-based fallback.", e)
            from pipeline.screening.filter3_efficiency import LNPTransductionFilter
            lnp_filter = LNPTransductionFilter()
            lnp_passed = [c for c in phase3_output["lnp"]
                         if lnp_filter.score(c) >= self.config.filters.min_transduction]

        self.run_stats["candidates"]["phase4_aav"] = len(aav_passed)
        self.run_stats["candidates"]["phase4_lnp"] = len(lnp_passed)

        logger.info("Phase 4 complete: AAV %d -> %d (%.2f%%)",
                     len(phase3_output["aav"]),
                     len(aav_passed),
                     100 * len(aav_passed) / max(len(phase3_output["aav"]), 1))

        return {"aav": aav_passed, "lnp": lnp_passed}

    def phase5_select_winners(self, phase4_output: dict) -> list:
        logger.info("PHASE 5: Selecting Top 50,000 Candidates")

        all_aav = sorted(
            phase4_output["aav"],
            key=lambda c: c.fitness,
            reverse=True
        )[:25000]

        all_lnp = sorted(
            phase4_output["lnp"],
            key=lambda c: c.fitness,
            reverse=True
        )[:25000]

        winners = []
        for c in all_aav:
            winners.append({
                "type": "aav",
                "candidate_id": c.candidate_id,
                "sequence": c.sequence,
                "mutations": c.mutations,
                "fitness": c.fitness,
                "esm_score": c.esm_score,
                "stability_score": c.stability_score,
                "surface_score": c.surface_score,
            })

        for c in all_lnp:
            winners.append({
                "type": "lnp",
                "candidate_id": c.candidate_id,
                "fitness": c.fitness,
                "ionizable_lipid": c.ionizable_lipid,
                "peg_lipid": c.peg_lipid,
                "helper_lipid": c.helper_lipid,
                "ionizable_frac": c.ionizable_frac,
                "peg_frac": c.peg_frac,
                "cholesterol_frac": c.cholesterol_frac,
                "pka": c.pka,
                "tail_length": c.tail_length,
                "unsaturation": c.unsaturation,
            })

        self.run_stats["candidates"]["winners"] = len(winners)

        self.checkpoint.save_checkpoint("phase5_winners", winners)
        logger.info("Phase 5 complete: %d winners selected", len(winners))

        return winners

    def phase6_lab_synthesis(self, winners: list) -> dict:
        logger.info("PHASE 6: Robotic Lab Synthesis")

        candidate_ids = [w["candidate_id"] for w in winners]
        barcode_design = self.barcoding.design_barcodes(candidate_ids)

        os.makedirs(self.config.lab.output_dir, exist_ok=True)

        protocol_path = os.path.join(self.config.lab.output_dir, "synthesis_protocol.py")
        self.protocol_gen.generate_synthesis_protocol(winners, protocol_path)

        barcode_path = os.path.join(self.config.lab.output_dir, "barcodes.fasta")
        self.barcoding.export_fasta(barcode_design, barcode_path)

        index_path = os.path.join(self.config.lab.output_dir, "index_table.csv")
        self.barcoding.export_index_table(barcode_design, index_path)

        plate_map_path = os.path.join(self.config.lab.output_dir, "plate_map.json")
        self.protocol_gen.generate_plate_map(winners, plate_map_path)

        pcr_path = os.path.join(self.config.lab.output_dir, "pcr_protocol.py")
        self.barcoding.generate_pcr_protocol(barcode_design, pcr_path)

        logger.info("Phase 6 complete: Lab files generated in %s", self.config.lab.output_dir)

        return {
            "barcode_design": barcode_design,
            "protocol_path": protocol_path,
            "barcode_path": barcode_path,
            "index_path": index_path,
            "winners": winners,
        }

    def phase7_sequencing_analysis(self, lab_output: dict) -> dict:
        logger.info("PHASE 7: Next-Gen Sequencing Analysis")

        fastq_path = os.path.join(self.config.lab.output_dir, "sequencing_data.fastq")
        if os.path.exists(fastq_path):
            barcode_decoder = self.barcoding._create_decoder(lab_output["barcode_design"])
            report = self.ngs_analyzer.analyze_sequencing_run(
                fastq_path, barcode_decoder
            )
            report_path = os.path.join(self.config.lab.output_dir, "sequencing_report")
            self.ngs_analyzer.export_report(report, report_path)

            winners = self.ngs_analyzer.select_winners(report)
            logger.info("Sequencing analysis complete: %d winners from NGS", len(winners))

            return {"report": report, "winners": winners}
        else:
            logger.info("No FASTQ data found. Using AI-predicted winners.")
            return {"report": None, "winners": lab_output["winners"]}

    def phase8_feedback_refinement(self, analysis_output: dict, original_winners: list) -> dict:
        logger.info("PHASE 8: Feedback Loop Refinement")

        sequencing_winners = analysis_output.get("winners", [])
        if not sequencing_winners:
            sequencing_winners = original_winners[:100]

        refined = self.feedback.refine_from_sequencing(
            sequencing_winners,
            self.aav_gen,
            self.lnp_gen,
            lambda candidates: [c for c in candidates if c.fitness > 0.5],
        )

        self.checkpoint.save_checkpoint("phase8_refined", refined)
        logger.info("Phase 8 complete: Feedback refinement done")

        return refined

    def phase9_wet_lab_validation(self, winners: list) -> dict:
        logger.info("PHASE 9: Wet-Lab Validation Protocol Generation")

        from pipeline.lab.validation_bridge import ValidationCandidate

        validation_candidates = []
        for w in winners[:10]:
            candidate = ValidationCandidate(
                candidate_id=w["candidate_id"],
                candidate_type=w["type"],
                sequence=w.get("sequence", ""),
                composition={
                    "ionizable_lipid": w.get("ionizable_lipid", ""),
                    "peg_lipid": w.get("peg_lipid", ""),
                    "ionizable_frac": w.get("ionizable_frac", 0),
                    "peg_frac": w.get("peg_frac", 0),
                },
                ai_score=w.get("fitness", 0.5),
                ai_predictions={
                    "immune_evasion": w.get("immune_evasion_score", 0.5),
                    "tropism": w.get("tropism_score", 0.5),
                    "transduction": w.get("transduction_score", 0.5),
                },
            )
            validation_candidates.append(candidate)

        protocol = self.cro_generator.generate_synthesis_protocol(
            validation_candidates,
            target_tissues=self.config.filters.target_tissues
            if hasattr(self.config.filters, "target_tissues") else ["cardiac", "neuronal"],
        )

        plan = self.cro_generator.generate_organoid_testing_plan(protocol)
        estimate = self.cro_generator.generate_cost_estimate(
            len(validation_candidates), 2
        )

        output_dir = os.path.join(self.config.lab.output_dir, "validation_package")
        self.cro_generator.export_protocol_package(protocol, plan, estimate, output_dir)

        logger.info("Phase 9 complete: Validation package at %s", output_dir)
        logger.info("Estimated cost: $%s", f"{estimate['total_estimated_usd']:,.0f}")

        return {
            "protocol": protocol,
            "plan": plan,
            "estimate": estimate,
            "output_dir": output_dir,
        }

    def _save_final_report(self, refined: dict):
        report = {
            "pipeline_run": self.run_stats,
            "refinement_result": {
                "aav_best_score": refined.get("aav_refinement", {}).get("best_score", 0)
                    if isinstance(refined.get("aav_refinement"), dict) else 0,
                "lnp_best_score": refined.get("lnp_refinement", {}).get("best_score", 0)
                    if isinstance(refined.get("lnp_refinement"), dict) else 0,
            },
        }

        report_path = os.path.join(self.config.checkpoint_dir, "final_report.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("Final report saved: %s", report_path)

    def _print_summary(self):
        logger.info("=" * 80)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 80)
        for phase, count in self.run_stats.get("candidates", {}).items():
            logger.info("  %s: %s", phase, f"{count:,}" if isinstance(count, int) else count)
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Longevity Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--backend", type=str, default="ray",
                        choices=["ray", "slurm", "kubernetes", "aws_batch"],
                        help="Compute backend")
    parser.add_argument("--workers", type=int, default=256, help="Number of workers")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size")
    args = parser.parse_args()

    config = PipelineConfig()
    config.compute.backend = args.backend
    config.compute.num_workers = args.workers
    config.compute.batch_size = args.batch_size

    if args.config:
        import yaml
        with open(args.config) as f:
            config_overrides = yaml.safe_load(f)
        config = PipelineConfig(**config_overrides)

    pipeline = LongevityPipeline(config)
    result = pipeline.run_full_pipeline()

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
