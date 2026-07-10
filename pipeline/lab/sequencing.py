import os
import json
import logging
import numpy as np
import csv
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SequencingRead:
    read_id: str
    sequence: str
    quality: str
    candidate_id: Optional[int] = None
    umi: str = ""
    is_passing: bool = True


@dataclass
class CandidateResult:
    candidate_id: int
    total_reads: int = 0
    unique_umis: int = 0
    mean_quality: float = 0.0
    coverage: float = 0.0
    mutation_detected: bool = False
    mutation_positions: list = field(default_factory=list)
    expression_level: float = 0.0
    is_winner: bool = False


@dataclass
class SequencingReport:
    total_reads: int = 0
    passing_reads: int = 0
    failed_reads: int = 0
    candidates_detected: int = 0
    candidate_results: dict = field(default_factory=dict)
    run_metadata: dict = field(default_factory=dict)


class NGSAnalyzer:
    def __init__(self):
        self.report = SequencingReport()

    def analyze_sequencing_run(
        self,
        fastq_path: str,
        barcode_decoder,
        reference_sequence: str = None,
        min_quality: float = 20.0,
    ) -> SequencingReport:
        logger.info("Starting NGS analysis: %s", fastq_path)

        results = defaultdict(lambda: {
            "reads": 0,
            "umis": set(),
            "qualities": [],
            "sequences": [],
        })

        read_count = 0
        passing_count = 0

        with open(fastq_path, "r") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
                if len(lines) == 4:
                    self._process_read(lines, barcode_decoder, results, min_quality)
                    read_count += 1
                    if results[lines[0].split()[0].replace("@", "")]["reads"] > 0:
                        passing_count += 1
                    lines = []

        self.report.total_reads = read_count
        self.report.passing_reads = passing_count
        self.report.failed_reads = read_count - passing_count

        for candidate_id, data in results.items():
            cr = CandidateResult(
                candidate_id=candidate_id,
                total_reads=data["reads"],
                unique_umis=len(data["umis"]),
                mean_quality=np.mean(data["qualities"]) if data["qualities"] else 0,
                coverage=data["reads"] / max(read_count, 1),
            )

            if reference_sequence and data["sequences"]:
                mutations = self._detect_mutations(data["sequences"], reference_sequence)
                cr.mutation_detected = len(mutations) > 0
                cr.mutation_positions = mutations

            cr.expression_level = self._estimate_expression_level(cr)
            self.report.candidate_results[candidate_id] = cr

        self.report.candidates_detected = len(results)
        logger.info(
            "NGS analysis complete: %d reads, %d candidates detected",
            self.report.total_reads,
            self.report.candidates_detected,
        )
        return self.report

    def _process_read(self, lines, barcode_decoder, results, min_quality):
        header = lines[0].replace("@", "").split()[0]
        sequence = lines[1]
        quality = lines[3]

        avg_qual = np.mean([ord(q) - 33 for q in quality])

        if avg_qual < min_quality:
            return

        if len(sequence) < 30:
            return

        i7 = sequence[0:6]
        umi = sequence[6:18]
        i5 = sequence[18:24]

        candidate_id = barcode_decoder.decode_read(i7, i5)
        if candidate_id is not None:
            results[candidate_id]["reads"] += 1
            results[candidate_id]["umis"].add(umi)
            results[candidate_id]["qualities"].append(avg_qual)
            results[candidate_id]["sequences"].append(sequence)

    def _detect_mutations(self, sequences: list, reference: str) -> list:
        mutation_counts = defaultdict(int)
        for seq in sequences:
            for i in range(min(len(seq), len(reference))):
                if i < len(seq) and seq[i] != reference[i]:
                    mutation_counts[i] += 1

        significant_mutations = [
            pos for pos, count in mutation_counts.items()
            if count / max(len(sequences), 1) > 0.01
        ]
        return significant_mutations

    def _estimate_expression_level(self, cr: CandidateResult) -> float:
        if cr.total_reads == 0:
            return 0.0
        coverage_score = min(1.0, cr.coverage * 100)
        umi_score = min(1.0, cr.unique_umis / 10)
        qual_score = cr.mean_quality / 40.0
        return float(np.clip(
            0.4 * coverage_score + 0.3 * umi_score + 0.3 * qual_score,
            0, 1
        ))

    def rank_candidates(self, report: SequencingReport) -> list[CandidateResult]:
        results = list(report.candidate_results.values())
        results.sort(key=lambda x: (
            x.expression_level,
            x.unique_umis,
            x.mean_quality,
            x.coverage,
        ), reverse=True)
        return results

    def select_winners(
        self, report: SequencingReport, top_n: int = 100
    ) -> list[CandidateResult]:
        ranked = self.rank_candidates(report)
        winners = ranked[:top_n]
        for w in winners:
            w.is_winner = True
        return winners

    def export_report(self, report: SequencingReport, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)

        summary_path = os.path.join(output_dir, "sequencing_summary.json")
        summary = {
            "total_reads": report.total_reads,
            "passing_reads": report.passing_reads,
            "failed_reads": report.failed_reads,
            "candidates_detected": report.candidates_detected,
            "run_metadata": report.run_metadata,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        csv_path = os.path.join(output_dir, "candidate_results.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "candidate_id", "total_reads", "unique_umis", "mean_quality",
                "coverage", "mutation_detected", "expression_level", "is_winner"
            ])
            for cr in report.candidate_results.values():
                writer.writerow([
                    cr.candidate_id, cr.total_reads, cr.unique_umis,
                    f"{cr.mean_quality:.2f}", f"{cr.coverage:.6f}",
                    cr.mutation_detected, f"{cr.expression_level:.4f}",
                    cr.is_winner
                ])

        winners_path = os.path.join(output_dir, "winners.json")
        winners = self.select_winners(report)
        winners_data = [{
            "candidate_id": w.candidate_id,
            "expression_level": w.expression_level,
            "unique_umis": w.unique_umis,
            "mean_quality": w.mean_quality,
        } for w in winners]
        with open(winners_path, "w") as f:
            json.dump(winners_data, f, indent=2)

        logger.info(
            "Report exported to %s (summary, %d candidates, %d winners)",
            output_dir, report.candidates_detected, len(winners)
        )

    def compare_to_baseline(
        self, report: SequencingReport, baseline_path: str
    ) -> dict:
        with open(baseline_path, "r") as f:
            baseline = json.load(f)

        improvements = {}
        for cid, cr in report.candidate_results.items():
            if cid in baseline:
                baseline_expr = baseline[cid].get("expression_level", 0)
                improvement = cr.expression_level - baseline_expr
                improvements[cid] = {
                    "expression_improvement": improvement,
                    "is_improved": improvement > 0,
                }

        return improvements
