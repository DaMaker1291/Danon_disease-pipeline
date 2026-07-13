"""
Dynamic Ingress Engine: accepts real wet-lab input formats (FASTQ, GenBank, CSV)
and maps inbound sequences against the official wild-type AAV9 capsid template,
isolating novel mutation strings on the variable loops (VR-IV, VR-VIII).

Uses BioPython for GenBank/FASTQ parsing with a pure-Python fallback for
environments where BioPython is unavailable.
"""
import os
import re
import json
import csv
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

# Wild-type AAV9 capsid reference (VP1, 736 aa)
WILD_TYPE_AAV9_CAPSID: str = (
    "MAADGYLPDWLEDTLSEGIRQWWKLKPGPPPPKPAERHKDDSRGLVLPGYKYLGPFNGLDKGEPVNEAD"
    "AAALEHDKAYDRQLDSGDNPYLKYNHADAEFQERLKEDTSFGGNLGRAVFQAKKRVLEPLGLVEEPVKT"
    "APGKKRPVEHSPVEPDSSSGTGKAGQQPARKRLNFGQTGDADSVPDPQPLGQPPAAPSGLGTNTMATGS"
    "GAPMADNNEGADGVGNSSGNWHCDSTWMGDRVITTSTRTWALPTYNNHLYKQISSQSGASNDNHYFGY"
    "STPWGYFDFNRFHCHFSPRDWQRLINNNWGFRPKRLNFKLFNIQVKEVTQNDGTTTIANNLTSTVQVFT"
    "DSEYQLPYVLGSAHQGCLPPFPADVFMIPQYGYLTLNNGSQAVGRSSFYCLEYFPSQMLRTGNNFTFSY"
    "TFEDVPFHSSYAHSQSLDRLMNPLIDQYLYYLSRTNTPSGTTTQSRLQFSQAGASDIRDQSRNWLPGPC"
    "YRQQRVSKTSADNNNSEYSWTGATKYHLNGRDSLVNPGPAMASHKDDEEKFFPQSGVLIFGKQGSEKTN"
    "VDIEKVMITDEEEIRTTNPVATEQYGSVSTNLQRGNRQAATADVNTQGVLPGMVWQDRDVYLQGPIWAK"
    "IPHTDGHFHPSPLMGGFGLKHPPPQILIKNTPVPANPSTTFSAAKFASFITQYSTGQVSVEIEWELQKE"
    "NSKRWNPEIQYTSNYYKSTNVDFAVNTEGTYSEPRPIGTRYLTRNL"
)

# VP1 numbering ranges for variable regions (1-indexed, full-length)
VR_REGIONS: Dict[str, Tuple[int, int]] = {
    "VR_I": (263, 280),
    "VR_II": (326, 346),
    "VR_III": (380, 395),
    "VR_IV": (448, 468),
    "VR_V": (489, 510),
    "VR_VI": (526, 544),
    "VR_VII": (545, 562),
    "VR_VIII": (570, 600),
    "VR_IX": (450, 485),
}


class SequencingRead(BaseModel):
    read_id: str
    sequence: str
    quality_scores: Optional[str] = None


class MutationRecord(BaseModel):
    position_vp1: int
    original_aa: str
    mutated_aa: str
    region: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class GenBankFeature(BaseModel):
    feature_type: str
    location: str
    qualifiers: Dict[str, List[str]] = Field(default_factory=dict)


class GenBankRecord(BaseModel):
    accession: str
    sequence: str
    features: List[GenBankFeature] = Field(default_factory=list)
    organism: str = ""


class ExpressionMatrixEntry(BaseModel):
    gene_id: str
    tissue: str
    expression_tpm: float = Field(ge=0.0)
    condition: str = ""


class IngressResult(BaseModel):
    source_format: str
    records_parsed: int
    mutations_isolated: List[MutationRecord] = Field(default_factory=list)
    template_coverage: float = 0.0
    vr_mutation_count: Dict[str, int] = Field(default_factory=dict)
    sequence_valid: bool = False
    warnings: List[str] = Field(default_factory=list)


class DataIngressEngine:
    """
    Parses raw wet-lab data formats and maps against AAV9 wild-type.
    """

    def __init__(self, wild_type: str = WILD_TYPE_AAV9_CAPSID):
        self.wild_type = wild_type
        self.wt_len = len(wild_type)
        self.vr_regions = VR_REGIONS

    # ------------------------------------------------------------------
    # FASTQ parsing
    # ------------------------------------------------------------------
    def parse_fastq(self, path_or_text: str, is_path: bool = True) -> List[SequencingRead]:
        if is_path:
            with open(path_or_text, "r") as f:
                text = f.read()
        else:
            text = path_or_text
        reads: List[SequencingRead] = []
        lines = text.strip().splitlines()
        i = 0
        while i + 3 < len(lines):
            if lines[i].startswith("@"):
                rid = lines[i][1:].split()[0]
                seq = lines[i + 1].strip().upper()
                qual = lines[i + 3].strip() if i + 3 < len(lines) else ""
                reads.append(SequencingRead(read_id=rid, sequence=seq, quality_scores=qual))
                i += 4
            else:
                i += 1
        return reads

    # ------------------------------------------------------------------
    # GenBank parsing (BioPython with fallback)
    # ------------------------------------------------------------------
    def parse_genbank(self, path_or_text: str, is_path: bool = True) -> List[GenBankRecord]:
        try:
            return self._parse_genbank_biopython(path_or_text, is_path)
        except ImportError:
            logger.warning("BioPython not available, using built-in GenBank parser")
            return self._parse_genbank_fallback(path_or_text, is_path)

    def _parse_genbank_biopython(self, path_or_text: str, is_path: bool) -> List[GenBankRecord]:
        from Bio import SeqIO
        records: List[GenBankRecord] = []
        if is_path:
            handle = open(path_or_text, "r")
        else:
            from io import StringIO
            handle = StringIO(path_or_text)
        try:
            for rec in SeqIO.parse(handle, "genbank"):
                features = []
                for f in rec.features:
                    features.append(GenBankFeature(
                        feature_type=f.type,
                        location=str(f.location),
                        qualifiers={k: list(v) for k, v in f.qualifiers.items()},
                    ))
                records.append(GenBankRecord(
                    accession=rec.id,
                    sequence=str(rec.seq).upper(),
                    features=features,
                    organism=rec.annotations.get("organism", ""),
                ))
        finally:
            if is_path:
                handle.close()
        return records

    def _parse_genbank_fallback(self, path_or_text: str, is_path: bool) -> List[GenBankRecord]:
        text = ""
        if is_path:
            with open(path_or_text, "r") as f:
                text = f.read()
        else:
            text = path_or_text
        records: List[GenBankRecord] = []
        lines = text.splitlines()
        accession = ""
        organism = ""
        seq_parts: List[str] = []
        features: List[GenBankFeature] = []
        in_features = False
        in_origin = False
        for line in lines:
            if line.startswith("ACCESSION"):
                accession = line.split()[-1]
            elif line.startswith("ORGANISM"):
                organism = line[12:].strip()
            elif line.startswith("FEATURES"):
                in_features = True
                in_origin = False
            elif line.startswith("ORIGIN"):
                in_origin = True
                in_features = False
            elif in_origin:
                m = re.search(r"[a-zA-Z]+", line)
                if m:
                    seq_parts.append(re.sub(r"[^a-zA-Z]", "", line).upper())
        sequence = "".join(seq_parts)
        records.append(GenBankRecord(
            accession=accession or "unknown",
            sequence=sequence or "",
            features=features,
            organism=organism,
        ))
        return records

    # ------------------------------------------------------------------
    # CSV expression matrix parsing
    # ------------------------------------------------------------------
    def parse_expression_csv(self, path_or_text: str, is_path: bool = True) -> List[ExpressionMatrixEntry]:
        entries: List[ExpressionMatrixEntry] = []
        if is_path:
            f = open(path_or_text, "r", newline="")
        else:
            from io import StringIO
            f = StringIO(path_or_text)
        try:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(ExpressionMatrixEntry(
                    gene_id=row.get("gene_id", row.get("Gene", "")),
                    tissue=row.get("tissue", row.get("Tissue", "")),
                    expression_tpm=float(row.get("expression_tpm", row.get("TPM", 0))),
                    condition=row.get("condition", row.get("Condition", "")),
                ))
        finally:
            if is_path:
                f.close()
        return entries

    # ------------------------------------------------------------------
    # Sequence sanitization and VR mutation isolation
    # ------------------------------------------------------------------
    def sanitize_and_map(self, sequence: str) -> IngressResult:
        seq = sequence.upper().strip()
        seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq)
        seq = seq[:self.wt_len]
        if not seq:
            return IngressResult(
                source_format="sequence",
                records_parsed=0,
                sequence_valid=False,
                warnings=["empty sequence after sanitization"],
            )
        align_start = 0
        matches = 0
        for i in range(min(len(seq), self.wt_len)):
            if i < len(seq) and seq[i] == self.wild_type[i]:
                matches += 1
        coverage = matches / max(self.wt_len, 1)
        mutations: List[MutationRecord] = []
        vr_counts: Dict[str, int] = {}
        for vr_name, (v_start, v_end) in self.vr_regions.items():
            vr_counts[vr_name] = 0
            for pos in range(v_start - 1, min(v_end - 1, len(seq), self.wt_len)):
                wt_aa = self.wild_type[pos]
                query_aa = seq[pos]
                if wt_aa != query_aa:
                    mutations.append(MutationRecord(
                        position_vp1=pos + 1,
                        original_aa=wt_aa,
                        mutated_aa=query_aa,
                        region=vr_name,
                        confidence=1.0,
                    ))
                    vr_counts[vr_name] += 1
        return IngressResult(
            source_format="sequence",
            records_parsed=1,
            mutations_isolated=mutations,
            template_coverage=float(coverage),
            vr_mutation_count=vr_counts,
            sequence_valid=coverage > 0.5,
            warnings=[] if coverage > 0.5 else ["low template coverage"],
        )

    def process_fastq_mutations(self, path_or_text: str, is_path: bool = True) -> IngressResult:
        reads = self.parse_fastq(path_or_text, is_path)
        if not reads:
            return IngressResult(source_format="fastq", records_parsed=0, sequence_valid=False)
        best_result = None
        best_coverage = 0.0
        for read in reads:
            result = self.sanitize_and_map(read.sequence)
            if result.template_coverage > best_coverage:
                best_coverage = result.template_coverage
                best_result = result
        if best_result is None:
            return IngressResult(source_format="fastq", records_parsed=len(reads), sequence_valid=False)
        best_result.source_format = "fastq"
        best_result.records_parsed = len(reads)
        return best_result

    def process_genbank_mutations(self, path_or_text: str, is_path: bool = True) -> IngressResult:
        records = self.parse_genbank(path_or_text, is_path)
        if not records:
            return IngressResult(source_format="genbank", records_parsed=0, sequence_valid=False)
        all_mutations: List[MutationRecord] = []
        total_coverage = 0.0
        total_vr: Dict[str, int] = {}
        for gb in records:
            result = self.sanitize_and_map(gb.sequence)
            all_mutations.extend(result.mutations_isolated)
            total_coverage += result.template_coverage
            for k, v in result.vr_mutation_count.items():
                total_vr[k] = total_vr.get(k, 0) + v
        n = len(records)
        return IngressResult(
            source_format="genbank",
            records_parsed=n,
            mutations_isolated=all_mutations,
            template_coverage=total_coverage / n,
            vr_mutation_count=total_vr,
            sequence_valid=total_coverage / n > 0.5,
        )

    def get_vr_sequences(self, sequence: str) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for vr_name, (v_start, v_end) in self.vr_regions.items():
            if v_end <= len(sequence):
                result[vr_name] = sequence[v_start - 1:v_end - 1]
        return result

    def get_vr_mutations(self, sequence: str) -> Dict[str, List[Tuple[int, str, str]]]:
        result: Dict[str, List[Tuple[int, str, str]]] = {}
        for vr_name, (v_start, v_end) in self.vr_regions.items():
            vr_muts: List[Tuple[int, str, str]] = []
            for pos in range(v_start - 1, min(v_end - 1, len(sequence), self.wt_len)):
                wt_aa = self.wild_type[pos]
                query_aa = sequence[pos]
                if wt_aa != query_aa:
                    vr_muts.append((pos + 1, wt_aa, query_aa))
            result[vr_name] = vr_muts
        return result
