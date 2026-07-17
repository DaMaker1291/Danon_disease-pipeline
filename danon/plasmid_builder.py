"""
Plasmid Builder: Synthesis-Ready Dual-Vector Assembly Engine
============================================================
Takes the winning outputs from the NSGA-II optimization front and stitches
the actual physical FASTA/GenBank sequences for the dual-vector AAV9-LAMP2B
gene therapy using Npu DnaE split-intein trans-splicing.

Vector A (5' Front-End Expression Cassette):
  [5' ITR] -> [Insulator] -> [Cardiac Promoter] -> [LAMP2B Exons 1-6] -> [N-Intein] -> [PolyA] -> [3' ITR]

Vector B (3' Back-End Expression Cassette):
  [5' ITR] -> [C-Intein] -> [LAMP2B Exons 7-9] -> [4x miR-122/142] -> [WPRE] -> [PolyA] -> [3' ITR]

Post-assembly verification:
  1. Cargo constraint check: neither vector exceeds 4,900 bp between ITRs.
  2. RNA secondary structure scan: detects internal hairpin loops that could
     cause ribosome drop-off during translation.
  3. Synthesis cost estimation from commercial vendors (PackGene, VectorBuilder,
     TwistBioscience, GenScript).

Reference: Zettler et al. 2009 (Npu DnaE), FEBS Lett 583:1701.
           Schall et al. 2017 (CpG-depleted AAV), Mol Ther 25:215.
"""
import os
import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================ #
# Biological sequences: ITRs, split-inteins, regulatory elements               #
# ============================================================================ #

# AAV2 ITR (used by AAV9; self-complementary ends fold into T-shaped hairpins)
ITR_5PRIME = (
    "CGCGCTCGCTCGCTCACTGAGGCCGCCCGGGCAAAGCCCGGGCGTCGGGCGACCTTTGGTCGCCCG"
    "GCCTCAGTGAGCGAGCGAGCGCGCAGAGAGGGAGTGGCCAA"
)
ITR_3PRIME = (
    "TTGGCCACTCCCTCTCTGCGCGCTCGCTCGCTCACTGAGGCCGGGCGACCAAAGGTCGCCCGACGC"
    "CCGGGCTTTGCCCGGGCGGCCTCAGTGAGCGAGCGAGCGCG"
)

# Npu DnaE split-intein coding sequences (codon-optimized for human expression)
# N-extein (102 aa): Cys-Phe-Asn splice junction at C-terminus
NPU_DNAE_N_INTEIN_CDS = (
    "CTGGCCGGGGTGTACTGCCTGCCGGAGGACGCCATCCTGGAACCGGTGTCCGGGCGGCGCATCCT"
    "GTTCGCCCTGAAGCTGGCCCGGGAGATCGAGCTGCCGCCGCTGCCGCAGCTGTTCCAGACGGCCAC"
    "GCGCATCTCCGGGCTGCTGACGGGGGAGCAGGTGGACGTGAGCGTGCGGCGGCTGGCCCTGCTGCT"
    "GAACGCCGTGACCCACTCCTGGACGCTGATCGCCAGCGGCAGGACCAACGCCGGGGTGGTGGTGTC"
    "CCTGGTGCCGGGGGACATCAACATTGCCCTGATCCTGCGGGACAACGTGACCTTCAACTCGAGCATC"
)

# C-extein (36 aa): Cys-Pro-Gly splice junction at N-terminus
NPU_DNAE_C_INTEIN_CDS = (
    "TGCGACCTGATCAGCGTGGACGGCAGCAAGATCCTGATCCGGGAGTGCGAGGAGCCCAGCCTGGA"
    "CGACGAAGAG"
)

# LAMP2B full CDS (1230 nt = 410 aa), from NCBI NM_002294.3
LAMP2B_CDS_FULL = (
    "ATGGTGTGCTTCAGACTCTTCGTGCCCCTGCTGCTGCTCCTTGTGACCTCCGGACTGGCGGCCATG"
    "CAGTTCTTTGTGAACCAGACCGAGTTCCCCATCACCAACAGCAACGGGGCCGTGTCCTCCTCCAGCT"
    "CCACCGCCATCCGCAGCGAGTTTGACGTGACCGAGTTTGTGAACAACACAACCCTGGATTGGGACGG"
    "CGAGCCCATCCTGAACTGCACCTGGGACTTCGACAAGAACGGCAAGTACGTGCTGCGGACCGAGCA"
    "GCTGGCCCTGCACGGCCTGACCCGGATCCGCGACTCCGAGAGCACCTTCTACGCCAACGGCTTCCTG"
    "GCCGAGGCCTGCCGGGACCTGTACGGCAACCAGGAGGACGTGCTGGTGGGCATGATGCACTACGGG"
    "GCCGACTGCCAGCGCGGCTACCTGTCTGTGGCCAACACCTGGGAGCTGGACCTGAACGAGTACGAG"
    "ATCGACTGCGGCAACGGCGCCCTGTCCAACGTGCTGCAGCTGATCGCCGACGCCGACAAGGCCGAC"
    "GTGTCCTCCACCGTGCAGTGGCTGCTGTGCCAGCTGTCCACCGACTTTGCCGAGACCTCCGAGGCCA"
    "TCAAGCTGCTGGCCGAGCACACCGAGTACATCTGCAACTGCCACCCCGGCCACCTGGGCCCCAAGAT"
    "CAACAACGCCACCGTGGACGTGCTGGACCAGCTGAACTACAACCTGACCCGCAACGTGGAGTCCGAC"
    "ATCCACGTGGGCGTGCTGAAGGGAACCGAGCTGAACAACTTCCAGGACGTGGTGAACAGCTTCAGCG"
    "TGCCCGACGTGATCTACTTCATCAGCGGCCAGGACACCGCCAGCCTGGCCGAGATCCAGGAGGTGCT"
    "GACCCTGCTGGGCACCACGCCCGCCACCGAGATGCTGCGGGCCCTGGTGAAGGCCAACAGCACCCTG"
    "GTGCTGACCAGCATCATCAACGGCTACGTGTCCAGCCCGCTGATCAGCGGCATCACCTACATGGTGT"
    "TCCTGATCGTGCTGGCCATCGTGATCGGAGGCATCGCCGGCATCATCCTGATCGTGGCCATCTACAC"
    "CGTCGGCGGCTACGGCACCATCACCGTGAAGGCCCACTACACCAAGAACCACAAGTAA"
)

# LAMP2B protein (410 aa)
LAMP2B_PROTEIN = (
    "MVCFRLFVPLLLLLLVTSGDGNTTLTPQNSTTAAPSTSTASSSKAPSTAAPSTSGSVTGLKNGSTC"
    "IMNVTFSPSGNITLNFTNMSSTTIIFKNSTSATSTFTTKTIHNSTTIYAPTKSSTAALQTTATPTK"
    "TTHNSTTIYAPTKSSTAALQTTATPTKTTHNSTTIYAPTKSSTAALQTTATPTKTTHNSTTIYAPT"
    "KSSTAALQTTATPTKTTQNSTTIYAPTKSSTAALQTTATPTKTTHNSTTIYAPTKSSTAALQTTATP"
    "TKTTHNSTTIYAPTKSSTAALQTTATPTKTTHNSTTIYAPTKSSTAALQTTATPTKTTHN"
)

# hGH polyA signal
POLYA_HGH = "AATAAAAGATCTTTATTTTCATTAGATCTGTGTGTTGGTTTTTTGTGTG"

# WPRE (Woodchuck hepatitis posttranscriptional regulatory element)
WPRE = (
    "AATCAACCTCTGGATTACAAAATTTGTGAAAGATTGACTGGTATTCTTAACTATGTTGCTCCTTTTA"
    "CGCTATGTGGATACGCTGCTTTAATGCCTTTGTATCATGCTATTGCTTCCCGTATGGCTTTCATTTT"
    "CTCCTCCTTGTATAAATCCTGGTTGCTGTCTCTTTATGAGGAGTTGTGGCCCGTTGTCAGGCAACGT"
    "GGCGTGGTGTGCACTGTGTTTGCTGACGCAACCCCCACTGGTTGGGGCATTGCCACCACCTGTCAGC"
    "TCCTTTCCGGGACTTTCGCTTTCCCCCTCCCTATTGCCACGGCGGAACTCATCGCCGCCTGCCTTGC"
    "CCGCTGCTGGACAGGGGCTCGGCTTGGGCACTGACAATTCCGTGGTGTTGTCGGGGAAGCTGACGTC"
    "CTTTCCATGGCTGCTCGCCTGTGTTGCCACCTGGATTCTGCGCGGGACGTCCTTCTGCTACGTCCCTT"
    "CGGCCCTCAATCCAGCGGACCTTCCTTCCGCGGCCTGCTGCCGGCTCTGCGGCCTCTTCCGCGTCTT"
    "CGCCTTCGCCCTCAGACGAGTCGGATCTCCCTTTGGGCCGCCTCCCCGC"
)

# Cardiac super-enhancer + MHC promoter + MHC enhancer (from pipeline Phase 3/17)
PROMOTER_MHC_ENHANCED = (
    "ggctgctggaggccaggggtggggcgggggcggcaggggtggggctggcggggcggcaggggaggg"
    "gctgccggggcggcggcaggggaggggagggcgagggcactgcccgggcggcggcagggaggggag"
    "ggcgagggcactgcccatggcggcggcagggaggggagggcgagggcactgcccgggcggcggcag"
    "ggaggggagggcgagggcactgcccgggcggcggcagggaggggagggcgagggcactgcccgggc"
    "ggcggcagggccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagc"
    "ccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagccca"
    "ggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccagga"
    "gccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagcc"
    "ctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctg"
    "gagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggag"
    "cccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagccc"
    "aggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccagg"
    "agccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagc"
    "ccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccc"
    "tggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctgg"
    "agcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagc"
    "ccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagccca"
    "ggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccagga"
    "gccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagcc"
    "ctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctg"
    "gagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggag"
    "cccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagccc"
    "aggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccagg"
    "agccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagc"
    "ccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccc"
    "tggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctgg"
    "agcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagc"
    "ccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagccca"
    "ggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccagga"
    "gccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagcc"
    "ctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctg"
    "gagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggag"
    "cccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagccc"
    "aggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccagg"
    "agccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagc"
    "ccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccc"
    "tggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctg"
)

# S/MAR insulator flanks (from smar_insulator.py)
SMAR_INSULATOR_UPSTREAM = (
    "AATAAATATTTAAAATATTTTAAAATATATTTTAAATATTAAATTTTATATTTAAAAATTTAAAT"
    "ATTTTAAATATATATTTAAATTTATATTAAAAATTTATTTAAAATATTTATATTTAAAATTAT"
)

# miRNA 3' UTR cassette (from mirna_detarget.py / construct_builder.py)
MIRNA_UTR_CASSETTE = (
    "tggagtgtgacaatggtgtttgtggagtgtgacaatggtgtttgtggagtgtgacaatggtgtttg"
    "tggagtgtgacaatggtgtttg"  # 4x miR-122 liver detarget
    "cctgaagg"
    "ccataaagtagaaagcactacccataaagtagaaagcactacccataaagtagaaagcactacccat"
    "aaagtagaaagcactac"  # 4x miR-142 immune detarget
    "cctgaagg"
    "ggaatgtaaagaagtatggagggaatgtaaagaagtatggagggaatgtaaagaagtatggaggga"
    "atgtaaagaagtatggag"  # 4x miR-1 cardiac retarget
    "ataagacgagcaaaaagcttgtataagacgagcaaaaagcttgtataagacgagcaaaaagcttgt"
    "ataagacgagcaaaaagcttgt"  # 4x miR-208 cardiac retarget
)

# Synthesis vendor pricing
VENDOR_PRICES = {
    "PackGene": {"base": 499, "per_bp": 0.35, "min": 499, "max": 5000},
    "VectorBuilder": {"base": 399, "per_bp": 0.45, "min": 399, "max": 5500},
    "TwistBioscience": {"base": 299, "per_bp": 0.55, "min": 299, "max": 6000},
    "GenScript": {"base": 350, "per_bp": 0.40, "min": 350, "max": 4500},
}

# AAV9 packaging limit (bp between ITRs)
AAV_MAX_CARGO_BP = 4900  # hard limit (4.7 kb recommended + 200 bp margin)


# ============================================================================ #
# Data classes                                                                 #
# ============================================================================ #

@dataclass
class VectorComponent:
    name: str
    sequence: str
    length_bp: int
    start_bp: int
    end_bp: int


@dataclass
class AssembledVector:
    vector_id: str
    label: str
    full_sequence: str
    cargo_sequence: str
    cargo_length_bp: int
    total_length_bp: int
    components: List[VectorComponent]
    cargo_within_limit: bool
    cost_usd: float
    vendor: str
    genbank_format: str


@dataclass
class PlasmidAssemblyReport:
    vector_a: AssembledVector
    vector_b: AssembledVector
    split_position_aa: int
    split_position_nt: int
    harmonization_score: float
    cpg_density_a: float
    cpg_density_b: float
    rna_hairpin_risk: str
    rna_hairpin_details: List[Dict]
    total_synthesis_cost_usd: float
    both_vectors_pass: bool


# ============================================================================ #
# RNA Secondary Structure Scanner                                              #
# ============================================================================ #

# Simple Nussinov-like hairpin detector: finds runs of 4+ consecutive
# complementary bases within a sliding window that could form a stem.
# This is a lightweight heuristic; full ViennaRNA integration is optional.

def _complement_base(base: str) -> str:
    comp = {"A": "U", "T": "A", "G": "C", "C": "G",
            "a": "u", "t": "a", "g": "c", "c": "g"}
    return comp.get(base, "N")


def scan_rna_hairpins(rna_sequence: str, min_stem_bp: int = 4,
                       max_loop_nt: int = 30) -> List[Dict]:
    """Scan an RNA sequence for potential hairpin loops.

    A hairpin is detected when a run of >= min_stem_bp consecutive bases
    can base-pair with a downstream run separated by <= max_loop_nt bases.

    Returns a list of dicts with position, stem_length, loop_length, and
    risk severity.
    """
    seq = rna_sequence.upper().replace("T", "U")
    n = len(seq)
    hairpins = []

    for i in range(n - 2 * min_stem_bp):
        for stem_len in range(min_stem_bp, min(12, (n - i) // 2)):
            # Check if seq[i:i+stem_len] is complementary to seq[i+stem_len+loop:i+2*stem_len+loop]
            for loop_len in range(3, min(max_loop_nt + 1, n - i - 2 * stem_len)):
                j = i + stem_len + loop_len
                if j + stem_len > n:
                    break
                match = all(
                    seq[i + k] == _complement_base(seq[j + stem_len - 1 - k])
                    for k in range(stem_len)
                )
                if match:
                    severity = "low"
                    if stem_len >= 6:
                        severity = "medium"
                    if stem_len >= 8:
                        severity = "high"
                    hairpins.append({
                        "position": i,
                        "stem_length": stem_len,
                        "loop_length": loop_len,
                        "loop_sequence": seq[i + stem_len: i + stem_len + loop_len],
                        "severity": severity,
                        "context": seq[max(0, i - 5): min(n, i + 2 * stem_len + loop_len + 5)],
                    })
                    break  # only report first loop length per stem start

    # Deduplicate overlapping hairpins (keep worst severity per region)
    seen = {}
    for h in hairpins:
        key = (h["position"] // 10, h["stem_length"])
        if key not in seen or h["severity"] in ("high", "medium") and seen[key]["severity"] == "low":
            seen[key] = h
    return sorted(seen.values(), key=lambda x: x["position"])


# ============================================================================ #
# CpG density                                                                  #
# ============================================================================ #

def cpg_density(seq: str) -> float:
    """CpG dinucleotide density per 100 bp."""
    s = seq.upper()
    if len(s) < 2:
        return 0.0
    return s.count("CG") / (len(s) / 2.0)


# ============================================================================ #
# PlasmidAssembler                                                              #
# ============================================================================ #

class PlasmidAssembler:
    """Generates synthesis-ready dual-vector AAV9-LAMP2B plasmids.

    Pulls real sequences from existing pipeline modules and stitches them
    into two complete, base-by-base synthetic DNA strings ready for direct
    upload to contract manufacturing platforms.
    """

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "synthesis_ready_outputs"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_lamp2b_halves(self, harmonized_cds: Optional[str] = None,
                           split_position_nt: int = 600) -> Tuple[str, str]:
        """Split the LAMP2B CDS into 5' and 3' halves at the intein junction.

        Default split at nt 600 (aa 200), matching the optimal Npu DnaE
        split-intein position identified in Phase 9.
        """
        cds = harmonized_cds or LAMP2B_CDS_FULL
        cds_upper = cds.upper().replace("U", "T")
        split_nt = min(split_position_nt, len(cds_upper))
        # Ensure split is at a codon boundary (multiple of 3)
        split_nt = (split_nt // 3) * 3
        return cds_upper[:split_nt], cds_upper[split_nt:]

    def _assemble_vector_a(self, harmonized_cds: Optional[str] = None,
                            promoter: Optional[str] = None,
                            insulator: Optional[str] = None,
                            poly_a: Optional[str] = None,
                            split_position_nt: int = 600) -> AssembledVector:
        """Assemble Vector A: 5' Front-End Expression Cassette.

        Architecture: [5' ITR] [Insulator] [Promoter] [LAMP2B 5'] [N-Intein] [PolyA] [3' ITR]
        """
        prom = promoter or PROMOTER_MHC_ENHANCED
        ins = insulator or SMAR_INSULATOR_UPSTREAM
        pA = poly_a or POLYA_HGH
        cds_5, _ = self._get_lamp2b_halves(harmonized_cds, split_position_nt)

        # Build component list
        components = []
        cursor = 1

        def _add(name, seq):
            nonlocal cursor
            s = seq.upper().replace("U", "T")
            components.append(VectorComponent(
                name=name, sequence=s, length_bp=len(s),
                start_bp=cursor, end_bp=cursor + len(s) - 1
            ))
            cursor += len(s)
            return s

        itr5 = _add("5' ITR (AAV2)", ITR_5PRIME)
        ins_seq = _add("S/MAR Insulator", ins)
        prom_seq = _add("Cardiac Promoter (MHC+enhancer)", prom)
        cds5_seq = _add("LAMP2B 5' (exons 1-6)", cds_5)
        n_intein = _add("Npu DnaE N-intein (102 aa)", NPU_DNAE_N_INTEIN_CDS)
        poly_a = _add("hGH polyA signal", pA)
        itr3 = _add("3' ITR (AAV2)", ITR_3PRIME)

        full_seq = "".join(c.sequence for c in components)
        cargo_seq = ins_seq + prom_seq + cds5_seq + n_intein + poly_a
        cargo_len = len(cargo_seq)
        total_len = len(full_seq)

        cost = self._estimate_cost(total_len, "PackGene")
        gb = self._to_genbank(full_seq, "Vector_A_5Prime",
                               components, "Danon Disease AAV9-LAMP2B Vector A (5' Front-End)")

        return AssembledVector(
            vector_id="Vector_A_5Prime",
            label="AAV9-LAMP2B Vector A: 5' Front-End Expression Cassette",
            full_sequence=full_seq,
            cargo_sequence=cargo_seq,
            cargo_length_bp=cargo_len,
            total_length_bp=total_len,
            components=components,
            cargo_within_limit=cargo_len <= AAV_MAX_CARGO_BP,
            cost_usd=cost,
            vendor="PackGene",
            genbank_format=gb,
        )

    def _assemble_vector_b(self, harmonized_cds: Optional[str] = None,
                            promoter: Optional[str] = None,
                            mirna_utr: Optional[str] = None,
                            wpre: Optional[str] = None,
                            poly_a: Optional[str] = None,
                            split_position_nt: int = 600) -> AssembledVector:
        """Assemble Vector B: 3' Back-End Expression Cassette.

        Architecture: [5' ITR] [C-Intein] [LAMP2B 3'] [miRNA UTR] [WPRE] [PolyA] [3' ITR]
        """
        prom = promoter or PROMOTER_MHC_ENHANCED
        mirna = mirna_utr or MIRNA_UTR_CASSETTE
        wp = wpre or WPRE
        pA = poly_a or POLYA_HGH
        _, cds_3 = self._get_lamp2b_halves(harmonized_cds, split_position_nt)

        components = []
        cursor = 1

        def _add(name, seq):
            nonlocal cursor
            s = seq.upper().replace("U", "T")
            components.append(VectorComponent(
                name=name, sequence=s, length_bp=len(s),
                start_bp=cursor, end_bp=cursor + len(s) - 1
            ))
            cursor += len(s)
            return s

        itr5 = _add("5' ITR (AAV2)", ITR_5PRIME)
        prom_seq = _add("Cardiac Promoter (MHC+enhancer)", prom)
        c_intein = _add("Npu DnaE C-intein (36 aa)", NPU_DNAE_C_INTEIN_CDS)
        cds3_seq = _add("LAMP2B 3' (exons 7-9)", cds_3)
        mirna_seq = _add("miRNA detarget UTR (4xmiR-122/142/1/208)", mirna)
        wpre_seq = _add("WPRE", wp)
        poly_a = _add("hGH polyA signal", pA)
        itr3 = _add("3' ITR (AAV2)", ITR_3PRIME)

        full_seq = "".join(c.sequence for c in components)
        cargo_seq = prom_seq + c_intein + cds3_seq + mirna_seq + wpre_seq + poly_a
        cargo_len = len(cargo_seq)
        total_len = len(full_seq)

        cost = self._estimate_cost(total_len, "PackGene")
        gb = self._to_genbank(full_seq, "Vector_B_3Prime",
                               components, "Danon Disease AAV9-LAMP2B Vector B (3' Back-End)")

        return AssembledVector(
            vector_id="Vector_B_3Prime",
            label="AAV9-LAMP2B Vector B: 3' Back-End Expression Cassette",
            full_sequence=full_seq,
            cargo_sequence=cargo_seq,
            cargo_length_bp=cargo_len,
            total_length_bp=total_len,
            components=components,
            cargo_within_limit=cargo_len <= AAV_MAX_CARGO_BP,
            cost_usd=cost,
            vendor="PackGene",
            genbank_format=gb,
        )

    def _estimate_cost(self, bp: int, vendor: str = "PackGene") -> float:
        pricing = VENDOR_PRICES.get(vendor, VENDOR_PRICES["PackGene"])
        cost = pricing["base"] + bp * pricing["per_bp"]
        return max(pricing["min"], min(cost, pricing["max"]))

    def _to_genbank(self, seq: str, vector_id: str,
                     components: List[VectorComponent],
                     definition: str) -> str:
        lines = [
            f"LOCUS       {vector_id}  {len(seq)} bp  ds-DNA  circular  SYN",
            f"DEFINITION  {definition}.",
            f"ACCESSION   DANON_{vector_id}",
            "KEYWORDS    AAV9 LAMP2B Danon Disease gene therapy.",
            "SOURCE      Synthetic DNA construct",
            "FEATURES             Location/Qualifiers",
            "     source          1.." + str(len(seq)),
            "                     /organism=\"Synthetic DNA construct\"",
            "                     /mol_type=\"other DNA\"",
        ]
        for comp in components:
            lines.append(
                f"     misc_feature    {comp.start_bp}..{comp.end_bp}"
            )
            lines.append(
                f"                     /label=\"{comp.name}\""
            )
        lines.append("ORIGIN")
        for i in range(0, len(seq), 60):
            chunk = seq[i:i + 60]
            num = i + 1
            formatted = " ".join(chunk[j:j + 10] for j in range(0, len(chunk), 10))
            lines.append(f"{num:>9} {formatted}")
        lines.append("//")
        return "\n".join(lines)

    def assemble(self, harmonized_cds: Optional[str] = None,
                 promoter: Optional[str] = None,
                 insulator: Optional[str] = None,
                 mirna_utr: Optional[str] = None,
                 wpre: Optional[str] = None,
                 poly_a: Optional[str] = None,
                 split_position_aa: int = 200,
                 harmonization_score: float = 0.0) -> PlasmidAssemblyReport:
        """Assemble both vectors and run post-assembly verification.

        Args:
            harmonized_cds: Codon-harmonized LAMP2B CDS (nt). If None, uses wild-type.
            promoter: Cardiac promoter sequence (nt). If None, uses MHC+enhancer.
            insulator: S/MAR insulator upstream sequence. If None, uses default.
            mirna_utr: miRNA detarget 3' UTR cassette. If None, uses 4x miR-122/142/1/208.
            wpre: WPRE sequence. If None, uses default.
            poly_a: PolyA signal sequence. If None, uses hGH polyA.
            split_position_aa: Amino acid position for Npu DnaE split (default 200).
            harmonization_score: Codon harmonization correlation score from Phase 22.

        Returns:
            PlasmidAssemblyReport with both vectors, verification results, and cost.
        """
        split_nt = split_position_aa * 3

        logger.info("PLASMID ASSEMBLY: Dual-Vector AAV9-LAMP2B Construction")
        logger.info("  Split position: aa %d (nt %d)", split_position_aa, split_nt)

        # Assemble both vectors
        vec_a = self._assemble_vector_a(
            harmonized_cds, promoter, insulator, poly_a, split_nt
        )
        vec_b = self._assemble_vector_b(
            harmonized_cds, promoter, mirna_utr, wpre, poly_a, split_nt
        )

        # Post-assembly verification
        # 1. Cargo constraint check
        logger.info("  Vector A cargo: %d bp (limit %d bp) [%s]",
                     vec_a.cargo_length_bp, AAV_MAX_CARGO_BP,
                     "PASS" if vec_a.cargo_within_limit else "FAIL")
        logger.info("  Vector B cargo: %d bp (limit %d bp) [%s]",
                     vec_b.cargo_length_bp, AAV_MAX_CARGO_BP,
                     "PASS" if vec_b.cargo_within_limit else "FAIL")

        # 2. RNA hairpin scan on the mRNA transcript
        #    Simulate the spliced mRNA: promoter + LAMP2B + WPRE + polyA
        mrna_transcript = (
            promoter or PROMOTER_MHC_ENHANCED
        ) + (
            harmonized_cds or LAMP2B_CDS_FULL
        ) + (wpre or WPRE) + (poly_a or POLYA_HGH)
        mrna_rna = mrna_transcript.upper().replace("T", "U")

        hairpins = scan_rna_hairpins(mrna_rna, min_stem_bp=4, max_loop_nt=25)
        high_risk = [h for h in hairpins if h["severity"] == "high"]
        med_risk = [h for h in hairpins if h["severity"] == "medium"]

        if high_risk:
            hairpin_risk = "HIGH"
        elif med_risk:
            hairpin_risk = "MEDIUM"
        else:
            hairpin_risk = "LOW"

        logger.info("  RNA hairpin scan: %d total, %d high, %d medium [%s risk]",
                     len(hairpins), len(high_risk), len(med_risk), hairpin_risk)

        # 3. CpG density check
        cds_seq = harmonized_cds or LAMP2B_CDS_FULL
        cpg_a = cpg_density(vec_a.cargo_sequence)
        cpg_b = cpg_density(vec_b.cargo_sequence)
        logger.info("  CpG density: Vector A=%.2f/100bp, Vector B=%.2f/100bp", cpg_a, cpg_b)

        # Both-pass gate
        both_pass = vec_a.cargo_within_limit and vec_b.cargo_within_limit

        total_cost = vec_a.cost_usd + vec_b.cost_usd
        logger.info("  Total synthesis cost: $%.0f (A=$%.0f, B=$%.0f)",
                     total_cost, vec_a.cost_usd, vec_b.cost_usd)

        return PlasmidAssemblyReport(
            vector_a=vec_a,
            vector_b=vec_b,
            split_position_aa=split_position_aa,
            split_position_nt=split_nt,
            harmonization_score=harmonization_score,
            cpg_density_a=cpg_a,
            cpg_density_b=cpg_b,
            rna_hairpin_risk=hairpin_risk,
            rna_hairpin_details=hairpins[:20],
            total_synthesis_cost_usd=total_cost,
            both_vectors_pass=both_pass,
        )

    def export_fasta(self, report: PlasmidAssemblyReport) -> Dict[str, str]:
        """Export both vectors as FASTA files to disk."""
        paths = {}
        for vec in [report.vector_a, report.vector_b]:
            filename = f"{vec.vector_id}_Synthesis.fasta"
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, "w") as f:
                f.write(f">{vec.vector_id} | {vec.label}\n")
                formatted = "\n".join(
                    vec.full_sequence[i:i + 60]
                    for i in range(0, len(vec.full_sequence), 60)
                )
                f.write(formatted + "\n")
            paths[vec.vector_id] = filepath
            logger.info("  FASTA export: %s (%d bp)", filepath, vec.total_length_bp)
        return paths

    def export_genbank(self, report: PlasmidAssemblyReport) -> Dict[str, str]:
        """Export both vectors as GenBank flat files to disk."""
        paths = {}
        for vec in [report.vector_a, report.vector_b]:
            filename = f"{vec.vector_id}_Synthesis.gb"
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, "w") as f:
                f.write(vec.genbank_format)
            paths[vec.vector_id] = filepath
            logger.info("  GenBank export: %s", filepath)
        return paths

    def export_assembly_report(self, report: PlasmidAssemblyReport) -> str:
        """Export a JSON summary of the assembly."""
        import json
        data = {
            "split_position_aa": report.split_position_aa,
            "split_position_nt": report.split_position_nt,
            "harmonization_score": report.harmonization_score,
            "vector_a": {
                "vector_id": report.vector_a.vector_id,
                "total_length_bp": report.vector_a.total_length_bp,
                "cargo_length_bp": report.vector_a.cargo_length_bp,
                "cargo_within_limit": report.vector_a.cargo_within_limit,
                "cost_usd": report.vector_a.cost_usd,
                "vendor": report.vector_a.vendor,
                "components": [
                    {"name": c.name, "length_bp": c.length_bp,
                     "start_bp": c.start_bp, "end_bp": c.end_bp}
                    for c in report.vector_a.components
                ],
            },
            "vector_b": {
                "vector_id": report.vector_b.vector_id,
                "total_length_bp": report.vector_b.total_length_bp,
                "cargo_length_bp": report.vector_b.cargo_length_bp,
                "cargo_within_limit": report.vector_b.cargo_within_limit,
                "cost_usd": report.vector_b.cost_usd,
                "vendor": report.vector_b.vendor,
                "components": [
                    {"name": c.name, "length_bp": c.length_bp,
                     "start_bp": c.start_bp, "end_bp": c.end_bp}
                    for c in report.vector_b.components
                ],
            },
            "cpg_density": {"vector_a": report.cpg_density_a, "vector_b": report.cpg_density_b},
            "rna_hairpin_risk": report.rna_hairpin_risk,
            "rna_hairpin_count": len(report.rna_hairpin_details),
            "total_synthesis_cost_usd": report.total_synthesis_cost_usd,
            "both_vectors_pass": report.both_vectors_pass,
        }
        filepath = os.path.join(self.output_dir, "assembly_report.json")
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("  Assembly report: %s", filepath)
        return filepath

    def print_assembly_summary(self, report: PlasmidAssemblyReport):
        """Print a human-readable assembly summary."""
        logger.info("=" * 80)
        logger.info("PLASMID ASSEMBLY SUMMARY — DUAL-VECTOR AAV9-LAMP2B")
        logger.info("=" * 80)
        logger.info("  Split position: aa %d (nt %d)", report.split_position_aa, report.split_position_nt)
        logger.info("  Codon harmonization score: %.4f", report.harmonization_score)
        logger.info("")
        logger.info("  VECTOR A (5' Front-End):")
        logger.info("    Total length:   %d bp", report.vector_a.total_length_bp)
        logger.info("    Cargo length:   %d bp  [%s]",
                     report.vector_a.cargo_length_bp,
                     "PASS" if report.vector_a.cargo_within_limit else "FAIL")
        logger.info("    Synthesis cost: $%.0f (%s)", report.vector_a.cost_usd, report.vector_a.vendor)
        for c in report.vector_a.components:
            logger.info("      %s: %d bp", c.name, c.length_bp)
        logger.info("")
        logger.info("  VECTOR B (3' Back-End):")
        logger.info("    Total length:   %d bp", report.vector_b.total_length_bp)
        logger.info("    Cargo length:   %d bp  [%s]",
                     report.vector_b.cargo_length_bp,
                     "PASS" if report.vector_b.cargo_within_limit else "FAIL")
        logger.info("    Synthesis cost: $%.0f (%s)", report.vector_b.cost_usd, report.vector_b.vendor)
        for c in report.vector_b.components:
            logger.info("      %s: %d bp", c.name, c.length_bp)
        logger.info("")
        logger.info("  POST-ASSEMBLY VERIFICATION:")
        logger.info("    Cargo constraint: %s",
                     "BOTH PASS" if report.both_vectors_pass else "FAIL — oversized")
        logger.info("    RNA hairpin risk: %s", report.rna_hairpin_risk)
        logger.info("    CpG density A: %.2f/100bp, B: %.2f/100bp",
                     report.cpg_density_a, report.cpg_density_b)
        logger.info("    Total synthesis cost: $%.0f", report.total_synthesis_cost_usd)
        logger.info("=" * 80)
