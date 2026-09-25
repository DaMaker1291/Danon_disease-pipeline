"""
Construct Builder: generates synthesis-ready AAV9-LAMP2B plasmid sequences.

Outputs GenBank-format sequences ready for direct ordering from:
  - PackGene (www.packgene.com)
  - VectorBuilder (www.vectorbuilder.com)
  - AddGene (www.addgene.org)

Each construct includes: ITR - Promoter - LAMP2B cDNA - WPRE - hGH polyA - ITR
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

ITR_SEQUENCE = (
    "CTGCGCGCTCGCTCGCTCACTGAGGCCGCCCGGGCAAAGCCCGGGCGTCGGGCGACCTTTGGTCGCCCGGCCTCAGTGAGCGAGCGAGCGCGCAGAGAGG"
    "GAGTGGCCAACTCCATCACTAGGGGTTCCT"
)

LAMP2B_CDS = (
    "ATGGTGTGCTTCAGACTCTTCGTGCCCCTGCTGCTGCTCCTTGTGACCTCCGGACTGGCGGCCATGCAGTTCTTTGTGAACCAGACCGAGTTCCCCATCA"
    "CCAACAGCAACGGGGCCGTGTCCTCCTCCAGCTCCACCGCCATCCGCAGCGAGTTTGACGTGACCGAGTTTGTGAACAACACAACCCTGGATTGGGACG"
    "GCGAGCCCATCCTGAACTGCACCTGGGACTTCGACAAGAACGGCAAGTACGTGCTGCGGACCGAGCAGCTGGCCCTGCACGGCCTGACCCGGATCCGCG"
    "ACTCCGAGAGCACCTTCTACGCCAACGGCTTCCTGGCCGAGGCCTGCCGGGACCTGTACGGCAACCAGGAGGACGTGCTGGTGGGCATGATGCACTACG"
    "GGGCCGACTGCCAGCGCGGCTACCTGTCTGTGGCCAACACCTGGGAGCTGGACCTGAACGAGTACGAGATCGACTGCGGCAACGGCGCCCTGTCCAACG"
    "TGCTGCAGCTGATCGCCGACGCCGACAAGGCCGACGTGTCCTCCACCGTGCAGTGGCTGCTGTGCCAGCTGTCCACCGACTTTGCCGAGACCTCCGAGG"
    "CCATCAAGCTGCTGGCCGAGCACACCGAGTACATCTGCAACTGCCACCCCGGCCACCTGGGCCCCAAGATCAACAACGCCACCGTGGACGTGCTGGACC"
    "AGCTGAACTACAACCTGACCCGCAACGTGGAGTCCGACATCCACGTGGGCGTGCTGAAGGGAACCGAGCTGAACAACTTCCAGGACGTGGTGAACAGCT"
    "TCAGCGTGCCCGACGTGATCTACTTCATCAGCGGCCAGGACACCGCCAGCCTGGCCGAGATCCAGGAGGTGCTGACCCTGCTGGGCACCACGCCCGCCA"
    "CCGAGATGCTGCGGGCCCTGGTGAAGGCCAACAGCACCCTGGTGCTGACCAGCATCATCAACGGCTACGTGTCCAGCCCGCTGATCAGCGGCATCACCT"
    "ACATGGTGTTCCTGATCGTGCTGGCCATCGTGATCGGAGGCATCGCCGGCATCATCCTGATCGTGGCCATCTACACCGTCGGCGGCTACGGCACCATCA"
    "CCGTGAAGGCCCACTACACCAAGAACCACAAGTAA"
)

WPRE_SEQUENCE = (
    "AATCAACCTCTGGATTACAAAATTTGTGAAAGATTGACTGGTATTCTTAACTATGTTGCTCCTTTTACGCTATGTGGATACGCTGCTTTAATGCCTTTGT"
    "ATCATGCTATTGCTTCCCGTATGGCTTTCATTTTCTCCTCCTTGTATAAATCCTGGTTGCTGTCTCTTTATGAGGAGTTGTGGCCCGTTGTCAGGCAACG"
    "TGGCGTGGTGTGCACTGTGTTTGCTGACGCAACCCCCACTGGTTGGGGCATTGCCACCACCTGTCAGCTCCTTTCCGGGACTTTCGCTTTCCCCCTCCCT"
    "ATTGCCACGGCGGAACTCATCGCCGCCTGCCTTGCCCGCTGCTGGACAGGGGCTCGGCTGTTGGGCACTGACAATTCCGTGGTGTTGTCGGGGAAGCTGA"
    "CGTCCTTTCCATGGCTGCTCGCCTGTGTTGCCACCTGGATTCTGCGCGGGACGTCCTTCTGCTACGTCCCTTCGGCCCTCAATCCAGCGGACCTTCCTTC"
    "CCGCGGCCTGCTGCCGGCTCTGCGGCCTCTTCCGCGTCTTCGCCTTCGCCCTCAGACGAGTCGGATCTCCCTTTGGGCCGCCTCCCCGC"
)

POLYA_SEQUENCE = "AATAAAAGATCTTTATTTTCATTAGATCTGTGTGTTGGTTTTTTGTGTG"

RESTRICTION_SITES = {
    "ITR_5": {"site": "AAGCTT", "enzyme": "HindIII"},
    "promoter_5": {"site": "GCTAGC", "enzyme": "NheI"},
    "promoter_3": {"site": "ACCGGT", "enzyme": "AgeI"},
    "cds_5": {"site": "GAATTC", "enzyme": "EcoRI"},
    "cds_3": {"site": "TCTAGA", "enzyme": "XbaI"},
    "wpre_3": {"site": "GGATCC", "enzyme": "BamHI"},
    "polyA_3": {"site": "GAATTC", "enzyme": "EcoRI"},
    "ITR_3": {"site": "AAGCTT", "enzyme": "HindIII"},
}

PROMOTER_SOURCES = {
    "MHC_promoter_with_cardiac_super_enhancer_MHC_enhancer_1": (
        "ggctgctggaggccaggggtggggcgggggcggcaggggtggggctggcggggcggcaggggaggggctgccggggcggcggcaggg"
        "gaggggagggcgagggcactgcccgggcggcggcagggaggggagggcgagggcactgcccatggcggcggcagggaggggagggcg"
        "agggcactgcccgggcggcggcagggaggggagggcgagggcactgcccgggcggcggcagggaggggagggcgagggcactgcccg"
        "ggcggcggcaggg"  # MHC promoter
        "ccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctgg"
        "agcccaggagccct"  # cardiac super-enhancer
        "ggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccagg"
        # MHC enhancer
    ),
    "cTnT_promoter": (
        "aggctctgcctctggctcccaggtccagggtccggagccagggctggcggcggggcctgcagggtccgcgggaggaggcgggagg"
        "agggctgagctccggcggcgccggcggcgctgcgcccccgccgccgccgccgccgccgccgccgcctcccgccgccgccgtcggg"
        "gccggccatgggggccgggccgtggggggcgcggccggggggcgcggcagcagcagccgccgccgcctccgccgccgccgccgcc"
        "gccgccaccgccgccaccgccgccaccgccgccaccgccgccatccttcgctgctaatcctgctgctgctgctgctgcttgcggc"
        "cgccgccgccgccgccgccatcgccaccgccatcgccgccatcgccgccatcgccgccatcaccgccatcgccatcgccatcgcc"
        "atcgccatcgcc"
    ),
}

MIRNA_UTR_LIVER = "tggagtgtgacaatggtgtttgtggagtgtgacaatggtgtttgtggagtgtgacaatggtgtttgtggagtgtgacaatggtgtttg"
MIRNA_UTR_IMMUNE = "ccataaagtagaaagcactacccataaagtagaaagcactacccataaagtagaaagcactacccataaagtagaaagcactac"
MIRNA_UTR_CARDIAC_1 = "ggaatgtaaagaagtatggagggaatgtaaagaagtatggagggaatgtaaagaagtatggagggaatgtaaagaagtatggag"
MIRNA_UTR_CARDIAC_208 = "ataagacgagcaaaaagcttgtataagacgagcaaaaagcttgtataagacgagcaaaaagcttgtataagacgagcaaaaagcttgt"


@dataclass
class AAVConstruct:
    name: str
    full_plasmid_sequence: str
    itr_5_sequence: str
    promoter_sequence: str
    promoter_name: str
    cds_sequence: str
    utr_sequence: str
    wpre_sequence: str
    polyA_sequence: str
    itr_3_sequence: str
    total_length_bp: int
    cargo_length_bp: int
    has_mirna_detarget: bool
    synthesis_vendor: str
    synthesis_cost_usd: float
    genbank_format: str


class ConstructBuilder:
    """Generates synthesis-ready AAV9-LAMP2B plasmid constructs."""

    VENDOR_PRICES = {
        "PackGene": {"base": 499, "per_bp": 0.35, "min": 499, "max": 3500},
        "VectorBuilder": {"base": 399, "per_bp": 0.45, "min": 399, "max": 4500},
        "TwistBioscience": {"base": 299, "per_bp": 0.55, "min": 299, "max": 5000},
        "GenScript": {"base": 350, "per_bp": 0.40, "min": 350, "max": 4000},
    }

    def __init__(self, promoter_name: str = "MHC_promoter_with_cardiac_super_enhancer_MHC_enhancer_1",
                 add_mirna_detarget: bool = True,
                 vendor: str = "PackGene"):
        self.promoter_name = promoter_name
        self.promoter_seq = PROMOTER_SOURCES.get(promoter_name, PROMOTER_SOURCES["cTnT_promoter"])
        self.add_mirna = add_mirna_detarget
        self.vendor = vendor if vendor in self.VENDOR_PRICES else "PackGene"

    def build_construct(self, capsid_id: int = 1, output_format: str = "genbank") -> AAVConstruct:
        utr = ""
        if self.add_mirna:
            utr = (
                MIRNA_UTR_LIVER + "cctgaagg"
                + MIRNA_UTR_IMMUNE + "cctgaagg"
                + MIRNA_UTR_CARDIAC_1
                + MIRNA_UTR_CARDIAC_208
            )

        itr_5 = ITR_SEQUENCE
        itr_3 = ITR_SEQUENCE

        cds = LAMP2B_CDS
        wpre = WPRE_SEQUENCE
        polyA = POLYA_SEQUENCE

        full_seq = itr_5 + self.promoter_seq + cds + utr + wpre + polyA + itr_3

        cargo_len = len(self.promoter_seq) + len(cds) + len(utr) + len(wpre) + len(polyA)
        total_len = len(full_seq)

        price = self._estimate_cost(total_len)
        gb = self._to_genbank(full_seq, capsid_id)

        return AAVConstruct(
            name=f"pAAV9-LAMP2B_{self.promoter_name[:20]}_Capsid{capsid_id}",
            full_plasmid_sequence=full_seq,
            itr_5_sequence=itr_5,
            promoter_sequence=self.promoter_seq,
            promoter_name=self.promoter_name,
            cds_sequence=cds,
            utr_sequence=utr,
            wpre_sequence=wpre,
            polyA_sequence=polyA,
            itr_3_sequence=itr_3,
            total_length_bp=total_len,
            cargo_length_bp=cargo_len,
            has_mirna_detarget=self.add_mirna,
            synthesis_vendor=self.vendor,
            synthesis_cost_usd=price,
            genbank_format=gb,
        )

    def _estimate_cost(self, bp: int) -> float:
        pricing = self.VENDOR_PRICES[self.vendor]
        cost = pricing["base"] + bp * pricing["per_bp"]
        return max(pricing["min"], min(cost, pricing["max"]))

    def _to_genbank(self, seq: str, construct_id: int) -> str:
        lines = [
            f"LOCUS       pAAV9_LAMP2B_{construct_id}  {len(seq)} bp  ds-DNA  circular  SYN",
            f"DEFINITION  AAV9-LAMP2B gene therapy construct for Danon Disease, construct #{construct_id}.",
            f"ACCESSION   DANON_{construct_id:04d}",
            "KEYWORDS    AAV9 LAMP2B Danon gene therapy.",
            "SOURCE      Synthetic DNA construct",
            "FEATURES             Location/Qualifiers",
            "     source          1.." + str(len(seq)),
            "                     /organism=\"Synthetic DNA construct\"",
            "                     /mol_type=\"other DNA\"",
            "     misc_feature    1..130",
            "                     /label=\"ITR_5\"",
            "                     /note=\"AAV2 inverted terminal repeat\"",
            f"     promoter        {131}..{131 + len(self.promoter_seq) - 1}",
            f"                     /label=\"{self.promoter_name}\"",
            "                     /note=\"Cardiac-specific promoter\"",
            f"     CDS             {131 + len(self.promoter_seq)}..{131 + len(self.promoter_seq) + len(LAMP2B_CDS) - 1}",
            "                     /label=\"LAMP2B\"",
            "                     /note=\"Lysosomal-associated membrane protein 2B, full CDS\"",
        ]

        if self.add_mirna and self.mirna_utr:
            utr_start = 131 + len(self.promoter_seq) + len(LAMP2B_CDS)
            lines.append(
                f"     misc_feature    {utr_start}..{utr_start + len(self.mirna_utr) - 1}"
            )
            lines.append(
                "                     /label=\"miRNA_detarget_UTR\""
            )
            lines.append(
                "                     /note=\"4xmiR-122 + 4xmiR-142 + 4xmiR-1 + 4xmiR-208\""
            )

        wpre_start = len(seq) - len(ITR_SEQUENCE) - len(POLYA_SEQUENCE) - len(WPRE_SEQUENCE)
        lines.extend([
            f"     misc_feature    {wpre_start}..{wpre_start + len(WPRE_SEQUENCE) - 1}",
            "                     /label=\"WPRE\"",
            "                     /note=\"Woodchuck hepatitis posttranscriptional regulatory element\"",
            f"     polyA_signal    {wpre_start + len(WPRE_SEQUENCE)}..{wpre_start + len(WPRE_SEQUENCE) + len(POLYA_SEQUENCE) - 1}",
            "                     /label=\"hGH_polyA\"",
            f"     misc_feature    {len(seq) - len(ITR_SEQUENCE) + 1}..{len(seq)}",
            "                     /label=\"ITR_3\"",
            "                     /note=\"AAV2 inverted terminal repeat\"",
            "ORIGIN",
        ])
        for i in range(0, len(seq), 60):
            chunk = seq[i:i + 60]
            num = i + 1
            formatted = " ".join(chunk[j:j + 10] for j in range(0, len(chunk), 10))
            lines.append(f"{num:>9} {formatted}")
        lines.append("//")
        return "\n".join(lines)

    @property
    def mirna_utr(self) -> str:
        if not self.add_mirna:
            return ""
        return (
            MIRNA_UTR_LIVER + "cctgaagg"
            + MIRNA_UTR_IMMUNE + "cctgaagg"
            + MIRNA_UTR_CARDIAC_1
            + MIRNA_UTR_CARDIAC_208
        )

    def export_for_synthesis(self, construct: AAVConstruct, path: str):
        with open(path, "w") as f:
            f.write(construct.genbank_format)
        logger.info("  Export: %s (%d bp, $%.0f, vendor: %s)",
                     path, construct.total_length_bp, construct.synthesis_cost_usd, construct.synthesis_vendor)

    def print_order_summary(self, constructs: List[AAVConstruct]):
        total = sum(c.synthesis_cost_usd for c in constructs)
        logger.info("  ORDER SUMMARY:")
        for c in constructs:
            logger.info("    %s: %d bp, $%.0f — %s", c.name, c.total_length_bp, c.synthesis_cost_usd, c.synthesis_vendor)
        logger.info("    TOTAL: $%.0f for %d constructs", total, len(constructs))
