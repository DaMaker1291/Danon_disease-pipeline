import os
import json
import logging
import hashlib
import numpy as np
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

I7_INDEX_SEQUENCES = [
    "ATCACG", "CGATGT", "TTAGGC", "TGACCA", "ACAGTG",
    "GCCAAT", "CAGATC", "ACTTGA", "GATCAG", "TAGCTT",
    "GGCTAC", "CTTGTA", "AGTCAA", "AGTTCC", "ATGTCA",
    "CCGTCC", "GTAGAG", "TCCGCG", "GTCCGC", "GCGTCC",
    "AGAGTG", "TCGGAG", "GCGGAC", "TGCGTG", "CGGAGT",
    "TCCGAG", "GCGATG", "CTCAGC", "ATCGCT", "GATCGC",
    "AGCTAG", "TCGACT",
]

I5_INDEX_SEQUENCES = [
    "TATCCT", "CGAGAT", "AGTTCC", "GCGTAC", "CTATTA",
    "AAGGAC", "TTCCGA", "GGATTC", "CCAAGT", "TTCGGA",
    "ATCGTG", "GGTACA", "CGAAAT", "ACTCGG", "TGACGC",
    "CTGCGA", "GACGTT", "TTCAGC", "CCGTGG", "AATCGG",
    "GAACCG", "TCAGAT", "GCTGAA", "CGTTGC", "AGCAGT",
    "TCAAGC", "GACTTG", "CTAGCA", "AGTCGC", "TGCAGC",
    "CATGAC", "GTCAGA",
]


@dataclass
class Barcode:
    candidate_id: int
    i7_index: str
    i5_index: str
    umi: str
    full_barcode: str = ""
    primer_fwd: str = ""
    primer_rev: str = ""

    def __post_init__(self):
        self.full_barcode = f"{self.i7_index}+{self.umi}+{self.i5_index}"
        self.primer_fwd = f"ATCTTCCGTCACAGTCTTTC{self.i7_index}"
        self.primer_rev = f"GACGTAACACCGTTCCAGCT{self.i5_index}"


@dataclass
class BarcodeDesign:
    total_candidates: int
    barcodes: list[Barcode] = field(default_factory=list)
    i7_pool_size: int = 32
    i5_pool_size: int = 32
    umi_length: int = 12
    collision_count: int = 0


class BarcodingDesigner:
    def __init__(self):
        self.rng = np.random.RandomState(42)

    def design_barcodes(
        self, candidate_ids: list[int], umi_length: int = 12
    ) -> BarcodeDesign:
        design = BarcodeDesign(total_candidates=len(candidate_ids))
        design.umi_length = umi_length

        used_combos = set()
        collisions = 0

        for cid in candidate_ids:
            i7_idx = self.rng.randint(0, len(I7_INDEX_SEQUENCES))
            i5_idx = self.rng.randint(0, len(I5_INDEX_SEQUENCES))
            umi = self._generate_umi(umi_length)

            combo = (i7_idx, i5_idx, umi)
            attempts = 0
            while combo in used_combos and attempts < 100:
                i7_idx = self.rng.randint(0, len(I7_INDEX_SEQUENCES))
                i5_idx = self.rng.randint(0, len(I5_INDEX_SEQUENCES))
                umi = self._generate_umi(umi_length)
                combo = (i7_idx, i5_idx, umi)
                attempts += 1

            if combo in used_combos:
                collisions += 1

            used_combos.add(combo)
            barcode = Barcode(
                candidate_id=cid,
                i7_index=I7_INDEX_SEQUENCES[i7_idx],
                i5_index=I5_INDEX_SEQUENCES[i5_idx],
                umi=umi,
            )
            design.barcodes.append(barcode)

        design.collision_count = collisions
        logger.info(
            "Barcode design: %d barcodes, %d collisions (%.4f%%)",
            len(design.barcodes),
            collisions,
            100 * collisions / max(len(candidate_ids), 1),
        )
        return design

    def _generate_umi(self, length: int) -> str:
        bases = "ACGT"
        return "".join(self.rng.choice(list(bases), size=length))

    def export_fasta(self, design: BarcodeDesign, output_path: str):
        with open(output_path, "w") as f:
            for bc in design.barcodes:
                f.write(f">candidate_{bc.candidate_id}_i7\n")
                f.write(f"{bc.i7_index}\n")
                f.write(f">candidate_{bc.candidate_id}_umi\n")
                f.write(f"{bc.umi}\n")
                f.write(f">candidate_{bc.candidate_id}_i5\n")
                f.write(f"{bc.i5_index}\n")
        logger.info("Barcode FASTA exported: %s", output_path)

    def export_index_table(self, design: BarcodeDesign, output_path: str):
        import csv
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "candidate_id", "i7_index", "i5_index", "umi",
                "full_barcode", "primer_fwd", "primer_rev"
            ])
            for bc in design.barcodes:
                writer.writerow([
                    bc.candidate_id,
                    bc.i7_index,
                    bc.i5_index,
                    bc.umi,
                    bc.full_barcode,
                    bc.primer_fwd,
                    bc.primer_rev,
                ])
        logger.info("Index table exported: %s", output_path)

    def generate_pcr_protocol(self, design: BarcodeDesign, output_path: str) -> str:
        protocol = f"""# PCR Amplification Protocol
# Longevity Pipeline - Barcoded Candidate Amplification
# Total candidates: {design.total_candidates}
# Barcodes designed: {len(design.barcodes)}
# Collisions: {design.collision_count}

import numpy as np

def pcr_amplification(template_amount_ng, primer_conc_um, cycle_count=25):
    efficiency = 0.92
    final_amount = template_amount_ng * (1 + efficiency) ** cycle_count
    return final_amount

def setup_pcr_reactions(candidates, plate):
    for i, candidate in enumerate(candidates):
        row = chr(65 + i // 12)
        col = i % 12 + 1
        well = f"{{row}}{{col}}"

        plate.wells()[well].add_reagent("template", 1.0)  # uL
        plate.wells()[well].add_reagent("forward_primer", 0.5)
        plate.wells()[well].add_reagent("reverse_primer", 0.5)
        plate.wells()[well].add_reagent("master_mix", 12.5)
        plate.wells()[well].add_reagent("water", 10.5)

# Amplification
for candidate in candidates:
    amount = pcr_amplification(10.0, 0.5)
    print(f"Candidate {{candidate.candidate_id}}: {{amount:.1f}} ng")
"""
        with open(output_path, "w") as f:
            f.write(protocol)
        return protocol


class BarcodeDecoder:
    def __init__(self, design: BarcodeDesign):
        self.design = design
        self.barcode_to_id = {}
        for bc in design.barcodes:
            key = (bc.i7_index, bc.i5_index)
            self.barcode_to_id[key] = bc.candidate_id

    def decode_read(self, i7_seq: str, i5_seq: str) -> Optional[int]:
        key = (i7_seq, i5_seq)
        return self.barcode_to_id.get(key)

    def decode_fastq(self, fastq_path: str) -> dict:
        results = {}
        current_id = None
        current_seq = None
        line_count = 0

        with open(fastq_path, "r") as f:
            for line in f:
                line_count += 1
                line = line.strip()

                if line_count % 4 == 1:
                    current_id = line
                elif line_count % 4 == 2:
                    current_seq = line
                elif line_count % 4 == 0:
                    if current_seq and len(current_seq) >= 30:
                        i7 = current_seq[0:6]
                        umi = current_seq[6:18]
                        i5 = current_seq[18:24]

                        candidate_id = self.decode_read(i7, i5)
                        if candidate_id is not None:
                            if candidate_id not in results:
                                results[candidate_id] = {"count": 0, "umis": set()}
                            results[candidate_id]["count"] += 1
                            results[candidate_id]["umis"].add(umi)

        for cid in results:
            results[cid]["unique_umis"] = len(results[cid]["umis"])
            del results[cid]["umis"]

        return results
