import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from pipeline.config import LabConfig

logger = logging.getLogger(__name__)


@dataclass
class MicrofluidicParameters:
    total_flow_rate_ul_min: float = 1000.0
    flow_rate_ratio: float = 3.0
    aqueous_flow_rate: float = 750.0
    organic_flow_rate: float = 250.0
    target_particle_size_nm: float = 80.0
    ethanol_pct: float = 80.0
    temperature_c: float = 22.0
    channel_diameter_um: float = 200.0
    mixing_cycles: int = 3


@dataclass
class DeckLayout:
    deck_positions: dict
    tip_racks: list
    reagent_racks: list
    plates: list


class OpentronsProtocol:
    def __init__(self, config: LabConfig):
        self.config = config
        self.deck_layout = self._setup_deck()
        self.wells_used = set()

    def _setup_deck(self) -> DeckLayout:
        return DeckLayout(
            deck_positions={
                1: {"type": "tiprack_300ul", "label": "p300_tips_1"},
                2: {"type": "tiprack_300ul", "label": "p300_tips_2"},
                3: {"type": "tiprack_20ul", "label": "p20_tips"},
                4: {"type": "reservoir_12ml", "label": "wash_buffer"},
                5: {"type": "reservoir_12ml", "label": "ethanol_80pct"},
                6: {"type": "plate_96_well", "label": "source_lnp"},
                7: {"type": "plate_96_well", "label": "destination"},
                8: {"type": "plate_96_well", "label": "reagents"},
            },
            tip_racks=[1, 2, 3],
            reagent_racks=[4, 5],
            plates=[6, 7, 8],
        )

    def _compute_microfluidic_params(self, candidate: dict) -> MicrofluidicParameters:
        comp = candidate.get("composition", {})
        pka = comp.get("pka", 6.35)
        ion_frac = comp.get("ionizable_frac", 0.40)
        peg_frac = comp.get("peg_frac", 0.015)
        chol_frac = comp.get("cholesterol_frac", 0.35)

        target_size = 50 + peg_frac * 2000 + (chol_frac - 0.35) * (-100)
        target_size = max(30, min(150, target_size))

        tfr = 1000.0 + (target_size - 80) * 5
        tfr = max(200, min(5000, tfr))

        frr = 3.0 + (ion_frac - 0.40) * 10
        frr = max(1.0, min(8.0, frr))

        aq_flow = tfr * frr / (1 + frr)
        org_flow = tfr / (1 + frr)

        return MicrofluidicParameters(
            total_flow_rate_ul_min=tfr,
            flow_rate_ratio=frr,
            aqueous_flow_rate=aq_flow,
            organic_flow_rate=org_flow,
            target_particle_size_nm=target_size,
            ethanol_pct=80.0,
            temperature_c=22.0,
            channel_diameter_um=200.0,
            mixing_cycles=3,
        )

    def generate_synthesis_protocol(self, candidates: list[dict], output_path: str) -> str:
        protocol_lines = [
            'from opentrons import protocol_api',
            'import time',
            '',
            'metadata = {',
            '    "apiLevel": "2.15",',
            '    "protocolName": "Longevity Drug Synthesis - Microfluidic LNP",',
            '    "description": "Automated LNP synthesis with microfluidic chip control",',
            '    "author": "Longevity Pipeline"',
            '}',
            '',
            'def run(protocol: protocol_api.ProtocolContext):',
            '    # ---- DECK SETUP ----',
            '    p300_tips_1 = protocol.load_labware("opentrons_96_tiprack_300ul", 1)',
            '    p300_tips_2 = protocol.load_labware("opentrons_96_tiprack_300ul", 2)',
            '    p20_tips = protocol.load_labware("opentrons_96_tiprack_20ul", 3)',
            '    wash_buffer = protocol.load_labware("nest_12_reservoir_15ml", 4)',
            '    ethanol = protocol.load_labware("nest_12_reservoir_15ml", 5)',
            '    source_plate = protocol.load_labware("nest_96_wellplate_2ml_deep", 6)',
            '    dest_plate = protocol.load_labware("nest_96_wellplate_2ml_deep", 7)',
            '    reagent_plate = protocol.load_labware("nest_96_wellplate_2ml_deep", 8)',
            '',
            '    p300 = protocol.load_instrument("p300_multi_gen2", "left", [p300_tips_1, p300_tips_2])',
            '    p20 = protocol.load_instrument("p20_multi_gen2", "right", [p20_tips])',
            '',
            '    # ---- MICROFLUIDIC CHIP PARAMETERS ----',
            '    # These values are read by Arduino/RPi controlling the chip',
            '    microfluidic_config = {}',
            '',
        ]

        for batch_idx in range(0, len(candidates), 96):
            batch = candidates[batch_idx:batch_idx + 96]
            protocol_lines.append(f'    # ---- BATCH {batch_idx // 96 + 1} ----')

            for i, candidate in enumerate(batch):
                row = chr(ord("A") + (i // 12))
                col = i % 12 + 1
                well = f"{row}{col}"
                mfd_params = self._compute_microfluidic_params(candidate)

                protocol_lines.extend([
                    f'    # Candidate {candidate["candidate_id"]}',
                    f'    # Particle size target: {mfd_params.target_particle_size_nm:.0f} nm',
                    f'    # TFR: {mfd_params.total_flow_rate_ul_min:.0f} uL/min | FRR: {mfd_params.flow_rate_ratio:.1f}:1',
                    f'    microfluidic_config["{well}"] = {{',
                    f'        "tfr": {mfd_params.total_flow_rate_ul_min:.1f},',
                    f'        "frr": {mfd_params.flow_rate_ratio:.2f},',
                    f'        "aq_flow": {mfd_params.aqueous_flow_rate:.1f},',
                    f'        "org_flow": {mfd_params.organic_flow_rate:.1f},',
                    f'        "target_nm": {mfd_params.target_particle_size_nm:.0f},',
                    f'        "ethanol_pct": {mfd_params.ethanol_pct:.0f},',
                    f'        "temp_c": {mfd_params.temperature_c:.0f},',
                    f'        "channel_um": {mfd_params.channel_diameter_um:.0f},',
                    f'        "mix_cycles": {mfd_params.mixing_cycles},',
                    f'    }}',
                    f'    p300.pick_up_tip()',
                    f'    p300.aspirate({self.config.synthesis_volume_ul}, wash_buffer.wells()["A1"])',
                    f'    p300.dispense({self.config.synthesis_volume_ul}, dest_plate.wells()["{well}"])',
                    f'    p300.drop_tip()',
                ])

                comp = candidate.get("composition", {})
                if comp.get("ionizable_lipid"):
                    protocol_lines.extend([
                        f'    p20.pick_up_tip()',
                        f'    p20.aspirate({comp.get("ionizable_frac", 0.4) * self.config.synthesis_volume_ul:.1f}, '
                        f'source_plate.wells()["A{col}"])',
                        f'    p20.dispense({comp.get("ionizable_frac", 0.4) * self.config.synthesis_volume_ul:.1f}, '
                        f'dest_plate.wells()["{well}"])',
                        f'    p20.drop_tip()',
                    ])

            protocol_lines.extend([
                f'    # Trigger microfluidic mixing for batch',
                f'    protocol.pause("Run microfluidic chip for batch {batch_idx // 96 + 1}")',
                f'    for well in microfluidic_config:',
                f'        cfg = microfluidic_config[well]',
                f'        print(f"Chip: TFR={{cfg[\'tfr\']}} uL/min, FRR={{cfg[\'frr\']}}:1, target={{cfg[\'target_nm\']}} nm")',
                '',
            ])

        protocol_lines.extend([
            '    # ---- FINAL STEPS ----',
            '    protocol.pause("Incubate LNPs at 4C for 30 minutes")',
            '    # Run DLS to verify particle size',
            '    protocol.pause("Measure particle size with DLS")',
            '',
        ])

        protocol_code = "\n".join(protocol_lines)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(protocol_code)
        logger.info("Opentrons protocol generated: %s", output_path)
        return protocol_code

    def generate_microfluidic_config(self, candidates: list[dict], output_path: str) -> dict:
        configs = {}
        for candidate in candidates:
            well_id = f"{candidate['candidate_id']}"
            mfd_params = self._compute_microfluidic_params(candidate)
            configs[well_id] = {
                "candidate_id": candidate["candidate_id"],
                "total_flow_rate_ul_min": mfd_params.total_flow_rate_ul_min,
                "flow_rate_ratio": mfd_params.flow_rate_ratio,
                "aqueous_flow_rate": mfd_params.aqueous_flow_rate,
                "organic_flow_rate": mfd_params.organic_flow_rate,
                "target_particle_size_nm": mfd_params.target_particle_size_nm,
                "ethanol_pct": mfd_params.ethanol_pct,
                "temperature_c": mfd_params.temperature_c,
                "channel_diameter_um": mfd_params.channel_diameter_um,
                "mixing_cycles": mfd_params.mixing_cycles,
            }

        with open(output_path, "w") as f:
            json.dump(configs, f, indent=2)
        logger.info("Microfluidic config generated: %s", output_path)
        return configs

    def generate_plate_map(self, candidates: list[dict], output_path: str) -> dict:
        plate_map = {}
        for i, candidate in enumerate(candidates[:384]):
            row = chr(ord("A") + (i // 12))
            col = i % 12 + 1
            well = f"{row}{col}"
            mfd_params = self._compute_microfluidic_params(candidate)
            plate_map[well] = {
                "candidate_id": candidate["candidate_id"],
                "plate": "destination",
                "volume_ul": self.config.synthesis_volume_ul,
                "target_particle_size_nm": mfd_params.target_particle_size_nm,
            }

        with open(output_path, "w") as f:
            json.dump(plate_map, f, indent=2)
        return plate_map
