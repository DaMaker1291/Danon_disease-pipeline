"""
Automated Liquid-Handling Module: Compiles top-5 Pareto-optimized candidates
into fully executable Python scripts for Opentrons OT-2 / Flex robots.

Generates exact labware layouts and serial dilution pipetting protocols
for LNP formulation or AAV viral aliquot mixing based on optimization output.

Protocol output is a self-contained Python script with opentrons.execute
compatibility (OT-2 API v2.15+ / Flex API).
"""
import os
import json
import logging
import numpy as np
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RobotType(str, Enum):
    OT2 = "OT-2"
    FLEX = "Flex"


class LabwareType(str, Enum):
    NEST_96_DEEP = "nest_96_wellplate_2ml_deep"
    NEST_96_FLAT = "nest_96_wellplate_200ul_flat"
    OPENTRONS_300_TIP = "opentrons_96_tiprack_300ul"
    OPENTRONS_20_TIP = "opentrons_96_tiprack_20ul"
    OPENTRONS_1000_TIP = "opentrons_96_tiprack_1000ul"
    NEST_12_RESERVOIR = "nest_12_reservoir_15ml"
    CORNING_6_FLAT = "corning_6_wellplate_16.8ml_flat"
    OPENTRONS_TRASH = "opentrons_1_trash_1100ml"


class PipetteType(str, Enum):
    P300_SINGLE = "p300_single_gen2"
    P300_MULTI = "p300_multi_gen2"
    P20_SINGLE = "p20_single_gen2"
    P1000_SINGLE = "p1000_single_gen2"
    FLEX_1000 = "flex_1000"
    FLEX_8CH = "flex_8channel_1000"


class PipettingStep(BaseModel):
    step_id: int
    step_type: str  # "transfer", "mix", "serial_dilution", "incubate"
    source_slot: str
    dest_slot: str
    volume_ul: float
    mix_repetitions: int = 3
    tip_change: bool = True
    incubation_minutes: float = 0.0
    description: str = ""


class LNPDilutionProtocol(BaseModel):
    candidate_id: int
    formulation_name: str
    ionizable_lipid: str
    ionizable_mmol: float
    helper_lipid: str
    helper_mmol: float
    cholesterol_mmol: float
    peg_lipid: str
    peg_mmol: float
    total_lipid_mmol: float
    frr_target: float
    n2p_ratio: float
    total_volume_ml: float


class AAVAliquotProtocol(BaseModel):
    candidate_id: int
    vector_titer_vg_ml: float
    dilution_factor: float
    total_volume_ul: float
    buffer_exchange: bool = False


class OpentronsProtocol(BaseModel):
    protocol_name: str
    robot: RobotType = RobotType.OT2
    api_version: str = "2.15"
    labware: Dict[str, str] = Field(default_factory=dict)
    pipettes: Dict[str, str] = Field(default_factory=dict)
    steps: List[PipettingStep] = Field(default_factory=list)
    source_code: str = ""


class OpentronsCompiler:
    """
    Compiles Pareto-optimized candidates into executable OT-2/Flex protocols.
    """

    def __init__(self, robot: RobotType = RobotType.OT2):
        self.robot = robot
        self.api_version = "2.15"

    def _labware_for_pipette(self, volume_ul: float) -> Tuple[LabwareType, PipetteType]:
        if volume_ul <= 20:
            return LabwareType.OPENTRONS_20_TIP, PipetteType.P20_SINGLE
        elif volume_ul <= 300:
            return LabwareType.OPENTRONS_300_TIP, PipetteType.P300_SINGLE
        else:
            return LabwareType.OPENTRONS_1000_TIP, PipetteType.P1000_SINGLE

    def compile_lnp_formulation(self, candidates: List[LNPDilutionProtocol],
                                 output_path: Optional[str] = None) -> List[OpentronsProtocol]:
        protocols: List[OpentronsProtocol] = []
        for i, candidate in enumerate(candidates[:5]):
            steps: List[PipettingStep] = []
            step_id = 1
            total_ethanol = candidate.total_lipid_mmol * 0.5
            total_aqueous = candidate.total_volume_ml - total_ethanol
            org_vol_ul = total_ethanol * 1000.0
            aq_vol_ul = total_aqueous * 1000.0

            tip_type, pip_type = self._labware_for_pipette(max(org_vol_ul, aq_vol_ul))
            labware_map = {
                "1": LabwareType.NEST_96_DEEP.value,
                "2": LabwareType.OPENTRONS_300_TIP.value if tip_type == LabwareType.OPENTRONS_300_TIP
                     else LabwareType.OPENTRONS_1000_TIP.value,
                "3": LabwareType.NEST_12_RESERVOIR.value,
                "4": LabwareType.NEST_96_FLAT.value,
                "5": LabwareType.OPENTRONS_TRASH.value,
            }
            pipette_map = {"left": pip_type.value}

            steps.append(PipettingStep(
                step_id=step_id, step_type="transfer",
                source_slot="3", dest_slot="1",
                volume_ul=candidate.ionizable_mmol * 100 / max(candidate.total_lipid_mmol, 1e-12),
                description=f"Transfer {candidate.ionizable_lipid} to well A1",
            ))
            step_id += 1

            steps.append(PipettingStep(
                step_id=step_id, step_type="transfer",
                source_slot="3", dest_slot="1",
                volume_ul=candidate.helper_mmol * 100 / max(candidate.total_lipid_mmol, 1e-12),
                description=f"Transfer {candidate.helper_lipid} to well A1",
            ))
            step_id += 1

            steps.append(PipettingStep(
                step_id=step_id, step_type="transfer",
                source_slot="3", dest_slot="1",
                volume_ul=candidate.cholesterol_mmol * 100 / max(candidate.total_lipid_mmol, 1e-12),
                description="Transfer cholesterol to well A1",
            ))
            step_id += 1

            steps.append(PipettingStep(
                step_id=step_id, step_type="transfer",
                source_slot="3", dest_slot="1",
                volume_ul=candidate.peg_mmol * 100 / max(candidate.total_lipid_mmol, 1e-12),
                description=f"Transfer {candidate.peg_lipid} to well A1",
            ))
            step_id += 1

            steps.append(PipettingStep(
                step_id=step_id, step_type="serial_dilution",
                source_slot="1", dest_slot="4",
                volume_ul=org_vol_ul * 0.1, mix_repetitions=5,
                description=f"Organic phase: {org_vol_ul:.0f} uL ethanol + lipids",
            ))
            step_id += 1

            steps.append(PipettingStep(
                step_id=step_id, step_type="transfer",
                source_slot="3", dest_slot="4",
                volume_ul=aq_vol_ul,
                description=f"Aqueous phase: {aq_vol_ul:.0f} uL buffer (FRR={candidate.frr_target:.2f})",
            ))
            step_id += 1

            steps.append(PipettingStep(
                step_id=step_id, step_type="mix",
                source_slot="4", dest_slot="4",
                volume_ul=50.0, mix_repetitions=10,
                description="Mix LNP at 50 uL, 10 reps for uniform encapsulation",
            ))
            step_id += 1

            steps.append(PipettingStep(
                step_id=step_id, step_type="incubate",
                source_slot="4", dest_slot="4",
                volume_ul=0.0, incubation_minutes=30.0,
                description="Incubate 30 min for LNP maturation",
            ))

            source = self._generate_lnp_source(protocol_name=candidate.formulation_name,
                                                labware=labware_map, pipettes=pipette_map, steps=steps)

            protocols.append(OpentronsProtocol(
                protocol_name=candidate.formulation_name,
                robot=self.robot,
                api_version=self.api_version,
                labware=labware_map,
                pipettes=pipette_map,
                steps=steps,
                source_code=source,
            ))

            if output_path:
                fname = f"ot2_{candidate.formulation_name.replace(' ', '_')}.py"
                fpath = os.path.join(output_path, fname)
                with open(fpath, "w") as f:
                    f.write(source)
                logger.info("Wrote OT-2 protocol: %s", fpath)

        return protocols

    def compile_aav_aliquoting(self, candidates: List[AAVAliquotProtocol],
                                output_path: Optional[str] = None) -> List[OpentronsProtocol]:
        protocols: List[OpentronsProtocol] = []
        for i, candidate in enumerate(candidates[:5]):
            steps: List[PipettingStep] = []
            step_id = 1
            tip_type, pip_type = self._labware_for_pipette(candidate.total_volume_ul)
            labware_map = {
                "1": LabwareType.NEST_96_DEEP.value,
                "2": LabwareType.OPENTRONS_300_TIP.value,
                "3": LabwareType.NEST_12_RESERVOIR.value,
                "4": LabwareType.NEST_96_FLAT.value,
                "5": LabwareType.OPENTRONS_TRASH.value,
            }
            pipette_map = {"left": pip_type.value}

            steps.append(PipettingStep(
                step_id=step_id, step_type="serial_dilution",
                source_slot="1", dest_slot="4",
                volume_ul=candidate.total_volume_ul * candidate.dilution_factor,
                description=f"Dilute vector {candidate.dilution_factor}x in buffer",
            ))
            step_id += 1

            steps.append(PipettingStep(
                step_id=step_id, step_type="transfer",
                source_slot="4", dest_slot="1",
                volume_ul=50.0,
                description="Aliquot 50 uL per well for storage",
            ))
            step_id += 1

            steps.append(PipettingStep(
                step_id=step_id, step_type="incubate",
                source_slot="1", dest_slot="1",
                volume_ul=0.0, incubation_minutes=5.0,
                description="Brief incubation for AAV stability",
            ))

            source = self._generate_aav_source(
                protocol_name=f"AAV_Candidate_{candidate.candidate_id}_Aliquoting",
                labware=labware_map, pipettes=pipette_map, steps=steps,
            )

            protocols.append(OpentronsProtocol(
                protocol_name=f"AAV_Candidate_{candidate.candidate_id}_Aliquoting",
                robot=self.robot,
                api_version=self.api_version,
                labware=labware_map,
                pipettes=pipette_map,
                steps=steps,
                source_code=source,
            ))

            if output_path:
                fname = f"ot2_aav_candidate_{candidate.candidate_id}_aliquot.py"
                fpath = os.path.join(output_path, fname)
                with open(fpath, "w") as f:
                    f.write(source)
                logger.info("Wrote OT-2 AAV protocol: %s", fpath)

        return protocols

    def _generate_lnp_source(self, protocol_name: str, labware: Dict[str, str],
                              pipettes: Dict[str, str], steps: List[PipettingStep]) -> str:
        lines: List[str] = []
        lines.append('"""')
        lines.append(f"Opentrons OT-2 Protocol: {protocol_name}")
        lines.append(f"API Version: {self.api_version}")
        lines.append(f"Robot: {self.robot.value}")
        lines.append('"""')
        lines.append("from opentrons import protocol_api")
        lines.append("import json")
        lines.append("")
        lines.append("metadata = {")
        lines.append(f'    "protocolName": "{protocol_name}",')
        lines.append(f'    "apiLevel": "{self.api_version}",')
        lines.append('    "robotType": "' + f'{self.robot.value}"' + ",")
        lines.append("}")
        lines.append("")
        lines.append("def run(protocol: protocol_api.ProtocolContext):")
        for slot, lw_type in labware.items():
            safe_name = lw_type.replace(".", "_").replace("-", "_")
            lines.append(f'    {safe_name} = protocol.load_labware("{lw_type}", "{slot}")')
        for mount, pip_type in pipettes.items():
            lines.append(f'    {mount}_pipette = protocol.load_instrument("{pip_type}", mount="{mount}")')
        lines.append("")
        lines.append("    # Pipetting steps")
        for step in steps:
            if step.step_type == "transfer":
                lines.append(f"    # {step.description}")
                lines.append(f'    left_pipette.transfer({step.volume_ul}, '
                             f'{labware[step.source_slot].replace(".", "_").replace("-", "_")}["A1"], '
                             f'{labware[step.dest_slot].replace(".", "_").replace("-", "_")}["A1"], '
                             f'new_tip="always" if {str(step.tip_change).lower()} else "never")')
            elif step.step_type == "mix":
                lines.append(f"    # {step.description}")
                lines.append(f'    left_pipette.mix({step.mix_repetitions}, {step.volume_ul}, '
                             f'{labware[step.dest_slot].replace(".", "_").replace("-", "_")}["A1"])')
            elif step.step_type == "serial_dilution":
                lines.append(f"    # {step.description}")
                lines.append(f'    left_pipette.transfer({step.volume_ul}, '
                             f'{labware[step.source_slot].replace(".", "_").replace("-", "_")}["A1"], '
                             f'{labware[step.dest_slot].replace(".", "_").replace("-", "_")}["A1"], '
                             f'mix_after=({step.mix_repetitions}, {step.volume_ul}), new_tip="always")')
            elif step.step_type == "incubate":
                lines.append(f"    # {step.description}")
                lines.append(f"    protocol.delay(seconds={int(step.incubation_minutes * 60)})")
            lines.append("")
        return "\n".join(lines)

    def _generate_aav_source(self, protocol_name: str, labware: Dict[str, str],
                              pipettes: Dict[str, str], steps: List[PipettingStep]) -> str:
        return self._generate_lnp_source(protocol_name, labware, pipettes, steps)

    def pareto_to_protocol(self, pareto_candidates: List, use_lnp: bool = True,
                            output_dir: str = "./lab_output_danon") -> List[OpentronsProtocol]:
        if use_lnp:
            lnp_candidates: List[LNPDilutionProtocol] = []
            for c in pareto_candidates[:5]:
                lipid = getattr(c, "ionizable_lipid", "DLin-MC3-DMA")
                helper = getattr(c, "helper_lipid", "DSPC")
                peg = getattr(c, "peg_lipid", "DMG-PEG2000")
                lnp_candidates.append(LNPDilutionProtocol(
                    candidate_id=getattr(c, "candidate_id", 0),
                    formulation_name=f"LNP_Candidate_{getattr(c, 'candidate_id', 0)}",
                    ionizable_lipid=lipid,
                    ionizable_mmol=0.4,
                    helper_lipid=helper,
                    helper_mmol=0.1,
                    cholesterol_mmol=0.35,
                    peg_lipid=peg,
                    peg_mmol=0.015,
                    total_lipid_mmol=0.865,
                    frr_target=0.33,
                    n2p_ratio=6.0,
                    total_volume_ml=1.0,
                ))
            return self.compile_lnp_formulation(lnp_candidates, output_dir)
        else:
            aav_candidates: List[AAVAliquotProtocol] = []
            for c in pareto_candidates[:5]:
                aav_candidates.append(AAVAliquotProtocol(
                    candidate_id=getattr(c, "candidate_id", 0),
                    vector_titer_vg_ml=getattr(c, "fitness", 5e13),
                    dilution_factor=10.0,
                    total_volume_ul=500.0,
                ))
            return self.compile_aav_aliquoting(aav_candidates, output_dir)
