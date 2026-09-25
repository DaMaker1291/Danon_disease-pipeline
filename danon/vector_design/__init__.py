"""The AAV9-LAMP2B vector-design and fabrication study.

This is a different study from the glioblastoma tumour twin that gives
``danon`` its name: it designs an AAV9 capsid carrying a LAMP2B payload for
Danon disease, and scores a competing lipid-nanoparticle formulation and the
microfluidic rig that would make either.

It lives here, in its own subpackage, because the two studies share a
codebase but not a disease, and because nothing on the tumour-twin path should
import a capsid or an LNP engine. ``danon/__init__.py`` deliberately does not
import this package; import it explicitly:

    from danon.vector_design import DanonAAVGenerator

Nothing in here has been synthesised, expressed, or dosed. Every gate is a
scorer in this repository, not an assay.
"""

from danon.vector_design.aav_generator import (
    WILD_TYPE_AAV9_CAPSID,
    DanonAAVCandidate,
    DanonAAVGenerator,
)
from danon.vector_design.cell_simulator import CellSimulationResult, DanonCellSimulator
from danon.vector_design.codon_elongation import (
    DEFAULT_LAMP2B_PEPTIDE,
    CodonElongationEngine,
    CodonElongationResult,
)
from danon.vector_design.construct_builder import ConstructBuilder
from danon.vector_design.data_ingress import DataIngressEngine, IngressResult
from danon.vector_design.dms_fitness import DMSFitnessLayer, DMSFitnessResult
from danon.vector_design.dual_vector_moi_optimizer import (
    DualVectorMOIOptimizer,
    DualVectorMoiDesign,
)
from danon.vector_design.epitope_masker import ChargeMaskDesign, EpitopeMasker
from danon.vector_design.hla_decoupler import HLADecoupler, HLADecouplerResult
from danon.vector_design.immunosuppression_protocol import (
    ImmunosuppressionAssessment,
    ImmunosuppressionProtocol,
)
from danon.vector_design.lnp_generator import DanonLNPCandidate, DanonLNPGenerator
from danon.vector_design.microfluidics_core import (
    FlowTelemetry,
    MicrofluidicConfig,
    MicrofluidicsCore,
)
from danon.vector_design.mirna_detarget import miRNADetargetEngine
from danon.vector_design.mouse_study_simulator import MouseStudyResult, MouseStudySimulator
from danon.vector_design.nab_assay_simulator import NAbAssayResult, NAbAssaySimulator
from danon.vector_design.opentrons_compiler import (
    OpentronsCompiler,
    OpentronsProtocol,
    RobotType,
)
from danon.vector_design.pareto_optimizer import ParetoOptimizer, ParetoPoint
from danon.vector_design.plasmid_builder import (
    AssembledVector,
    PlasmidAssembler,
    PlasmidAssemblyReport,
)
from danon.vector_design.platform_validator import (
    CLINICAL_ASSUMED_INPUTS,
    PlatformValidator,
    RegulatoryAssessment,
)
from danon.vector_design.promoter_spec import PromoterSpec, PromoterSpecEngine
from danon.vector_design.safety_engine import (
    DanonSafetyEngine,
    DanonSafetyProfile,
    print_global_regulatory_disclaimer,
)
from danon.vector_design.smar_insulator import (
    CpGDepletionReport,
    CpGOptimizationEngine,
    SMARInsulatorEngine,
    calculate_cpg_density,
)
from danon.vector_design.solvation_energy import SolvationEnergyEngine, SolvationResult
from danon.vector_design.stoichiometric_calc import (
    DecoyOptimizationResult,
    StoichiometricCalculator,
)
from danon.vector_design.synthesis_guard import SynthesisGuard, SynthesisResult
from danon.vector_design.translational_readiness import (
    PreclinicalValidationMilestones,
    TranslationalReadinessEngine,
    TranslationalReadinessResult,
)
from danon.vector_design.tropism_filter import DanonTropismFilter
from danon.vector_design.wetlab_lims_tracker import (
    AssayResult,
    ConstructOrder,
    WetlabLIMSTracker,
)

__all__ = [
    # Capsid design
    "DanonAAVGenerator", "DanonAAVCandidate", "WILD_TYPE_AAV9_CAPSID",
    "DanonTropismFilter",
    "EpitopeMasker", "ChargeMaskDesign",
    "DMSFitnessLayer", "DMSFitnessResult",
    "SolvationEnergyEngine", "SolvationResult",
    "DualVectorMOIOptimizer", "DualVectorMoiDesign",
    # Payload and construct
    "PlasmidAssembler", "PlasmidAssemblyReport", "AssembledVector",
    "ConstructBuilder",
    "CodonElongationEngine", "CodonElongationResult", "DEFAULT_LAMP2B_PEPTIDE",
    "PromoterSpecEngine", "PromoterSpec",
    "miRNADetargetEngine",
    "SMARInsulatorEngine", "CpGOptimizationEngine", "CpGDepletionReport",
    "calculate_cpg_density",
    "HLADecoupler", "HLADecouplerResult",
    "SynthesisGuard", "SynthesisResult",
    # Safety, immunology, translation
    "DanonSafetyEngine", "DanonSafetyProfile",
    "print_global_regulatory_disclaimer",
    "NAbAssaySimulator", "NAbAssayResult",
    "ImmunosuppressionProtocol", "ImmunosuppressionAssessment",
    "StoichiometricCalculator", "DecoyOptimizationResult",
    "PlatformValidator", "RegulatoryAssessment", "CLINICAL_ASSUMED_INPUTS",
    "TranslationalReadinessEngine", "TranslationalReadinessResult",
    "PreclinicalValidationMilestones",
    "MouseStudySimulator", "MouseStudyResult",
    "DanonCellSimulator", "CellSimulationResult",
    # Delivery vehicle and fabrication
    "DanonLNPGenerator", "DanonLNPCandidate",
    "MicrofluidicsCore", "FlowTelemetry", "MicrofluidicConfig",
    "ParetoOptimizer", "ParetoPoint",
    "WetlabLIMSTracker", "ConstructOrder", "AssayResult",
    "OpentronsCompiler", "OpentronsProtocol", "RobotType",
    # Input formats
    "DataIngressEngine", "IngressResult",
]
