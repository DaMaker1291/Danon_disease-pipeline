from pydantic import BaseModel, Field
from typing import List


class DanonConfig(BaseModel):
    target_disease: str = "Danon Disease (Monogenic Hypertrophic Cardiomyopathy)"
    gene_defect: str = "LAMP2 (Xq22)"
    inheritance: str = "X-linked dominant"
    therapeutic_payload: str = "LAMP2B_Transgene"
    vector_backbone: str = "AAV9_Recombinant_Capsid"

    target_tissues: List[str] = Field(
        default=["cardiac_myocytes", "skeletal_myocytes", "vascular_endothelium"],
        description="Primary therapeutic delivery destinations requiring lysosomal pathway correction.",
    )
    avoid_tissues: List[str] = Field(
        default=["hepatic", "renal"],
        description="Off-target clearance sinks requiring high fitness penalties.",
    )

    regulatory_framework: str = "MHRA_ILAP_FastTrack (UK)"
    primary_surrogate_endpoint: str = "Cardiomyocyte_LAMP2_Protein_Expression"
    secondary_clinical_endpoint: str = "Left_Ventricular_Mass_Index_Reduction"

    lamp2b_expression_target: float = 0.70
    max_hepatic_accumulation: float = 0.30
    min_cardiac_tropism: float = 0.50

    # ---- Horizon-2: 24-phase hyper-dimensional pipeline controls ----
    total_pipeline_phases: int = 24
    max_solvation_delta_g: float = Field(
        default=2.5, description="Maximum allowable ΔΔG_solv destabilization (kcal/mol)."
    )
    min_codon_elongation_index: float = Field(
        default=0.88, description="Minimum tRNA adaptation index across the LAMP2B transcript."
    )
    hla_binding_cutoff_nm: float = Field(
        default=500.0, description="IC50 threshold below which an MHC-II peptide is rejected."
    )
    synthesis_gc_window: List[float] = Field(
        default=[40.0, 65.0], description="Allowable %GC bounds for commercial synthesis."
    )
    dms_lethal_threshold: float = Field(
        default=-0.55, description="DMS fitness below which a substitution is lethal."
    )
    min_smar_at_content: float = Field(
        default=0.65, description="Minimum AT-content for a functional S/MAR insulator flank."
    )
    lamp2b_split_position: int = Field(
        default=200, description="Npu DnaE split-intein position in LAMP2B (aa)."
    )

    api_proxy_port: int = 6006
    backend_server_port: int = 8000
    primary_pdb_template: str = "3J1S"

    aav_total_candidates: int = 1_000_000_000
    lnp_total_candidates: int = 1_000_000_000
    batch_size: int = 10_000
    num_workers: int = 4
    max_seq_len: int = 750

    antibody_panel: List[str] = Field(
        default=[
            "AAV2_Ab4", "AAV2_Ab58", "AAV8_Ab1", "AAV9_Ab3",
            "human_igG_pool", "human_IgM_pool", "anti-AAV9_serum",
        ]
    )

    checkpoint_dir: str = "./checkpoints_danon"
    diagnostics_dir: str = "./diagnostics_danon"
    output_dir: str = "./lab_output_danon"
    log_level: str = "INFO"
    random_seed: int = 42


danon_config = DanonConfig()
