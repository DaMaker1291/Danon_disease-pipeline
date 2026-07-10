import os
from dataclasses import dataclass, field
from typing import Optional
from pydantic import BaseModel


class ComputeConfig(BaseModel):
    backend: str = "ray"  # ray | slurm | kubernetes | aws_batch
    num_workers: int = 256
    gpus_per_worker: int = 1
    cpus_per_worker: int = 8
    memory_per_worker_gb: int = 32
    batch_size: int = 10_000
    max_concurrent: int = 500
    slurm_partition: str = "gpu"
    slurm_account: str = ""
    aws_region: str = "us-east-1"
    aws_job_queue: str = "longevity-gpu"
    aws_job_definition: str = "longevity-worker"
    k8s_namespace: str = "longevity"
    k8s_image: str = "longevity-pipeline:latest"


class GenerationConfig(BaseModel):
    aav_total_candidates: int = 10_000_000_000
    aav_variable_regions: list[int] = field(default_factory=lambda: [
        263, 264, 265, 266, 267, 268, 269, 270, 271, 272,
        273, 274, 275, 276, 277, 278, 279, 280, 281, 282,
        283, 284, 285, 286, 287, 288, 289, 290, 291, 292,
        293, 294, 295, 296, 297, 298, 299, 300, 301, 302,
        303, 304, 305, 306, 307, 308, 309, 310, 311, 312,
        313, 314, 315, 316, 317, 318, 319, 320, 321, 322,
        323, 324, 325, 326, 327, 328, 329, 330, 331, 332,
        333, 334, 335, 336, 337, 338, 339, 340, 341, 342,
        343, 344, 345, 346, 347, 348, 349, 350, 351, 352,
        353, 354, 355, 356, 357, 358, 359, 360, 361, 362,
        363, 364, 365, 366, 367, 368, 369, 370, 371, 372,
        373, 374, 375, 376, 377, 378, 379, 380, 381, 382,
        383, 384, 385, 386, 387, 388, 389, 390, 391, 392,
        393, 394, 395, 396, 397, 398, 399, 400, 401, 402,
        403, 404, 405, 406, 407, 408, 409, 410, 411, 412,
        413, 414, 415, 416, 417, 418, 419, 420, 421, 422,
        423, 424, 425, 426, 427, 428, 429, 430, 431, 432,
        433, 434, 435, 436, 437, 438, 439, 440, 441, 442,
        443, 444, 445, 446, 447, 448, 449, 450, 451, 452,
        453, 454, 455, 456, 457, 458, 459, 460, 461, 462,
        463, 464, 465, 466, 467, 468, 469, 470, 471, 472,
        473, 474, 475, 476, 477, 478, 479, 480, 481, 482,
        483, 484, 485, 486, 487, 488, 489, 490, 491, 492,
        493, 494, 495, 496, 497, 498, 499, 500, 501, 502,
        503, 504, 505, 506, 507, 508, 509, 510, 511, 512,
        513, 514, 515, 516, 517, 518, 519, 520, 521, 522,
        523, 524, 525, 526, 527, 528, 529, 530, 531, 532,
        533, 534, 535, 536, 537, 538, 539, 540, 541, 542,
        543, 544, 545, 546, 547, 548, 549, 550, 551, 552,
        553, 554, 555, 556, 557, 558, 559, 560, 561, 562,
        563, 564, 565, 566, 567, 568, 569, 570, 571, 572,
        573, 574, 575, 576, 577, 578, 579, 580, 581, 582,
        583, 584, 585, 586, 587, 588, 589, 590, 591, 592,
        593, 594, 595, 596, 597, 598, 599, 600, 601, 602,
        603, 604, 605, 606, 607, 608, 609, 610, 611, 612,
        613, 614, 615, 616, 617, 618, 619, 620, 621, 622,
        623, 624, 625, 626, 627, 628, 629, 630, 631, 632,
        633, 634, 635, 636, 637, 638, 639, 640, 641, 642,
        643, 644, 645, 646, 647, 648, 649, 650, 651, 652,
        653, 654, 655, 656, 657, 658, 659, 660, 661, 662,
        663, 664, 665, 666, 667, 668, 669, 670, 671, 672,
        673, 674, 675, 676, 677, 678, 679, 680, 681, 682,
        683, 684, 685, 686, 687, 688, 689, 690, 691, 692,
        693, 694, 695, 696, 697, 698, 699, 700, 701, 702,
        703, 704, 705, 706, 707, 708, 709, 710, 711, 712,
        713, 714, 715, 716, 717, 718, 719, 720, 721, 722,
        723, 724, 725, 726, 727, 728, 729, 730, 731, 732,
    ])
    aav_mutation_rate: float = 0.05
    lnp_total_candidates: int = 10_000_000_000
    lnp_components: dict = field(default_factory=lambda: {
        "ionizable_lipids": ["DLin-MC3-DMA", "SM-102", "ALC-0315", "DODAP", "DLin-DMA"],
        "peg_lipids": ["DMG-PEG2000", "DSPC-PEG2000", "DSPE-PEG2000"],
        "helper_lipids": ["DSPC", "DPPC", "DOPE", "CHOL"],
        "target_pka_range": [6.2, 6.5],
    })
    protein_model: str = "esm2_t33_650M_UR50S"
    chemical_model: str = "chemberta"


class FilterConfig(BaseModel):
    filter1_target: int = 500_000_000  # Immune evasion survivors
    filter2_target: int = 10_000_000   # Tropism survivors
    filter3_target: int = 50_000       # Transduction efficiency survivors
    antibody_panel: list[str] = field(default_factory=lambda: [
        "AAV2_Ab4", "AAV2_Ab58", "AAV8_Ab1", "AAV9_Ab3",
        "human_igG_pool", "human_IgM_pool",
    ])
    target_tissues: list[str] = field(default_factory=lambda: [
        "cardiac", "neuronal", "joint_cartilage", "skeletal_muscle",
    ])
    avoid_tissues: list[str] = field(default_factory=lambda: [
        "hepatic", "renal",
    ])
    min_transduction: float = 0.5
    max_transduction: float = 0.8
    immune_threshold: float = 0.48  # escape score (0-1, higher = better evasion)
    tropism_threshold: float = 0.32


class LabConfig(BaseModel):
    opentrons_host: str = "http://localhost:31950"
    deck_layout: str = "standard_96"
    synthesis_volume_ul: float = 100.0
    barcoding_scheme: str = "dual_index_i7_i5"
    sequencing_platform: str = "illumina_novaseq"
    sequencing_depth: int = 100  # million reads
    output_dir: str = "./lab_output"


class PipelineConfig(BaseModel):
    compute: ComputeConfig = ComputeConfig()
    generation: GenerationConfig = GenerationConfig()
    filters: FilterConfig = FilterConfig()
    lab: LabConfig = LabConfig()
    checkpoint_dir: str = "./checkpoints"
    log_level: str = "INFO"
    random_seed: int = 42
