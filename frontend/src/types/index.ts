export interface ElectrostaticProfile {
  region: string;
  residueProfiles: ResidueProfile[];
  netChargeChange: number;
  surfacePotentialBefore: number;
  surfacePotentialAfter: number;
  antibodyDisruptionScore: number;
}

export interface ResidueProfile {
  position: number;
  aa: string;
  charge: number;
  surfaceAccessibility: number;
  localElectrostaticField: number;
  bornSolvationEnergy: number;
  poissonBoltzmannPotential: number;
  isEpitope: boolean;
  epitopeSite: string | null;
  isDockingResidue: boolean;
}

export interface FlowTelemetry {
  reynoldsNumberAqueous: number;
  reynoldsNumberOrganic: number;
  reynoldsNumberMixed: number;
  wallShearStressPa: number;
  maxShearStressPa: number;
  mixingEfficiency: number;
  pecletNumber: number;
  pressureDropPa: number;
  flowRegime: string;
  shearStressSafe: boolean;
  frr: number;
}

export interface ParetoPoint {
  candidateId: number;
  cardiacTropism: number;
  hepaticAvoidance: number;
  immuneEvasion: number;
  lamp2bExpression: number;
  promoterScore: number;
  mirnaScore: number;
  paretoRank: number;
  crowdingDistance: number;
}

export interface ClinicalOutcome {
  mutationType: string;
  months: number[];
  lvmiPredicted: number[];
  survivalProbability: number[];
  survivalLower: number[];
  survivalUpper: number[];
}

export interface UploadResult {
  sourceFormat: string;
  recordsParsed: number;
  mutationsIsolated: {
    positionVp1: number;
    originalAa: string;
    mutatedAa: string;
    region: string;
    confidence: number;
  }[];
  templateCoverage: number;
  sequenceValid: boolean;
  warnings: string[];
}

export interface PipelineState {
  candidates: ParetoPoint[];
  flowTelemetry: FlowTelemetry | null;
  electrostatics: ElectrostaticProfile | null;
  clinicalOutcomes: ClinicalOutcome[];
  uploadResult: UploadResult | null;
  isRunning: boolean;
}

export interface PipelineConstraints {
  maxHepaticAccumulation: number;
  minCardiacTropism: number;
  minImmuneEvasion: number;
  lamp2bExpressionTarget: number;
  candidatePool: number;
  maxMutationsVrIv: number;
  maxMutationsVrViii: number;
  randomSeed: number;
}

export interface Mutation {
  position: number;
  positionVp1: number;
  original: string;
  mutated: string;
  region: string;
  chargeBefore: number;
  chargeAfter: number;
  chargeReversal: boolean;
  isEpitope: boolean;
  isDocking: boolean;
}

export interface RegionResidue {
  position: number;
  positionVp1: number;
  aa: string;
  charge: number;
  surfaceAccessibility: number;
  localElectrostaticField: number;
  bornSolvationEnergy: number;
  poissonBoltzmannPotential: number;
  isEpitope: boolean;
  epitopeSite: string | null;
  isDockingResidue: boolean;
}

export interface RegionProfile {
  region: string;
  residueProfiles: RegionResidue[];
  netCharge: number;
  surfacePotential: number;
  antibodyDisruptionScore: number;
}

export interface PipelineScores {
  cardiacTropism: number;
  hepaticAvoidance: number;
  immuneEvasion: number;
  lamp2bExpression: number;
  structural: number;
  overall: number;
  maskScoreVrIv: number;
  maskScoreVrViii: number;
  chargeReversalRatio: number;
  improvementVsUcl: number;
  dockingPreserved: boolean;
}

export interface Phase {
  id: number;
  name: string;
  category: string;
  score: number;
  status: 'pass' | 'warn' | 'fail';
  metric: string;
  selectivityFactor: number;
  horizon: number;
}

export interface AdvancedMetrics {
  dms: {
    capsidViable: boolean;
    minDmsFitness: number;
    meanDmsFitness: number;
    conservedPocketViolations: number;
    lethalMutations: number;
    boundaryMargin: number;
    mutationScores: {
      position: number; original_aa: string; mutated_aa: string;
      blosum_score: number; conservation_weight: number; in_conserved_pocket: boolean;
      dms_fitness: number; lethal: boolean;
    }[];
  };
  solvation: {
    ddgSolvTotal: number; plasmaSoluble: boolean; aggregationRisk: number; maxAllowedDdg: number;
    regions: { region: string; ddgSolv: number; dgWildType: number; dgMutant: number }[];
  };
  smar: {
    cpgDensityRaw: number; cpgDensityDepleted: number; cpgReductionPct: number;
    cpgWithinThreshold: boolean; primaryStrategy: string;
    secondarySmAR: { combinedAtContent: number; meanStrength: number; shieldingPredicted: boolean };
  };
  vectorCapacity: {
    strategy: string; cargoLengthBp: number; toxicityRiskMultiplier: number; clinicalJustification: string;
  };
  codon: {
    tai: number; minWindowTai: number; nCodons: number; stallSites: number;
    codonOptimized: boolean; threshold: number; meanElongationRate: number;
  };
  hla: {
    chosenSplitPosition: number; strongestIc50Nm: number; highAffinityHits: number;
    decoupled: boolean; cutoffNm: number; peptidesEvaluated: number;
    binders: { segment: string; core_9mer: string; binding_score: number; predicted_ic50_nm: number; immunogenic: boolean }[];
  };
  synthesis: {
    lengthBp: number; gcContent: number; gcMinWindow: number; gcMaxWindow: number;
    outOfBoundsWindows: number; homopolymerRuns: number; invertedRepeats: number;
    directRepeats: number; synthesizable: boolean; hardFailures: number; gcBounds: number[];
  };
}

export interface ReadinessGate {
  id: number;
  tier: string;
  name: string;
  description: string;
  verified: boolean;
  tokenField: string;
  pendingLabel: string;
}

export interface TranslationalReadiness {
  currentStage: string;
  clinicalTrialEligibility: boolean;
  translationalCompletion: number;
  readinessScorePct: number;
  gates: ReadinessGate[];
  requiredNextStep: string;
  verifiedCount: number;
  totalGates: number;
}

export interface PipelineResult {
  sequence: string;
  wildTypeSequence: string;
  vpOffset: number;
  candidateId: number;
  mutations: Mutation[];
  regions: { VR_IV: RegionProfile; VR_VIII: RegionProfile };
  regionResidueRanges: {
    VR_IV: { start: number; end: number; startVp1: number; endVp1: number };
    VR_VIII: { start: number; end: number; startVp1: number; endVp1: number };
  };
  scores: PipelineScores;
  paretoFront: ParetoPoint[];
  spikeCount: number;
  poolEvaluated: number;
  phases: Phase[];
  advancedMetrics: AdvancedMetrics;
  combinatorialAdvantage: number;
  phasesPassed: number;
  totalPhases: number;
  translationalReadiness: TranslationalReadiness;
}
