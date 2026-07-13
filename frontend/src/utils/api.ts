import type {
  ElectrostaticProfile,
  FlowTelemetry,
  ParetoPoint,
  ClinicalOutcome,
  UploadResult,
  PipelineConstraints,
  PipelineResult,
} from '../types';

// Dev: Vite proxies /api to localhost:8000.
// Production (GitHub Pages): call local backend directly.
const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const BASE = import.meta.env.VITE_API_URL || (isLocal ? '/api' : 'http://localhost:8000/api');

export async function runPipeline(constraints: PipelineConstraints): Promise<PipelineResult> {
  const res = await fetch(`${BASE}/run-pipeline`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      max_hepatic_accumulation: constraints.maxHepaticAccumulation,
      min_cardiac_tropism: constraints.minCardiacTropism,
      min_immune_evasion: constraints.minImmuneEvasion,
      lamp2b_expression_target: constraints.lamp2bExpressionTarget,
      candidate_pool: constraints.candidatePool,
      max_mutations_vr_iv: constraints.maxMutationsVrIv,
      max_mutations_vr_viii: constraints.maxMutationsVrViii,
      random_seed: constraints.randomSeed,
    }),
  });
  if (!res.ok) throw new Error(`run-pipeline API error: ${res.status}`);
  return res.json();
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} API error: ${res.status}`);
  return res.json();
}

// Horizon-2 individual phase endpoints (phases 19-24)
export const fetchDmsFitness = (mutations: { position: number; original: string; mutated: string }[]) =>
  postJson('/phase/dms', { mutations });
export const fetchSolvation = (wildType: string, mutant: string, maxDeltaG = 2.5) =>
  postJson('/phase/solvation', { wild_type: wildType, mutant, max_delta_g: maxDeltaG });
export const fetchSmar = (upstream?: string, downstream?: string) =>
  postJson('/phase/smar', { upstream, downstream });
export const fetchCodon = (protein?: string, cds?: string, minTai = 0.88) =>
  postJson('/phase/codon', { protein, cds, min_tai: minTai });
export const fetchHla = (segments: string[], cutoffNm = 500) =>
  postJson('/phase/hla', { segments, cutoff_nm: cutoffNm });
export const fetchSynthesis = (dna: string, gcLow = 40, gcHigh = 65) =>
  postJson('/phase/synthesis', { dna, gc_low: gcLow, gc_high: gcHigh });

export async function fetchElectrostaticProfile(sequence: string, region: string): Promise<ElectrostaticProfile> {
  const res = await fetch(`${BASE}/electrostatics`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sequence, region }),
  });
  if (!res.ok) throw new Error(`electrostatics API error: ${res.status}`);
  return res.json();
}

export async function fetchFlowTelemetry(config: Record<string, number>): Promise<FlowTelemetry> {
  const res = await fetch(`${BASE}/microfluidics`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!res.ok) throw new Error(`microfluidics API error: ${res.status}`);
  return res.json();
}

export async function fetchParetoFront(): Promise<ParetoPoint[]> {
  const res = await fetch(`${BASE}/pareto`);
  if (!res.ok) throw new Error(`pareto API error: ${res.status}`);
  return res.json();
}

export async function fetchClinicalOutcomes(): Promise<ClinicalOutcome[]> {
  const res = await fetch(`${BASE}/clinical`);
  if (!res.ok) throw new Error(`clinical API error: ${res.status}`);
  return res.json();
}

export async function uploadSequencingData(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/upload`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`upload API error: ${res.status}`);
  return res.json();
}
