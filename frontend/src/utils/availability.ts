/**
 * Census of a longitudinal MRI availability manifest, computed in the browser.
 *
 * MIRROR OF danon/availability.py
 * -------------------------------
 * The deployed app is static, so a manifest dropped into this portal cannot be
 * censused by the Python pipeline. Rather than report an invented result --
 * which is what this portal used to do for every unrecognised file -- the same
 * counting rules are implemented here, and tests/test_availability.py pins the
 * numbers for the committed CFB-GBM manifest, so if either side drifts the
 * suite fails.
 *
 * A pair counts when one patient carries the reference sequence at two
 * timepoints at least MIN_INTERVAL_WEEKS apart. Pairs are split by scanner
 * because a pair spanning two machines is a confound for a delta-volume
 * measurement. Nothing here interprets image content: a flag records that a
 * series exists, not that it is registered or artefact-free.
 */
import type { AvailabilityCensus } from '../types';

export const REFERENCE_SEQUENCE = 't1gd';
export const MIN_INTERVAL_WEEKS = 4;
export const SEQUENCE_COLUMNS = [
  't1gd', 't1eg', 't1tse', 't2star', 't2tse', 'flair', 'adc',
] as const;

const REQUIRED_COLUMNS = ['id_patient', 'temporality'];
const MACHINE_COLUMN = 'mri_machine';

/** True when the header carries the manifest's own columns. */
export function looksLikeAvailabilityManifest(text: string): boolean {
  const first = text.split(/\r?\n/, 1)[0] ?? '';
  if (!first.includes('\t')) return false;
  const cols = first.split('\t').map(c => c.trim().toLowerCase());
  return (
    REQUIRED_COLUMNS.every(c => cols.includes(c)) &&
    cols.some(c => c.startsWith('time_diff'))
  );
}

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const ordered = [...values].sort((a, b) => a - b);
  const mid = ordered.length >> 1;
  return ordered.length % 2 ? ordered[mid] : (ordered[mid - 1] + ordered[mid]) / 2;
}

interface Row {
  patient: string;
  timepoint: string;
  offset: number;
  sequences: Record<string, boolean>;
  machine: string;
}

export function censusAvailabilityManifest(text: string): AvailabilityCensus {
  const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
  const header = (lines[0] ?? '').split('\t').map(c => c.trim());
  const lower = header.map(c => c.toLowerCase());
  const at = (name: string) => lower.indexOf(name);
  const iPatient = at('id_patient');
  const iTimepoint = at('temporality');
  const iOffset = lower.findIndex(c => c.startsWith('time_diff'));
  const iMachine = at(MACHINE_COLUMN);
  const sequenceIndex: Record<string, number> = {};
  for (const col of SEQUENCE_COLUMNS) sequenceIndex[col] = at(col);

  const missingColumns = [...REQUIRED_COLUMNS.filter(c => at(c) === -1)];
  if (iOffset === -1) missingColumns.push('time_diff_t0');

  const rows: Row[] = [];
  let malformed = 0;
  for (const line of lines.slice(1)) {
    const cells = line.split('\t');
    const patient = (cells[iPatient] ?? '').trim();
    if (!patient) continue;
    const offset = Number.parseFloat(cells[iOffset] ?? '');
    if (!Number.isFinite(offset)) {
      malformed += 1;
      continue;
    }
    const sequences: Record<string, boolean> = {};
    for (const col of SEQUENCE_COLUMNS) {
      const idx = sequenceIndex[col];
      sequences[col] = idx >= 0 && (cells[idx] ?? '').trim() === '1';
    }
    rows.push({
      patient,
      timepoint: (cells[iTimepoint] ?? '').trim(),
      offset: Math.trunc(offset),
      sequences,
      machine: iMachine >= 0 ? (cells[iMachine] ?? '').trim() : '',
    });
  }

  const byPatient = new Map<string, Row[]>();
  const timepoints: Record<string, number> = {};
  const scanners: Record<string, number> = {};
  for (const row of rows) {
    const list = byPatient.get(row.patient);
    if (list) list.push(row);
    else byPatient.set(row.patient, [row]);
    timepoints[row.timepoint] = (timepoints[row.timepoint] ?? 0) + 1;
    if (row.machine) scanners[row.machine] = (scanners[row.machine] ?? 0) + 1;
  }

  let nPatientsMultiTimepoint = 0;
  let nUsablePairs = 0;
  let nSameMachinePairs = 0;
  let nCrossMachinePairs = 0;
  const intervals: number[] = [];
  const patientsWithPair = new Set<string>();
  const patientsOnlySame = new Set<string>();
  const patientsCross = new Set<string>();
  const patientsPerSequence: Record<string, number> = {};
  for (const col of SEQUENCE_COLUMNS) patientsPerSequence[col] = 0;

  for (const [patient, list] of byPatient) {
    if (list.length > 1) nPatientsMultiTimepoint += 1;
    const ordered = [...list].sort((a, b) => a.offset - b.offset);
    let hadPair = false;
    let hadCross = false;
    for (let i = 0; i < ordered.length; i += 1) {
      for (let j = i + 1; j < ordered.length; j += 1) {
        const gap = ordered[j].offset - ordered[i].offset;
        if (gap < MIN_INTERVAL_WEEKS) continue;
        for (const col of SEQUENCE_COLUMNS) {
          if (ordered[i].sequences[col] && ordered[j].sequences[col]) {
            patientsPerSequence[col] += 1;
          }
        }
        if (!(ordered[i].sequences[REFERENCE_SEQUENCE] && ordered[j].sequences[REFERENCE_SEQUENCE])) {
          continue;
        }
        hadPair = true;
        nUsablePairs += 1;
        intervals.push(gap);
        const a = ordered[i].machine;
        const b = ordered[j].machine;
        if (a && a === b) nSameMachinePairs += 1;
        else {
          nCrossMachinePairs += 1;
          hadCross = true;
        }
      }
    }
    if (hadPair) {
      patientsWithPair.add(patient);
      if (hadCross) patientsCross.add(patient);
      else patientsOnlySame.add(patient);
    }
  }

  return {
    sourceFormat: 'mri_availability_manifest',
    columns: header,
    nRows: rows.length,
    nPatients: byPatient.size,
    timepoints,
    nPatientsMultiTimepoint,
    referenceSequence: REFERENCE_SEQUENCE,
    minIntervalWeeks: MIN_INTERVAL_WEEKS,
    nUsablePairs,
    nPatientsWithUsablePair: patientsWithPair.size,
    nSameMachinePairs,
    nCrossMachinePairs,
    intervalWeeksMin: intervals.length ? Math.min(...intervals) : null,
    intervalWeeksMedian: median(intervals),
    intervalWeeksMax: intervals.length ? Math.max(...intervals) : null,
    scanners,
    patientsPerSequence,
    missingColumns,
    malformedRows: malformed,
  };
}
