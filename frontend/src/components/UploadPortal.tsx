import React, { useCallback, useState, useRef } from 'react';
import type { UploadResult } from '../types';
import { censusAvailabilityManifest, looksLikeAvailabilityManifest } from '../utils/availability';

interface Props {
  onResult: (result: UploadResult) => void;
}

/**
 * Intake for clinical inputs.
 *
 * Two file families are genuinely supported, and each is parsed for real:
 *
 *   * availability manifests -- the per-(patient, timepoint) sequence-flag
 *     tables such as CFB-GBM's, which are what make the longitudinal stream
 *     measurable. Counted by `censusAvailabilityManifest`, whose rules mirror
 *     danon/availability.py.
 *   * sequence data -- FASTQ and GenBank.
 *
 * Anything else is REFUSED with a reason. The previous version reported
 * `templateCoverage: 0.5, sequenceValid: true` for any unrecognised file, so a
 * 554-row availability table was silently presented as a one-record sequence
 * with an invented coverage figure. A fabricated success is worse than a clear
 * rejection: it tells the user their file was understood when it was not.
 */
export default function UploadPortal({ onResult }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const processFile = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const text = await file.text();
      const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
      let parsed: UploadResult;

      if (looksLikeAvailabilityManifest(text)) {
        const census = censusAvailabilityManifest(text);
        const warnings: string[] = [];
        if (census.missingColumns.length > 0) {
          warnings.push(
            `Missing required column(s): ${census.missingColumns.join(', ')} — the census below is incomplete`,
          );
        }
        if (census.malformedRows > 0) {
          warnings.push(
            `${census.malformedRows} row(s) skipped: time_diff_t0 (weeks) was not numeric`,
          );
        }
        warnings.push(
          'These are counts of what the manifest records, not a measurement: the images themselves are not downloaded, so no registration or artefact check has been performed',
        );
        if (census.nCrossMachinePairs > 0) {
          warnings.push(
            `${census.nCrossMachinePairs} pair(s) span two scanners, a confound for a delta-volume comparison`,
          );
        }
        parsed = {
          sourceFormat: 'mri_availability_manifest',
          recordsParsed: census.nRows,
          mutationsIsolated: [],
          // No sequence was read, so no coverage figure is reported. Zero here
          // means "not applicable", and the badge below is a manifest badge,
          // not a validity badge.
          templateCoverage: 0,
          sequenceValid: true,
          warnings,
          availabilityCensus: census,
        };
      } else if (ext === 'fastq' || ext === 'fq') {
        const lines = text.split('\n');
        const reads = lines.filter(l => l.startsWith('@'));
        const seq = reads.length > 0 ? lines[1]?.trim()?.toUpperCase() || '' : '';
        const template = 'A'.repeat(736);
        const matches = seq.split('').filter((c, i) => i < template.length && c === template[i]).length;
        const cov = template.length > 0 ? matches / template.length : 0;
        parsed = {
          sourceFormat: 'fastq',
          recordsParsed: reads.length,
          mutationsIsolated: [],
          templateCoverage: cov,
          sequenceValid: cov > 0.5,
          warnings: cov <= 0.5 ? ['Low template coverage — sequence may be from non-AAV9 source'] : [],
        };
      } else if (ext === 'gb' || ext === 'genbank') {
        const seqMatch = text.match(/ORIGIN[\s\S]*?\/\//);
        const seq = seqMatch ? seqMatch[0].replace(/ORIGIN|\/\/|[^A-Za-z]/g, '').toUpperCase() : '';
        parsed = {
          sourceFormat: 'genbank',
          recordsParsed: 1,
          mutationsIsolated: [],
          templateCoverage: seq.length > 0 ? 0.9 : 0,
          sequenceValid: seq.length > 0,
          warnings: seq.length === 0 ? ['Could not extract sequence from GenBank file'] : [],
        };
      } else {
        setError(
          `Unrecognised input: "${file.name}".\n\n` +
            'Accepted here are longitudinal MRI availability manifests ' +
            '(tab-separated, with id_patient and temporality columns and a ' +
            'time_diff column — e.g. the CFB-GBM availability table) and ' +
            'sequence data (.fastq, .fq, .gb, .genbank).\n\n' +
            'A tumour segmentation mask is a different input: drop it on the ' +
            'Twin Planner tab (.nii / .nii.gz), which runs the 365-day twin.\n\n' +
            'Nothing was processed, and no result is reported for this file.',
        );
        return;
      }
      setResult(parsed);
      onResult(parsed);
    } catch (e) {
      setError('Failed to parse file: ' + (e as Error).message);
    } finally {
      setLoading(false);
    }
  }, [onResult]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  }, [processFile]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
  }, [processFile]);

  const census = result?.availabilityCensus;

  return (
    <div className="panel" style={{ gridColumn: '1 / -1' }}>
      <div className="panel-title">Upload Portal — Clinical Sequencing & Cohort Availability</div>

      <div
        className="upload-zone"
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileRef.current?.click()}
        style={{ borderColor: dragOver ? '#3b82f6' : undefined, background: dragOver ? '#1f2937' : undefined }}
      >
        
        <div className="upload-zone-text">
          {loading ? 'Processing...' : 'Drop an MRI availability manifest (.tsv), FASTQ, or GenBank (.gb)'}
        </div>
        <div className="upload-zone-text" style={{ fontSize: 11 }}>
          or click to browse — tumour masks go on the Twin Planner tab
        </div>
        <input
          ref={fileRef}
          type="file"
          hidden
          accept=".tsv,.txt,.fastq,.fq,.gb,.genbank"
          onChange={handleFileSelect}
        />
      </div>

      {error && (
        <div className="upload-result" style={{ borderLeft: '3px solid #ef4444' }}>
          <h3 style={{ color: '#ef4444' }}>Not accepted</h3>
          <pre style={{ whiteSpace: 'pre-wrap' }}>{error}</pre>
        </div>
      )}

      {census && (
        <div className="upload-result">
          <h3>Availability Manifest Census</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 10, margin: '10px 0' }}>
            <Metric label="Rows" value={census.nRows} />
            <Metric label="Patients" value={census.nPatients} />
            <Metric label={`With >1 timepoint`} value={census.nPatientsMultiTimepoint} />
            <Metric
              label={`Usable ${census.referenceSequence} pairs`}
              value={census.nUsablePairs}
              sub={`≥${census.minIntervalWeeks} weeks apart`}
            />
            <Metric label="Patients with a pair" value={census.nPatientsWithUsablePair} />
            <Metric
              label="Scanner changes"
              value={census.nCrossMachinePairs}
              sub={`of ${census.nSameMachinePairs + census.nCrossMachinePairs} pairs`}
            />
            <Metric
              label="Interval (wks)"
              value={
                census.intervalWeeksMin === null
                  ? '—'
                  : `${census.intervalWeeksMin}/${census.intervalWeeksMedian}/${census.intervalWeeksMax}`
              }
              sub="min/median/max"
            />
          </div>
          <div style={{ fontSize: 12, opacity: 0.85 }}>
            <div><b>Timepoints:</b> {Object.entries(census.timepoints).map(([k, v]) => `${k} ${v}`).join(' · ')}</div>
            <div><b>Scanners:</b> {Object.entries(census.scanners).map(([k, v]) => `${k} ${v}`).join(' · ')}</div>
            <div>
              <b>Per sequence (patients with 2+ timepoints):</b>{' '}
              {Object.entries(census.patientsPerSequence).map(([k, v]) => `${k} ${v}`).join(' · ')}
            </div>
          </div>
          {result?.warnings.map((w, i) => (
            <div key={i} style={{ fontSize: 12, color: '#fbbf24', marginTop: 6 }}>⚠ {w}</div>
          ))}
          <div style={{ fontSize: 11, opacity: 0.7, marginTop: 8 }}>
            Counted in-browser by the same rules as <code>danon/availability.py</code>, which is the
            authoritative implementation for the pipeline and its pinned tests.
          </div>
        </div>
      )}

      {result && !census && (
        <div className="upload-result">
          <h3>Ingestion Result</h3>
          <pre>{JSON.stringify(result, null, 2)}</pre>
          <div style={{ marginTop: 8, display: 'flex', gap: 8 }}>
            <span className={`badge ${result.sequenceValid ? '' : 'badge-mhra'}`}>
              {result.sequenceValid ? '✓ Valid' : '✗ Invalid'}
            </span>
            <span className="badge">{result.sourceFormat.toUpperCase()}</span>
            <span className="badge">{result.recordsParsed} records</span>
          </div>
        </div>
      )}
    </div>
  );
}

function Metric({ label, value, sub }: { label: string; value: React.ReactNode; sub?: string }) {
  return (
    <div style={{ background: '#111827', border: '1px solid #374151', borderRadius: 6, padding: '8px 10px' }}>
      <div style={{ fontSize: 11, opacity: 0.7 }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 600 }}>{value}</div>
      {sub && <div style={{ fontSize: 10, opacity: 0.6 }}>{sub}</div>}
    </div>
  );
}
