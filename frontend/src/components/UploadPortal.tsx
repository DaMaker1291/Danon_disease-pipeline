import React, { useCallback, useState, useRef } from 'react';
import type { AvailabilityCensus } from '../types';
import { censusAvailabilityManifest, looksLikeAvailabilityManifest } from '../utils/availability';

/**
 * Data intake.
 *
 * ONE file family is accepted here, and it is parsed for real: the
 * per-(patient, timepoint) MRI availability manifests such as CFB-GBM's, which
 * are what make the longitudinal imaging stream countable. The rules mirror
 * `danon/availability.py`, which is the authoritative implementation, and the
 * same bytes get the same verdict on both paths.
 *
 * Two things this component used to do and no longer does, both because they
 * reported numbers that nothing measured:
 *
 *   * a FASTQ branch scored each read against a locally constructed
 *     736-adenine string standing in for the AAV9 VP1 template, and published
 *     the result as "template coverage";
 *   * a GenBank branch reported `templateCoverage: 0.9` as a constant.
 *
 * Neither is a measurement, and sequence intake is not this study's input: it
 * belongs to the vector-design pipeline (`danon/vector_design/data_ingress.py`,
 * reachable through the API). A tumour segmentation mask is a different input
 * again, and it goes to the Twin Planner tab, which runs the twin.
 *
 * Anything else is refused with a reason. A fabricated success is worse than a
 * clear rejection: it tells the user their file was understood when it was not.
 */
export default function UploadPortal() {
  const [dragOver, setDragOver] = useState(false);
  const [census, setCensus] = useState<AvailabilityCensus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const processFile = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setCensus(null);
    try {
      const text = await file.text();

      if (!looksLikeAvailabilityManifest(text)) {
        setError(
          `Unrecognised input: "${file.name}".\n\n` +
            'Accepted here is the longitudinal MRI availability manifest: a tab-separated ' +
            'table with the columns of the TCIA/CFB-GBM table (id_patient, temporality, ' +
            'time_diff_t0 (weeks), the sequence flags, mri_machine).\n\n' +
            'A tumour segmentation mask is a different input: drop it on the Twin Planner ' +
            'tab (.nii / .nii.gz), which runs the 365-day twin.\n\n' +
            'Nothing was processed, and no result is reported for this file.',
        );
        return;
      }

      const c = censusAvailabilityManifest(text);
      // danon/availability.py refuses these three cases outright, and this
      // portal must not report a census computed from the rows that happened to
      // parse: a number derived from a subset reads as the file's number.
      const refusal = c.nRows === 0
        ? 'the manifest has no usable rows (header only, or every row was blank)'
        : c.missingColumns.length > 0
          ? `required column(s) missing: ${c.missingColumns.join(', ')} ` +
            `(found: ${c.columns.join(', ')})`
          : c.malformedRows > 0
            ? `${c.malformedRows} row(s) have a non-numeric time_diff_t0 (weeks)`
            : null;
      if (refusal) {
        setError(
          `Refused: ${refusal}.\n\n` +
            'The reference intake (danon/availability.py) refuses the same file, so no census ' +
            'is reported here rather than one computed from the rows that happened to parse.\n\n' +
            'Nothing was processed.',
        );
        return;
      }
      setCensus(c);
    } catch (e) {
      setError('Failed to parse file: ' + (e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) processFile(file);
  }, [processFile]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
    e.target.value = '';
  }, [processFile]);

  const warnings: string[] = [];
  if (census) {
    warnings.push(
      'These are counts of what the manifest records, not a measurement: the images ' +
      'themselves are not downloaded, so no registration or artefact check has been performed',
    );
    if (census.nCrossMachinePairs > 0) {
      warnings.push(
        `${census.nCrossMachinePairs} pair(s) span two scanners, a confound for a ` +
        'delta-volume comparison',
      );
    }
  }

  return (
    <div className="panel">
      <div className="panel-title">Cohort availability intake</div>

      <div
        className={`upload-zone ${dragOver ? 'drag' : ''}`}
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileRef.current?.click()}
      >
        <div className="upload-zone-text">
          {loading ? 'Processing…' : 'Drop an MRI availability manifest (.tsv, .txt)'}
        </div>
        <div className="upload-zone-text" style={{ fontSize: 11 }}>
          or click to browse — a tumour mask goes on the MRI → Plan tab
        </div>
        <input
          ref={fileRef}
          type="file"
          hidden
          accept=".tsv,.txt"
          onChange={handleFileSelect}
        />
      </div>

      {error && (
        <div className="intake-result intake-refused">
          <h3>Not accepted</h3>
          <pre>{error}</pre>
        </div>
      )}

      {census && (
        <div className="intake-result">
          <h3>Availability census</h3>
          <div className="intake-metrics">
            <Metric label="Rows" value={census.nRows} />
            <Metric label="Patients" value={census.nPatients} />
            <Metric label="With >1 timepoint" value={census.nPatientsMultiTimepoint} />
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
          <dl className="intake-facts">
            <div>
              <dt>Timepoints</dt>
              <dd>{Object.entries(census.timepoints).map(([k, v]) => `${k} ${v}`).join(' · ')}</dd>
            </div>
            <div>
              <dt>Scanners</dt>
              <dd>{Object.entries(census.scanners).map(([k, v]) => `${k} ${v}`).join(' · ')}</dd>
            </div>
            <div>
              <dt>Patients per sequence</dt>
              <dd>
                {Object.entries(census.patientsPerSequence).map(([k, v]) => `${k} ${v}`).join(' · ')}
                <span className="intake-note"> (a pair at two qualifying timepoints)</span>
              </dd>
            </div>
          </dl>
          {warnings.map((w, i) => <div key={i} className="intake-warning">⚠ {w}</div>)}
          <div className="intake-note">
            Counted in-browser by the same rules as <code>danon/availability.py</code>, which is the
            authoritative implementation for the pipeline and its pinned tests.
          </div>
        </div>
      )}
    </div>
  );
}

function Metric({ label, value, sub }: { label: string; value: React.ReactNode; sub?: string }) {
  return (
    <div className="intake-metric">
      <div className="intake-metric-label">{label}</div>
      <div className="intake-metric-value">{value}</div>
      {sub && <div className="intake-metric-sub">{sub}</div>}
    </div>
  );
}
