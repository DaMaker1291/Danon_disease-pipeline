import { useArtifact, artifactStamp } from '../utils/artifacts';
import type {
  EvolutionForecastCensus,
  EvolutionForecastCrossValidated,
  EvolutionForecastData,
  EvolutionForecastTask,
} from '../types/artifacts';

/**
 * The one screen in this app whose numbers are measured in real patients.
 *
 * The page is organised around a single question -- is this interval clear of
 * the nothing it has to beat? -- and everything else is subordinate to that.
 * Each task shows one range strip: the cross-validated estimate, its
 * patient-clustered interval as a band, and the null marked on the same axis. The
 * strip is the page's one bold element because it encodes the finding rather
 * than decorating it; a reader who looks at nothing else sees whether the band
 * sits to one side of the marker or across it.
 *
 * Two estimators are shown for each task, never one. The single seeded split
 * came first and is kept because it shows what one split is worth: on task A its
 * interval swallowed the null, and on task B its positive advantage reversed
 * across folds. Dropping either number would hide the fragility the record was
 * built to expose.
 *
 * Nothing is computed here. The record is read, and the artifact's own verdict
 * sentence is rendered verbatim -- a UI that paraphrased these could only soften
 * them.
 */

const NULL_MARK = 0.5;

function num(x: number | boolean | null | undefined, d = 3): string {
  if (x === null || x === undefined || typeof x === 'boolean') return '—';
  if (Number.isNaN(x)) return '—';
  return x.toFixed(d);
}

function pct(x: number | boolean | null | undefined, d = 1): string {
  if (x === null || x === undefined || typeof x === 'boolean') return '—';
  if (Number.isNaN(x)) return '—';
  return `${(x * 100).toFixed(d)}%`;
}

/**
 * The interval against the null, on one axis.
 *
 * `low`/`high` are the patient-level interval; `point` is the estimate. The
 * marker is where the estimate would have to be for the claim to be nothing (0.5
 * for an AUC, 0 for a difference). When the band crosses the marker the model is
 * not shown to beat its baseline, and the band is drawn as such -- the styling
 * follows the numbers, never the other way round.
 */
function IntervalStrip({
  point,
  low,
  high,
  nullAt,
  floor,
  ceiling,
  resolved,
  caption,
}: {
  point: number;
  low: number;
  high: number;
  nullAt: number;
  floor: number;
  ceiling: number;
  resolved: boolean;
  caption: string;
}) {
  const span = ceiling - floor || 1;
  const at = (v: number) =>
    `${Math.max(0, Math.min(100, ((v - floor) / span) * 100))}%`;
  return (
    <div className="fc-strip">
      <div className="fc-strip-track">
        <span className="fc-strip-null" style={{ left: at(nullAt) }} />
        <span
          className={resolved ? 'fc-strip-band is-resolved' : 'fc-strip-band'}
          style={{ left: at(low), width: `${Math.max(1.2, ((high - low) / span) * 100)}%` }}
        />
        <span className="fc-strip-point" style={{ left: at(point) }} />
      </div>
      <div className="fc-strip-caption">{caption}</div>
    </div>
  );
}

/** One task: the cross-validated estimate first, the single split beside it. */
function TaskBlock({
  title,
  question,
  cv,
  single,
  verdict,
  resolved,
  unit,
}: {
  title: string;
  question: string;
  cv: EvolutionForecastCrossValidated;
  single: EvolutionForecastTask;
  verdict: string;
  resolved: boolean;
  unit: 'auc' | 'mae';
}) {
  const m = cv.metrics;
  const isAuc = unit === 'auc';

  const point = isAuc ? Number(m.oof_auc) : Number(m.oof_mae_advantage);
  const low = isAuc ? Number(m.oof_auc_ci_low) : Number(m.oof_mae_advantage_ci_low);
  const high = isAuc ? Number(m.oof_auc_ci_high) : Number(m.oof_mae_advantage_ci_high);

  const singleResolved = isAuc
    ? typeof single.metrics.auc_fitted_logistic_ci_low === 'number' &&
      single.metrics.auc_fitted_logistic_ci_low > NULL_MARK
    : single.metrics.mae_advantage_resolvable === true;
  const singlePoint = isAuc
    ? Number(single.metrics.auc_fitted_logistic)
    : Number(single.metrics.mae_advantage_over_persistence);
  const singleLow = isAuc
    ? Number(single.metrics.auc_fitted_logistic_ci_low)
    : Number(single.metrics.mae_advantage_ci_low);
  const singleHigh = isAuc
    ? Number(single.metrics.auc_fitted_logistic_ci_high)
    : Number(single.metrics.mae_advantage_ci_high);

  const foldsAbove = isAuc
    ? Number(m.folds_above_chance)
    : Number(m.folds_beating_persistence);
  const foldTotal = Number(m.n_folds_scored);

  const floorAuc = Math.min(0.4, low - 0.05);
  const floor = isAuc ? floorAuc : Math.min(0, low, singleLow) - 0.004;
  const ceiling = isAuc ? Math.max(0.9, high + 0.05) : Math.max(0.02, high, singleHigh) + 0.004;

  return (
    <section className="fc-task glass">
      <h3 className="fc-task-title">{title}</h3>
      <p className="fc-question">{question}</p>

      <IntervalStrip
        point={point}
        low={low}
        high={high}
        nullAt={isAuc ? NULL_MARK : 0}
        floor={floor}
        ceiling={ceiling}
        resolved={resolved}
        caption={
          isAuc
            ? `pooled out-of-fold AUC ${num(point)} over ${Number(m.pooled_rows)} rows · interval [${num(low)}, ${num(high)}], resampled by patient · the marker is chance`
            : `per-patient advantage ${num(point, 5)} · interval [${num(low, 5)}, ${num(high, 5)}] · the marker is no advantage`
        }
      />

      <dl className="fc-figures">
        <div>
          <dt>{isAuc ? 'majority class' : 'no-change persistence'}</dt>
          <dd className="mono">
            {isAuc
              ? pct(m.baseline_majority_accuracy)
              : num(m.persistence_mae_per_patient, 4)}
          </dd>
        </div>
        <div>
          <dt>{isAuc ? 'the model, pooled out-of-fold' : 'the model, per patient'}</dt>
          <dd className="mono">
            {isAuc ? num(m.oof_auc) : num(m.model_mae_per_patient, 4)}
          </dd>
        </div>
        {isAuc && (
          <div>
            <dt>the pooled unit — one score per</dt>
            <dd className="mono">{Number(m.pooled_rows)} rows</dd>
          </div>
        )}
        {isAuc && (
          <div>
            <dt>the patient endpoint, declared separately</dt>
            <dd className="mono">
              {num(m.patient_endpoint_auc)} · [{num(m.patient_endpoint_ci_low)},{' '}
              {num(m.patient_endpoint_ci_high)}]
            </dd>
          </div>
        )}
        <div>
          <dt>
            folds {isAuc ? 'above chance' : 'beating persistence'} of {foldTotal}
          </dt>
          <dd className="mono">
            {foldsAbove}
            <span> / {foldTotal}</span>
          </dd>
        </div>
      </dl>

      <div className={resolved ? 'fc-verdict is-resolved' : 'fc-verdict'}>{verdict}</div>

      {cv.calibration && (
        <div className="fc-calibration">
          <div className="fc-calibration-head">
            Above, whether the ranking separates. Here, whether a probability from this model can
            be read at all — the probability it gave against the rate that actually followed, over
            equal-count bins of the same rows the ranking was measured on.
          </div>
          <table className="fc-reliability">
            <thead>
              <tr>
                <th>rows</th>
                <th>predicted</th>
                <th>observed</th>
                <th>difference</th>
              </tr>
            </thead>
            <tbody>
              {cv.calibration.bins.map((b) => (
                <tr key={b.bin}>
                  <td>{b.n}</td>
                  <td>{num(b.mean_predicted)}</td>
                  <td>{num(b.observed_rate)}</td>
                  <td>{num(b.difference)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="fc-calibration-head">
            Brier {num(cv.calibration.summary.brier_score, 4)} against a constant predictor&rsquo;s{' '}
            {num(cv.calibration.summary.brier_score_constant_prevalence, 4)} — skill{' '}
            {num(cv.calibration.summary.brier_skill_score, 4)} · curve slope{' '}
            {num(cv.calibration.summary.calibration_slope)} (ideal 1) · intercept{' '}
            {num(cv.calibration.summary.calibration_intercept)} (ideal 0)
          </div>
          <p className="fc-calibration-reading">{cv.calibration.reading}</p>
        </div>
      )}

      {cv.clinical_use && (
        <div className="fc-clinical fc-pooling">
          <div className="fc-pooling-head">
            <strong>Clinical report</strong> — the same out-of-fold scores, read as a decision
            instead of a ranking: one call at a threshold declared before this cohort was ever
            read, counted against the rows it finds and the rows it burns. Nothing here is computed
            in the browser; every figure and every sentence below comes from the record.
          </div>
          <p className="fc-calibration-reading">{cv.clinical_use.intended_use}</p>

          <table className="fc-reliability">
            <thead>
              <tr>
                <th>threshold</th>
                <th>sensitivity</th>
                <th>specificity</th>
                <th>PPV</th>
                <th>NPV</th>
                <th>missed</th>
                <th>false alarms</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>{num(cv.clinical_use.operating_point.threshold)}</td>
                <td>{pct(cv.clinical_use.operating_point.sensitivity)}</td>
                <td>{pct(cv.clinical_use.operating_point.specificity)}</td>
                <td>{pct(cv.clinical_use.operating_point.ppv)}</td>
                <td>{pct(cv.clinical_use.operating_point.npv)}</td>
                <td>{cv.clinical_use.operating_point.fn}</td>
                <td>{cv.clinical_use.operating_point.fp}</td>
              </tr>
            </tbody>
          </table>
          <p className="fc-calibration-reading">
            {cv.clinical_use.operating_point.reading}
          </p>

          <div className="fc-pooling-head">
            Is acting on the call better than the two options a clinician already has? Net benefit
            over thresholds, against treating every row and treating none — {cv.clinical_use.decision_curve.axis}.
          </div>
          <table className="fc-reliability">
            <thead>
              <tr>
                <th>threshold</th>
                <th>the model</th>
                <th>treat all</th>
                <th>treat none</th>
                <th>beats both</th>
              </tr>
            </thead>
            <tbody>
              {cv.clinical_use.decision_curve.rows.map((r) => (
                <tr
                  key={r.threshold}
                  style={r.beats_both ? { color: 'var(--success, #7ee787)' } : undefined}
                >
                  <td>{num(r.threshold, 2)}</td>
                  <td className="mono">{num(r.model)}</td>
                  <td className="mono">{num(r.treat_all)}</td>
                  <td className="mono">{num(r.treat_none)}</td>
                  <td>{r.beats_both ? 'yes' : 'no'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="fc-pooling-reading">{cv.clinical_use.decision_curve.reading}</p>

          <div className="fc-pooling-head">On whose authority this is used</div>
          <ul className="fc-notes">
            {cv.clinical_use.conditions.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        </div>
      )}

      {cv.pooling && (
        <div className="fc-pooling">
          <div className="fc-pooling-head">
            What a pooled score is on this page — the unit and the label, counted rather than
            asserted. It is stated here because an earlier version of this estimator got it wrong,
            and the numbers below are what that cost.
          </div>
          <dl className="fc-figures">
            <div>
              <dt>pooled unit</dt>
              <dd className="mono">{cv.pooling.unit}</dd>
            </div>
            <div>
              <dt>pooled rows</dt>
              <dd className="mono">{cv.pooling.n_pooled_rows}</dd>
            </div>
            <div>
              <dt>patients contributing them</dt>
              <dd className="mono">{cv.pooling.n_patients}</dd>
            </div>
            <div>
              <dt>patients holding both labels</dt>
              <dd className="mono">{cv.pooling.mixed_label_patients}</dd>
            </div>
            <div>
              <dt>positives by “last row wins”</dt>
              <dd className="mono">{cv.pooling.patients_positive_last_row}</dd>
            </div>
            <div>
              <dt>positives by “a re-detection anywhere”</dt>
              <dd className="mono">{cv.pooling.patients_positive_any_row}</dd>
            </div>
          </dl>
          <p className="fc-pooling-reading">{cv.pooling.reading}</p>
        </div>
      )}

      <div className="fc-crosscheck">
        <div className="fc-crosscheck-head">
          the same task on one seeded split — kept as the check on what one split is worth
        </div>
        <IntervalStrip
          point={singlePoint}
          low={singleLow}
          high={singleHigh}
          nullAt={isAuc ? NULL_MARK : 0}
          floor={floor}
          ceiling={ceiling}
          resolved={singleResolved}
          caption={
            isAuc
              ? `single split AUC ${num(singlePoint)} · interval [${num(singleLow)}, ${num(singleHigh)}] — it crosses chance`
              : `single split advantage ${num(singlePoint, 5)} · interval [${num(singleLow, 5)}, ${num(singleHigh, 5)}] — it crosses zero`
          }
        />
      </div>

      <p className="fc-meta">
        {cv.n_repeats} repeats of {cv.n_folds}-fold, split by patient · {cv.n_patients} patients,{' '}
        {cv.n_rows} rows, each carrying its own measured label · every patient tested once per
        repeat by a model that never saw it
      </p>
    </section>
  );
}

/** The assumption underneath every pair: counted, with what it cannot prove. */
function PairingBasis({ census }: { census: EvolutionForecastCensus }) {
  const declared = Object.entries(census.declared_types);
  return (
    <section className="fc-task glass">
      <h3 className="fc-task-title">What a “recurrence” is, and what proves it</h3>
      <p className="fc-question">
        Every pair on this page joins a primary specimen to a later draw from the same patient. The
        release declares{' '}
        <strong>
          {census.n_specimen_rows.toLocaleString()} specimen rows, all of them “
          {declared.map(([label]) => label).join('”, “')}”
        </strong>{' '}
        — one declared kind, so its own specimen-type column carries no timepoint at all and cannot
        be the basis for the pairing. The basis is {census.basis}.{' '}
        {census.n_excluded_by_suffix} specimen with an unrecognised suffix is excluded rather than
        guessed into a fold.
      </p>
      <p className="fc-question fc-question-strong">{census.cannot_establish}</p>
      <p className="fc-meta">
        declared kinds: {declared.map(([label, n]) => `${label} ×${n}`).join(', ')} · {' '}
        {census.declared_type_carries_timepoint
          ? 'the release now declares more than one kind, so the pairing must be re-derived from it'
          : 'a single declared kind, counted here rather than assumed away'}
      </p>
    </section>
  );
}

export default function EvolutionForecast() {
  const { artifact, loading, error } = useArtifact<EvolutionForecastData>(
    'evolution_forecast',
    'artifacts',
  );

  if (loading) return <div className="panel glass">loading the forecast record…</div>;
  if (error)
    return (
      <div className="panel glass">
        <strong>The forecast record is absent.</strong>
        <p className="fc-question">
          {error} — rebuild it with <code>_glass_forecast_cohort.py</code> over the cached GLASS
          records. This screen will not draw a forecast it does not have.
        </p>
      </div>
    );
  if (!artifact) return null;

  const d = artifact.data;
  const ex = d.worked_example;

  return (
    <div className="content-grid" style={{ gridTemplateColumns: '1fr' }}>
      <div className="panel glass">
        <div className="panel-title">
          Cancer evolution, measured in real patient specimens
        </div>
        <p className="fc-question">
          Every allele fraction here is computed from reads in the public GLASS release — the tumour
          allele counts beside the reference counts, in a primary tumour and in a later draw from the
          same patient. Two questions are asked of them, each scored against the dumbest honest
          competitor it has to beat, and each reported with the patient-level interval that decides
          whether beating it is resolvable at all.
        </p>
        <dl className="fc-figures fc-figures-cohort">
          <div>
            <dt>mutation rows</dt>
            <dd className="mono">{d.source.records_used.toLocaleString()}</dd>
          </div>
          <div>
            <dt>patients</dt>
            <dd className="mono">{d.source.patients.toLocaleString()}</dd>
          </div>
          <div>
            <dt>specimens</dt>
            <dd className="mono">{d.source.spectra.toLocaleString()}</dd>
          </div>
          <div>
            <dt>primary → later pairs</dt>
            <dd className="mono">{d.source.paired_specimen_pairs.toLocaleString()}</dd>
          </div>
          <div>
            <dt>dropped for missing counts</dt>
            <dd className="mono">{d.source.records_dropped.toLocaleString()}</dd>
          </div>
        </dl>
      </div>

      <TaskBlock
        title="Is a variant measured in the primary seen again later?"
        question="Rows are the variants the PRIMARY specimen itself measured, split into those found again in the later draw and those that are not, with the primary's own reading as the only input — nothing that could carry the future. Variants seen only later are excluded, because a variant is 'gained' precisely by not appearing in the primary, and scoring those rows would let a model read the construction instead of the measurement."
        cv={d.task_a_cross_validated}
        single={d.task_a_detection_at_recurrence}
        verdict={d.verdict.task_a}
        resolved={d.verdict.task_a_resolved}
        unit="auc"
      />

      <TaskBlock
        title="How far does a variant's fraction move between the two draws?"
        question="Rows are the variants found in both specimens, with the primary's own fraction and depth as the only inputs. The competitor is persistence — the fraction does not move — and the comparison is reported per patient as well as per variant, because the many variants of one specimen are not independent draws."
        cv={d.task_b_cross_validated}
        single={d.task_b_allele_fraction_change}
        verdict={d.verdict.task_b}
        resolved={d.verdict.task_b_resolved}
        unit="mae"
      />

      <PairingBasis census={d.timepoint_census} />

      {ex && (
        <div className="panel glass">
          <div className="panel-title">One worked example, checkable by hand</div>
          <p className="fc-question">
            {ex.patient_id} — <span className="mono">{ex.variant}</span>, read at depth{' '}
            {ex.primary_depth.toLocaleString()}. Its fraction in the primary was{' '}
            <span className="mono">{num(ex.primary_af, 4)}</span> and it moved by{' '}
            <span className="mono">{num(ex.delta_af, 4)}</span> to the later draw.
          </p>
          <p className="fc-question">
            Because variant copies can never exceed total copies, that fraction is a provable LOWER
            bound on the share of tumour cells carrying the variant:{' '}
            <span className="mono">
              {num(ex.lower_bound_primary, 4)} → {num(ex.lower_bound_recurrence, 4)}
            </span>
            . No purity and no copy number is needed for a bound, which is why a bound is what this
            repository publishes and never a clone size.
          </p>
        </div>
      )}

      <div className="panel glass">
        <div className="panel-title">What this screen cannot say</div>
        <p className="fc-question">{d.cannot_say}</p>
        <ul className="fc-notes">
          {d.limitations.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
        <div className="fc-meta">{artifactStamp(artifact)}</div>
      </div>
    </div>
  );
}
