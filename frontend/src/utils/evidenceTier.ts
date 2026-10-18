/**
 * Display form for a claim's `tier`.
 *
 * `tier` is `danon.cure.EvidenceStatus`: how well the *mechanism* a sentence is
 * about is supported in humans. The artifact ships the raw enum member
 * (`proven_failed`, `emerging_human`), which is fine in a filter dropdown and
 * unreadable in a chip on the landing page.
 *
 * Two rules, both deliberate:
 *
 * 1. **The member names are kept, only made readable.** `danon/claims.py` opens
 *    by describing how "using one vocabulary while citing the other is how a
 *    provenance claim becomes decoration", so inventing friendlier synonyms here
 *    would reintroduce exactly that drift. `proven_failed` stays
 *    "proven failed"; the words are separated, not replaced.
 *
 * 2. **No per-tier definition is shown.** The enum's own comments gloss
 *    `proven_failed` as "failed in adequately powered human trials", which
 *    describes the cure study it was written for. The tumour twin reuses the
 *    member for absent-or-refuted mechanisms (`egfrviii_not_in_open_data` is
 *    `proven_failed` because no open structural-variant call exists, not because
 *    one failed a trial), so quoting that gloss would assert something untrue.
 *    The tooltip therefore names the vocabulary and its source instead of
 *    paraphrasing it.
 *
 * The tier is NOT the audit result. Whether this repository's claim currently
 * passes is `supported`, and the pages show that separately -- a tier chip must
 * never be read as "verified".
 */

/** Colour family for a chip. `real` is the green used for measured data. */
export type TierTone = 'real' | 'neutral' | 'warn';

export interface TierInfo {
  /** Readable form of the enum member, for a chip label. */
  label: string;
  /** Names the vocabulary and where it is defined. Never a paraphrase. */
  source: string;
  tone: TierTone;
}

const SOURCE = 'danon/cure.py::EvidenceStatus';

export const EVIDENCE_TIERS: Record<string, TierInfo> = {
  established: { label: 'established', source: `${SOURCE}.established`, tone: 'real' },
  emerging_human: { label: 'emerging human', source: `${SOURCE}.emerging_human`, tone: 'neutral' },
  animal_only: { label: 'animal-only', source: `${SOURCE}.animal_only`, tone: 'warn' },
  proven_failed: { label: 'proven failed', source: `${SOURCE}.proven_failed`, tone: 'warn' },
  unknown: { label: 'unknown', source: `${SOURCE}.unknown`, tone: 'warn' },
};

/**
 * Readable form of an EvidenceStatus member.
 *
 * An unrecognised member is separated rather than hidden, and is reported as
 * `warn` -- a tier this build does not know must not render as if it were
 * established evidence.
 */
export function tierInfo(tier: string): TierInfo {
  return (
    EVIDENCE_TIERS[tier] ?? {
      label: tier.replace(/_/g, ' '),
      source: `${SOURCE} (member not in this build's vocabulary)`,
      tone: 'warn',
    }
  );
}

/** One sentence naming the vocabulary, for the landing page. */
export const TIER_VOCABULARY_NOTE =
  'Tier names come from danon/cure.py::EvidenceStatus — the vocabulary this ' +
  'repository uses for how well a mechanism is supported in humans. It is not ' +
  'the audit result: whether a claim currently passes is reported separately.';
