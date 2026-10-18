"""Fetch the REAL paired GLASS release and measure its allele-fraction spectra.

WHAT THIS DOES
--------------
Reads the public processed GLASS release (`difg_glass`) on cBioPortal through its
open API, for a DECLARED gene panel and the sequenced sample list, and caches the
raw mutation records. Nothing here is simulated: every number written to
`outputs_real/glass_variants.json` comes from patient specimens that were
sequenced by the GLASS Consortium, and every limitation is recorded with it.

WHAT IT HONESTLY IS NOT
-----------------------
* Not the full variant space. The API requires a gene filter for this profile
  (an unfiltered query returns HTTP 400), so the panel below is DECLARED and the
  record says so rather than implying genome-wide coverage.
* Not a clonal reconstruction and not a clone size. The release carries no
  purity and no copy number, so `danon.variants` publishes a provable LOWER
  bound on the fraction of tumour cells carrying a variant, and refuses a point
  estimate, an upper bound and any k-bin -> two-bin summarisation.
* Not a treatment or outcome dataset. Nothing here is joined to therapy, and no
  engine run consumes it.

Writes
  data/glass_paired/glass_mutations.json   raw records + per-call provenance
  outputs_real/glass_variants.json         the measured record
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone

from danon.variants import (
    LOWER_BOUND_DERIVATION,
    POINT_FRACTION_REFUSED,
    TWO_BIN_SUMMARY_REFUSED,
    UPPER_BOUND_REFUSED,
    VARIANT_PANEL,
    clone_fraction_lower_bound,
    paired_changes,
    spectra_from_records,
    variant_channel_provenance,
    VariantRecord,
    what_this_channel_cannot_know,
)

API = "https://www.cbioportal.org/api"
STUDY = "difg_glass"
SEQUENCED_LIST = f"{STUDY}_sequenced"
MUTATIONS = f"{API}/molecular-profiles/{STUDY}_mutations/mutations"

CACHE_DIR = os.path.join("data", "glass_paired")
CACHE = os.path.join(CACHE_DIR, "glass_mutations.json")
OUT = os.path.join("outputs_real", "glass_variants.json")


def _get(url: str, timeout: int = 90):
    """GET JSON, recording the failure instead of raising it."""
    headers = {"User-Agent": "danon-variant-channel/1.0", "Accept": "application/json"}
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except Exception as exc:  # noqa: BLE001 -- the record carries failures too
        return None, {"error": f"{type(exc).__name__}: {exc}"}


def label_timepoint(sample_id: str) -> str:
    """The specimen's KIND, read from its own id.

    DECLARED RULE, and the only one used: GLASS ids name the specimen kind in the
    suffix (`-TP` primary, `-R1`/`-R2`/`-R3` recurrence, `-RM` metastasis). The
    API's own `sampleType` string is recorded beside the label so the rule can be
    checked rather than trusted, and a specimen the rule cannot label is recorded
    as `Unlabelled` rather than guessed into a pair.
    """
    suffix = sample_id.rsplit("-", 1)[-1].upper()
    if suffix == "TP":
        return "Primary"
    if suffix == "RM":
        return "Metastasis"
    if len(suffix) >= 2 and suffix[0] == "R" and suffix[1:].isdigit():
        return "Recurrence"
    return "Unlabelled"


def main() -> dict:
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs("outputs_real", exist_ok=True)
    fetched_utc = datetime.now(timezone.utc).isoformat()

    # -- specimen table (sample id -> patient id + the API's own sample type)
    status, samples = _get(f"{API}/studies/{STUDY}/samples?pageSize=10000")
    if status != 200 or not samples:
        raise SystemExit(f"GLASS samples unreadable: status={status}")
    sample_type = {s["sampleId"]: s.get("sampleType", "") for s in samples}
    patient_of = {s["sampleId"]: s.get("patientId", "") for s in samples}

    # -- the declared panel, one call per gene
    calls = []
    raw_records = []
    for gene, entrez in VARIANT_PANEL:
        url = f"{MUTATIONS}?sampleListId={SEQUENCED_LIST}&entrezGeneId={entrez}&pageSize=100000"
        code, payload = _get(url)
        records = payload if isinstance(payload, list) else []
        calls.append(
            {
                "gene": gene,
                "entrez_gene_id": entrez,
                "url": url,
                "status": code,
                "n_records": len(records),
            }
        )
        for rec in records:
            rec = dict(rec)
            rec["_gene"] = gene
            raw_records.append(rec)
        print(f"  {gene:8s} status={code} records={len(records)}")

    with open(CACHE, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "source": f"{API}/molecular-profiles/{STUDY}_mutations/mutations",
                "study_id": STUDY,
                "sample_list_id": SEQUENCED_LIST,
                "fetched_utc": fetched_utc,
                "panel": [list(p) for p in VARIANT_PANEL],
                "calls": calls,
                "samples": samples,
                "records": raw_records,
            },
            fh,
            indent=1,
        )

    # -- records -> the measured layer
    dropped = 0
    records = []
    for rec in raw_records:
        alt, ref = rec.get("tumorAltCount"), rec.get("tumorRefCount")
        if not isinstance(alt, int) or not isinstance(ref, int):
            dropped += 1
            continue
        sample_id = rec.get("sampleId", "")
        try:
            records.append(
                VariantRecord(
                    patient_id=rec.get("patientId") or patient_of.get(sample_id, ""),
                    sample_id=sample_id,
                    timepoint=label_timepoint(sample_id),
                    gene=rec["_gene"],
                    protein_change=rec.get("proteinChange") or "",
                    chromosome=str(rec.get("chr") or ""),
                    start=int(rec.get("startPosition") or 0),
                    ref_allele=str(rec.get("referenceAllele") or ""),
                    alt_allele=str(rec.get("variantAllele") or ""),
                    alt_count=alt,
                    ref_count=ref,
                    source=f"{API}/molecular-profiles/{STUDY}_mutations/mutations",
                )
            )
        except Exception:  # noqa: BLE001 -- zero-depth or malformed -> dropped
            dropped += 1

    spectra = spectra_from_records(records)
    pairs = paired_changes(spectra)

    # -- measurement, not summary prose
    label_counts = {}
    for spec in spectra:
        label_counts[spec.timepoint] = label_counts.get(spec.timepoint, 0) + 1
    type_counts = {}
    for sid in sample_type:
        type_counts[sample_type[sid]] = type_counts.get(sample_type[sid], 0) + 1

    afs = [r.allele_fraction for r in records]
    af_sorted = sorted(afs)
    shared_total = sum(p.n_shared for p in pairs)
    gained_total = sum(p.n_gained for p in pairs)
    lost_total = sum(p.n_lost for p in pairs)
    deltas = sorted(
        d for p in pairs for d in p.af_delta.values()
    )
    largest_rise = None
    for p in pairs:
        if not p.af_delta:
            continue
        key = max(p.af_delta, key=lambda k: p.af_delta[k])
        entry = (p.af_delta[key], p.patient_id, key, p.primary, p.recurrence)
        if largest_rise is None or entry[0] > largest_rise[0]:
            largest_rise = entry

    example = None
    if pairs:
        p = max(pairs, key=lambda x: (x.n_shared, x.n_gained))
        if p.af_delta:
            key = max(p.af_delta, key=lambda k: p.af_delta[k])
            example = {
                "patient_id": p.patient_id,
                "primary": p.primary,
                "recurrence": p.recurrence,
                "variant": key,
                "af_primary": p.af_primary[key],
                "af_recurrence": p.af_recurrence[key],
                "af_delta": p.af_delta[key],
                "lower_bound_primary": clone_fraction_lower_bound(p.af_primary[key])[0],
                "lower_bound_recurrence": clone_fraction_lower_bound(p.af_recurrence[key])[0],
            }

    def median(values):
        if not values:
            return None
        n = len(values)
        mid = n // 2
        return values[mid] if n % 2 else 0.5 * (values[mid - 1] + values[mid])

    record = {
        "record_version": "glass_variants_v1",
        "source": {
            "api": API,
            "study_id": STUDY,
            "public_study": True,
            "sample_list_id": SEQUENCED_LIST,
            "fetched_utc": fetched_utc,
            "raw_cache": CACHE,
            "panel": [f"{g} (entrez {e})" for g, e in VARIANT_PANEL],
            "panel_is_declared_not_genome_wide": True,
            "unfiltered_query_status": 400,
            "timepoint_label_rule": (
                "suffix of the sample id: '-TP' primary, '-R<n>' recurrence, "
                "'-RM' metastasis, anything else Unlabelled -- with the API's own "
                "sampleType recorded beside it"
            ),
        },
        "measured": {
            "n_mutation_records": len(raw_records),
            "n_records_with_allele_counts": len(records),
            "n_records_dropped_no_counts_or_zero_depth": dropped,
            "n_spectra": len(spectra),
            "n_patients": len({r.patient_id for r in records}),
            "n_samples": len({r.sample_id for r in records}),
            "n_genes_with_records": len({r.gene for r in records}),
            "spectra_by_timepoint_label": label_counts,
            "api_sample_types": type_counts,
            "allele_fraction": {
                "n": len(afs),
                "min": min(afs) if afs else None,
                "median": median(af_sorted),
                "max": max(afs) if afs else None,
                "definition": "alt / (alt + ref), computed from read counts",
            },
            "paired_change": {
                "n_pairs": len(pairs),
                "n_patients_with_a_pair": len({p.patient_id for p in pairs}),
                "variants_shared": shared_total,
                "variants_gained": gained_total,
                "variants_lost": lost_total,
                "median_af_delta_shared": median(deltas),
                "largest_rise": (
                    {
                        "af_delta": largest_rise[0],
                        "patient_id": largest_rise[1],
                        "variant": largest_rise[2],
                        "primary": largest_rise[3],
                        "recurrence": largest_rise[4],
                    }
                    if largest_rise
                    else None
                ),
            },
            "worked_example": example,
        },
        "calls": calls,
        "what_is_refused": {
            "point_clone_fraction": POINT_FRACTION_REFUSED,
            "upper_bound_on_clone_fraction": UPPER_BOUND_REFUSED,
            "two_bin_summarisation": TWO_BIN_SUMMARY_REFUSED,
            "lower_bound_derivation": LOWER_BOUND_DERIVATION,
        },
        "provenance": {
            k: v.value if hasattr(v, "value") else str(v)
            for k, v in variant_channel_provenance().items()
        },
        "cannot_know": what_this_channel_cannot_know(),
        "limitations": [
            "a DECLARED gene panel, not the whole variant space: the mutations "
            "endpoint of this profile requires a gene filter",
            "no purity and no copy number are carried, so only a lower bound on "
            "the clone fraction is published",
            "specimens are labelled by KIND, not by day, so 'primary versus "
            "recurrence' is the only pairing and no day-0/day-N mapping exists",
            "no treatment record and no outcome is joined to these specimens",
            "the panel is genome-filtered by gene, so a change in a gene outside "
            "it is invisible here",
        ],
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)

    m = record["measured"]
    print("\n--- measured from the real GLASS release ---")
    print(f"records {m['n_mutation_records']} | with counts {m['n_records_with_allele_counts']} "
          f"| dropped {m['n_records_dropped_no_counts_or_zero_depth']}")
    print(f"patients {m['n_patients']} | samples {m['n_samples']} | spectra {m['n_spectra']} "
          f"| genes {m['n_genes_with_records']}")
    print(f"labels {m['spectra_by_timepoint_label']}")
    print(f"api sample types {m['api_sample_types']}")
    print(f"AF n={m['allele_fraction']['n']} median={m['allele_fraction']['median']} "
          f"max={m['allele_fraction']['max']}")
    pc = m["paired_change"]
    print(f"pairs {pc['n_pairs']} over {pc['n_patients_with_a_pair']} patient(s) | "
          f"shared {pc['variants_shared']} gained {pc['variants_gained']} lost {pc['variants_lost']}")
    print(f"largest rise {pc['largest_rise']}")
    print(f"\nwrote {CACHE} and {OUT}")


if __name__ == "__main__":
    main()
