"""Record what longitudinal pairs exist, from the repository, TCIA and Figshare.

The repository-side counts are re-derivable offline at any time (see
`danon.walkforward.longitudinal_census`). The network side is not: it depends
on what a public API answers today, so it is RECORDED here with the exact
query string and the time it was made, and the claim reads the record rather
than re-querying the network.

Three network sources, because they answer different questions:
  * TCIA/NBIA -- collections that carry imaging, probed for patients and
    segmentation series;
  * Figshare -- the LUMIERE collection, which is longitudinal imaging the
    TCIA wrappers do not surface. Its article list is recorded because it is
    what tells the two assets apart: the collection ships 'MRI data and
    automated segmentations' (two U-Net tools) as one article and the expert
    'Expert RANO rating' as another, so a Dice against its masks would score
    this engine against another algorithm rather than against expert truth;
  * cBioPortal -- the PUBLIC GLASS release, probed for whether it is open, for
    how its specimens pair, and for whether the allele counts that a clonal
    frequency would be read from are populated at all.

Run:  python _walkforward_census.py            # queries the APIs, then writes
      python _walkforward_census.py --offline  # skips the network
"""

import argparse
import datetime
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict

from danon.walkforward import longitudinal_census

OUT = os.path.join("outputs_real", "walkforward_census.json")
API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/"

#: LUMIERE is hosted on Figshare, which the NBIA API cannot answer for. The
#: collection id is the one the dataset's own landing page uses.
LUMIERE_COLLECTION = 5904905
FIGSHARE_LUMIERE = f"https://api.figshare.com/v2/collections/{LUMIERE_COLLECTION}/articles"

#: GLASS is one of the two genomics sources a walk-forward route names, and the
#: NBIA probe above cannot answer for it at all: it is genomics, not imaging.
#: cBioPortal publishes a PROCESSED GLASS release as an open study, so the
#: question is not reachability but whether a clonal FREQUENCY can be read from
#: it and mapped onto the engine's two bins.
CBIOPORTAL = "https://www.cbioportal.org/api/"
GLASS_STUDY = "difg_glass"

#: The genes a resistance route names, with the entrez ids the API requires.
GLASS_GENES = (("MSH6", 2956), ("MGMT", 4255), ("TP53", 7157))

#: The collections named in the longitudinal-validation proposal, plus the one
#: this repository has always probed, so the record shows all three together.
QUERIES = (
    "MU-Glioma-Post",
    "Brain-Tumor-Progression",
    "CFB-GBM",
)


def _fetch_url(url: str, timeout: int = 40):
    """Fetch an absolute URL and KEEP the body, for records that need it."""
    headers = {"User-Agent": "danon-audit/1.0", "Accept": "application/json"}
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            return {"status": response.status, "bytes": len(body), "body": body}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "bytes": 0, "error": f"HTTPError {exc.code}"}
    except Exception as exc:  # noqa: BLE001 -- the record carries failures too
        return {"status": None, "bytes": 0, "error": f"{type(exc).__name__}: {exc}"}


def _get_url(url: str, timeout: int = 25):
    """Probe an absolute URL, recording the failure rather than raising it."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read()
            return {"status": response.status, "bytes": len(body),
                    "records": len(json.loads(body)) if body.strip() else 0}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "bytes": 0, "records": None,
                "error": f"HTTPError {exc.code}"}
    except Exception as exc:  # noqa: BLE001 -- the census records failures too
        return {"status": None, "bytes": 0, "records": None,
                "error": f"{type(exc).__name__}: {exc}"}


def _get(path: str, timeout: int = 25):
    return _get_url(API + path, timeout=timeout)


def probe_lumiere():
    """Read the LUMIERE collection's article list from the Figshare API.

    The titles are the record: they are what separates the automated
    segmentations from the expert RANO rating, and the annotation in
    `longitudinal_census` is checked against them.
    """
    entry = {
        "collection_id": LUMIERE_COLLECTION,
        "host": "figshare.com (Springer Nature)",
        "query": FIGSHARE_LUMIERE,
        "articles": [],
    }
    try:
        with urllib.request.urlopen(FIGSHARE_LUMIERE, timeout=25) as response:
            body = response.read()
            entry["status"] = response.status
            entry["bytes"] = len(body)
            payload = json.loads(body)
    except Exception as exc:  # noqa: BLE001 -- the record carries the failure
        entry.update({"status": None, "bytes": 0,
                      "error": f"{type(exc).__name__}: {exc}"})
        return entry
    entry["n_articles"] = len(payload)
    entry["articles"] = [
        {"id": a.get("id"), "title": a.get("title"), "doi": a.get("doi")}
        for a in payload
    ]
    titles = [str(a.get("title") or "").lower() for a in payload]
    entry["segmentation_article_is_automated"] = any(
        "automated segmentation" in t for t in titles
    )
    entry["expert_article_is_rano"] = any("expert rano" in t for t in titles)
    return entry


def probe_glass_public_release() -> dict:
    """Read the public GLASS release: its samples, its pairing, its allele data.

    `tumorAltCount`/`tumorRefCount` are in the API's own record schema. Whether
    they are POPULATED decides whether a clonal frequency is readable, which is
    the whole question a resistance-anchoring route turns on.
    """
    entry = {"api": CBIOPORTAL, "study_id": GLASS_STUDY, "mutations": {}}
    study = _fetch_url(CBIOPORTAL + f"studies/{GLASS_STUDY}")
    entry["status"] = study.get("status")
    if not study.get("body"):
        entry["error"] = study.get("error", "study metadata unreadable")
        return entry
    meta = json.loads(study["body"])
    entry["name"] = meta.get("name")
    entry["public_study"] = meta.get("publicStudy")
    entry["all_samples"] = meta.get("allSampleCount")
    entry["sequenced_samples"] = meta.get("sequencedSampleCount")
    entry["pmid"] = meta.get("pmid")

    # Pairing is the point: primary and recurrent specimens of the same patient.
    kinds = _fetch_url(
        CBIOPORTAL + f"studies/{GLASS_STUDY}/clinical-data"
        "?clinicalDataType=SAMPLE&attributeId=SAMPLE_TYPE&pageSize=100000"
    )
    if kinds.get("body"):
        counts: Dict[str, int] = {}
        for row in json.loads(kinds["body"]):
            value = str(row.get("value"))
            counts[value] = counts.get(value, 0) + 1
        entry["sample_types"] = counts

    # Per gene: are the records there, and are the allele counts populated?
    for gene, entrez in GLASS_GENES:
        got = _fetch_url(
            CBIOPORTAL + f"molecular-profiles/{GLASS_STUDY}_mutations/mutations"
            f"?sampleListId={GLASS_STUDY}_sequenced&entrezGeneId={entrez}&pageSize=5"
        )
        record: Dict[str, Any] = {"entrez_id": entrez, "status": got.get("status")}
        if got.get("body"):
            records = json.loads(got["body"])
            record["records"] = len(records)
            record["allele_counts_populated"] = any(
                r.get("tumorAltCount") is not None for r in records
            )
            record["mutation_statuses"] = sorted(
                {str(r.get("mutationStatus")) for r in records}
            )
            record["sample_ids"] = sorted({str(r.get("sampleId")) for r in records})[:4]
        else:
            record["error"] = got.get("error", "no body")
        entry["mutations"][gene] = record
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    repo = longitudinal_census()
    record = {
        "recorded_on": datetime.datetime.now().isoformat(timespec="seconds"),
        "api": API,
        "offline": bool(args.offline),
        "repository": {
            "scorable_pairs": repo["scorable_pairs"],
            "sources": repo["sources"],
        },
        "tcia": {},
        "figshare": {},
        "cbioportal": {},
    }
    if not args.offline:
        for collection in QUERIES:
            record["tcia"][collection] = {
                "query": f"getPatient?Collection={collection}",
                "get_patient": _get(f"getPatient?Collection={collection}"),
                "get_series": _get(f"getSeries?Collection={collection}&Modality=SEG"),
            }
        record["figshare"]["LUMIERE"] = probe_lumiere()
        record["cbioportal"]["GLASS"] = probe_glass_public_release()
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)
    print(f"wrote {OUT}")
    for name, entry in record["tcia"].items():
        print(
            f"  {name:26s} patients={entry['get_patient']['records']} "
            f"seg-series={entry['get_series']['records']} "
            f"{entry['get_patient'].get('error') or ''}"
        )
    print(
        f"  repository: {repo['scorable_pairs']} scorable pair(s); "
        f"CFB-GBM declares {repo['sources']['CFB-GBM']['usable_pairs_declared']} "
        f"over {repo['sources']['CFB-GBM']['patients_with_usable_pair']} patients, "
        f"images cached: {repo['sources']['CFB-GBM']['images_present']}"
    )
    lumiere = record["figshare"].get("LUMIERE")
    if lumiere:
        print(
            f"  LUMIERE (figshare collection {lumiere['collection_id']}): "
            f"status={lumiere.get('status')} "
            f"articles={lumiere.get('n_articles')} "
            f"automated-segmentation article={lumiere.get('segmentation_article_is_automated')} "
            f"expert-RANO article={lumiere.get('expert_article_is_rano')}"
        )
    glass = record["cbioportal"].get("GLASS")
    if glass:
        genes = glass.get("mutations") or {}
        populated = {g: v.get("allele_counts_populated") for g, v in genes.items()}
        print(
            f"  GLASS (cBioPortal {glass.get('study_id')}): "
            f"public={glass.get('public_study')} "
            f"samples={glass.get('all_samples')}/{glass.get('sequenced_samples')} "
            f"sequenced; sample types={glass.get('sample_types')}; "
            f"allele counts populated {populated}"
        )


if __name__ == "__main__":
    main()
