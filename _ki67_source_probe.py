"""Record whether the two external sources named for a Ki-67 index carry one.

A common "use only public data" route to the growth rate says: parse the TCGA
clinical XMLs for a Ki-67 field, or pull Ivy GAP's regional proliferation index
from its public API. Both are checkable without downloading an atlas or a
slide, so they are checked here and RECORDED, for the same reason the
walk-forward census is recorded -- the answer depends on what a public API
carries today, and a claim must be able to read the answer without the network.

Three things are probed:

  * GDC -- TCGA-GBM's Clinical Supplement (BCR XML) files, which are the
    structured per-patient clinical record a parser would read. The smallest is
    downloaded and EVERY field name in it is recorded. If a Ki-67/MIB-1 field
    exists anywhere in that schema, `proliferation_tokens_found` is non-empty
    and the refusal in danon/histology.py has to be revisited rather than kept.
  * Ivy GAP -- the header of each of its four public CSV tables, recorded column
    by column. The trap this guards against: the only proliferation-named column
    in the whole atlas is `normalized_area_ctmvp`, the area fraction of
    MICROVASCULAR proliferation -- a vessel-structure measurement, not a Ki-67
    labelling index -- so a keyword search would "find" proliferation here.
  * TCIA -- whether any collection in the v1 public list is ivy-named, which is
    the only machine-readable answer to whether the atlas cohort's imaging is
    reachable through the API this repository already queries.

Run:  python _ki67_source_probe.py            # queries the APIs, then writes
      python _ki67_source_probe.py --offline  # skips the network
"""

import argparse
import csv
import datetime
import io
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

OUT = os.path.join("outputs_real", "ki67_source_probe.json")
GDC_API = "https://api.gdc.cancer.gov/"
IVYGAP_API = "https://glioblastoma.alleninstitute.org/api/v2/gbm/"
TCIA_API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/"

#: Ivy GAP's four downloadable tables, as linked from its own download page.
IVYGAP_TABLES = (
    "tumor_details.csv",
    "sub_block_details.csv",
    "rna_seq_samples_details.csv",
    "gene_expression_details.csv",
)

#: Every spelling of a proliferation index worth searching a field list for.
#: 'ctmvp' is deliberately NOT here: it is the microvascular-proliferation
#: STRUCTURE fraction, and matching it would be the mistake this records.
PROLIFERATION_TOKENS = ("ki67", "ki_67", "ki-67", "mib", "labelling", "labeling")


#: The GDC's file endpoint closes a connection that arrives without a
#: user-agent rather than answering it, and all three hosts sometimes drop the
#: first attempt, so a fetch is retried before its failure is recorded.
_HEADERS = {"User-Agent": "danon-audit/1.0", "Accept": "*/*"}


def _get(url: str, timeout: int = 40, head_bytes: int = 0, attempts: int = 3):
    """Fetch a URL, retrying transient drops, and record a real failure."""
    last = None
    for _ in range(attempts):
        try:
            request = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read(head_bytes) if head_bytes else response.read()
                return {"status": response.status, "bytes": len(body), "body": body}
        except urllib.error.HTTPError as exc:
            last = {"status": exc.code, "bytes": 0, "error": f"HTTPError {exc.code}"}
        except Exception as exc:  # noqa: BLE001 -- the record carries failures too
            last = {"status": None, "bytes": 0, "error": f"{type(exc).__name__}: {exc}"}
    return last


def _tokens_in(names) -> list:
    """Which of `names` contains a proliferation token, case-insensitively."""
    found = []
    for name in names:
        low = str(name).lower()
        if any(tok in low for tok in PROLIFERATION_TOKENS):
            found.append(name)
    return found


def probe_gdc_clinical_schema() -> dict:
    """Download one TCGA-GBM clinical supplement and record every field name."""
    entry = {"api": GDC_API, "file_name": None, "field_names": [],
             "n_distinct_tags": 0, "proliferation_tokens_found": []}
    filt = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-GBM"]}},
        {"op": "in", "content": {"field": "data_category", "value": ["Clinical"]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Clinical Supplement"]}},
    ]}
    query = ("files?" + urllib.parse.urlencode(
        {"filters": json.dumps(filt), "fields": "file_id,file_name", "size": "1"}))
    entry["query"] = query
    listing = _get(GDC_API + query)
    entry["listing_status"] = listing.get("status")
    if not listing.get("body"):
        entry["error"] = listing.get("error", "no listing body")
        return entry
    hits = json.loads(listing["body"])["data"]["hits"]
    if not hits:
        entry["error"] = "no TCGA-GBM clinical supplement is listed"
        return entry
    entry["file_name"] = hits[0]["file_name"]
    entry["file_id"] = hits[0]["file_id"]
    fetch = _get(f"{GDC_API}data/{hits[0]['file_id']}")
    entry["status"] = fetch.get("status")
    if not fetch.get("body"):
        entry["error"] = fetch.get("error", "no file body")
        return entry
    text = fetch["body"].decode("utf-8", "replace")
    # The file is `gbm:tcga_bcr` with namespace prefixes, so a naive
    # `<([a-z_0-9]+)>` finds nothing: the colon has to be in the name pattern.
    seen, uniq = set(), []
    for tag in re.findall(r"<([A-Za-z_0-9:]+)[ >/]", text):
        if tag.lower() not in seen:
            seen.add(tag.lower())
            uniq.append(tag)
    entry["field_names"] = uniq
    entry["n_distinct_tags"] = len(uniq)
    entry["proliferation_tokens_found"] = _tokens_in(uniq)
    # The schema is the evidence, so its own version is recorded with it.
    version = re.search(r'schemaVersion="([^"]+)"', text)
    entry["schema_version"] = version.group(1) if version else None
    return entry


def probe_ivygap_tables() -> dict:
    """Read each Ivy GAP table's header and record its columns."""
    entry = {"api": IVYGAP_API, "tables": {}}
    for name in IVYGAP_TABLES:
        got = _get(IVYGAP_API + name, head_bytes=4000)
        table = {"status": got.get("status"), "columns": [],
                 "proliferation_named_columns": []}
        if got.get("body"):
            text = got["body"].decode("utf-8", "replace")
            header = text.split("\n")[0].strip()
            table["columns"] = next(csv.reader(io.StringIO(header)), [])
            table["proliferation_named_columns"] = _tokens_in(table["columns"])
        else:
            table["error"] = got.get("error", "no body")
        entry["tables"][name] = table
    return entry


def probe_tcia_collections() -> dict:
    """Record whether the public collection list has an ivy-named collection."""
    entry = {"api": TCIA_API, "query": "getCollectionValues"}
    got = _get(TCIA_API + "getCollectionValues", head_bytes=200_000)
    entry["status"] = got.get("status")
    if not got.get("body"):
        entry["error"] = got.get("error", "no body")
        return entry
    names = [c["Collection"] for c in json.loads(got["body"])]
    entry["n_collections"] = len(names)
    entry["ivy_named_collections"] = [n for n in names if "ivy" in n.lower()]
    entry["brain_named_collections"] = [
        n for n in names
        if any(tok in n.lower() for tok in ("brain", "glioma", "gbm", "astro"))
    ]
    return entry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    record = {
        "recorded_on": datetime.datetime.now().isoformat(timespec="seconds"),
        "offline": bool(args.offline),
        "question": (
            "do the sources a public-data route names for the growth rate carry a "
            "Ki-67 labelling index, which is the numerator danon/histology.py refuses "
            "to substitute a nuclei density for?"
        ),
        "gdc_tcga_gbm_clinical_schema": {},
        "ivygap_tables": {},
        "tcia_collections": {},
    }
    if not args.offline:
        record["gdc_tcga_gbm_clinical_schema"] = probe_gdc_clinical_schema()
        record["ivygap_tables"] = probe_ivygap_tables()
        record["tcia_collections"] = probe_tcia_collections()

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)

    gdc = record["gdc_tcga_gbm_clinical_schema"]
    ivy_tables = (record["ivygap_tables"] or {}).get("tables") or {}
    tcia = record["tcia_collections"]
    print("wrote", OUT)
    if args.offline:
        print("  --offline: network blocks left empty")
        return
    print(f"  GDC {gdc.get('file_name')}: {gdc.get('n_distinct_tags')} distinct "
          f"fields, schema {gdc.get('schema_version')}, "
          f"proliferation-named: {gdc.get('proliferation_tokens_found')}")
    for name, table in ivy_tables.items():
        print(f"  Ivy GAP {name}: {len(table.get('columns') or [])} columns, "
              f"proliferation-named: {table.get('proliferation_named_columns')}")
    print(f"  TCIA: {tcia.get('n_collections')} collections, "
          f"ivy-named: {tcia.get('ivy_named_collections')}")


if __name__ == "__main__":
    main()
