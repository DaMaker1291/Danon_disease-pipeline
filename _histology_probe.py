"""Stream 2 probe: what the biopsy archive actually holds, and what it cannot.

Writes `outputs_real/histology_probe.json`. Three things happen, in this order,
so that no number depends on a later choice:

  1. GDC is queried live for TCGA-GBM slide images: how many, of which strategy,
     and how many are named as Ki-67 or IHC (the stain stream 2's mapping needs).
  2. The smallest slide by reported file size is downloaded and read: its own
     MPP and AppMag, and the stain called from hue fractions over real pixels.
  3. Nuclei density is measured over several crops, and `ki67_fraction` is
     called so the *refusal* is recorded as data rather than as a comment.

The file-size rule is fixed here, before any measurement: smallest first.
"""

import json
import os
import re
import urllib.parse
import urllib.request

from danon.histology import (
    HistologyError,
    classify_stain,
    ki67_fraction,
    measure_nuclei_density,
    read_slide_metadata,
)

GDC_FILES = "https://api.gdc.cancer.gov/files"
GDC_DATA = "https://api.gdc.cancer.gov/data/{}"
PROJECT = "TCGA-GBM"
CACHE = "data/tcia/wsiprode"
OUT = "outputs_real/histology_probe.json"
CROPS = [(500, 500), (3000, 1500), (6000, 1200), (2500, 3200), (7000, 3000), (1200, 1800)]


def gdc_slide_census(project: str = PROJECT) -> dict:
    """Live count of a project's slide images, and how many name Ki-67 or IHC."""
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {"field": "cases.project.project_id", "value": [project]},
            },
            {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
        ],
    }
    query = urllib.parse.urlencode(
        {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,experimental_strategy,file_size",
            "format": "JSON",
            "size": 2100,
            "sort": "file_size:asc",
        }
    )
    with urllib.request.urlopen(f"{GDC_FILES}?{query}", timeout=90) as response:
        payload = json.load(response)
    hits = payload["data"]["hits"]
    total = int(payload["data"]["pagination"]["total"])
    by_strategy: dict = {}
    for hit in hits:
        key = hit.get("experimental_strategy") or "unknown"
        by_strategy[key] = by_strategy.get(key, 0) + 1
    names = [str(h.get("file_name", "")).lower() for h in hits]
    return {
        "project": project,
        "endpoint": GDC_FILES,
        "n_slide_images": total,
        "n_tabulated": len(hits),
        "by_strategy": by_strategy,
        "n_named_ki67": sum("ki67" in n or "ki-67" in n for n in names),
        "n_named_ihc": sum("ihc" in n for n in names),
        "note": (
            "TCGA file names carry a barcode, not a marker, so 'named' is a lower "
            "bound on what is identifiable from the index alone; the stain is "
            "settled from pixels below"
        ),
        "sizes_mb": {
            "smallest": round(min(int(h["file_size"]) for h in hits) / 1e6, 2),
            "largest_tabulated": round(max(int(h["file_size"]) for h in hits) / 1e6, 2),
        },
    }


def fetch_smallest(hits: list, cache: str = CACHE) -> str:
    os.makedirs(cache, exist_ok=True)
    row = hits[0]
    dest = os.path.join(cache, re.sub(r"[^A-Za-z0-9._-]", "_", row["file_name"]))
    if not os.path.exists(dest):
        urllib.request.urlretrieve(GDC_DATA.format(row["file_id"]), dest)
    return dest


def measure_slide(path: str) -> dict:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    meta = read_slide_metadata(path)
    im = Image.open(path)
    im.seek(0)
    sample = im.crop((3000, 1500, 3000 + 1024, 1500 + 1024)).convert("RGB")
    import numpy as np

    stain = classify_stain(np.asarray(sample))
    density = measure_nuclei_density(path, crops=CROPS, meta=meta)
    record = {
        "file": os.path.basename(path),
        "bytes": os.path.getsize(path),
        "slide": meta.provenance(),
        "stain": stain,
        "density": density,
    }
    try:
        record["ki67_fraction"] = ki67_fraction(stain)
        record["ki67_refusal"] = None
    except HistologyError as exc:
        record["ki67_fraction"] = None
        record["ki67_refusal"] = str(exc)
    return record


def main() -> None:
    census = gdc_slide_census()
    print(
        f"GDC {census['project']}: {census['n_slide_images']} slide images; "
        f"named ki67={census['n_named_ki67']}, ihc={census['n_named_ihc']}",
        flush=True,
    )
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {"field": "cases.project.project_id", "value": [PROJECT]},
            },
            {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
        ],
    }
    query = urllib.parse.urlencode(
        {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,file_size,experimental_strategy",
            "format": "JSON",
            "size": 3,
            "sort": "file_size:asc",
        }
    )
    with urllib.request.urlopen(f"{GDC_FILES}?{query}", timeout=90) as response:
        smallest = json.load(response)["data"]["hits"]
    path = fetch_smallest(smallest)
    print(f"downloaded {os.path.basename(path)} ({os.path.getsize(path)/1e6:.2f} MB)", flush=True)
    measured = measure_slide(path)
    print(
        f"stain {measured['stain']['stain']} | AppMag {measured['slide']['app_mag']} "
        f"MPP {measured['slide']['mpp_um']} | {measured['density']['nuclei_per_mm2']} "
        f"nuclei/mm^2 over {measured['density']['crops_used']} crops",
        flush=True,
    )
    print("ki67 refusal:", (measured["ki67_refusal"] or "")[:160], flush=True)

    record = {
        "census": census,
        "slide_measured": measured,
        "selection": "the smallest TCGA-GBM slide image by reported file size",
        "stream2_status": "MEASURED for cellularity; REFUSED for the Ki-67 ratio",
        "regenerate": "python _histology_probe.py",
    }
    with open(OUT, "w") as fh:
        json.dump(record, fh, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
