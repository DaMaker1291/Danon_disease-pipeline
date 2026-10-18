"""Exercise the MRI Scans page's real surface end-to-end.

The page claims three things that can each be false in a way a screenshot cannot
catch:

  1. the PNG it serves is the committed BraTS render (byte-identical mirror),
  2. the mask it offers to upload is the engine-ready mask, and the voxel count
     printed on the card is the voxel count inside that file,
  3. POSTing that exact served file to the running API returns that same geometry.

This script asserts all three against the files on disk and the live API, and
exits non-zero on the first disagreement. Run:

    python _verify_scan_surface.py [--base http://127.0.0.1:8000]
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request

from danon.imaging import read_nifti

ARTIFACT = "frontend/public/artifacts/scans.json"
IMG_MIRROR = "frontend/public/artifacts/scans"
MASK_MIRROR = "frontend/public/artifacts/masks"
ENGINE_DIR = "data/engine_ready"

FAILURES = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILURES.append(msg)


def sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def post_mask(url: str, path: str) -> dict:
    """POST a .nii.gz the way the browser does (multipart/form-data)."""
    boundary = "----verifyboundary7f3a"
    with open(path, "rb") as fh:
        blob = fh.read()
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="file"; filename="',
            os.path.basename(path).encode(),
            b'"\r\nContent-Type: application/gzip\r\n\r\n',
            blob,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    req = urllib.request.Request(
        f"{url}/api/twin/plan",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        return json.loads(resp.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--skip-api", action="store_true")
    args = ap.parse_args()

    artifact = json.load(open(ARTIFACT, encoding="utf-8"))
    data = artifact["data"]
    cases = data["cases"]

    print(f"artifact: {ARTIFACT}  cases={len(cases)}  standard={data['standard_case']}")
    check(len(cases) == 6, f"artifact lists 6 cases (found {len(cases)})")

    served_masks = set()
    for c in cases:
        case = c["case"]
        print(f"\n[{case}]")

        # ---- 1. the rendered scan the page shows -------------------------
        canonical = os.path.join("outputs_real/scans", c["image"])
        served = os.path.join(IMG_MIRROR, c["image"])
        check(os.path.exists(canonical), f"canonical PNG exists: {canonical}")
        check(os.path.exists(served), f"served PNG exists: {served}")
        if os.path.exists(canonical) and os.path.exists(served):
            check(
                sha(canonical) == sha(served),
                "served PNG is byte-identical to the committed render",
            )
        montage_file = data["montage"]["file"]
        check(
            os.path.exists(os.path.join(IMG_MIRROR, montage_file)),
            f"montage PNG served: {montage_file}",
        )

        # ---- 2. the mask the button uploads ------------------------------
        mask_name = c.get("engine_mask_file")
        check(bool(mask_name), "artifact names the engine mask to upload")
        if not mask_name:
            continue
        served_masks.add(mask_name)
        engine_src = os.path.join(ENGINE_DIR, mask_name)
        served_mask = os.path.join(MASK_MIRROR, mask_name)
        check(os.path.exists(engine_src), f"engine mask exists: {engine_src}")
        check(os.path.exists(served_mask), f"served mask exists: {served_mask}")
        if not (os.path.exists(engine_src) and os.path.exists(served_mask)):
            continue
        check(
            sha(engine_src) == sha(served_mask),
            "served mask is byte-identical to data/engine_ready/",
        )

        arr, _ = read_nifti(served_mask)
        vox = int((arr > 0).sum())
        check(vox > 0, f"served mask parses as NIfTI with tumour voxels ({vox})")
        check(
            vox == c["engine_mask_voxels"],
            f"card's engine_mask_voxels ({c['engine_mask_voxels']}) == voxels in the "
            f"served file ({vox})",
        )
        check(
            [int(v) for v in arr.shape] == c["engine_grid_xyz"],
            f"card's engine_grid_xyz ({c['engine_grid_xyz']}) == mask shape "
            f"{[int(v) for v in arr.shape]}",
        )

        # ---- 3. the live API on that exact file --------------------------
        if args.skip_api:
            continue
        try:
            payload = post_mask(args.base, served_mask)
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            check(False, f"live POST /api/twin/plan raised {type(exc).__name__}: {exc}")
            continue
        g = payload["geometry"]
        check(
            g["tumourVoxels"] == vox,
            f"API tumourVoxels ({g['tumourVoxels']}) == voxels in the served mask ({vox})",
        )
        check(
            g["gridShapeXyz"] == c["engine_grid_xyz"],
            f"API gridShapeXyz ({g['gridShapeXyz']}) == card's engine_grid_xyz "
            f"({c['engine_grid_xyz']})",
        )
        check(bool(payload["scheduleText"].strip()), "API returned a schedule")
        check(
            "ASSUMED" in json.dumps(payload.get("provenance", {})),
            "API provenance labels its assumed inputs rather than hiding them",
        )

    orphan = {
        f
        for f in os.listdir(MASK_MIRROR)
        if f.endswith(".nii.gz") and f not in served_masks
    }
    check(not orphan, f"no stale masks served that no card references ({sorted(orphan)})")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED")
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("scan surface: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())