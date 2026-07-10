"""
Export real data for Google Colab training.
Run this locally first, then upload the generated zip to Colab.

Usage:
    python export_for_colab.py
"""
import os
import json
import zipfile
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_ZIP = os.path.join(os.path.dirname(__file__), "colab_data.zip")

FILES_TO_INCLUDE = [
    "real_screening_aav_tropism.json",
    "real_screening_lnp_delivery.json",
    "real_screening_immune_escape.json",
    "LNPDB_real.csv",
    "fit4function_screens.csv",
]

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    existing = []
    missing = []
    for fname in FILES_TO_INCLUDE:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            existing.append((fname, size_mb))
        else:
            missing.append(fname)

    logger.info("=== Data files found ===")
    for fname, size_mb in existing:
        logger.info("  %s  (%.1f MB)", fname, size_mb)

    if missing:
        logger.warning("=== Missing files (will be skipped) ===")
        for fname in missing:
            logger.warning("  %s", fname)

    logger.info("Creating zip: %s", OUT_ZIP)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname, _ in existing:
            fpath = os.path.join(DATA_DIR, fname)
            zf.write(fpath, fname)
            logger.info("  Added: %s", fname)

    zip_size_mb = os.path.getsize(OUT_ZIP) / (1024 * 1024)
    logger.info("Done! Zip size: %.1f MB", zip_size_mb)
    logger.info("Upload '%s' to Colab via: from google.colab import files; files.upload()", OUT_ZIP)

    logger.info("")
    logger.info("=== Dataset stats ===")
    for fname, _ in existing:
        fpath = os.path.join(DATA_DIR, fname)
        if fname.endswith(".json"):
            with open(fpath) as f:
                data = json.load(f)
            logger.info("  %s: %d samples", fname, len(data))
        elif fname.endswith(".csv"):
            with open(fpath) as f:
                lines = f.readlines()
            logger.info("  %s: %d rows", fname, len(lines) - 1)

if __name__ == "__main__":
    main()
