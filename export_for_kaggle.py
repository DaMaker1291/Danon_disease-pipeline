"""
Upload real data to Kaggle as a dataset.

SETUP (one-time):
  1. pip install kaggle
  2. Go to kaggle.com -> Account -> Create API Token -> downloads kaggle.json
  3. Place kaggle.json in USERPROFILE/.kaggle/
  4. Run: python export_for_kaggle.py

This creates a Kaggle dataset you can attach to any notebook.
"""
import os
import json
import subprocess
import shutil

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
KAGGLE_DIR = os.path.join(os.path.dirname(__file__), "kaggle_dataset")
DATASET_NAME = "longevity-real-data"

def main():
    os.makedirs(KAGGLE_DIR, exist_ok=True)

    files = {
        "real_screening_aav_tropism.json": os.path.join(DATA_DIR, "real_screening_aav_tropism.json"),
        "real_screening_lnp_delivery.json": os.path.join(DATA_DIR, "real_screening_lnp_delivery.json"),
        "real_screening_immune_escape.json": os.path.join(DATA_DIR, "real_screening_immune_escape.json"),
        "LNPDB_real.csv": os.path.join(DATA_DIR, "LNPDB_real.csv"),
        "fit4function_screens.csv": os.path.join(DATA_DIR, "fit4function_screens.csv"),
    }

    for fname, src in files.items():
        if os.path.exists(src):
            dst = os.path.join(KAGGLE_DIR, fname)
            shutil.copy2(src, dst)
            size_mb = os.path.getsize(src) / (1024 * 1024)
            print(f"  Copied: {fname} ({size_mb:.1f} MB)")
        else:
            print(f"  MISSING: {fname}")

    # Create dataset-metadata.json
    meta = {
        "title": "Longevity Pipeline - Real Screening Data",
        "id": f"supro/{DATASET_NAME}",
        "licenses": [{"name": "CC0-1.0"}],
        "description": (
            "Real screening data for longevity drug discovery pipeline. "
            "Includes 100K AAV capsid variants (Fit4Function/Harvard Church Lab), "
            "773 LNP formulations (LNPDB), with fitness scores, tissue tropism, "
            "immune escape scores, and antibody response data."
        ),
    }
    meta_path = os.path.join(KAGGLE_DIR, "dataset-metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Created: dataset-metadata.json")

    print(f"\nDataset folder: {KAGGLE_DIR}")
    print(f"\nTo upload:")
    print(f"  cd {KAGGLE_DIR}")
    print(f"  kaggle datasets create -p .")
    print(f"\nOr update:")
    print(f"  kaggle datasets version -p . -m 'Updated data'")

if __name__ == "__main__":
    main()
