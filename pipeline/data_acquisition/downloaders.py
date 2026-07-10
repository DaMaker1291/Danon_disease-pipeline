import os
import json
import logging
import subprocess
from urllib.request import urlretrieve
from urllib.error import URLError

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data")


class DatasetDownloader:
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)

    def download_file(self, url: str, filename: str, force: bool = False) -> str:
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath) and not force:
            logger.info("Already downloaded: %s", filepath)
            return filepath
        logger.info("Downloading %s -> %s", url, filepath)
        try:
            urlretrieve(url, filepath)
            logger.info("Downloaded: %s (%d bytes)", filepath, os.path.getsize(filepath))
            return filepath
        except URLError as e:
            logger.error("Download failed: %s", e)
            return None

    def download_with_wget(self, url: str, filename: str) -> str:
        filepath = os.path.join(self.data_dir, filename)
        cmd = ["wget", "-q", "--show-progress", "-O", filepath, url]
        try:
            subprocess.run(cmd, check=True, timeout=3600)
            return filepath
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error("wget failed: %s", e)
            return None

    def download_with_curl(self, url: str, filename: str) -> str:
        filepath = os.path.join(self.data_dir, filename)
        cmd = ["curl", "-L", "-o", filepath, url]
        try:
            subprocess.run(cmd, check=True, timeout=3600)
            return filepath
        except subprocess.CalledProcessError as e:
            logger.error("curl failed: %s", e)
            return None


class AgingAtlasDownloader(DatasetDownloader):
    DATASETS = {
        "aging_atlas_bulk": {
            "url": "https://aging.pku.edu.cn/api/download/bulk_rnaseq",
            "filename": "aging_atlas_bulk_rnaseq.h5ad",
            "description": "Bulk RNA-seq from aging human tissues",
        },
        "aging_atlas_single_cell": {
            "url": "https://aging.pku.edu.cn/api/download/single_cell",
            "filename": "aging_atlas_scRNAseq.h5ad",
            "description": "Single-cell RNA-seq from aging human tissues",
        },
        "aging_atlas_spatial": {
            "url": "https://aging.pku.edu.cn/api/download/spatial_transcriptomics",
            "filename": "aging_atlas_spatial.h5ad",
            "description": "Spatial transcriptomics from aging human tissues",
        },
    }

    def download_all(self) -> dict:
        results = {}
        for key, info in self.DATASETS.items():
            filepath = self.download_with_wget(info["url"], info["filename"])
            if filepath:
                results[key] = {"path": filepath, "description": info["description"]}
        return results


class HumanCellAtlasDownloader(DatasetDownloader):
    PUBLIC_DATASETS = {
        "aging_heart": {
            "matrix": "https://data.humancellatlas.org/projects/matrix/heart_aging/filtered.h5ad",
            "metadata": "https://data.humancellatlas.org/projects/metadata/heart_aging.csv",
        },
        "aging_brain": {
            "matrix": "https://data.humancellatlas.org/projects/matrix/brain_aging/filtered.h5ad",
            "metadata": "https://data.humancellatlas.org/projects/metadata/brain_aging.csv",
        },
        "aging_joint": {
            "matrix": "https://data.humancellatlas.org/projects/matrix/joint_aging/filtered.h5ad",
            "metadata": "https://data.humancellatlas.org/projects/metadata/joint_aging.csv",
        },
    }

    def download_all(self) -> dict:
        results = {}
        for key, files in self.PUBLIC_DATASETS.items():
            results[key] = {}
            for file_type, url in files.items():
                ext = "h5ad" if file_type == "matrix" else "csv"
                filename = f"hca_{key}_{file_type}.{ext}"
                filepath = self.download_with_wget(url, filename)
                if filepath:
                    results[key][file_type] = filepath
        return results


class ScreeningDataDownloader(DatasetDownloader):
    SCREENING_DATASETS = {
        "aav_tropism_screen": {
            "barcodes": "https://pubmed.ncbi.nlm.nih.gov/aav_tropism_barcode_counts.csv",
            "metadata": "https://pubmed.ncbi.nlm.nih.gov/aav_tropism_metadata.csv",
        },
        "lnp_delivery_screen": {
            "barcodes": "https://pubmed.ncbi.nlm.nih.gov/lnp_barcode_counts.csv",
            "metadata": "https://pubmed.ncbi.nlm.nih.gov/lnp_metadata.csv",
        },
        "immune_escape_screen": {
            "sequences": "https://pubmed.ncbi.nlm.nih.gov/aav_escape_sequences.fasta",
            "binding": "https://pubmed.ncbi.nlm.nih.gov/antibody_binding_data.csv",
        },
    }

    def download_all(self) -> dict:
        results = {}
        for key, files in self.SCREENING_DATASETS.items():
            results[key] = {}
            for file_type, url in files.items():
                ext = "csv" if "csv" in url else "fasta"
                filename = f"screening_{key}_{file_type}.{ext}"
                filepath = self.download_with_wget(url, filename)
                if filepath:
                    results[key][file_type] = filepath
        return results
