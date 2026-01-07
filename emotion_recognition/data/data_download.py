"""Data download functionality for RAVDESS dataset."""

import logging
import os
import urllib.request
import zipfile
from typing import Optional

log = logging.getLogger(__name__)


def download_ravdess_dataset(
    target_dir: str = "data",
    url: str = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1",
    zip_filename: str = "Audio_Speech_Actors_01-24.zip",
) -> str:
    """Download RAVDESS dataset from Zenodo.

    Args:
        target_dir: Directory to save dataset
        url: URL to download dataset from
        zip_filename: Name of the zip file

    Returns:
        Path to downloaded zip file
    """
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, zip_filename)

    log.info(f"Downloading RAVDESS dataset from {url}")
    urllib.request.urlretrieve(url, zip_path)
    log.info(f"Dataset downloaded to {zip_path}")

    return zip_path


def extract_ravdess_dataset(zip_path: str, extract_dir: str = "data") -> str:
    """Extract RAVDESS dataset from zip file.

    Args:
        zip_path: Path to zip file
        extract_dir: Directory to extract to

    Returns:
        Path to extracted directory
    """
    log.info(f"Extracting dataset from {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    extracted_dir = os.path.join(extract_dir, "Audio_Speech_Actors_01-24")
    log.info(f"Dataset extracted to {extracted_dir}")

    return extracted_dir


def ensure_data_available(
    data_dir: str = "data/Audio_Speech_Actors_01-24", download_if_missing: bool = True
) -> bool:
    """Ensure RAVDESS data is available, download if missing.

    Args:
        data_dir: Expected data directory
        download_if_missing: Whether to download if data is missing

    Returns:
        True if data is available, False otherwise
    """
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
        log.info(f"Data already available at {data_dir}")
        return True

    if not download_if_missing:
        log.error(f"Data not found at {data_dir} and download_if_missing=False")
        return False

    log.info(f"Data not found at {data_dir}, downloading...")

    try:
        # Download and extract
        zip_path = download_ravdess_dataset()
        extract_ravdess_dataset(zip_path)
        return True
    except Exception as e:
        log.error(f"Failed to download dataset: {e}")
        return False
