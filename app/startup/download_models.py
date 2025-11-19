import os
import re
import logging
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import requests

logger = logging.getLogger(__name__)


def _get_google_drive_file_id(url: str) -> str | None:
    # Patterns: /d/<id>/ or ?id=<id> or id in query
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if "id" in qs and qs["id"]:
        return qs["id"][0]
    return None


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    # fallback: search response text
    m = re.search(r"confirm=([0-9A-Za-z_\-]+)&", response.text)
    if m:
        return m.group(1)
    return None


def _save_response_content(response, destination, chunk_size=32768):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


def _download_from_google_drive(file_id: str, destination: Path):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    logger.info("Requesting Google Drive file id=%s", file_id)
    response = session.get(URL, params={"id": file_id}, stream=True, timeout=60)
    token = _get_confirm_token(response)
    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True, timeout=60)
    response.raise_for_status()
    _save_response_content(response, destination)


def _download_direct(url: str, destination: Path):
    logger.info("Downloading: %s -> %s", url, destination)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        _save_response_content(r, destination)


def _derive_filename_from_url(url: str, default_name: str) -> str:
    parsed = urlparse(url)
    last = os.path.basename(parsed.path)
    if last and last not in ("", "view", "file") and "." in last:
        return last
    qs = parse_qs(parsed.query)
    if "id" in qs:
        return qs["id"][0]
    # fallback: use default name
    return default_name


def download_file_from_url(url: str, destination_dir: Path, default_name: str) -> Path:
    """
    Download a file from a URL to destination_dir. If URL points to Google Drive,
    uses the Drive download flow. Skips download if file already exists.
    Returns the destination Path.
    """
    if not url:
        raise ValueError("No URL provided")
    filename = _derive_filename_from_url(url, default_name)
    dest = destination_dir / filename
    if dest.exists():
        logger.info("File already exists, skipping download: %s", dest)
        return dest

    # Try Google Drive flow first when possible
    file_id = _get_google_drive_file_id(url)
    try:
        if file_id:
            _download_from_google_drive(file_id, dest)
        else:
            _download_direct(url, dest)
    except Exception:
        # If Google Drive flow fails and url looks like drive, try direct as fallback
        logger.exception("Primary download method failed for %s, attempting direct GET", url)
        _download_direct(url, dest)

    logger.info("Downloaded file to %s", dest)
    return dest


def download_models_from_env(root_dir: Path | str):
    """Read ONNX_MODEL and ONNX_MODEL_DATA from environment and download to root_dir.
    Skips downloads for missing env vars or if files already exist.
    """
    root = Path(root_dir)
    model_a = os.environ.get("ONNX_MODEL")
    model_b = os.environ.get("ONNX_MODEL_DATA")

    results = {}
    if model_a:
        try:
            results["ONNX_MODEL"] = download_file_from_url(model_a, root, "clip_vision_static.onnx")
        except Exception as e:
            logger.exception("Failed to download ONNX_MODEL: %s", e)
    else:
        logger.info("Environment variable ONNX_MODEL not set; skipping")

    if model_b:
        try:
            results["ONNX_MODEL_DATA"] = download_file_from_url(model_b, root, "clip_vision_static.onnx.data")
        except Exception as e:
            logger.exception("Failed to download ONNX_MODEL_DATA: %s", e)
    else:
        logger.info("Environment variable ONNX_MODEL_DATA not set; skipping")

    return results
