import shutil
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests

ZIP_MAGIC = (
    b"PK\x03\x04",  # standard (non-empty) archive
    b"PK\x05\x06",  # empty archive
    b"PK\x07\x08",  # spanned archive
)

ZIP_CONTENT_TYPE = {
    "application/zip",
    "application/x-zip-compressed",
    "application/x-zip",
}


def _remote_is_zip_by_magic(path: str) -> bool | None:
    """
    Check the first bytes of a remote file for the ZIP magic number using a
    Range request. Returns True/False if determinable, or None if the server
    doesn't support ranged requests / the check fails.
    """
    try:
        with requests.get(
            path, headers={"Range": "bytes=0-3"}, stream=True, timeout=10
        ) as resp:
            resp.raise_for_status()
            magic = next(resp.iter_content(4))
    except Exception:
        return None
    return magic.startswith(ZIP_MAGIC)


def _is_remote(path: str) -> bool:
    """Return True if `path` looks like a remote URL."""
    return urlparse(path).scheme in ("http", "https", "ftp")


def extract_zip_archive(zip_path: Path, extract_dir: Path) -> None:
    """Extract `zip_path` into `extract_dir`, cleaning up on failure."""
    print("extract")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    except Exception:
        # Don't leave a half-extracted dir around that would look "cached".
        shutil.rmtree(extract_dir, ignore_errors=True)
        raise


def create_zip_archive(
    zip_dir: Path, zarr_source_dir: Path, zip_name: str, *args
) -> Path:
    """
    Archives all contents of `zarr_source_dir` into a zip file stored inside
    `zip_dir`, then deletes everything except the created zip file.

    Args:
        zarr_source_dir: Path to the directory to archive.
        zip_dir: Path to the dir where the zip archive will be created
        zip_name: Filename of the zip archive to be created, e.g. results.zip
        *args: delayed object can be passed to this function if executed in a delayed
            context as args to ensure this function executes only after other computations
            have finnished

    Returns:
        Path to the created zip file.
    """
    source_path = zarr_source_dir.resolve()

    if not source_path.is_dir():
        raise ValueError(f"{zarr_source_dir} is not a valid directory")

    zip_path = zip_dir / zip_name

    # Create zip archive
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in source_path.rglob("*"):
            # Skip the zip file itself if it already exists
            if item == zip_path:
                continue

            # Store paths relative to source_dir
            arcname = item.relative_to(source_path)
            zf.write(item, arcname)

    shutil.rmtree(source_path)

    return zip_path


def is_zip(path: str) -> bool:
    """
    Detects whether a given path is a zip archive, can be a local path, or a remote url
    :param path: Path to the zip archive, local path or URL
    :return:
    """
    if path.endswith(".zip"):
        return True

    if _is_remote(path):
        try:
            req = urllib.request.Request(path, method="HEAD")
            with urllib.request.urlopen(req) as resp:
                content_type = (
                    resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
                )
        except Exception:
            content_type = ""

        if content_type in ZIP_CONTENT_TYPE:
            return True

        # Inconclusive (e.g. octet-stream): fall back to magic-byte sniffing.
        if content_type in {"", "application/octet-stream"}:
            magic_result = _remote_is_zip_by_magic(path)
            if magic_result is not None:
                return magic_result

    return False
