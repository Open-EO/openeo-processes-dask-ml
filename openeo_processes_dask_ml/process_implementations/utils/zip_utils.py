import shutil
import zipfile
from pathlib import Path


def extract_zip_archive(zip_path: Path, extract_dir: Path) -> None:
    """Extract `zip_path` into `extract_dir`, cleaning up on failure."""
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
