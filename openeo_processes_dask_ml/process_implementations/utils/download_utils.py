from pathlib import Path

import boto3
import botocore.exceptions
import requests
from botocore import UNSIGNED
from botocore.config import Config

from openeo_processes_dask_ml.process_implementations.constants import (
    S3_MODEL_REPO_ACCESS_KEY_ID,
    S3_MODEL_REPO_ENDPOINT,
    S3_MODEL_REPO_SECRET_ACCESS_KEY,
)


def download_http(url: str, target_path: str):
    chunk_size = 8192
    total_size = 0
    # todo: replace sys.out.write() and print() with logger actions
    try:
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()

            # Check for Content-Length header to estimate size (optional)
            # content_length = response.headers.get('content-length')
            # if content_length:
            #     total_expected_size = int(content_length)
            #     print(f"Downloading {url}")
            #     print(f"Total expected size: {total_expected_size / (1024 * 1024):.2f} MB")
            # else:
            #     total_expected_size = None
            #     print(f"Downloading {url} (size unknown)")

            with open(target_path, "wb") as f:
                # Iterate over the response data in chunks
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        # total_size += len(chunk)
                        # # Optional: Print progress
                        # if total_expected_size:
                        #     done = int(50 * total_size / total_expected_size)
                        #     sys.stdout.write(
                        #         f"\r[{'=' * done}{' ' * (50 - done)}] {total_size / (1024 * 1024):.2f} MB / {total_expected_size / (1024 * 1024):.2f} MB")
                        #     sys.stdout.flush()
                        #
                        # else:
                        #     sys.stdout.write(
                        #         f"\rDownloaded: {total_size / (1024 * 1024):.2f} MB")
                        #     sys.stdout.flush()

        # print("\nDownload complete.")  # Newline after progress bar

    except requests.exceptions.RequestException as e:
        raise Exception(f"\nError downloading {url}: {e}")

    except Exception as e:
        raise Exception(f"\nAn unexpected error occurred: {e}")


def download_s3(url: str, target_path: str):
    object_path = url[5:].split("/")  # remove s3://, then split by /
    bucket_name = object_path.pop(0)
    object_key = "/".join(object_path)

    try:
        if S3_MODEL_REPO_ACCESS_KEY_ID and S3_MODEL_REPO_SECRET_ACCESS_KEY:
            s3 = boto3.client(
                "s3",
                aws_access_key_id=S3_MODEL_REPO_ACCESS_KEY_ID,
                aws_secret_access_key=S3_MODEL_REPO_SECRET_ACCESS_KEY,
                endpoint_url=S3_MODEL_REPO_ENDPOINT,
            )
        else:
            s3 = boto3.client(
                "s3",
                endpoint_url=S3_MODEL_REPO_ENDPOINT,
                config=Config(signature_version=UNSIGNED),
            )
        s3.download_file(bucket_name, object_key, target_path)

    except FileNotFoundError:
        raise Exception(f"Could not locate file with {object_key=} in {bucket_name=}")

    except botocore.exceptions.ClientError:
        raise Exception(f"Error connecting to s3 storage to download model")


def download(url: str, target_path: str | Path):
    if isinstance(target_path, Path):
        target_path = str(target_path)

    protocol = url.split("://")[0]

    # download the model
    if protocol == "http" or protocol == "https":
        download_http(url, target_path)
    elif protocol == "s3":
        download_s3(url, target_path)
