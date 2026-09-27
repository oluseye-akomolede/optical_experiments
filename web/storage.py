"""MinIO access for the web app, plus the local-directory fallback tests use.

The bucket is private, so the browser never talks to MinIO and nothing is
presigned: every result file is proxied through this app. That also means the
browser cannot be handed a URL it could have used to enumerate the bucket.
"""

from __future__ import annotations

import os
from pathlib import Path


class StorageError(Exception):
    """A storage operation failed in a way worth telling the owner about."""


def bucket_from_env() -> str:
    return os.environ.get("MSE_BUCKET") or "mse-web"


class S3Store:
    """MinIO/S3 backend. Credentials come from the environment, never a file."""

    def __init__(self, bucket: str | None = None, endpoint: str | None = None):
        import boto3
        from botocore.config import Config

        self.bucket = bucket or bucket_from_env()
        self.endpoint = (endpoint or os.environ.get("MINIO_ENDPOINT")
                         or "http://minio.minio.svc.cluster.local:9000")
        access = os.environ.get("MINIO_ACCESS_KEY")
        secret = os.environ.get("MINIO_SECRET_KEY")
        if not access or not secret:
            raise StorageError(
                "MINIO_ACCESS_KEY and MINIO_SECRET_KEY must be set in the environment"
            )
        self._client = boto3.client(
            "s3", endpoint_url=self.endpoint,
            aws_access_key_id=access, aws_secret_access_key=secret,
            config=Config(s3={"addressing_style": "path"},
                          retries={"max_attempts": 3}),
        )

    def read(self, key: str) -> bytes:
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:  # noqa: BLE001
            if _is_missing(exc):
                raise FileNotFoundError(key) from exc
            raise StorageError(f"could not read {key} from storage") from exc
        return response["Body"].read()

    def write(self, key: str, data: bytes) -> None:
        try:
            self._client.put_object(Bucket=self.bucket, Key=key, Body=data)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"could not upload the file to storage") from exc

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:  # noqa: BLE001
            return False

    def size(self, key: str) -> int | None:
        """The object's length in bytes, or ``None`` if it is not there.

        A HEAD, not a GET: the caller wants to *check* a file, and downloading
        it to measure it would be the very cost this exists to avoid.
        """
        try:
            return int(self._client.head_object(
                Bucket=self.bucket, Key=key)["ContentLength"])
        except Exception as exc:  # noqa: BLE001
            if _is_missing(exc):
                return None
            raise StorageError(f"could not check {key} in storage") from exc


def _is_missing(exc: Exception) -> bool:
    response = getattr(exc, "response", None) or {}
    code = str(response.get("Error", {}).get("Code", ""))
    return code in ("NoSuchKey", "404", "NoSuchBucket")


class LocalStore:
    """Directory-backed store, so tests and local runs need no MinIO.

    ``root`` stands in for the bucket, and every key is confined to it -- a key
    is derived from a job id, and a job id must never be able to escape.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        root = self.root.resolve()
        if root not in path.parents and path != root:
            raise StorageError(f"refusing to touch {key!r} outside {self.root}")
        return path

    def read(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        return path.read_bytes()

    def write(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def size(self, key: str) -> int | None:
        """The file's length in bytes, or ``None`` if it is not there."""
        path = self._path(key)
        if not path.is_file():
            return None
        return path.stat().st_size


def job_key(job_id, name: str) -> str:
    """Object key for a file belonging to a job (matches the solver's layout)."""
    return f"jobs/{job_id}/{name}"
