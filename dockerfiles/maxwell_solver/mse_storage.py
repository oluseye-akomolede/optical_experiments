"""MinIO access for the MSE solver, with a filesystem path for tests.

The web app in task 2 proxies these objects to the browser, so the bucket stays
private and nothing is presigned. The solver only ever reads one input and
writes a handful of results, so this is deliberately thin.

``--local-in`` / ``--local-out`` (see :mod:`mse_maxwell_solver`) use the same
interface against a directory, which is how the tests and the Jetson bench run
without MinIO or a database.
"""

from __future__ import annotations

import os
from pathlib import Path

#: default endpoint inside the cluster
DEFAULT_ENDPOINT = "http://minio.minio.svc.cluster.local:9000"

#: default bucket for this project
DEFAULT_BUCKET = "mse-web"


class StorageError(Exception):
    """A storage operation failed in a way the user should be told about."""


def bucket_from_env() -> str:
    return os.environ.get("MSE_BUCKET", DEFAULT_BUCKET)


class ObjectStore:
    """Read/write objects under a bucket, keyed by relative path.

    Implementations must expose the same ``read``/``write``/``exists`` surface so
    the solver is identical under MinIO and under ``--local-out``.
    """

    def read(self, key: str) -> bytes:                        # pragma: no cover
        raise NotImplementedError

    def write(self, key: str, data: bytes) -> None:           # pragma: no cover
        raise NotImplementedError

    def exists(self, key: str) -> bool:                       # pragma: no cover
        raise NotImplementedError


def job_key(job_id: str, name: str) -> str:
    """Object key for a file belonging to a job."""
    return f"jobs/{job_id}/{name}"


class S3Store(ObjectStore):
    """MinIO/S3 backend.

    Credentials come from the environment (``MINIO_ACCESS_KEY`` /
    ``MINIO_SECRET_KEY``); they are never logged or persisted.
    """

    def __init__(self, bucket: str | None = None, endpoint: str | None = None):
        import boto3
        from botocore.config import Config

        self.bucket = bucket or bucket_from_env()
        self.endpoint = endpoint or os.environ.get("MINIO_ENDPOINT", DEFAULT_ENDPOINT)
        access = os.environ.get("MINIO_ACCESS_KEY")
        secret = os.environ.get("MINIO_SECRET_KEY")
        if not access or not secret:
            raise StorageError(
                "MINIO_ACCESS_KEY and MINIO_SECRET_KEY must be set in the environment"
            )
        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=access,
            aws_secret_access_key=secret,
            # MinIO needs path-style addressing
            config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 3}),
        )

    def read(self, key: str) -> bytes:
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:  # noqa: BLE001
            if _is_missing(exc):
                raise StorageError(
                    "The uploaded file could not be found in storage. "
                    "Please try uploading it again."
                ) from exc
            raise StorageError(f"could not read the uploaded file: {exc}") from exc
        return response["Body"].read()

    def write(self, key: str, data: bytes) -> None:
        try:
            self._client.put_object(Bucket=self.bucket, Key=key, Body=data)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"could not write {key}: {exc}") from exc

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:  # noqa: BLE001
            return False


def _is_missing(exc: Exception) -> bool:
    response = getattr(exc, "response", None) or {}
    code = str(response.get("Error", {}).get("Code", ""))
    return code in ("NoSuchKey", "404", "NoSuchBucket")


class LocalStore(ObjectStore):
    """Directory-backed store for ``--local-in``/``--local-out``.

    ``root`` is the directory that stands in for the bucket, so a key
    ``jobs/<id>/input.obj`` maps to ``<root>/jobs/<id>/input.obj``.
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
            raise StorageError(
                "The uploaded file could not be found in storage. "
                "Please try uploading it again."
            )
        return path.read_bytes()

    def write(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()
