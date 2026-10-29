"""
cloudflare_r2.py — Cloudflare R2 (S3 API) storage for scan jobs.

One bucket, one prefix per job:
    face_jobs/<job_id>/input/rig/0/f_150.jpg      face pair (uploaded by the Pi / laptop)
    face_jobs/<job_id>/input/calib/0/f_1000.jpg   free AprilTag disc photos for turntable_poses.py (optional)
    face_jobs/<job_id>/output/output.zip          deliverables + logs + stats (uploaded by the pod)
    face_jobs/<job_id>/output/status.json         written LAST by the pod: the completion signal
    face_jobs/<job_id>/output/run_log.txt         also outside the zip, so a failed run still reports why

The pod never gets the account keys: create_temp_credentials() mints short-lived S3 credentials
limited to one prefix (read-only on input/, read-write on output/).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import boto3
import requests
from boto3.exceptions import S3UploadFailedError
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

try:  # Pi / laptop: keys in .env (the pod gets them as environment variables)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("FaceScan.R2")

DEFAULT_BUCKET = "photogrammetry-pipeline"
JOBS_ROOT = "face_jobs"   # own prefix: object scans (Turntable Studio) may use jobs/ in the same bucket
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
CLOUDFLARE_API = "https://api.cloudflare.com/client/v4"


class R2Error(Exception):
    """Raised for R2 problems the caller must react to (failed deletes, failed listing, failed preflight)."""


def _build_r2_config() -> Config:
    """
    Build the botocore Config used for the R2 client.

    boto3 >= 1.36 sends CRC checksums by default, which R2 can reject (CLOUD-16); the "when_required" options
    turn that off. They do not exist on older botocore, which raises TypeError, so fall back to the options it knows.
    """
    base = dict(
        connect_timeout=10,  # Fail fast if unable to open socket
        read_timeout=60,  # Fail if transfer stalls (a full-resolution JPEG can take a while on a slow uplink)
        retries={'max_attempts': 1},  # Disable internal blind retries; our worker controls backoff
    )
    attempts = (
        dict(base, region_name="auto",
             request_checksum_calculation="when_required",
             response_checksum_validation="when_required"),
        dict(base, region_name="auto"),
        base,
    )
    for options in attempts[:-1]:
        try:
            return Config(**options)
        except TypeError:
            continue
    return Config(**attempts[-1])


R2_CONFIG = _build_r2_config()


# ── Client ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def make_client(endpoint_url: str, access_key_id: str, secret_access_key: str, session_token: Optional[str] = None):
    """S3 client for R2. session_token is required for the temporary credentials of create_temp_credentials()."""
    return boto3.client(
        service_name="s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        aws_session_token=session_token or None,
        config=R2_CONFIG,
    )


def client_from_env(prefix: str = "R2_"):
    """
    Client from <prefix>ENDPOINT_URL / ACCESS_KEY_ID / SECRET_ACCESS_KEY [/ SESSION_TOKEN].
    The endpoint is shared (R2_ENDPOINT_URL) when <prefix>ENDPOINT_URL is not set.
    """
    endpoint = os.environ.get(prefix + "ENDPOINT_URL") or os.environ.get("R2_ENDPOINT_URL")
    key_id = os.environ.get(prefix + "ACCESS_KEY_ID")
    secret = os.environ.get(prefix + "SECRET_ACCESS_KEY")
    missing = [n for n, v in (("R2_ENDPOINT_URL", endpoint), (prefix + "ACCESS_KEY_ID", key_id),
                               (prefix + "SECRET_ACCESS_KEY", secret)) if not (v or "").strip()]
    if missing:
        raise R2Error("Cloudflare R2 is not configured, missing environment variable(s): " + ", ".join(missing))
    return make_client(endpoint, key_id, secret, os.environ.get(prefix + "SESSION_TOKEN"))


def bucket_from_env() -> str:
    return os.environ.get("R2_BUCKET") or DEFAULT_BUCKET


_default_client = None


def _client(client=None):
    """The given client, or a lazily created one from the R2_* environment (Pi / laptop)."""
    global _default_client
    if client is not None:
        return client
    if _default_client is None:
        _default_client = client_from_env()
    return _default_client


# ── Keys ────────────────────────────────────────────────────────────────────────────────────────────────────────────

def new_job_id() -> str:
    """Sortable, unique job id: 20261027-153012-a1b2c3."""
    import time
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6]


def job_prefix(job_id: str) -> str:
    if not job_id or "/" in job_id or job_id in (".", ".."):
        raise ValueError(f"Invalid job id: {job_id!r}")
    return f"{JOBS_ROOT}/{job_id}/"


def input_prefix(job_id: str) -> str:
    return job_prefix(job_id) + "input/"


def output_prefix(job_id: str) -> str:
    return job_prefix(job_id) + "output/"


def input_key(job_id: str, relative_path: str) -> str:
    """input_key("j", "rig/0/f_1000.jpg") -> face_jobs/j/input/rig/0/f_1000.jpg"""
    return input_prefix(job_id) + relative_path.replace("\\", "/").lstrip("/")


def output_key(job_id: str, name: str) -> str:
    return output_prefix(job_id) + name


# ── Basic object operations ─────────────────────────────────────────────────────────────────────────────────────────

def upload_file(file_name: str, bucket: Optional[str] = None, object_name: Optional[str] = None, client=None) -> bool:
    """
    Upload a file. Never raises: every failure is logged and reported as False (the caller, e.g. the scan upload
    queue, decides whether to retry; the photo always stays on disk).
    """
    bucket = bucket or bucket_from_env()
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Do not touch the network for a file that is missing or empty (a 0-byte photo is a failed capture)
    try:
        size = os.path.getsize(file_name) if os.path.isfile(file_name) else None
    except OSError as e:
        logger.error("[R2] Cannot stat %s: %s", file_name, e)
        return False
    if size is None:
        logger.error("[R2] Not uploading %s: file does not exist", file_name)
        return False
    if size == 0:
        logger.error("[R2] Not uploading %s: file is empty (0 bytes)", file_name)
        return False

    try:
        _client(client).upload_file(file_name, bucket, object_name)
    except (ClientError, BotoCoreError, S3UploadFailedError, OSError, R2Error) as e:
        logger.error("[R2] Upload of %s as %s failed: %s", file_name, object_name, e)
        return False
    return True


def put_json(data: dict, object_name: str, bucket: Optional[str] = None, client=None) -> bool:
    """Small JSON object (status.json). Never raises."""
    try:
        _client(client).put_object(Bucket=bucket or bucket_from_env(), Key=object_name,
                                   Body=json.dumps(data, indent=2).encode("utf-8"), ContentType="application/json")
    except (ClientError, BotoCoreError, OSError, R2Error) as e:
        logger.error("[R2] Writing %s failed: %s", object_name, e)
        return False
    return True


def get_json(object_name: str, bucket: Optional[str] = None, client=None) -> Optional[dict]:
    """Small JSON object, or None when it does not exist (other errors are raised as R2Error)."""
    try:
        body = _client(client).get_object(Bucket=bucket or bucket_from_env(), Key=object_name)["Body"].read()
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            return None
        raise R2Error(f"Could not read {object_name}: {e}") from e
    except BotoCoreError as e:
        raise R2Error(f"Could not read {object_name}: {e}") from e
    return json.loads(body.decode("utf-8"))


def download_file(object_name: str, file_name: str, bucket: Optional[str] = None, client=None) -> bool:
    """Download one object. Never raises."""
    try:
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        _client(client).download_file(bucket or bucket_from_env(), object_name, file_name)
    except (ClientError, BotoCoreError, OSError, R2Error) as e:
        logger.error("[R2] Download of %s failed: %s", object_name, e)
        return False
    return True


def list_keys(prefix: str = "", bucket: Optional[str] = None, client=None) -> set[str]:
    """
    Every object key under `prefix` (paginated, so more than 1000 objects are handled).
    :raises R2Error: if the listing fails
    """
    bucket = bucket or bucket_from_env()
    keys: set[str] = set()
    try:
        paginator = _client(client).get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                keys.add(obj['Key'])
    except (ClientError, BotoCoreError) as e:
        raise R2Error(f"Could not list bucket '{bucket}': {e}") from e
    return keys


def verify_keys(expected: Iterable[str], bucket: Optional[str] = None, prefix: str = "", client=None) -> list[str]:
    """Sorted list of expected keys missing from the bucket (empty = everything is there). One listing."""
    present = list_keys(prefix=prefix, bucket=bucket, client=client)
    return sorted(set(expected) - present)


# delete_objects accepts at most 1000 keys per request
_DELETE_CHUNK = 1000


def delete_prefix(prefix: str, bucket: Optional[str] = None, client=None) -> int:
    """
    Delete every object under `prefix` (a job: face_jobs/<id>/). Refuses an empty prefix.
    :return: Number of objects deleted
    :raises R2Error: if listing or deleting fails, or if R2 reports per-key errors (the message lists the keys)
    """
    if not prefix or not prefix.startswith(JOBS_ROOT + "/"):
        raise R2Error(f"Refusing to delete outside '{JOBS_ROOT}/': {prefix!r}")
    bucket = bucket or bucket_from_env()
    cl = _client(client)
    keys = sorted(list_keys(prefix=prefix, bucket=bucket, client=cl))
    deleted = 0
    failed: list[str] = []
    for start in range(0, len(keys), _DELETE_CHUNK):
        chunk = keys[start:start + _DELETE_CHUNK]
        try:
            response = cl.delete_objects(Bucket=bucket, Delete={'Objects': [{'Key': k} for k in chunk], 'Quiet': True})
        except (ClientError, BotoCoreError) as e:
            raise R2Error(f"Could not delete objects from bucket '{bucket}': {e}") from e
        errors = response.get('Errors') or []
        failed.extend(err.get('Key', '?') for err in errors)
        deleted += len(chunk) - len(errors)
    if failed:
        raise R2Error(f"Failed to delete {len(failed)} object(s) from bucket '{bucket}': " + ", ".join(sorted(failed)))
    logger.info("[R2] Deleted %d object(s) under %s", deleted, prefix)
    return deleted


def preflight(bucket: Optional[str] = None, client=None, prefix: str = "_preflight/") -> None:
    """
    Verify that uploads to R2 will work before a scan starts: a small write + head + delete round trip on a
    throw-away key (credentials, bucket permissions, checksum behaviour CLOUD-16).
    :raises R2Error: with a human readable message when anything is wrong
    """
    bucket = bucket or bucket_from_env()
    cl = _client(client)
    key = f"{prefix}{uuid.uuid4().hex}.txt"
    written = False
    try:
        cl.put_object(Bucket=bucket, Key=key, Body=b"preflight")
        written = True
        cl.head_object(Bucket=bucket, Key=key)
    except (ClientError, BotoCoreError, OSError) as e:
        raise R2Error(
            f"R2 write test failed on bucket '{bucket}' (check credentials, bucket name and permissions): {e}"
        ) from e
    finally:
        if written:
            try:
                cl.delete_object(Bucket=bucket, Key=key)
            except (ClientError, BotoCoreError, OSError) as e:
                raise R2Error(
                    f"R2 delete test failed on bucket '{bucket}' (probe object '{key}' may be left behind): {e}"
                ) from e


# ── Job transfers ───────────────────────────────────────────────────────────────────────────────────────────────────

def local_image_files(input_dir: str) -> Dict[str, str]:
    """{relative path with '/': absolute path} of every image under input_dir (rig/0/..., rig/1/... or flat)."""
    root = Path(input_dir).resolve()
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and "masks" not in p.relative_to(root).parts:
            files[p.relative_to(root).as_posix()] = str(p)
    return files


def upload_job_images(job_id: str, input_dir: str, bucket: Optional[str] = None, client=None,
                      workers: int = 8) -> Dict[str, List[str]]:
    """Uploads every image of input_dir to face_jobs/<id>/input/<relative path>. Returns {"uploaded": [...], "failed": [...]}."""
    files = local_image_files(input_dir)
    cl = _client(client)

    def one(item):
        rel, path = item
        key = input_key(job_id, rel)
        return key, upload_file(path, bucket, key, client=cl)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        results = list(ex.map(one, files.items()))
    return {"uploaded": [k for k, ok in results if ok], "failed": [k for k, ok in results if not ok]}


def download_job_images(job_id: str, dest_dir: str, bucket: Optional[str] = None, client=None,
                        workers: int = 8) -> List[str]:
    """
    Downloads face_jobs/<id>/input/** into dest_dir keeping the relative layout (rig/0/f_1000.jpg).
    :raises R2Error: when nothing is there or any download fails
    """
    prefix = input_prefix(job_id)
    cl = _client(client)
    keys = sorted(list_keys(prefix=prefix, bucket=bucket, client=cl))
    if not keys:
        raise R2Error(f"No input images under {prefix}")
    dest = Path(dest_dir).resolve()

    def one(key):
        rel = key[len(prefix):]
        target = (dest / rel).resolve()
        if dest not in target.parents:
            return key, False  # a key like ../../x must not escape dest_dir
        return key, download_file(key, str(target), bucket, client=cl)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        results = list(ex.map(one, keys))
    failed = [k for k, ok in results if not ok]
    if failed:
        raise R2Error(f"{len(failed)} input image(s) could not be downloaded: " + ", ".join(failed[:5]))
    return [k for k, _ in results]


def download_job_output(job_id: str, dest_dir: str, bucket: Optional[str] = None, client=None) -> Path:
    """
    Downloads face_jobs/<id>/output/output.zip into dest_dir (a folder for this job only) and extracts it.
    Also fetches status.json and run_log.txt when present.
    :raises R2Error: when output.zip is missing or not a valid zip
    """
    dest = Path(dest_dir).resolve()
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    cl = _client(client)
    for name in ("status.json", "run_log.txt"):
        download_file(output_key(job_id, name), str(dest / name), bucket, client=cl)
    zip_path = dest / "output.zip"
    if not download_file(output_key(job_id, "output.zip"), str(zip_path), bucket, client=cl):
        raise R2Error(f"output.zip of job {job_id} could not be downloaded")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():  # no absolute paths or .. escaping dest
                target = (dest / member).resolve()
                if dest != target and dest not in target.parents:
                    raise R2Error(f"Unsafe path in output.zip: {member}")
            zf.extractall(dest)
    except zipfile.BadZipFile as e:
        raise R2Error(f"output.zip of job {job_id} is not a valid zip: {e}") from e
    return dest


def delete_stale_jobs(older_than_s: float, bucket: Optional[str] = None, client=None,
                      keep: Iterable[str] = (), now: Optional[float] = None) -> list[str]:
    """
    Deletes the folders face_jobs/<id>/ whose NEWEST object is older than `older_than_s` (photos of failed jobs that
    were kept for a retry, jobs interrupted on this machine). A running job writes started.json / status.json, and
    a pod never runs longer than its runtime cap, so a job untouched for longer than that cannot be running.
    Nothing outside face_jobs/ is touched. Returns the deleted job ids.
    """
    import time as _time
    bucket = bucket or bucket_from_env()
    cl = _client(client)
    newest: Dict[str, float] = {}
    try:
        for page in cl.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=JOBS_ROOT + "/"):
            for obj in page.get('Contents', []):
                parts = obj['Key'].split("/")
                if len(parts) < 3 or not parts[1]:
                    continue
                stamp = obj['LastModified'].timestamp()
                newest[parts[1]] = max(newest.get(parts[1], 0.0), stamp)
    except (ClientError, BotoCoreError) as e:
        raise R2Error(f"Could not list bucket '{bucket}': {e}") from e
    limit = (now if now is not None else _time.time()) - older_than_s
    stale = sorted(j for j, t in newest.items() if t < limit and j not in set(keep))
    for job_id in stale:
        delete_prefix(job_prefix(job_id), bucket, client=cl)
    return stale


# ── Temporary, prefix-scoped credentials for the pod ───────────────────────────────────────────────────────────────

@dataclass
class TempCredentials:
    access_key_id: str
    secret_access_key: str
    session_token: str

    def as_env(self, prefix: str) -> Dict[str, str]:
        return {
            prefix + "ACCESS_KEY_ID": self.access_key_id,
            prefix + "SECRET_ACCESS_KEY": self.secret_access_key,
            prefix + "SESSION_TOKEN": self.session_token,
        }


def temp_credentials_request(bucket: str, parent_access_key_id: str, permission: str,
                             prefixes: List[str], ttl_seconds: int) -> dict:
    """Body of POST /accounts/{account_id}/r2/temp-access-credentials."""
    if permission not in ("object-read-only", "object-read-write"):
        raise ValueError(f"Pod credentials must be object-level, got {permission!r}")
    if not prefixes or any(not p.startswith(JOBS_ROOT + "/") for p in prefixes):
        raise ValueError("Pod credentials must be limited to job prefixes")
    return {
        "bucket": bucket,
        "parentAccessKeyId": parent_access_key_id,
        "permission": permission,
        "prefixes": prefixes,
        "ttlSeconds": int(ttl_seconds),
    }


def create_temp_credentials(account_id: str, api_token: str, bucket: str, parent_access_key_id: str,
                            permission: str, prefixes: List[str], ttl_seconds: int,
                            session: Optional[requests.Session] = None) -> TempCredentials:
    """
    Short-lived S3 credentials limited to `prefixes` (Cloudflare R2 temp-access-credentials API).
    The API token needs only R2 permissions; it never leaves the Pi / laptop.
    """
    body = temp_credentials_request(bucket, parent_access_key_id, permission, prefixes, ttl_seconds)
    http = session or requests
    r = http.post(
        f"{CLOUDFLARE_API}/accounts/{account_id}/r2/temp-access-credentials",
        headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"},
        json=body,
        timeout=30,
    )
    try:
        payload = r.json()
    except ValueError:
        payload = {}
    if r.status_code >= 400 or not payload.get("success", False):
        raise R2Error(f"Cloudflare temp-access-credentials failed ({r.status_code}): {payload.get('errors') or r.text[:300]}")
    res = payload["result"]
    return TempCredentials(res["accessKeyId"], res["secretAccessKey"], res["sessionToken"])


def pod_credentials_env(job_id: str, ttl_seconds: int, account_id: Optional[str] = None,
                        api_token: Optional[str] = None, bucket: Optional[str] = None,
                        parent_access_key_id: Optional[str] = None) -> Dict[str, str]:
    """
    Environment for the pod: read-only credentials on input/, read-write on output/, both expiring after
    ttl_seconds, plus the endpoint and the bucket. Values default to CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_R2_API_TOKEN, R2_BUCKET and R2_ACCESS_KEY_ID.
    """
    account_id = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = api_token or os.environ.get("CLOUDFLARE_R2_API_TOKEN")
    parent = parent_access_key_id or os.environ.get("R2_ACCESS_KEY_ID")
    bucket = bucket or bucket_from_env()
    missing = [n for n, v in (("CLOUDFLARE_ACCOUNT_ID", account_id), ("CLOUDFLARE_R2_API_TOKEN", api_token),
                              ("R2_ACCESS_KEY_ID", parent), ("R2_ENDPOINT_URL", os.environ.get("R2_ENDPOINT_URL")))
               if not v]
    if missing:
        raise R2Error("Cannot create pod credentials, missing: " + ", ".join(missing))
    read = create_temp_credentials(account_id, api_token, bucket, parent, "object-read-only",
                                   [input_prefix(job_id)], ttl_seconds)
    write = create_temp_credentials(account_id, api_token, bucket, parent, "object-read-write",
                                    [output_prefix(job_id)], ttl_seconds)
    env = {"R2_ENDPOINT_URL": os.environ["R2_ENDPOINT_URL"], "R2_BUCKET": bucket}
    env.update(read.as_env("R2_IN_"))
    env.update(write.as_env("R2_OUT_"))
    return env
