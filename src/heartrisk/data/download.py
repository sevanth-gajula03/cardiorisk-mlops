"""Download the Cleveland heart disease data from the UCI repository.

The official UCI archive is tried first; if it is unreachable the direct
file link is used. A MANIFEST.json is written next to the raw file with the
source, timestamp and SHA-256 so a later run can tell if the data changed.

Usage:
    python -m heartrisk.data.download [--force]
"""

import argparse
import hashlib
import io
import json
import logging
import ssl
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone

import truststore

from heartrisk.config import load_config, resolve

log = logging.getLogger(__name__)
TIMEOUT_SECONDS = 30

# Verify TLS against the operating system's certificate store so the download
# also works behind corporate proxies that re-sign HTTPS traffic.
SSL_CONTEXT = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS, context=SSL_CONTEXT) as resp:
        return resp.read()


def fetch_from_archive(url: str, member: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(fetch(url))) as zf:
        return zf.read(member)


def download(force: bool = False) -> dict:
    cfg = load_config()["data"]
    raw_path = resolve(cfg["raw_file"])
    manifest_path = resolve(cfg["manifest_file"])

    if raw_path.exists() and not force:
        log.info("Raw file already present at %s (use --force to refresh)", raw_path)
        return json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    try:
        payload = fetch_from_archive(cfg["source_url"], cfg["member"])
        used_url = cfg["source_url"]
    except (urllib.error.URLError, zipfile.BadZipFile, KeyError) as exc:
        log.warning("Archive download failed (%s); trying direct link", exc)
        payload = fetch(cfg["fallback_url"])
        used_url = cfg["fallback_url"]

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(payload)

    manifest = {
        "source": used_url,
        "file": cfg["member"],
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bytes": len(payload),
        "rows": payload.decode().strip().count("\n") + 1,
        "sha256": sha256_of(payload),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Saved %d rows to %s", manifest["rows"], raw_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    download(force=args.force)


if __name__ == "__main__":
    main()
