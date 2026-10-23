"""Encrypt/decrypt the private GitHub Actions state cache."""
from __future__ import annotations

import base64
import hashlib
import io
import os
import tarfile
from pathlib import Path, PurePosixPath

from cryptography.fernet import Fernet, InvalidToken

CACHE_FILE = Path(".agent-state.enc")
ALLOWED = {"agent.db", "agent.db-wal", "agent.db-shm"}


def _fernet() -> Fernet:
    raw = os.environ.get("STATE_ENCRYPTION_KEY", "").strip()
    if not raw:
        raw = os.environ.get("FEEDBACK_SIGNING_KEY", "").strip()
    key = base64.urlsafe_b64encode(hashlib.sha256(raw.encode()).digest()) if raw else b""
    if not key:
        raise SystemExit("STATE_ENCRYPTION_KEY is required to access the private state cache")
    try:
        return Fernet(key)
    except (ValueError, TypeError) as exc:
        raise SystemExit("Could not derive a valid key for the private state cache") from exc


def encrypt() -> None:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for name in sorted(ALLOWED):
            path = Path(name)
            if path.is_file():
                archive.add(path, arcname=name, recursive=False)
        cache_dir = Path("data/cache")
        if cache_dir.is_dir():
            for path in sorted(p for p in cache_dir.rglob("*") if p.is_file()):
                archive.add(path, arcname=path.as_posix(), recursive=False)
    CACHE_FILE.write_bytes(_fernet().encrypt(payload.getvalue()))


def decrypt() -> None:
    try:
        data = _fernet().decrypt(CACHE_FILE.read_bytes())
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not (
                    member.name in ALLOWED or member.name.startswith("data/cache/")
                ) or not member.isfile():
                    raise SystemExit(f"Refusing unexpected cache entry: {member.name}")
                destination = Path(*path.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise SystemExit(f"Missing cache payload: {member.name}")
                destination.write_bytes(source.read())
    except (OSError, InvalidToken, tarfile.TarError) as exc:
        raise SystemExit(f"Could not restore encrypted state cache: {exc}") from exc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("encrypt", "decrypt"))
    mode = parser.parse_args().mode
    if mode == "encrypt":
        encrypt()
    else:
        decrypt()
