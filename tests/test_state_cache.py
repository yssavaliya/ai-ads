from __future__ import annotations

from pathlib import Path

import pytest

from scripts import state_cache


def test_encrypted_state_round_trips_without_plaintext(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STATE_ENCRYPTION_KEY", "a long private key value")
    Path("agent.db").write_bytes(b"private sqlite data")
    Path("data/cache").mkdir(parents=True)
    Path("data/cache/profile.json").write_text('{"email":"private@example.com"}')
    Path("data/site_manifests").mkdir(parents=True)
    Path("data/site_manifests/biz.json").write_text('{"path_key":"secret-path-key"}')

    state_cache.encrypt()
    encrypted = state_cache.CACHE_FILE.read_bytes()
    assert b"private sqlite data" not in encrypted
    assert b"private@example.com" not in encrypted
    assert b"secret-path-key" not in encrypted

    Path("agent.db").unlink()
    Path("data/cache/profile.json").unlink()
    Path("data/site_manifests/biz.json").unlink()
    state_cache.decrypt()
    assert Path("agent.db").read_bytes() == b"private sqlite data"
    assert "private@example.com" in Path("data/cache/profile.json").read_text()
    assert "secret-path-key" in Path("data/site_manifests/biz.json").read_text()


def test_state_cache_requires_a_secret(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("STATE_ENCRYPTION_KEY", raising=False)
    monkeypatch.delenv("FEEDBACK_SIGNING_KEY", raising=False)
    with pytest.raises(SystemExit, match="required"):
        state_cache.encrypt()
