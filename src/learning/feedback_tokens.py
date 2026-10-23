"""Signed feedback tokens (audit C-03).

A feedback link used to identify its prompt by a bare, sequential SQLite id that is
recycled whenever the Actions-cached database is evicted, so a stale or hand-written
issue could mutate the wrong prompt or the learned weights. Links now carry
`<uid>.<signature>`: the prompt's durable random uid plus an HMAC bound to the button
kind, so a token cannot be re-pointed at another prompt or turned from GOOD into BAD.
"""
from __future__ import annotations

import hashlib
import hmac
import re

from src.shared.config import settings

KINDS = ("good", "bad", "regenerate", "posted")
TOKEN_RE = re.compile(r"\bt:([0-9a-f]{32})\.([0-9a-f]{20})\b")
SIG_HEX_CHARS = 20  # 80 bits: ample against online guessing through GitHub issues


def _key() -> bytes | None:
    key = settings.feedback_signing_key
    return key.encode() if key else None


def enabled() -> bool:
    return _key() is not None


def _sig(uid: str, kind: str, key: bytes) -> str:
    return hmac.new(key, f"{uid}|{kind}".encode(), hashlib.sha256).hexdigest()[:SIG_HEX_CHARS]


def sign(uid: str, kind: str) -> str | None:
    """`t:<uid>.<sig>` for the issue title, or None when no signing key is configured."""
    key = _key()
    if key is None or kind not in KINDS or not re.fullmatch(r"[0-9a-f]{32}", uid or ""):
        return None
    return f"t:{uid}.{_sig(uid, kind, key)}"


def verify(text: str, kind: str) -> str | None:
    """The prompt uid if `text` carries a valid token for `kind`, else None."""
    key = _key()
    match = TOKEN_RE.search(text or "")
    if key is None or match is None or kind not in KINDS:
        return None
    uid, signature = match.groups()
    return uid if hmac.compare_digest(signature, _sig(uid, kind, key)) else None
