"""PII redaction for anything that leaves the process towards a third-party model (audit H-01).

Business profiles are scraped from websites and contain contact e-mails and phone numbers;
the weekly report prompt used to include the client's own address. None of that is needed
to write a caption, so it is stripped (keys) and masked (free text) before any LLM call.
"""
from __future__ import annotations

import re
from typing import Any

PII_KEYS = frozenset({
    "email", "emails", "contact_email", "contact_emails", "e_mail", "mail",
    "phone", "phones", "telephone", "mobile", "whatsapp", "contact_phone",
})
# Quantifiers are BOUNDED (RFC 5321: local part <= 64, domain <= 255, phone numbers <= ~20 chars). An
# unbounded `[...]+@` re-scans the whole remaining run from every start position: a 50 KB token with no
# '@' took 11 s. Bounded, the worst case is O(64 x n).
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9.\-]{1,255}\.[A-Za-z]{2,24}")
# +91 98765 43210 / (079) 2345-6789 / 98765-43210 ... at least 9 digits once separators are removed
PHONE_RE = re.compile(r"(?<![\w.])\+?\d[\d\s().\-]{7,20}\d(?![\w])")


def redact_text(text: str) -> str:
    text = EMAIL_RE.sub("[email removed]", text)

    def _mask(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        return "[phone removed]" if 9 <= len(digits) <= 15 else match.group(0)

    return PHONE_RE.sub(_mask, text)


def redact(value: Any) -> Any:
    """Deep copy with PII keys dropped and PII-looking substrings masked."""
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items() if str(k).lower() not in PII_KEYS}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return redact_text(value)
    return value
