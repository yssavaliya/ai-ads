"""Verify the LLM + creative pipeline works with the repo's real API keys.

This bypasses GitHub Actions so you can confirm prompts generate locally before
relying on the scheduled workflow. Set GROQ_API_KEY / GEMINI_API_KEY in your
environment, then:

    PYTHONPATH=. python scripts/check_llm.py

It exercises the SAME code paths the pipeline uses (LLMClient._try_groq /
_try_gemini), so a pass here means the workflow will also
produce prompts. Exit code 0 = at least one provider produced output.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")

from src.shared.llm_client import LLMClient  # noqa: E402

PROMPT = (
    "Business: Aimagix Studio, an AI content studio in Ahmedabad. "
    "Write one short, viral Instagram caption hook (max 15 words)."
)


def main() -> int:
    print("GROQ_API_KEY set:   ", bool(os.environ.get("GROQ_API_KEY")))
    print("GEMINI_API_KEY set: ", bool(os.environ.get("GEMINI_API_KEY")))
    print("POLLINATIONS:       keyless (free tier, no key required)")
    print("-" * 60)

    client = LLMClient()
    ok = False
    for name, method in [
        ("groq", client._try_groq),
        ("gemini", client._try_gemini),
    ]:
        print(f"\n=== {name} ===")
        try:
            res = method(PROMPT, "check", 0.7)
        except Exception as exc:  # noqa: BLE001
            print(f"EXCEPTION: {type(exc).__name__}: {str(exc).splitlines()[0]}")
            continue
        if res is None:
            print("FAILED (no result — see ERROR log line above for the reason)")
        elif res.stdout and res.stdout.strip():
            ok = True
            print("OK ->", res.stdout.strip()[:200])
        else:
            print("FAILED (empty output)")

    print("\n" + "=" * 60)
    if ok:
        print("RESULT: at least one LLM provider produced output. Pipeline is healthy.")
        return 0
    print("RESULT: ALL LLM providers failed. Check the ERROR lines above (likely model/key).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
