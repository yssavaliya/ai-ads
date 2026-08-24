"""Verify the LLM + creative pipeline works with the repo's real API keys.

This bypasses GitHub Actions so you can confirm prompts generate locally before
relying on the scheduled workflow. Set GROQ_API_KEY (and optionally
GEMINI_API_KEY, POLLINATIONS_API_KEY) in your environment, then:

    python scripts/check_llm.py

Exit code 0 = at least one provider produced a prompt. Non-zero = all failed.
"""

from __future__ import annotations

import os
import sys

from src.shared.llm_client import LLMClient, LLMResult
from src.shared.config import settings

TASK = "audience-mapper"
TEST_PROMPT = (
    "Business: Aimagix Studio — AI content studio in Ahmedabad. "
    "Niche: AI ad creative. Audience: small business owners. "
    "Write one short, viral Instagram caption hook (max 20 words)."
)


def main() -> int:
    print("GROQ_API_KEY set:   ", bool(os.environ.get("GROQ_API_KEY") or settings.groq_api_key))
    print("GEMINI_API_KEY set: ", bool(os.environ.get("GEMINI_API_KEY") or settings.gemini_api_key))
    print("POLLINATIONS:       keyless (free tier, no key required)")
    print("-" * 60)

    client = LLMClient()
    providers = ["groq", "gemini", "pollinations"]
    ok = False
    for provider in providers:
        print(f"\n=== Trying provider: {provider} ===")
        try:
            result: LLMResult = client.run(
                provider=provider,
                task=TASK,
                prompt=TEST_PROMPT,
                timeout=90,
                max_retries=2,
            )
            if result.success:
                ok = True
                text = result.stdout.strip()
                print(f"SUCCESS ({result.model or 'default'})")
                print("OUTPUT:", text[:300])
            else:
                print("FAILED")
                print("STDERR:", (result.stderr or "").strip()[:500])
        except Exception as exc:  # noqa: BLE001
            print(f"EXCEPTION: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 60)
    if ok:
        print("RESULT: at least one LLM provider produced output. Pipeline is healthy.")
        return 0
    print("RESULT: ALL LLM providers failed. Check the secrets in repo Settings > Secrets.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
