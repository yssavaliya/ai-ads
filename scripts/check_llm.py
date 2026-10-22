"""Verify the LLM provider chain works with the repo's real API keys.

This bypasses the pipelines so you can confirm prompts generate before relying on
the scheduled workflows (the Drift Check workflow runs it daily). It probes EVERY
configured provider through the same code path the pipelines use, so a pass here
means the workflows will produce prompts, and the report shows which fallbacks are
healthy.

    python scripts/check_llm.py

Exit code 0 = at least one provider produced output. Providers that fail while
others work are reported as warnings (the chain still has a working fallback).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")

from src.shared import llm_providers  # noqa: E402
from src.shared.llm_client import LLMClient  # noqa: E402

PROMPT = (
    "Business: Aimagix Studio, an AI content studio in Ahmedabad. "
    "Write one short, viral Instagram caption hook (max 15 words)."
)


def main() -> int:
    configured = llm_providers.configured_providers()
    print("Provider chain:", llm_providers.describe_setup())
    print("-" * 60)
    if not configured:
        print("No provider is configured. Set GROQ_API_KEY (free at console.groq.com) or any other key.")
        return 1

    client = LLMClient()
    healthy: list[str] = []
    failing: list[str] = []
    for provider in configured:
        print(f"\n=== {provider.name} ({', '.join(provider.models)}) ===")
        try:
            res = client._try_provider(provider.name, PROMPT, "check", 0.7, 1)  # one attempt: a health probe
        except Exception as exc:  # noqa: BLE001
            print(f"EXCEPTION: {type(exc).__name__}: {str(exc).splitlines()[0]}")
            failing.append(provider.name)
            continue
        if res is not None and res.stdout.strip():
            healthy.append(provider.name)
            print(f"OK via {res.model} -> {res.stdout.strip()[:200]}")
        else:
            failing.append(provider.name)
            print("FAILED (no usable answer — see the ERROR log line above for the reason)")

    print("\n" + "=" * 60)
    if failing:
        print(f"::warning::providers failing: {', '.join(failing)}")
    if healthy:
        print(f"RESULT: {len(healthy)}/{len(configured)} providers healthy ({', '.join(healthy)}). Pipeline is healthy.")
        return 0
    print("RESULT: ALL LLM providers failed. Check the ERROR lines above (likely model/key).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
