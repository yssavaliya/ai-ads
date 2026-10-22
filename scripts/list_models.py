"""List the models each configured provider actually offers this API key.

Free-tier model names rotate and access differs per account (a Groq org may get a 404
for a model another org can use), so default fallback lists are only guesses. Run this
(locally, or via the Drift Check workflow, which prints it in its log) and put the ids
you want into GROQ_FALLBACK_MODELS / GEMINI_FALLBACK_MODELS / OPENROUTER_MODELS / ...

    python scripts/list_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from src.shared import llm_providers  # noqa: E402


def list_models(provider: llm_providers.Provider) -> list[str]:
    headers = {"Authorization": f"Bearer {provider.api_key}", **provider.headers}
    resp = httpx.get(provider.base_url.rstrip("/") + "/models", headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    items = payload.get("data", payload) if isinstance(payload, dict) else payload
    ids = []
    for item in items if isinstance(items, list) else []:
        ident = item.get("id") if isinstance(item, dict) else str(item)
        if ident:
            ids.append(str(ident).removeprefix("models/"))
    return sorted(ids)


def main() -> int:
    shown = 0
    for provider in llm_providers.configured_providers():
        if provider.api_key == "keyless":
            continue
        print(f"\n=== {provider.name} (configured: {', '.join(provider.models)}) ===")
        try:
            ids = list_models(provider)
        except Exception as exc:  # noqa: BLE001 - diagnostics must never fail the workflow
            print(f"could not list models: {type(exc).__name__}: {str(exc)[:200]}")
            continue
        shown += 1
        if provider.name == "openrouter":
            ids = [i for i in ids if i.endswith(":free")]
        for ident in ids:
            mark = "*" if ident in provider.models else " "
            print(f" {mark} {ident}")
        missing = [m for m in provider.models if m not in ids]
        if missing:
            print(f"::warning::{provider.name}: configured but not offered to this key: {', '.join(missing)}")
    print("\n(* = in the configured fallback list)")
    return 0 if shown or not llm_providers.configured_providers() else 0


if __name__ == "__main__":
    sys.exit(main())
