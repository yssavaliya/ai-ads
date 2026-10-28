"""Diagnostic tool to verify Buffer API credentials and list connected social profiles.

Uses Buffer's modern GraphQL API (https://api.buffer.com).

Usage:
    # If BUFFER_ACCESS_TOKEN is set in environment or .env:
    python scripts/check_buffer.py

    # Pass token directly:
    python scripts/check_buffer.py --token "your_buffer_token"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.shared.config import settings  # noqa: E402

BUFFER_GQL_URL = "https://api.buffer.com"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Buffer credentials and list profile IDs.")
    parser.add_argument("--token", type=str, help="Buffer Access Token")
    args = parser.parse_args()

    token = args.token or getattr(settings, "buffer_access_token", None) or os.getenv("BUFFER_ACCESS_TOKEN")
    if not token:
        print("Error: No Buffer Access Token found.")
        print("Set BUFFER_ACCESS_TOKEN in .env or pass via --token.")
        sys.exit(1)

    print(f"Checking Buffer GraphQL API with token: {token[:6]}...{token[-4:]}")

    # 1. Fetch organization ID
    account_query = """
    query {
      account {
        id
        email
        organizations {
          id
          name
        }
      }
    }
    """
    try:
        resp = httpx.post(
            BUFFER_GQL_URL,
            headers={"Authorization": f"Bearer {token}"},
            json={"query": account_query},
            timeout=30,
        )
        if resp.status_code == 401:
            print("❌ Authentication failed (401 Unauthorized). Please check your BUFFER_ACCESS_TOKEN.")
            sys.exit(1)
        resp.raise_for_status()
        payload = resp.json()
        if "errors" in payload:
            print("❌ Buffer API returned errors:", payload["errors"])
            sys.exit(1)
    except Exception as exc:
        print(f"❌ Connection error: {exc}")
        sys.exit(1)

    account = payload.get("data", {}).get("account", {})
    email = account.get("email", "unknown")
    orgs = account.get("organizations", [])
    if not orgs:
        print(f"✅ Connected to Buffer as {email}, but no organization found.")
        return

    org_id = orgs[0].get("id")
    org_name = orgs[0].get("name", "Organization")
    print(f"✅ Authenticated as: {email} (Org: {org_name}, ID: {org_id})")

    # 2. Fetch connected channels
    channels_query = """
    query GetChannels($input: ChannelsInput!) {
      channels(input: $input) {
        id
        name
        service
      }
    }
    """
    resp_channels = httpx.post(
        BUFFER_GQL_URL,
        headers={"Authorization": f"Bearer {token}"},
        json={"query": channels_query, "variables": {"input": {"organizationId": org_id}}},
        timeout=30,
    )
    channels_data = resp_channels.json().get("data", {}).get("channels", [])

    if not channels_data:
        print("\n⚠️ Connected to Buffer, but no social channels found.")
        print("Please connect your Instagram or LinkedIn account at https://buffer.com")
        return

    print(f"\nFound {len(channels_data)} connected social channel(s):\n")
    print(f"{'SERVICE':<15} {'CHANNEL NAME':<25} {'CHANNEL ID / PROFILE ID'}")
    print("-" * 70)
    for c in channels_data:
        service = c.get("service", "unknown")
        name = c.get("name", "unknown")
        cid = c.get("id", "")
        print(f"{service:<15} {name:<25} {cid}")

    print("\nTo set your channel in GitHub Actions:")
    print(f"  gh secret set BUFFER_PROFILE_ID --body \"{channels_data[0]['id']}\" -R yssavaliya/ai-ads\n")


if __name__ == "__main__":
    main()
