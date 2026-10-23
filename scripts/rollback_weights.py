"""Inspect or roll back the learned weights using the learning ledger.

The weights are pure counts, so the ledger (data/learning_ledger.jsonl) reproduces every past
state exactly. Rolling back is append-only: the rollback is itself recorded.

    python scripts/rollback_weights.py                 # show the current version + recent updates
    python scripts/rollback_weights.py --to 120        # DRY RUN: what version 120 looked like
    python scripts/rollback_weights.py --to 120 --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.learning.bayesian_updater import (  # noqa: E402
    BayesianUpdater,
    effective_updates,
    read_ledger,
    replay_ledger,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--to", type=int, help="target version (number of updates to keep)")
    parser.add_argument("--apply", action="store_true", help="write the rolled-back weights (default: dry run)")
    args = parser.parse_args()

    updater = BayesianUpdater()
    events = read_ledger(updater.ledger_path())
    _, updates = effective_updates(events)
    print(f"ledger: {updater.ledger_path()}  |  current version: {len(updates)}")
    if args.to is None:
        for e in updates[-10:]:
            print(f"  {e['ts']}  {e['business']:<24} {e['feedback']:<5} {e.get('prompt_type')}/{e.get('format')}  ({e.get('source')})")
        return 0
    if not 0 <= args.to <= len(updates):
        print(f"error: --to must be between 0 and {len(updates)}", file=sys.stderr)
        return 2
    target = replay_ledger(events, upto=args.to)
    print(f"version {args.to} would hold: {json.dumps({k: {d: len(v) for d, v in biz.items()} for k, biz in target.items()})}")
    if args.apply:
        updater.rollback(args.to)
        print(f"rolled back to version {args.to}. Commit data/weights.json and data/learning_ledger.jsonl.")
    else:
        print("dry run: nothing written (add --apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
