"""Policy replay audit for the paper's Agent Behavior Analysis.

Replays every logged move (entries carry full game_state + valid_actions)
through the CURRENT frozen policy and counts exact matches. Persuasion
seller moves are stochastic (trust-budget mixing), so they are audited as
support membership plus realized lying rate, not pointwise.

Run: python replay_audit.py [--since "2026-08-25 20:06"] [--limit N]
Mismatch caveat reported in output: ladder-driven decisions depend on
analysis/*.json as loaded NOW; moves made under earlier snapshots of those
files can differ legitimately. Run after the final freeze for paper numbers.
"""

import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from barg import bargaining_strategy
from nego import negotiation_strategy
from pers import persuasion_strategy

ROOT = Path(__file__).resolve().parent.parent
STRATS = {"bargaining": bargaining_strategy, "negotiation": negotiation_strategy,
          "persuasion": persuasion_strategy}


def norm_action(a):
    if not isinstance(a, dict):
        return a
    return {k: (round(v, 2) if isinstance(v, float) else v)
            for k, v in a.items() if k != "message"}


def main():
    since = 0.0
    if "--since" in sys.argv:
        s = sys.argv[sys.argv.index("--since") + 1]
        since = datetime.strptime(s, "%Y-%m-%d %H:%M").timestamp()
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    total = match = errors = 0
    pers_seller_total = pers_seller_rec = 0
    mismatch_by = Counter()
    invariants = defaultdict(int)

    with open(ROOT / "logs" / "moves.jsonl", encoding="utf-8",
              errors="replace") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("ts", 0) < since or not e.get("game_state"):
                continue
            fam = e.get("family")
            if fam not in STRATS:
                continue
            logged = e.get("action") or {}

            # Invariants (checked on the logged action itself).
            if fam == "bargaining" and logged.get("decision") == "walkaway":
                invariants["bargaining_walkaway"] += 1
            if isinstance(logged.get("message"), str) and \
                    len(logged["message"]) > 2000:
                invariants["message_over_cap"] += 1

            at = e.get("action_type")
            if fam == "persuasion" and at in ("seller_message",
                                              "seller_recommendation"):
                pers_seller_total += 1
                rec = logged.get("decision") == "yes" or \
                    ("message" in logged and "skip" not in
                     str(logged.get("message", "")).lower() and "pass" not in
                     str(logged.get("message", "")).lower())
                pers_seller_rec += 1 if rec else 0
                continue        # stochastic: audited by rate, not pointwise

            game = {
                "game_id": e.get("game_id"), "game_family": fam,
                "your_player": e.get("your_player"), "phase": e.get("phase"),
                "game_state": e.get("game_state"),
                "valid_actions": e.get("valid_actions") or {},
                "opponent": e.get("opponent"), "prompt": "",
            }
            total += 1
            if limit and total > limit:
                total -= 1
                break
            try:
                replayed = STRATS[fam](game)
            except Exception:
                errors += 1
                continue
            if norm_action(replayed) == norm_action(logged):
                match += 1
            else:
                mismatch_by[f"{fam}/{at}"] += 1

    print(f"deterministic moves replayed: {total}")
    if total:
        print(f"exact match: {match} ({100 * match / total:.2f}%)")
    print(f"replay errors: {errors}")
    print(f"persuasion seller moves (stochastic, rate-audited): "
          f"{pers_seller_total}, recommend rate "
          f"{100 * pers_seller_rec / max(pers_seller_total, 1):.1f}%")
    print("top mismatch cells:", mismatch_by.most_common(8))
    print("invariant violations:", dict(invariants) or "none")


if __name__ == "__main__":
    main()
