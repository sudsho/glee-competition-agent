"""Capture the close-of-competition record for the paper and prizes.

Saves analysis/final_snapshot.json with: the full agent leaderboard, our
account's entries, all four agents' stats, and season totals from the
logs. Run immediately at (and shortly after) the close.
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
from glee_sdk import GleeClient

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis" / "final_snapshot.json"

KEYS = [("sudsho", "key.txt"), ("Test 1", "keys/test1.txt"),
        ("Test 2", "keys/test2.txt"), ("Test 3", "keys/test3.txt")]


def main():
    snap = {"captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "captured_unix": time.time()}

    lb = requests.get(
        "https://glee-competition.com/api/leaderboard/overall",
        params={"player_type": "agent", "show_baselines": "true",
                "limit": 200}, timeout=60).json()
    snap["leaderboard_top200"] = lb
    snap["our_entries"] = [r for r in lb if r.get("player_name") in
                           ("sudsho", "Test 1", "Test 2", "Test 3")]

    snap["agents"] = {}
    for name, kf in KEYS:
        try:
            snap["agents"][name] = GleeClient(
                api_key=(ROOT / kf).read_text().strip()).stats()
        except Exception as exc:
            snap["agents"][name] = {"error": str(exc)}

    fam_moves = Counter()
    games = set()
    with open(ROOT / "logs" / "moves.jsonl", encoding="utf-8",
              errors="replace") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            fam_moves[e.get("family")] += 1
            games.add(e.get("game_id"))
    snap["main_agent_log_totals"] = {
        "moves_by_family": dict(fam_moves),
        "distinct_games_in_moves_log": len(games),
    }

    OUT.write_text(json.dumps(snap, indent=1, default=str), encoding="utf-8")
    print(f"snapshot -> {OUT}")
    for r in snap["our_entries"]:
        print(" ", r.get("rank"), r.get("player_name"),
              round(r.get("rating", 0), 1), "owner_best:",
              r.get("is_owner_best"))


if __name__ == "__main__":
    main()
