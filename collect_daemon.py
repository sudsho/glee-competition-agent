"""Persistent throttled results collector.

Runs forever: fetches finished-game outcomes at ~5 requests/minute (the
runner needs the rest of the shared 60/min budget) and appends them to
logs/results.jsonl. Newest games first, so the live-quantile blend always
reflects the current pool. Safe to restart anytime; skips already-collected
ids.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from glee_sdk import GleeClient, GleeAPIError

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
RESULTS = LOGS / "results.jsonl"
SLEEP = 12.0        # ~5 requests/minute


def load_done():
    done = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["game_id"])
            except (KeyError, json.JSONDecodeError):
                pass
    return done


def main():
    key = (ROOT / "key.txt").read_text(encoding="utf-8").strip()
    client = GleeClient(api_key=key)
    done = load_done()
    print(f"collector daemon up; {len(done)} already collected", flush=True)

    while True:
        try:
            ids = [g for g in (LOGS / "game_ids.txt").read_text().split()
                   if g and g not in done]
        except OSError:
            ids = []
        if not ids:
            time.sleep(60)
            continue
        for gid in reversed(ids):          # newest first
            try:
                g = client.game_state(gid)
            except GleeAPIError:
                time.sleep(30)
                continue
            except Exception:
                time.sleep(60)
                continue
            status = g.get("status")
            if status == "active":
                # Not done yet: leave it in the queue for a later pass
                # (marking it done here would drop the game forever).
                time.sleep(SLEEP)
                continue
            done.add(gid)
            state = g.get("game_state") or {}
            rec = {
                "game_id": gid,
                "family": g.get("game_family"),
                "your_player": g.get("your_player"),
                "status": status,
                "result": g.get("result"),
                "opponent": g.get("opponent"),
                "rounds": state.get("round"),
                "cfg": {k: state.get(k) for k in (
                    "money_to_divide", "delta_1", "delta_2", "max_rounds",
                    "horizon_known", "complete_information", "messages_allowed",
                    "player_1_value", "player_2_value",
                    "p", "v", "u", "product_price", "total_rounds",
                    "seller_message_type") if state.get(k) is not None},
            }
            try:
                with open(RESULTS, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
            except OSError:
                pass
            time.sleep(SLEEP)


if __name__ == "__main__":
    main()
