"""Fetch outcomes of finished games for tuning and the competition paper.

Heavily throttled: the 60 req/min budget is shared with the live runner,
so this tool spends at most ~10 requests/min and stops after --max games
per invocation. Results accumulate in logs/results.jsonl.

Run: python collect_results.py [--max 60]
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


def main(max_fetch: int) -> None:
    key = (ROOT / "key.txt").read_text(encoding="utf-8").strip()
    client = GleeClient(api_key=key)

    done = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["game_id"])
            except (KeyError, json.JSONDecodeError):
                pass

    ids = []
    idfile = LOGS / "game_ids.txt"
    if idfile.exists():
        ids = [g for g in idfile.read_text().split() if g and g not in done]
    if "--recent" in sys.argv:
        ids = list(reversed(ids))

    print(f"{len(done)} already collected, {len(ids)} to check")
    fetched = 0
    with open(RESULTS, "a", encoding="utf-8") as out:
        for gid in ids:
            if fetched >= max_fetch:
                break
            try:
                g = client.game_state(gid)
            except GleeAPIError as e:
                print(f"{gid}: {e}")
                time.sleep(10)
                continue
            status = g.get("status")
            if status == "active":
                continue
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
            out.write(json.dumps(rec, default=str) + "\n")
            fetched += 1
            time.sleep(6)          # ~10 req/min, leave the runner its budget
    print(f"fetched {fetched} new results -> {RESULTS}")


if __name__ == "__main__":
    n = 60
    if "--max" in sys.argv:
        n = int(sys.argv[sys.argv.index("--max") + 1])
    main(n)
