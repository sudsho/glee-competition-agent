"""Harvest finished-game results from runner.log at zero API cost.

The runner logs 'Game <id> finished! Result: {...}' for every game that
ends while we hold the last move's connection; joined with moves.jsonl
(family, role, opponent, cfg from game_state) these are exact outcomes,
validated 2,313/2,313 against the API. Run at every checkup before
build_live_quantiles.py; the API collector daemon fills the remainder.
"""

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
RESULTS = LOGS / "results.jsonl"

CFG_KEYS = ("money_to_divide", "delta_1", "delta_2", "max_rounds",
            "horizon_known", "complete_information", "messages_allowed",
            "player_1_value", "player_2_value",
            "p", "v", "u", "product_price", "total_rounds",
            "seller_message_type")

FIN = re.compile(r"INFO Game ([0-9a-f-]{36}) .*finished! Result: (\{.*\})\s*$")


def main():
    done = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["game_id"])
            except (KeyError, json.JSONDecodeError):
                pass

    # Latest move context per game.
    ctx = {}
    with open(LOGS / "moves.jsonl", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            gid = e.get("game_id")
            if not gid:
                continue
            st = e.get("game_state") or {}
            ctx[gid] = {
                "family": e.get("family"),
                "your_player": e.get("your_player"),
                "opponent": e.get("opponent"),
                "rounds": e.get("round"),
                "ts": e.get("ts"),
                "action": e.get("action"),
                "action_type": e.get("action_type"),
                "cfg": {k: st.get(k) for k in CFG_KEYS
                        if st.get(k) is not None},
            }

    added = 0
    with open(LOGS / "runner.log", encoding="utf-8", errors="replace") as f, \
            open(RESULTS, "a", encoding="utf-8") as out:
        for line in f:
            m = FIN.search(line)
            if not m:
                continue
            gid = m.group(1)
            if gid in done:
                continue
            try:
                result = ast.literal_eval(m.group(2))
            except (ValueError, SyntaxError):
                continue
            c = ctx.get(gid)
            if not c or not c.get("family"):
                continue
            done.add(gid)
            out.write(json.dumps({
                "game_id": gid,
                "family": c["family"],
                "your_player": c["your_player"],
                "status": "completed",
                "result": result,
                "opponent": c["opponent"],
                "rounds": c["rounds"],
                "cfg": c["cfg"],
                "source": "log",
            }, default=str) + "\n")
            added += 1

    # Second pass: infer outcomes for stale games whose last logged move
    # was our own non-final-round offer. We never moved again, so the
    # opponent's accept ended the game before the runner could log it:
    # these are OUR WINS, and omitting them loss-biases every live pool
    # (validated ~90-98% exact against API truth).
    import time as _time
    now = _time.time()
    inferred = 0
    with open(RESULTS, "a", encoding="utf-8") as out:
        for gid, c in ctx.items():
            if gid in done or not c.get("family"):
                continue
            ts = c.get("ts") or now
            if now - ts < 3600:
                continue                     # possibly still active
            act = c.get("action") or {}
            cfg = c.get("cfg") or {}
            t = int(c.get("rounds") or 0)
            T = cfg.get("max_rounds")
            if T is not None and t >= int(T):
                continue                     # final-round offer: ambiguous
            me = c.get("your_player")
            result = None
            if c["family"] == "bargaining" and "alice_gain" in act:
                d1, d2 = cfg.get("delta_1"), cfg.get("delta_2")
                p1 = act.get("alice_gain")
                p2 = act.get("bob_gain")
                result = {
                    "player_1_payoff": (float(p1) * float(d1) ** (t - 1)
                                        if d1 is not None else None),
                    "player_2_payoff": (float(p2) * float(d2) ** (t - 1)
                                        if d2 is not None else None),
                    "outcome": "agreement",
                    "agreed_round": t,
                }
            elif c["family"] == "negotiation" and \
                    act.get("product_price") is not None:
                price = float(act["product_price"])
                v1 = cfg.get("player_1_value")
                v2 = cfg.get("player_2_value")
                result = {
                    "player_1_payoff": (price - float(v1)
                                        if v1 is not None else None),
                    "player_2_payoff": (float(v2) - price
                                        if v2 is not None else None),
                    "outcome": "agreement",
                    "agreed_price": price,
                    "agreed_round": t,
                }
            if result is None:
                continue
            done.add(gid)
            out.write(json.dumps({
                "game_id": gid, "family": c["family"], "your_player": me,
                "status": "completed", "result": result,
                "opponent": c.get("opponent"), "rounds": t,
                "cfg": cfg, "source": "inferred",
            }, default=str) + "\n")
            inferred += 1

    print(f"harvested {added} results from runner.log, inferred {inferred} "
          f"opponent-ended wins ({len(done)} total collected)")


if __name__ == "__main__":
    main()
