"""Build drift-corrected quantiles from observed competition outcomes.

The static percentile tables describe the public 2025 research corpus, but
the live scoring pool is dominated by competition games and keeps
strengthening. This script distills logs/results.jsonl into normalized
payoff quantiles at exactly the aggregation keys targets.py looks up, so
Ladder.q can blend static and live views.

Keys (must mirror targets.py):
  barg|{role}|{d_me:.2f}|{d_op:.2f or h}|{tclass}
  nego|{role}|{my_f:.2f}|{tclass}
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "logs" / "results.jsonl"
OUT = ROOT / "analysis" / "live_quantiles.json"
QS = [5, 10, 25, 50, 75, 90, 95, 99]


def tclass(max_rounds):
    try:
        v = float(max_rounds)
    except (TypeError, ValueError):
        return "inf"
    return "inf" if v >= 90 else f"T{int(v)}"


def scale(myv):
    for k in range(0, 13):
        m = 10.0 ** k
        if 0.75 <= myv / m <= 1.55:
            return m
    return myv


def main():
    pools = defaultdict(list)
    n_recs = 0
    seen = set()
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        gid = r.get("game_id")
        if gid in seen:
            continue
        seen.add(gid)
        res = r.get("result") or {}
        cfg = r.get("cfg") or {}
        fam = r.get("family")
        me = r.get("your_player")
        if not fam or me not in ("player_1", "player_2"):
            continue
        opp = "player_2" if me == "player_1" else "player_1"
        tc = tclass(cfg.get("max_rounds"))
        n_recs += 1

        if fam == "bargaining":
            M = float(cfg.get("money_to_divide") or 0)
            if M <= 0:
                continue
            d1, d2 = cfg.get("delta_1"), cfg.get("delta_2")
            d_me = d1 if me == "player_1" else d2
            d_op = d2 if me == "player_1" else d1
            pay_me = res.get(f"{me}_payoff")
            pay_op = res.get(f"{opp}_payoff")
            if d_me is not None and pay_me is not None:
                dop_s = f"{float(d_op):.2f}" if d_op is not None else "h"
                pools[f"barg|{me}|{float(d_me):.2f}|{dop_s}|{tc}"].append(
                    float(pay_me) / M)
            # Opponent side is a population sample too, usable when both
            # discounts were visible to us (complete information).
            if d_me is not None and d_op is not None and pay_op is not None:
                pools[f"barg|{opp}|{float(d_op):.2f}|{float(d_me):.2f}|{tc}"] \
                    .append(float(pay_op) / M)

        elif fam == "negotiation":
            myv = cfg.get(f"{me}_value")
            pay_me = res.get(f"{me}_payoff")
            if myv is None or pay_me is None:
                continue
            myv = float(myv)
            M = scale(myv)
            pools[f"nego|{me}|{myv / M:.2f}|{tc}"].append(float(pay_me) / M)
            oppv = cfg.get(f"{opp}_value")
            if oppv is not None:       # complete info: exact-pair live cell
                pools[f"nego|{me}|{myv / M:.2f}|{float(oppv) / M:.2f}|{tc}"] \
                    .append(float(pay_me) / M)

    out = {}
    for key, vals in pools.items():
        v = sorted(vals)
        n = len(v)
        if n < 5:
            continue
        qs = []
        for q in QS:
            idx = min(n - 1, max(0, int(round(q / 100.0 * (n - 1)))))
            qs.append(round(v[idx], 4))
        out[key] = {"q": qs, "n": n}

    OUT.write_text(json.dumps({"quantile_levels": QS, "pools": out}),
                   encoding="utf-8")
    print(f"{n_recs} records -> {len(out)} live cells (n>=5) -> {OUT}")


if __name__ == "__main__":
    main()
