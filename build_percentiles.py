"""Reconstruct per-configuration payoff distributions from the public GLEE
research dataset (the seed of the competition's scoring reference pool).

Output: analysis/percentiles.json
  {family: {config_key: {"player_1"|"player_2": {"q": [p5..p99], "n": N}}}}

Percentiles are computed on raw absolute payoffs; the competition's
percentile scoring is invariant to any monotone transform, so absolute
money works for targeting regardless of the exact normalization used
server-side.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "GLEE" / "Data" / "llm_vs_llm"
OUT = ROOT / "analysis" / "percentiles.json"

QS = [5, 10, 25, 50, 75, 90, 95, 99]

CONFIG_KEYS = {
    "bargaining": ("money_to_divide", "max_rounds", "complete_information",
                   "messages_allowed", "delta_1", "delta_2"),
    "negotiation": ("seller_value", "buyer_value", "product_price_order",
                    "max_rounds", "complete_information", "messages_allowed"),
    "persuasion": ("p", "v", "c", "product_price", "total_rounds",
                   "is_seller_know_cv", "seller_message_type", "is_myopic"),
}


def cfg_key(family: str, args: dict) -> str:
    return "|".join(f"{k}={args.get(k)}" for k in CONFIG_KEYS[family])


def read_rows(path: Path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def payoff_bargaining(rows, args):
    d1 = float(args.get("delta_1", 1.0))
    d2 = float(args.get("delta_2", 1.0))
    last_offer = None
    for r in rows:
        dec = (r.get("decision") or "").strip().lower()
        if r.get("alice_gain") not in (None, "",):
            try:
                last_offer = (float(r["alice_gain"]), float(r["bob_gain"]),
                              int(float(r.get("round") or 1)))
            except (TypeError, ValueError):
                pass
        if dec == "accept" and last_offer:
            a, b, t = last_offer
            return a * d1 ** (t - 1), b * d2 ** (t - 1)
    return 0.0, 0.0


def payoff_negotiation(rows, args):
    m = float(args.get("product_price_order") or 1.0)
    va = float(args.get("seller_value") or 0.0) * m
    vb = float(args.get("buyer_value") or 0.0) * m
    last_price = None
    for r in rows:
        dec = (r.get("decision") or "").strip().lower()
        if r.get("product_price") not in (None, ""):
            try:
                last_price = float(r["product_price"])
            except (TypeError, ValueError):
                pass
        if dec == "acceptoffer" and last_price is not None:
            return last_price - va, vb - last_price
    return 0.0, 0.0


def payoff_persuasion(rows, args):
    price = float(args.get("product_price") or 1.0)
    seller = buyer = 0.0
    quality_worth = {}
    for r in rows:
        rnd = r.get("round")
        q = (r.get("round_quality") or "").strip().lower()
        if q:
            try:
                quality_worth[rnd] = float(r.get("product_worth") or 0.0)
            except (TypeError, ValueError):
                quality_worth[rnd] = 0.0
        player = (r.get("player") or "").strip().lower()
        dec = (r.get("decision") or "").strip().lower()
        if player in ("player_2", "bob", "2") and dec in ("yes", "no"):
            if dec == "yes":
                seller += price
                buyer += quality_worth.get(rnd, 0.0) - price
    return seller, buyer


PAYOFF = {
    "bargaining": payoff_bargaining,
    "negotiation": payoff_negotiation,
    "persuasion": payoff_persuasion,
}


def quantiles(values, qs):
    v = sorted(values)
    n = len(v)
    out = []
    for q in qs:
        idx = min(n - 1, max(0, int(round(q / 100.0 * (n - 1)))))
        out.append(round(v[idx], 2))
    return out


def main():
    result = {}
    for family in ("bargaining", "negotiation", "persuasion"):
        fam_dir = DATA / family
        if not fam_dir.is_dir():
            print(f"skip {family}: {fam_dir} missing")
            continue
        pools = defaultdict(lambda: ([], []))
        games = 0
        for cfg_path in fam_dir.glob("*/*/*/*/config.json"):
            gdir = cfg_path.parent
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8",
                                                    errors="replace"))
                args = cfg.get("game_args") or {}
                rows = read_rows(gdir / "game.csv")
                pa, pb = PAYOFF[family](rows, args)
            except Exception:
                continue
            key = cfg_key(family, args)
            pools[key][0].append(pa)
            pools[key][1].append(pb)
            games += 1
            if games % 5000 == 0:
                print(f"{family}: {games} games processed", flush=True)
        fam_out = {}
        for key, (p1, p2) in pools.items():
            fam_out[key] = {
                "player_1": {"q": quantiles(p1, QS), "n": len(p1)},
                "player_2": {"q": quantiles(p2, QS), "n": len(p2)},
            }
        result[family] = fam_out
        print(f"{family}: {games} games, {len(fam_out)} configs", flush=True)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"quantile_levels": QS, "families": result}),
                   encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
