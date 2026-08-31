"""Distill percentiles.json into policy targets.

Prints, per strategic situation, what normalized payoff the reference pool's
50th / 90th / 95th percentile represents, aggregated over money scales and
secondary flags. This is what 'winning play' means per configuration.
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = json.loads((ROOT / "analysis" / "percentiles.json").read_text())
QL = D["quantile_levels"]          # [5,10,25,50,75,90,95,99]
IQ = {q: i for i, q in enumerate(QL)}


def parse_key(key):
    out = {}
    for part in key.split("|"):
        k, v = part.split("=", 1)
        out[k] = v
    return out


def fnum(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def agg(rows):
    """Mean of each quantile across configs (already normalized)."""
    if not rows:
        return None
    n = len(rows[0])
    return [sum(r[i] for r in rows) / len(rows) for i in range(n)]


def show(title, groups, fmt="{:6.3f}"):
    print(f"\n== {title} ==")
    print(f"{'group':<28} {'n_cfg':>5}  " +
          "  ".join(f"q{q:>2}" for q in (25, 50, 75, 90, 95)))
    for g in sorted(groups):
        rows = groups[g]
        a = agg(rows)
        if a is None:
            continue
        vals = "  ".join(fmt.format(a[IQ[q]]) for q in (25, 50, 75, 90, 95))
        print(f"{str(g):<28} {len(rows):>5}  {vals}")


# ---------------- bargaining: normalize by pot M ----------------
for role in ("player_1", "player_2"):
    groups = defaultdict(list)
    for key, v in D["families"]["bargaining"].items():
        cfg = parse_key(key)
        M = fnum(cfg.get("money_to_divide"), 1.0)
        qs = [x / M for x in v[role]["q"]]
        d1, d2 = cfg.get("delta_1"), cfg.get("delta_2")
        horizon = "T12" if cfg.get("max_rounds") not in ("99", "None") else "inf"
        groups[(f"d_me={d1 if role=='player_1' else d2}",
                f"d_op={d2 if role=='player_1' else d1}", horizon)].append(qs)
    show(f"BARGAINING {role} (payoff / M) by (my delta, opp delta, horizon)",
         groups)

# ---------------- negotiation: normalize by M ----------------
for role in ("player_1", "player_2"):
    groups = defaultdict(list)
    for key, v in D["families"]["negotiation"].items():
        cfg = parse_key(key)
        M = fnum(cfg.get("product_price_order"), 1.0)
        qs = [x / M for x in v[role]["q"]]
        fa, fb = cfg.get("seller_value"), cfg.get("buyer_value")
        T = cfg.get("max_rounds")
        T = {"1": "T1", "10": "T10"}.get(T, "inf")
        groups[(f"Fs={fa}", f"Fb={fb}", T)].append(qs)
    show(f"NEGOTIATION {role} (payoff / M) by (seller F, buyer F, horizon)",
         groups)

# ---------------- persuasion ----------------
groups = defaultdict(list)
for key, v in D["families"]["persuasion"].items():
    cfg = parse_key(key)
    price = fnum(cfg.get("product_price"), 1.0)
    T = fnum(cfg.get("total_rounds"), 20.0)
    qs = [x / (price * T) for x in v["player_1"]["q"]]   # sales rate
    pv = fnum(cfg.get("p"), 0) * fnum(cfg.get("v"), 0)
    regime = ("blind-buy" if pv > 1.02 else
              "no-trade" if fnum(cfg.get("v"), 0) <= 1.0 else "trust")
    groups[(f"p={cfg.get('p')[:4]}", f"v={cfg.get('v')}", regime,
            cfg.get("seller_message_type"))].append(qs)
show("PERSUASION seller (sales rate) by (p, v, regime, msg type)", groups)

groups = defaultdict(list)
for key, v in D["families"]["persuasion"].items():
    cfg = parse_key(key)
    price = fnum(cfg.get("product_price"), 1.0)
    T = fnum(cfg.get("total_rounds"), 20.0)
    vv = fnum(cfg.get("v"), 1.0)
    denom = max((vv - 1.0) * price * T, 1e-9)   # max possible buyer surplus
    qs = [x / denom for x in v["player_2"]["q"]]
    pv = fnum(cfg.get("p"), 0) * vv
    regime = ("blind-buy" if pv > 1.02 else
              "no-trade" if vv <= 1.0 else "trust")
    groups[(f"p={cfg.get('p')[:4]}", f"v={cfg.get('v')}", regime)].append(qs)
show("PERSUASION buyer (surplus / max possible) by (p, v, regime)", groups)
