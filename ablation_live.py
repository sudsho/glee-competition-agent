"""Concurrent live ablation (bargaining): arm A vs arms B and C.

From Aug 24 to Aug 26 arm A (replica 1) ran a modified bargaining policy
(even split opener, q25 accept in rounds 1-2) while arms B and C
(replicas 2 and 3, which changed only negotiation and persuasion) ran the
baseline bargaining code against the same pool in the same hours. Each
game's realized payoff is scored by its reference-pool percentile on the
game-rating scale; arms are compared by mean game rating, raw and with
configuration-cell stratification.

Usage: python ablation_live.py
"""

import math
import pickle
from collections import defaultdict
from pathlib import Path

from ablation_pool import Pool

ROOT = Path(__file__).resolve().parent.parent
ABL = ROOT / "analysis" / "ablation"


def load(tag):
    with open(ABL / f"traces_{tag}.pkl", "rb") as f:
        return pickle.load(f)


def cell(r):
    return (r["role"], r["d_me"], r["d_op"], r["T"], r["ci"], r["ma"])


def scored(rows, pool):
    out = []
    for r in rows:
        p = pool.pct(r["pay"], r["role"], r["d_me"], r["d_op"], r["T"], r["ci"], r["ma"])
        if p is not None:
            out.append((cell(r), Pool.rating(p), r))
    return out


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(v / n)


def stratified_diff(a, b, key=None):
    """Weighted difference of means over strata present in both arms.

    Strata are configuration cells by default, or key(record) if given."""
    ca, cb = defaultdict(list), defaultdict(list)
    for c, x, r in a:
        ca[key(r) if key else c].append(x)
    for c, x, r in b:
        cb[key(r) if key else c].append(x)
    num = var = wsum = 0.0
    for c in ca:
        if c not in cb or len(ca[c]) < 2 or len(cb[c]) < 2:
            continue
        w = len(ca[c]) + len(cb[c])
        ma, sa = mean_se(ca[c])
        mb, sb = mean_se(cb[c])
        num += w * (ma - mb)
        var += w * w * (sa ** 2 + sb ** 2)
        wsum += w
    return num / wsum, math.sqrt(var) / wsum


def main():
    pool = Pool()
    arms = {t: load(t) for t in ("vA", "vB", "vC")}
    t0 = max(min(r["ts0"] for r in rows) for rows in arms.values())
    t1 = min(max(r["ts0"] for r in rows) for rows in arms.values())
    arms = {t: [r for r in rows if t0 <= r["ts0"] <= t1] for t, rows in arms.items()}
    sc = {t: scored(rows, pool) for t, rows in arms.items()}
    print(f"common window {t1 - t0:.0f} s ({(t1 - t0) / 3600:.1f} h)")
    for t, rows in sc.items():
        m, se = mean_se([x for _, x, _ in rows])
        agree = sum(1 for _, _, r in rows if r["outcome"] == "agreement") / len(rows)
        r1 = sum(1 for _, _, r in rows if r["outcome"] == "agreement"
                 and r["agreed_round"] == 1) / len(rows)
        print(f"{t}: n={len(rows)} mean rating {m:.1f} (se {se:.1f}) "
              f"agreement {agree:.3f} round-1 deals {r1:.3f}")
    ctrl = sc["vB"] + sc["vC"]
    ma_, sa = mean_se([x for _, x, _ in sc["vA"]])
    mc, sc_ = mean_se([x for _, x, _ in ctrl])
    d, se = ma_ - mc, math.sqrt(sa ** 2 + sc_ ** 2)
    print(f"A - (B+C) raw: {d:.1f} +/- {1.96 * se:.1f} (95% CI)")
    ds, ses = stratified_diff(sc["vA"], ctrl)
    print(f"A - (B+C) stratified by cell: {ds:.1f} +/- {1.96 * ses:.1f}")
    mb, sb = mean_se([x for _, x, _ in sc["vB"]])
    mc2, sc2 = mean_se([x for _, x, _ in sc["vC"]])
    print(f"B - C raw (identical code, placebo): {mb - mc2:.1f} +/- "
          f"{1.96 * math.sqrt(sb ** 2 + sc2 ** 2):.1f}")
    db, seb = stratified_diff(sc["vB"], sc["vC"])
    print(f"B - C stratified by cell (placebo): {db:.1f} +/- {1.96 * seb:.1f}")

    def opp(r):
        return (r["opp_type"], (r["opp_name"] or "").lower())
    do, seo = stratified_diff(sc["vA"], ctrl, key=opp)
    print(f"A - (B+C) stratified by named opponent: {do:.1f} +/- {1.96 * seo:.1f}")
    dpo, sepo = stratified_diff(sc["vB"], sc["vC"], key=opp)
    print(f"B - C stratified by named opponent (placebo): {dpo:.1f} +/- "
          f"{1.96 * sepo:.1f}")
    for t, rows in sc.items():
        n = len(rows)
        types = defaultdict(int)
        for _, _, r in rows:
            types[r["opp_type"]] += 1
        print(f"{t} opponent types: " + ", ".join(
            f"{k} {v / n:.3f}" for k, v in sorted(types.items())))


if __name__ == "__main__":
    main()
