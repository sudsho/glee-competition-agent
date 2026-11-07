"""Decompose the pool-aware clamp on the latest frozen pool.

Usage: python ablation_decomp.py <part>
  lo      no lower bound on the equilibrium demand (KEEP_LO = 0)
  hi      no upper bound on the equilibrium demand (KEEP_HI = 1)
  cont    no clamp on the equilibrium continuation (CONT_LO = 0, CONT_HI = 1)
  targets no pool-based clamp parts (targets and pool accept bars), bounds kept
  bounds  no numeric bounds (demand, continuation, ceiling, greed limits),
          pool parts kept
  all     the whole clamp removed (the Table 2 no-clamp arm)
"""

import pickle
import random
import sys

import ablation_sim as S
import barg
from ablation_pool import Pool


def main(part, sample=3000, window="Aug 28-30"):
    pool = Pool()
    allrows = S.load(["main", "vA", "vB", "vC", "rep"])
    a, b = S.WINDOWS[window]
    fit_rows = [r for r in allrows if a <= r["ts0"] < b]
    F = S.ThresholdModel(fit_rows)
    first = S.first_offer_pools(fit_rows)
    games = [r for r in allrows if r["agent"] == "main" and a <= r["ts0"] < b
             and S.eligible(r, pool)]
    random.Random(7).shuffle(games)
    games = games[:sample]
    base = S.run_version("final", games, F, first, pool)
    if part == "lo":
        barg.KEEP_LO = 0.0
    elif part == "hi":
        barg.KEEP_HI = 1.0
    elif part == "cont":
        barg.CONT_LO, barg.CONT_HI = 0.0, 1.0
    elif part == "bounds":
        # Numeric bounds off; the greed cap still follows the pool's q90.
        barg.KEEP_LO, barg.KEEP_HI, barg.KEEP_MAX = 0.0, 1.0, 1.0
        barg.CONT_LO, barg.CONT_HI = 0.0, 1.0
        barg.CAP_GREED, barg.GREED_MAX = 0.0, 1.0
    if part == "all":
        res = S.run_version("no clamp", games, F, first, pool)
    elif part == "targets":
        res = S.run_version("final", games, F, first, pool, extra={"targets"})
    else:
        res = S.run_version("final", games, F, first, pool)
    dm, se = S.paired(res, base)
    print(f"{part}: {dm:+.1f} +/- {1.96 * se:.1f} (n={len(games)})", flush=True)
    with open(S.ABL / f"decomp_{part}.pkl", "wb") as f:
        pickle.dump((dm, se, len(games)), f)


if __name__ == "__main__":
    main(sys.argv[1])
