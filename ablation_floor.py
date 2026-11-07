"""How much of the clamp effect is the patient pool-median accept floor.

Usage: python ablation_floor.py <window>
Prints, against the final policy: the floor alone removed, and the clamp
removed with the floor kept (filed under the safeguards).
"""

import pickle
import random
import sys

import ablation_sim as S
from ablation_pool import Pool


def main(window, sample=4000):
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
    out = {}
    for name, version, extra in (("floor only removed", "final", {"floor"}),
                                 ("clamp removed, floor kept", "no clamp", {"floor-safe"})):
        res = S.run_version(version, games, F, first, pool, extra=extra)
        dm, se = S.paired(res, base)
        out[name] = (dm, se)
        print(f"{window}: {name}: {dm:+.1f} +/- {1.96 * se:.1f}", flush=True)
    with open(S.ABL / f"floor_{window.replace(' ', '_')}.pkl", "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    main(sys.argv[1])
