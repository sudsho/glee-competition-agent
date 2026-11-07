"""Out-of-policy check: replay arm A's live rule set on the frozen pools.

Arm A (even-split opener, early q25 accepts) was measured live against
the baselines on Aug 24-26. Replaying the same rule switch on top of the
final policy, against later frozen pools, tests whether the replay gets the
sign and the size of a live effect right.

Usage: python ablation_variant_a.py <window>
"""

import pickle
import random
import sys

import ablation_sim as S
import var_cfg
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
    var_cfg.VARIANT = "A"
    alt = S.run_version("final", games, F, first, pool)
    var_cfg.VARIANT = ""
    dm, se = S.paired(alt, base)
    print(f"{window}: variant A rules minus final {dm:+.1f} +/- {1.96 * se:.1f} "
          f"(n={len(games)})", flush=True)
    with open(S.ABL / f"variantA_{window.replace(' ', '_')}.pkl", "wb") as f:
        pickle.dump((dm, se, len(games)), f)


if __name__ == "__main__":
    main(sys.argv[1])
