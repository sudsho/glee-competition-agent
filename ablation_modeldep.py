"""How much of each replay difference rests on the fitted opponent model.

A replay is log-pinned when every opponent decision it needed was fixed by
that opponent's logged accepts and rejects and every opponent offer came
from the log. Differences against the final policy are split into the part
from log-pinned replays and the part from replays that drew on the model.

Usage: python ablation_modeldep.py <window>   (e.g. "Aug 28-30")
       python ablation_modeldep.py <window> givecap
"""

import pickle
import random
import sys

import ablation_sim as S
import barg
from ablation_pool import Pool


class TracedOpponent(S.Opponent):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.generated = 0

    def offer(self, t):
        if t not in self.offers:
            self.generated += 1
        return super().offer(t)


def run(games, F, first, pool, flags, policy=None):
    barg.ABLATE.clear()
    if flags:
        barg.ABLATE.update(flags)
    pol = policy or barg.bargaining_strategy
    out = []
    for r in games:
        reps = []
        for rep in range(S.REPS):
            opp = TracedOpponent(r, F, first, rep)
            x, t = S.play(r, pol, opp)
            pc = pool.pct(x, r["role"], r["d_me"], r["d_op"], r["T"], r["ci"], r["ma"])
            reps.append((Pool.rating(pc), opp.free > 0 or opp.generated > 0, t is None))
        out.append(reps)
    barg.ABLATE.clear()
    return out


def split(alt, base):
    pairs = [(a, b) for A, B in zip(alt, base) for a, b in zip(A, B)]
    k = len(pairs)
    total = sum(a[0] - b[0] for a, b in pairs) / k
    modeled = sum(a[0] - b[0] for a, b in pairs if a[1] or b[1]) / k
    share = sum(1 for a, b in pairs if a[1] or b[1]) / k
    nodeal = sum(1 for a, _ in pairs if a[2]) / k
    return total, total - modeled, modeled, share, nodeal


def main(window, mode="versions", sample=4000):
    pool = Pool()
    allrows = S.load(["main", "vA", "vB", "vC", "rep"])
    a, b = S.WINDOWS[window]
    fit_rows = [r for r in allrows if a <= r["ts0"] < b]
    F = S.ThresholdModel(fit_rows)
    first = S.first_offer_pools(fit_rows)
    games = [r for r in allrows if r["agent"] == "main" and a <= r["ts0"] < b
             and S.eligible(r, pool)]
    random.Random(7).shuffle(games)
    games = games[:(sample if mode == "versions" else 3000)]
    base = run(games, F, first, pool, set())
    b_nodeal = sum(1 for B in base for x in B if x[2]) / (len(base) * S.REPS)
    print(f"{window} ({mode}): n={len(games)}, final no-deal share {b_nodeal:.4f}",
          flush=True)
    res = {"base_nodeal": b_nodeal, "rows": {}}
    if mode == "givecap":
        mode = "set:GIVE_CAP=0.49"
    if mode.startswith("set:"):
        name, val = mode[4:].split("=")
        val = float(val)
        if name == "FLOORS":
            old = dict(barg.FLOORS)
            barg.FLOORS.update({k: x * val for k, x in old.items()})
            alt = run(games, F, first, pool, set())
            barg.FLOORS.update(old)
        else:
            old = getattr(barg, name)
            setattr(barg, name, val)
            alt = run(games, F, first, pool, set())
            setattr(barg, name, old)
        variants = {f"{name} {val:g}": alt}
    else:
        variants = {
            "no clamp": run(games, F, first, pool, {"clamp"}),
            "no safeguards": run(games, F, first, pool, {"safe"}),
            "equilibrium only": run(games, F, first, pool, {"clamp", "safe"}),
            "even split": run(games, F, first, pool, None, policy=S.even_policy),
        }
    for name, alt in variants.items():
        total, pinned, modeled, share, nodeal = split(alt, base)
        res["rows"][name] = (total, pinned, modeled, share, nodeal)
        print(f"  {name:17s} total {total:+8.1f} | from log-pinned replays "
              f"{pinned:+8.1f} | from modeled replays {modeled:+8.1f} "
              f"({share:.1%} of replays) | no-deal {nodeal:.3f}", flush=True)
    tag = window.replace(" ", "_") + ("" if mode == "versions" else "_" + mode.replace(":", "_").replace("=", "_"))
    with open(S.ABL / f"modeldep_{tag}.pkl", "wb") as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "versions")
