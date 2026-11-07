"""Offline controlled ablation (bargaining) against frozen opponent pools.

Every logged game is replayed under each policy version. The opponent of
that game is held fixed. Its offers are the logged ones, continued after
the log runs out by small fixed concessions (STEP, at most 0.017 of the
pot in total) and then frozen. Its accept decisions follow one fixed
threshold per game: tau ~ Logistic(mu(x), sigma(x)) is fitted by
interval-censored maximum likelihood (ThresholdModel, one interval per
game), and each opponent's tau is drawn inside the interval its own
logged accepts and rejects imply. Decisions the log already fixes are
taken from the log. Three draws per game are shared by all versions, so
differences are paired. Payoffs are scored by mid-rank percentile in the
corpus reference pool, on the game-rating scale 2000 + 8000 (pct - 0.5).

Usage:
  python ablation_sim.py versions [window]   # version table, three pools
  python ablation_sim.py sweep [CONSTANT ...] # one-at-a-time sensitivity
"""

import hashlib
import math
import pickle
import random
import sys
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import barg
import var_cfg
from ablation_pool import Pool
from norm import offer_gain, split_action

ROOT = Path(__file__).resolve().parent.parent
ABL = ROOT / "analysis" / "ablation"
HIDDEN_T = 99          # hidden horizons ran out near round 100 in our logs
REPS = 3


def _utc(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp()


WINDOWS = {
    "Aug 22-24": (_utc("2026-08-22 00:00"), _utc("2026-08-25 00:00")),
    "Aug 25-27": (_utc("2026-08-25 00:00"), _utc("2026-08-28 00:00")),
    "Aug 28-30": (_utc("2026-08-28 00:00"), _utc("2026-08-30 12:00")),
}


def unif(*key):
    h = hashlib.md5("|".join(map(str, key)).encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def load(tags):
    rows = []
    for t in tags:
        with open(ABL / f"traces_{t}.pkl", "rb") as f:
            rows.extend(pickle.load(f))
    return rows


# ---------------------------------------------------------------- opponent

def ctx_feats(r):
    ci = float(r["ci"])
    we_p1 = 1.0 if r["role"] == "player_1" else 0.0
    dv = (r["d_op"] if (r["ci"] and r["d_op"] is not None) else 0.0) or 0.0
    return [1.0, ci, float(r["hk"]), float(r["ma"]),
            1.0 if r["opp_type"] == "agent" else 0.0,
            1.0 if r["opp_type"] == "human" else 0.0,
            we_p1, dv, (r["d_me"] or 0.0) * ci,
            math.log10(max(r["M"], 1.0)) / 6.0, float(r["hk"]) * we_p1]


def _interval(r):
    rej = [s for (_, s, d) in r["ours"] if d == "reject"]
    acc = [s for (_, s, d) in r["ours"] if d == "accept"]
    lo = max(rej) if rej else -math.inf
    hi = min(acc) if acc else math.inf
    return lo, hi


def _lam(z):
    return 1.0 / (1.0 + np.exp(-z))


class ThresholdModel:
    """Opponent acceptance thresholds: tau ~ Logistic(mu(x), sigma(x)).

    Interval-censored maximum likelihood with one interval per game:
    tau lies above the largest share the opponent rejected and at or
    below the smallest share it accepted. A stonewaller therefore counts
    once, not once per rejected offer.
    """

    def __init__(self, rows):
        from scipy.optimize import minimize
        X, lo, hi = [], [], []
        self.inconsistent = 0
        for r in rows:
            if not any(d in ("accept", "reject") for (_, _, d) in r["ours"]):
                continue
            l, h = _interval(r)
            if l >= h:
                self.inconsistent += 1
                continue
            X.append(ctx_feats(r))
            lo.append(l)
            hi.append(h)
        X, lo, hi = np.array(X), np.array(lo), np.array(hi)
        self.n = len(lo)
        k = X.shape[1]
        idx = np.random.RandomState(0).permutation(self.n)
        tr, te = idx[: int(0.8 * self.n)], idx[int(0.8 * self.n):]

        def nll(theta, sel):
            beta, g0, g1 = theta[:k], theta[k], theta[k + 1]
            mu = X[sel] @ beta
            sig = np.exp(g0 + g1 * X[sel, 1])
            ph = np.where(np.isinf(hi[sel]), 1.0, _lam((hi[sel] - mu) / sig))
            pl = np.where(np.isinf(lo[sel]), 0.0, _lam((lo[sel] - mu) / sig))
            return -np.sum(np.log(np.clip(ph - pl, 1e-12, None)))

        x0 = np.zeros(k + 2)
        x0[0], x0[k] = 0.45, math.log(0.1)
        fit_tr = minimize(nll, x0, args=(tr,), method="L-BFGS-B")
        self.heldout_nll = float(nll(fit_tr.x, te) / len(te))
        base = minimize(lambda th: nll(np.r_[th[0], np.zeros(k - 1), th[1], 0.0], tr),
                        [0.45, math.log(0.1)], method="L-BFGS-B")
        self.heldout_nll_null = float(
            nll(np.r_[base.x[0], np.zeros(k - 1), base.x[1], 0.0], te) / len(te))
        fit = minimize(nll, fit_tr.x, args=(np.arange(self.n),), method="L-BFGS-B")
        self.theta = fit.x
        self.k = k

    def params(self, r):
        x = np.array(ctx_feats(r))
        mu = float(x @ self.theta[: self.k])
        sig = float(math.exp(self.theta[self.k] + self.theta[self.k + 1] * x[1]))
        return mu, sig

    def cdf(self, r, s):
        mu, sig = self.params(r)
        return 1.0 / (1.0 + math.exp(-(s - mu) / sig))


class Opponent:
    """Frozen opponent of one logged game: its logged offers, and a fixed
    acceptance threshold drawn inside the interval its decisions imply."""

    STEP = {1: 0.0109, 2: 0.0021, 3: 0.0038}   # median observed concessions

    def __init__(self, r, F, first_offers, rep):
        self.r = r
        self.real = {t: (s, d) for (t, s, d) in r["ours"] if d in ("accept", "reject")}
        lo, hi = _interval(r)
        mu, sig = F.params(r)
        self.conflict = lo >= hi
        if self.conflict:
            lo = -math.inf                      # softened: honor its accept
        clo = 0.0 if math.isinf(lo) else 1.0 / (1.0 + math.exp(-(lo - mu) / sig))
        chi = 1.0 if math.isinf(hi) else 1.0 / (1.0 + math.exp(-(hi - mu) / sig))
        v = unif(r["gid"], "tau", rep)
        q = min(max(clo + v * (chi - clo), 1e-12), 1.0 - 1e-12)
        self.tau = mu + sig * math.log(q / (1.0 - q))
        self.lo, self.hi = lo, hi
        self.pinned = self.free = 0
        self.offers = {t: x for (t, x, _) in r["theirs"]}
        self.first = None
        if not self.offers:
            pool = first_offers.get(_ctx(r)) or first_offers.get("all")
            j = min(len(pool) - 1, int((1.0 - q) * len(pool)))
            self.first = pool[j]

    def accepts(self, s, t):
        if t in self.real:
            s_real, d_real = self.real[t]
            if d_real == "accept" and s >= s_real - 1e-9:
                self.pinned += 1
                return True
            if d_real == "reject" and s <= s_real + 1e-9:
                self.pinned += 1
                return False
        if s <= self.lo + 1e-9 or s >= self.hi - 1e-9:
            self.pinned += 1
        else:
            self.free += 1
        return s >= self.tau - 1e-12

    def offer(self, t):
        if t in self.offers:
            return self.offers[t]
        prev = [tt for tt in self.offers if tt < t]
        if prev:
            t0 = max(prev)
            x, k = self.offers[t0], sum(1 for tt in self.offers if tt <= t0)
        else:
            x, k, t0 = self.first, 1, t
        tt = t0
        while tt + 2 <= t:
            x += self.STEP.get(k, 0.0)
            k += 1
            tt += 2
        return min(max(x, 0.0), 1.0)


def _ctx(r):
    return (r["role"], r["opp_type"], r["ci"], r["hk"])


def first_offer_pools(rows):
    pools = defaultdict(list)
    for r in rows:
        if r["theirs"]:
            x = sorted(r["theirs"])[0][1]
            pools[_ctx(r)].append(x)
            pools["all"].append(x)
    return {k: sorted(v) for k, v in pools.items() if len(v) >= 20}


# ----------------------------------------------------------------- policies

def even_policy(game):
    st, va, me = game["game_state"], game["valid_actions"], game["your_player"]
    M = float(st["money_to_divide"])
    if va.get("type") == "offer":
        return split_action(M / 2.0, M, me)
    offered = offer_gain(st.get("last_offer") or {}, me) or 0.0
    T, t = st.get("max_rounds"), int(st.get("round") or 1)
    if T is not None and t >= int(T):
        return {"decision": "accept" if offered > 0 else "reject"}
    return {"decision": "accept" if offered >= 0.5 * M - 1e-9 else "reject"}


VERSIONS = {
    "final": set(),
    "no clamp": {"clamp"},
    "no safeguards": {"safe"},
    "equilibrium only": {"clamp", "safe"},
    "even split": None,
}


def _state(r, t, hist, phase, last_offer):
    me = r["role"]
    st = {"game_family": "bargaining", "round": t, "phase": phase,
          "horizon_known": r["hk"], "current_player": me,
          "proposer": "player_1" if t % 2 == 1 else "player_2",
          "money_to_divide": r["M"], "complete_information": r["ci"],
          "messages_allowed": r["ma"], "history": hist, "last_offer": last_offer}
    st[f"delta_{1 if me == 'player_1' else 2}"] = r["d_me"]
    if r["ci"] and r["d_op"] is not None:
        st[f"delta_{2 if me == 'player_1' else 1}"] = r["d_op"]
    if r["hk"] and r["T"] is not None:
        st["max_rounds"] = r["T"]
    return st


def play(r, policy, opp):
    """Replay one game; returns (normalized payoff, agreed round or None)."""
    me = r["role"]
    op = "player_2" if me == "player_1" else "player_1"
    M, d = r["M"], r["d_me"]
    T = r["T"] if r["T"] is not None else HIDDEN_T
    game = {"your_player": me, "opponent": {"type": r["opp_type"], "name": r["opp_name"]}}
    hist = []
    for t in range(1, T + 1):
        if ("player_1" if t % 2 == 1 else "player_2") == me:
            game["game_state"] = _state(r, t, hist, "offer", None)
            # Fields without "message" skip text generation only.
            game["valid_actions"] = {"type": "offer",
                                     "fields": {"alice_gain": "", "bob_gain": ""}}
            a = policy(game)
            mine = float(a["alice_gain"] if me == "player_1" else a["bob_gain"])
            s = 1.0 - mine / M
            ok = opp.accepts(s, t)
            hist.append({"round": t, "proposer": me,
                         "offer": {f"{me}_gain": mine, f"{op}_gain": M - mine},
                         "decision": "accept" if ok else "reject"})
            if ok:
                return (1.0 - s) * d ** (t - 1), t
        else:
            x = opp.offer(t)
            lo = {f"{me}_gain": x * M, f"{op}_gain": (1.0 - x) * M,
                  "proposer": op, "round": t}
            game["game_state"] = _state(r, t, hist, "decision", lo)
            game["valid_actions"] = {"type": "decision", "fields": {}}
            dec = policy(game).get("decision")
            hist.append({"round": t, "proposer": op, "offer": lo, "decision": dec})
            if dec == "accept":
                return x * d ** (t - 1), t
    return 0.0, None


PIN = defaultdict(lambda: [0, 0])     # version -> [pinned, model-drawn]


def run_version(name, games, F, first, pool, reps=REPS, extra=()):
    flags = VERSIONS[name]
    if flags is None:
        policy = even_policy
    else:
        barg.ABLATE.clear()
        barg.ABLATE.update(flags)
        barg.ABLATE.update(extra)
        policy = barg.bargaining_strategy
    out = []
    for r in games:
        vals = []
        for rep in range(reps):
            opp = Opponent(r, F, first, rep)
            x, _ = play(r, policy, opp)
            PIN[name][0] += opp.pinned
            PIN[name][1] += opp.free
            pc = pool.pct(x, r["role"], r["d_me"], r["d_op"], r["T"], r["ci"], r["ma"])
            vals.append(Pool.rating(pc))
        out.append(sum(vals) / len(vals))
    barg.ABLATE.clear()
    return out


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
    return m, se


def eligible(r, pool):
    return (r["d_me"] is not None and r["M"] > 0
            and pool.sample(r["role"], r["d_me"], r["d_op"], r["T"], r["ci"], r["ma"]))


def versions(sample=4000, only=None):
    pool = Pool()
    allrows = load(["main", "vA", "vB", "vC", "rep"])
    main_rows = [r for r in allrows if r["agent"] == "main"]
    report = {}
    for wname, (a, b) in WINDOWS.items():
        if only and wname != only:
            continue
        fit_rows = [r for r in allrows if a <= r["ts0"] < b]
        F = ThresholdModel(fit_rows)
        first = first_offer_pools(fit_rows)
        games = [r for r in main_rows if a <= r["ts0"] < b and eligible(r, pool)]
        random.Random(7).shuffle(games)
        games = games[:sample]
        conflicts = sum(Opponent(r, F, first, 0).conflict for r in games)
        PIN.clear()
        res = {v: run_version(v, games, F, first, pool) for v in VERSIONS}
        realized = [Pool.rating(pool.pct(r["pay"], r["role"], r["d_me"], r["d_op"],
                                          r["T"], r["ci"], r["ma"])) for r in games]
        print(f"\n== pool {wname}: threshold fit on {F.n} opponents "
              f"({F.inconsistent} inconsistent skipped); held-out NLL "
              f"{F.heldout_nll:.4f} vs {F.heldout_nll_null:.4f} null; games "
              f"{len(games)}; softened opponents {conflicts}")
        base = res["final"]
        for v in VERSIONS:
            m = sum(res[v]) / len(res[v])
            pin = PIN[v][0] / max(1, sum(PIN[v]))
            if v == "final":
                print(f"  {v:18s} {m:7.1f}   pinned {pin:.3f}")
            else:
                dm, se = paired(res[v], base)
                print(f"  {v:18s} {m:7.1f}   vs final {dm:+7.1f} +/- "
                      f"{1.96 * se:5.1f}   pinned {pin:.3f}")
        mr = sum(realized) / len(realized)
        dm, se = paired(base, realized)
        print(f"  realized (logged)  {mr:7.1f}   final replay minus realized "
              f"{dm:+.1f} +/- {1.96 * se:.1f}", flush=True)
        report[wname] = {"F": (F.n, F.inconsistent, F.heldout_nll, F.heldout_nll_null),
                         "n": len(games),
                         "res": {v: (sum(x) / len(x)) for v, x in res.items()},
                         "diff": {v: paired(res[v], base) for v in VERSIONS if v != "final"},
                         "pinned": {v: PIN[v][0] / max(1, sum(PIN[v])) for v in VERSIONS},
                         "realized": mr, "realized_diff": (dm, se),
                         "conflicts": conflicts}
    name = "versions.pkl" if not only else f"versions_{only.replace(' ', '_')}.pkl"
    with open(ABL / name, "wb") as f:
        pickle.dump(report, f)


SWEEP = {
    "KEEP_LO": [0.48, 0.50, 0.52, 0.54, 0.56],
    "KEEP_HI": [0.60, 0.65, 0.70, 0.75, 0.80],
    "KEEP_MAX": [0.70, 0.75, 0.80, 0.85, 0.90],
    "CONT_LO": [0.30, 0.35, 0.40, 0.45, 0.50],
    "HAIRCUT": [0.80, 0.85, 0.90, 0.95, 1.00],
    "CAP_GREED": [0.52, 0.55, 0.58, 0.61, 0.64],
    "RHO": [0.93, 0.95, 0.97, 0.99, 1.00],
    "GIVE_CAP": [0.49, 0.52, 0.55, 0.58, 0.61],
    "FLOORS": [0.8, 0.9, 1.0, 1.1, 1.2],
}


def sweep(names=None, sample=3000, window="Aug 28-30"):
    pool = Pool()
    allrows = load(["main", "vA", "vB", "vC", "rep"])
    a, b = WINDOWS[window]
    fit_rows = [r for r in allrows if a <= r["ts0"] < b]
    F = ThresholdModel(fit_rows)
    first = first_offer_pools(fit_rows)
    games = [r for r in allrows if r["agent"] == "main" and a <= r["ts0"] < b
             and eligible(r, pool)]
    random.Random(7).shuffle(games)
    games = games[:sample]
    base = run_version("final", games, F, first, pool)
    mb = sum(base) / len(base)
    print(f"sweep on pool {window}: n={len(games)} final {mb:.1f}", flush=True)
    floors0 = dict(barg.FLOORS)
    for name in (names or list(SWEEP)):
        row = []
        chosen = 1.0 if name == "FLOORS" else getattr(barg, name)
        for v in SWEEP[name]:
            if abs(v - chosen) < 1e-12:
                row.append((v, 0.0, 0.0))
                continue
            if name == "FLOORS":
                barg.FLOORS.update({k: x * v for k, x in floors0.items()})
            else:
                setattr(barg, name, v)
            res = run_version("final", games, F, first, pool)
            if name == "FLOORS":
                barg.FLOORS.update(floors0)
            else:
                setattr(barg, name, chosen)
            dm, se = paired(res, base)
            row.append((v, dm, se))
        print(f"  {name:10s} " + "  ".join(f"{v}:{dm:+.1f}(+/-{1.96 * se:.1f})"
                                           for v, dm, se in row), flush=True)
        with open(ABL / f"sweep_{name}.pkl", "wb") as f:
            pickle.dump({"base": mb, "row": row, "n": len(games)}, f)


if __name__ == "__main__":
    assert var_cfg.VARIANT == "", "run without GLEE_VARIANT"
    if sys.argv[1] == "sweep":
        sweep(sys.argv[2:] or None)
    else:
        versions(only=sys.argv[2] if len(sys.argv) > 2 else None)
