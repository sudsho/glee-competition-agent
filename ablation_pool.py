"""Reference-pool scoring for offline ablation (bargaining).

Builds, from the public GLEE corpus, the exact normalized payoff samples of
every configuration-role cell and scores a payoff by its mid-rank
percentile in the matching cell, mapped to the competition's game-rating
scale 2000 + 8000 (pct - 0.5). A hidden opponent discount pools the cells
over that discount, the same aggregation the agent's own targets use.

Usage: python ablation_pool.py   (writes analysis/ablation/pool_barg.pkl)
"""

import bisect
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "GLEE" / "Data" / "llm_vs_llm" / "bargaining"
OUT = ROOT / "analysis" / "ablation" / "pool_barg.pkl"


def tclass(T):
    if T is None:
        return "inf"
    T = int(T)
    return "inf" if T >= 90 else f"T{T}"


def _payoffs(rows, d1, d2):
    last = None
    for r in rows:
        if r.get("alice_gain") not in (None, ""):
            try:
                last = (float(r["alice_gain"]), float(r["bob_gain"]),
                        int(float(r.get("round") or 1)))
            except (TypeError, ValueError):
                pass
        if (r.get("decision") or "").strip().lower() == "accept" and last:
            a, b, t = last
            return a * d1 ** (t - 1), b * d2 ** (t - 1)
    return 0.0, 0.0


def build():
    cells = defaultdict(list)
    n = 0
    for cfg_path in DATA.glob("*/*/*/*/config.json"):
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8", errors="replace"))
            a = cfg.get("game_args") or {}
            M = float(a["money_to_divide"])
            d1, d2 = float(a["delta_1"]), float(a["delta_2"])
            with open(cfg_path.parent / "game.csv", newline="", encoding="utf-8",
                      errors="replace") as f:
                rows = list(csv.DictReader(f))
            p1, p2 = _payoffs(rows, d1, d2)
        except Exception:
            continue
        tc = tclass(a.get("max_rounds"))
        ci, ma = bool(a.get("complete_information")), bool(a.get("messages_allowed"))
        cells[("player_1", round(d1, 2), round(d2, 2), tc, ci, ma)].append(p1 / M)
        cells[("player_2", round(d2, 2), round(d1, 2), tc, ci, ma)].append(p2 / M)
        n += 1
    pool = {k: sorted(v) for k, v in cells.items()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump(pool, f)
    print(f"{n} corpus games, {len(pool)} role cells -> {OUT}")


class Pool:
    """Mid-rank percentile of a normalized payoff in its reference cell."""

    def __init__(self, path=OUT):
        with open(path, "rb") as f:
            self.cells = pickle.load(f)
        self._cache = {}

    def sample(self, role, d_me, d_op, T, ci, ma):
        key = (role, None if d_me is None else round(d_me, 2),
               None if d_op is None else round(d_op, 2), tclass(T), ci, ma)
        if key in self._cache:
            return self._cache[key]
        out = None
        for strict in (True, False):
            vals = []
            for (r, dm, do, tc, c, m), v in self.cells.items():
                if r != role or tc != key[3]:
                    continue
                if d_me is not None and abs(dm - d_me) > 0.011:
                    continue
                if d_op is not None and abs(do - d_op) > 0.011:
                    continue
                if strict and (c != ci or m != ma):
                    continue
                vals.extend(v)
            if vals:
                out = sorted(vals)
                break
        self._cache[key] = out
        return out

    def pct(self, x, role, d_me, d_op, T, ci, ma):
        s = self.sample(role, d_me, d_op, T, ci, ma)
        if not s:
            return None
        lo = bisect.bisect_left(s, x - 1e-9)
        hi = bisect.bisect_right(s, x + 1e-9)
        return (lo + 0.5 * (hi - lo)) / len(s)

    @staticmethod
    def rating(p):
        return 2000.0 + 8000.0 * (p - 0.5)


if __name__ == "__main__":
    build()
