"""Percentile targets from the reconstructed GLEE reference pool.

Provides, per live game situation, the normalized payoff (share of M) that
each pool percentile represents, so strategies can aim at q90 play instead
of guessing. All lookups degrade gracefully: no table, no match -> None,
and callers fall back to their theory-driven defaults.
"""

import json
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "analysis" / "percentiles.json"
_LIVE_PATH = Path(__file__).resolve().parent.parent / "analysis" / "live_quantiles.json"

_QL = []
_BARG = []      # (d1, d2, tclass, ci, ma, {"player_1": qnorm, "player_2": qnorm})
_NEGO = []      # (fs, fb, tclass, {"player_1": qnorm, "player_2": qnorm})
_LIVE = {}      # aggregation key -> {"q": [...], "n": int}


def _f(v, d=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def _b(v):
    return str(v).lower() in ("true", "1")


def _parse(key):
    out = {}
    for part in key.split("|"):
        k, v = part.split("=", 1)
        out[k] = v
    return out


def _tclass(max_rounds):
    v = _f(max_rounds)
    if v is None or v >= 90:
        return "inf"
    return f"T{int(v)}"


def _load():
    global _QL, _LIVE
    if _QL or not _PATH.exists():
        return
    data = json.loads(_PATH.read_text(encoding="utf-8"))
    _QL = data["quantile_levels"]
    if _LIVE_PATH.exists():
        try:
            _LIVE = json.loads(_LIVE_PATH.read_text(encoding="utf-8"))["pools"]
        except (json.JSONDecodeError, KeyError):
            _LIVE = {}
    for key, v in data["families"].get("bargaining", {}).items():
        cfg = _parse(key)
        M = _f(cfg.get("money_to_divide"), 1.0) or 1.0
        _BARG.append((
            _f(cfg.get("delta_1")), _f(cfg.get("delta_2")),
            _tclass(cfg.get("max_rounds")),
            _b(cfg.get("complete_information")), _b(cfg.get("messages_allowed")),
            {r: [x / M for x in v[r]["q"]] for r in ("player_1", "player_2")},
        ))
    for key, v in data["families"].get("negotiation", {}).items():
        cfg = _parse(key)
        M = _f(cfg.get("product_price_order"), 1.0) or 1.0
        _NEGO.append((
            _f(cfg.get("seller_value")), _f(cfg.get("buyer_value")),
            _tclass(cfg.get("max_rounds")),
            {r: [x / M for x in v[r]["q"]] for r in ("player_1", "player_2")},
        ))


def _avg(rows):
    if not rows:
        return None
    n = len(rows[0])
    return [sum(r[i] for r in rows) / len(rows) for i in range(n)]


def _q(vec, q):
    if vec is None:
        return None
    try:
        return vec[_QL.index(q)]
    except ValueError:
        return None


class Ladder:
    """Static pool quantiles, optionally blended with live competition
    observations. The live pool dominates actual scoring by now, so its
    weight grows with sample size up to 0.6."""

    def __init__(self, vec, live_key=None, ratchet=False):
        self.vec = vec
        self.ratchet = ratchet
        cell = _LIVE.get(live_key) if live_key else None
        if cell is None and live_key and live_key.startswith("barg|"):
            parts = live_key.split("|")
            parts[3] = "h"                      # fall back to hidden-opp cell
            cell = _LIVE.get("|".join(parts))
        self.live = cell["q"] if cell else None
        self.live_n = cell["n"] if cell else 0

    def q(self, level):
        base = _q(self.vec, level)
        if base is None:
            return None
        live = _q(self.live, level) if self.live else None
        if live is None:
            return base
        w = min(0.6, self.live_n / (self.live_n + 40.0))
        blended = (1.0 - w) * base + w * live
        # Ratchet mode: our live cells are a self-sample of our own play, so
        # they may only RAISE a bar above the static pool, never lower it
        # (a lowered bar converges to our own past mistakes).
        return max(base, blended) if self.ratchet else blended


def barg_ladder(role, d_me, d_op_known, T, ci, ma):
    """Normalized payoff ladder for my role; d_op_known=None aggregates
    over the hidden opponent discount."""
    _load()
    if not _BARG:
        return None
    tc = _tclass(T)
    my_i, op_i = (0, 1) if role == "player_1" else (1, 0)

    def match(e, strict):
        if abs((e[my_i] or 0) - d_me) > 0.011:
            return False
        if d_op_known is not None and abs((e[op_i] or 0) - d_op_known) > 0.011:
            return False
        if e[2] != tc:
            return False
        if strict and (e[3] != ci or e[4] != ma):
            return False
        return True

    dop_s = f"{d_op_known:.2f}" if d_op_known is not None else "h"
    live_key = f"barg|{role}|{d_me:.2f}|{dop_s}|{_tclass(T)}"
    for strict in (True, False):
        rows = [e[5][role] for e in _BARG if match(e, strict)]
        if rows:
            return Ladder(_avg(rows), live_key)
    return None


def nego_ladder(role, my_f, T, opp_f=None):
    """Normalized surplus ladder for my role and valuation multiplier.

    With opp_f (complete information) match the exact valuation pair.
    Without it, average ONLY over alive configs (buyer value > seller
    value): dead configs have all-zero ladders that would drag accept
    thresholds and concession floors to half their correct level.
    """
    _load()
    if not _NEGO:
        return None
    tc = _tclass(T)
    fi = 0 if role == "player_1" else 1
    oi = 1 - fi

    def rows_for(tc_filter):
        out = []
        for e in _NEGO:
            if tc_filter and e[2] != tc:
                continue
            if abs((e[fi] or 0) - my_f) > 0.011:
                continue
            if opp_f is not None:
                if abs((e[oi] or 0) - opp_f) > 0.011:
                    continue
            else:
                fs, fb = (e[0] or 0), (e[1] or 0)
                if fb <= fs:            # dead config: no surplus possible
                    continue
            out.append(e[3][role])
        return out

    rows = rows_for(True) or rows_for(False)
    if not rows and opp_f is not None:      # e.g. dead pair: fall back
        return nego_ladder(role, my_f, T, None)
    if not rows:
        return None
    # A known-opponent ladder must never blend the pooled-across-opponents
    # live cell: it halves the accept bar exactly where the exact pair pays
    # double (measured pct 21.6 in the complete-info cells it poisoned).
    if opp_f is not None:
        live_key = f"nego|{role}|{my_f:.2f}|{opp_f:.2f}|{_tclass(T)}"
    else:
        live_key = f"nego|{role}|{my_f:.2f}|{_tclass(T)}"
    return Ladder(_avg(rows), live_key, ratchet=(opp_f is not None))
