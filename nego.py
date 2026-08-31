"""Negotiation (bilateral trade, private valuations, no discounting).

Anchor high (seller) / low (buyer), concede on a Boulware schedule toward
an inferred estimate of the opponent's reservation, accept whenever the
standing offer beats our own plan's continuation, and always accept any
positive surplus on the final round. Valuations sit on a known grid
{0.8, 1.0, 1.2, 1.5} x M (M a power of 10), which makes reservation
inference nearly discrete. Walking away is played only to save clock in
clearly ZOPA-free unlimited games.
"""

from norm import norm_player, other, allow_message
from msgs import nego_seller_msg, nego_buyer_msg
import oppmodel
import targets
import var_cfg

ZOPA_SHARE = 0.50 if var_cfg.VARIANT == "B" else 0.60
GAMBIT_FRAC = 0.75 if var_cfg.VARIANT == "B" else 0.60
CROSS_LATE = 5 if var_cfg.VARIANT == "B" else 4
CROSS_FRAC = 0.94 if var_cfg.VARIANT == "B" else 0.96


def _sched_q(t, T):
    """Which pool percentile of surplus to insist on by round. Alive-only
    ladders sit roughly 2x higher than the old dead-inclusive ones, so q25
    here corresponds to the old q50 bar."""
    if t <= 2:
        return 90
    if T is None:
        if t <= 6:
            return 75
        if t <= 12:
            return 50
        return 25
    if t <= 4:
        return 75
    if t == T - 1:
        return 50     # the complete-info final-counter gambit backstops this
    if t >= T - 2:
        return 25
    return 50

GRID = [0.8, 1.0, 1.2, 1.5]
BETA = 0.35          # Boulware concession exponent (<1 = stubborn early)
T_EFF_UNLIMITED = 20


def _scale(myv: float) -> float:
    for k in range(0, 13):
        m = 10.0 ** k
        if 0.75 <= myv / m <= 1.55:
            return m
    return myv


def _opp_prices(hist, opp):
    """Chronological price signals from the opponent (offers + counters)."""
    out = []
    for h in hist or []:
        offer = h.get("offer") or {}
        if norm_player(offer.get("from_player")) == opp and offer.get("price") is not None:
            out.append(float(offer["price"]))
        co = h.get("counteroffer")
        if co is not None and norm_player(h.get("decided_by")) == opp:
            try:
                out.append(float(co if not isinstance(co, dict) else co.get("price")))
            except (TypeError, ValueError):
                pass
    return out


def _aitken(prices):
    if len(prices) < 3:
        return None
    o1, o2, o3 = prices[-3], prices[-2], prices[-1]
    d1, d2 = o2 - o1, o3 - o2
    if abs(d1) < 1e-12:
        return o3
    r = d2 / d1
    if not (0.0 < r < 0.95):
        return o3
    return o3 + r * d2 / (1.0 - r)


def _est_opp_reserve(opp_prices, i_am_seller, M, oppv):
    """Point estimate of the opponent's reservation price."""
    if oppv is not None:
        return oppv
    cands = [f * M for f in GRID]
    if i_am_seller:
        # Opponent is the buyer: any price they offered is below their max.
        lb = max(opp_prices) if opp_prices else None
        if lb is not None:
            cands = [c for c in cands if c >= lb - 1e-9] or [GRID[-1] * M]
    else:
        ub = min(opp_prices) if opp_prices else None
        if ub is not None:
            cands = [c for c in cands if c <= ub + 1e-9] or [GRID[0] * M]
    # A frozen opponent's prices are their ANCHOR, not their reservation;
    # extrapolating them would let a stonewaller dictate our estimate.
    frozen = (len(opp_prices) >= 3
              and abs(opp_prices[-1] - opp_prices[-3]) < 0.01 * M)
    ext = _aitken(opp_prices) if not frozen else None
    if ext is not None:
        lo, hi = min(cands), max(cands)
        ext = min(max(ext, lo), hi)
        return min(cands, key=lambda c: abs(c - ext))
    return sum(cands) / len(cands)


def _planned(t, T, p0, dest, beta=BETA):
    """Faratin time-dependent offer curve from anchor p0 toward dest."""
    t_eff = T if T else T_EFF_UNLIMITED
    f = (min(t, t_eff) / t_eff) ** (1.0 / beta)
    return dest + (p0 - dest) * (1.0 - f)


def negotiation_strategy(game: dict) -> dict:
    state = game["game_state"]
    va = game["valid_actions"]
    me = norm_player(game.get("your_player") or state.get("current_player"))
    opp = other(me)
    role = (state.get(f"{me}_role") or
            ("seller" if me == "player_1" else "buyer")).lower()
    seller = role == "seller"
    myv = float(state[f"{me}_value"])
    oppv = state.get(f"{opp}_value")
    oppv = float(oppv) if oppv is not None else None
    M = _scale(myv)
    t = int(state.get("round") or 1)
    T = state.get("max_rounds")
    T = int(T) if T else None
    hist = state.get("history") or []
    opp_prices = _opp_prices(hist, opp)
    reserve_hat = _est_opp_reserve(opp_prices, seller, M, oppv)

    if va.get("type") == "offer":
        price = _offer_price(seller, myv, oppv, reserve_hat, M, t, T)
        price = round(price, 2)
        action = {"product_price": price}
        if allow_message(state, va):
            action["message"] = (nego_seller_msg(price, t) if seller
                                 else nego_buyer_msg(price, t))
        return action

    return _decision(game, state, va, seller, myv, oppv, reserve_hat, M, t, T,
                     opp_prices)


def _margin(seller, myv, M, T, oppv=None):
    """Concession floor margin: hold out for at least the pool-median surplus."""
    m = 0.04
    ladder = targets.nego_ladder("player_1" if seller else "player_2",
                                 myv / M, T,
                                 oppv / M if oppv is not None else None)
    if ladder is not None:
        q50 = ladder.q(50)
        if q50 is not None:
            m = max(m, q50)
    return m * M


def _f_next(myv, M):
    """Lowest grid value strictly above my value: the price the seller must
    cross for the cheapest viable buyer to ever accept."""
    for f in GRID:
        if f * M > myv * 1.001:
            return f * M
    return None


def _offer_price(seller, myv, oppv, reserve_hat, M, t, T):
    last_chance = (T is not None and t >= T - 1) or (T is None and t >= 18)
    if seller:
        if T == 1:
            return _tioli_seller(myv, oppv, M)
        if last_chance:
            # Season data: counters that become the final-round offer are
            # rejected 89% of the time when they hold the pool-median
            # margin. A thin-margin close dominates the near-certain zero
            # on finite horizons; unlimited games keep a partial margin
            # (their standing offers were already screened by our accepts).
            m = _margin(seller, myv, M, T, oppv)
            hold = 0.005 * M if T is not None else 0.4 * m
            return max(reserve_hat * 0.995, myv + hold, myv * 1.01)
        anchor = (oppv * 0.995 if oppv is not None else GRID[-1] * M * 0.99)
        anchor = max(anchor, myv * 1.05)
        m = _margin(seller, myv, M, T, oppv)
        if oppv is not None and oppv > myv:
            # Complete info: a destination pinned at 0.97x the buyer's value
            # was accepted 0/349 times; concede to a ZOPA share so soft
            # opponents can actually close mid-game.
            dest = myv + max(m, ZOPA_SHARE * (oppv - myv))
        else:
            dest = max(myv + m, min(reserve_hat * 0.97, anchor))
        dest = min(dest, anchor)
        price = max(_planned(t, T, anchor, dest), myv * 1.01)
        # Guarantee the cheapest viable buyer sees an acceptable price with
        # real surplus, twice, before the game dies.
        if oppv is None:
            fn = _f_next(myv, M)
            late = (T is not None and t >= T - CROSS_LATE) or \
                   (T is None and t >= 8)
            if fn is not None and late and reserve_hat <= fn * 1.05:
                price = min(price, CROSS_FRAC * fn)
        return max(price, myv * 1.01)
    # buyer
    if T == 1:
        p = GRID[0] * M * 0.99
        return min(p, myv * 0.98)
    if last_chance:
        # Mirror of the seller rule above: thin-margin close on finite
        # horizons (final-round counters holding real margin die 89% of
        # the time), partial margin in unlimited games.
        m = _margin(seller, myv, M, T, oppv)
        hold = 0.005 * M if T is not None else 0.4 * m
        return min(reserve_hat * 1.005, myv - hold, myv * 0.99)
    # 0.82M rather than an insulting lowball: three opponent sellers have
    # rage-quit round-2 against 0.75M anchors, and the pool's top buyer
    # prices sit near or above M anyway.
    anchor = (oppv * 1.005 if oppv is not None else 0.82 * M)
    anchor = min(anchor, myv * 0.95)
    m = _margin(seller, myv, M, T, oppv)
    if oppv is not None and oppv < myv:
        dest = myv - max(m, ZOPA_SHARE * (myv - oppv))
    else:
        dest = min(myv - m, max(reserve_hat * 1.03, anchor))
    dest = max(dest, anchor)
    return min(_planned(t, T, anchor, dest), myv * 0.99)


def _tioli_seller(myv, oppv, M):
    """Single-round take-it-or-leave-it ask."""
    if oppv is not None:
        return oppv * 0.99 if oppv > myv else myv * 1.3
    # Low own value: percentile beats money-EV. Pool fs<=1.0 sellers who ask
    # just under the NEXT grid value close reliably and land q90-level;
    # our old EV-max ask of 1.188M was rejected every time (0-for-3).
    if myv / M <= 1.01:
        fn = _f_next(myv, M)
        if fn is not None:
            return 0.995 * fn
    best, best_ev = myv * 1.3, 0.0
    n = len(GRID)
    for i, f in enumerate(GRID):
        p = f * M * 0.99
        if p <= myv:
            continue
        p_accept = (n - i) / n          # uniform prior over the buyer grid
        ev = (p - myv) * p_accept
        if ev > best_ev:
            best, best_ev = p, ev
    return best


def _decision(game, state, va, seller, myv, oppv, reserve_hat, M, t, T,
              opp_prices):
    last = state.get("last_offer") or {}
    p = last.get("price")
    if p is None:
        return {"decision": "AcceptOffer"}
    p = float(p)
    surplus = (p - myv) if seller else (myv - p)
    final = T is not None and t >= T

    if surplus > 0:
        if final:
            return {"decision": "AcceptOffer"}
        zopa = abs(oppv - myv) if oppv is not None else None
        stonewaller = bool(oppmodel.profile(game).get("nego_stonewaller"))
        # Named stonewallers never accept our counters, but with no
        # discounting their frozen price is still there near the deadline,
        # so wait for it: the option that they concede stays alive.
        if stonewaller:
            stone_time = (T is not None and t >= T - 2) or \
                         (T is None and t >= 12)
            floor_s = max(0.03 * M, 0.20 * zopa) if zopa else 0.03 * M
            if stone_time and surplus >= floor_s:
                return {"decision": "AcceptOffer"}
        # Hold for the final-counter gambit: when our counter will become
        # the round-T offer (parity: T-1-t even) in an alive complete-info
        # game, no earlier accept path may fire below the gambit share.
        # Measured close rate of the gambit: 251/263 (95.4%) at pct ~94.
        zopa_signed = ((oppv - myv) if seller else (myv - oppv)) \
            if oppv is not None else None
        hold = (T is not None and zopa_signed is not None and zopa_signed > 0
                and not stonewaller and t < T and (T - 1 - t) % 2 == 0
                and surplus < GAMBIT_FRAC * zopa_signed)
        nxt = _offer_price(seller, myv, oppv, reserve_hat, M, t + 1, T)
        nxt_surplus = (nxt - myv) if seller else (myv - nxt)
        if not hold and surplus >= nxt_surplus * 0.98:
            return {"decision": "AcceptOffer"}
        # Pool-percentile accept: in thin-margin configurations (e.g. equal
        # valuations) even a tiny surplus is top-decile play; the ladder
        # knows that, a fixed fraction rule does not.
        ladder = targets.nego_ladder("player_1" if seller else "player_2",
                                     myv / M, T,
                                     oppv / M if oppv is not None else None)
        if ladder is not None:
            tgt = ladder.q(_sched_q(t, T))
            if not hold and tgt is not None \
                    and surplus >= max(tgt, 0.004) * M:
                return {"decision": "AcceptOffer"}
        # Deep-round salvage in unlimited games: any real crumb beats the
        # certain zero of a stalemate walk. With a KNOWN wide ZOPA, crumbs
        # are premature until much later: waiting is free and their price
        # may still cross.
        if T is None and t >= 22 and surplus >= 0.005 * M:
            if zopa is not None and zopa >= 0.1 * M:
                if t >= 35 and surplus >= 0.02 * M:
                    return {"decision": "AcceptOffer"}
            else:
                return {"decision": "AcceptOffer"}
        potential = max((reserve_hat - myv) if seller else (myv - reserve_hat),
                        surplus)
        need = max(0.45 if T is None else 0.30, 0.75 - 0.05 * (t - 1))
        if T is not None and t >= T - 1:
            # Final-counter gambit (complete info): the decider at T-1
            # proposes at T, and a thin-margin close there has been accepted
            # 10/10 times ever fired. Fall through to reject-and-counter
            # unless the standing offer already gives us most of the ZOPA.
            gambit = (zopa is not None and t == T - 1 and not stonewaller
                      and surplus < GAMBIT_FRAC * zopa)
            if not gambit:
                need = 0.05
        if hold:
            need = max(need, GAMBIT_FRAC)
        if potential <= 0 or surplus / potential >= need:
            return {"decision": "AcceptOffer"}

    if final:
        return {"decision": "RejectOffer"}   # no counter exists on the last round

    # Walking never beats continuing under no discounting, so walk ONLY when
    # the config is structurally dead for us (no grid value the opponent
    # could hold would ever yield surplus) or a deep stalemate with a frozen
    # opponent. A fs=0.8 seller has 75% prior ZOPA and must never walk.
    if T is None and opp_prices:
        dead_for_me = (seller and myv / M >= 1.45) or \
                      (not seller and myv / M <= 0.85)
        best = (max(opp_prices) - myv) if seller else (myv - min(opp_prices))
        if dead_for_me and t >= 4 and best < 0.005 * M:
            return {"decision": "WalkAway"}
        frozen = len(opp_prices) >= 3 and \
            abs(opp_prices[-1] - opp_prices[-3]) < 0.002 * M
        if t >= 25 and frozen and best < 0.02 * M and surplus < 0.005 * M:
            return {"decision": "WalkAway"}

    counter = _offer_price(seller, myv, oppv, reserve_hat, M, t + 1, T)
    counter = round(max(counter, myv * 1.01) if seller
                    else min(counter, myv * 0.99), 2)
    action = {"decision": "RejectOffer", "product_price": counter}
    if allow_message(state, va):
        action["message"] = (nego_seller_msg(counter, t) if seller
                             else nego_buyer_msg(counter, t))
    return action
