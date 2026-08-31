"""Bargaining (alternating offers, per-round inflation).

Anchors on subgame-perfect equilibrium math (Stahl backward induction on
known finite horizons, Rubinstein closed form otherwise) but clamps to
pool-aware bounds: the score is a payoff percentile against a reference
pool that mostly plays near 50/50, so we never concede the extreme SPE
shares theory alone would justify. Behavioral floors keep human-like
opponents from rejecting out of fairness. Walking away is never played:
rejecting weakly dominates it (continuation value is always >= 0).
"""

from norm import (norm_player, other, pidx, offer_gain, split_action,
                  opp_bucket, allow_message)
from msgs import barg_offer_msg
import oppmodel
import targets
import var_cfg


def _sched(t):
    """Ambition schedule: which pool percentile to target by round."""
    if t <= 2:
        return 90
    if t <= 4:
        return 75
    if t <= 8:
        return 50
    return 25

RHO = 0.97           # per-round survival belief when the horizon is unknown
D_PRIOR = 0.90       # prior on a hidden opponent discount
CAP_GREED = 0.58     # accept any offer at/above this share of the pot
GIVE_CAP = 0.55      # never offer the opponent more than this share early
FLOORS = {"human": 0.37, "hidden": 0.32, "agent": 0.26}


def _deltas(state, me):
    d_me = state.get(f"delta_{pidx(me)}")
    d_me = 1.0 if d_me is None else float(d_me)
    d_op = state.get(f"delta_{pidx(other(me))}")
    d_op = None if d_op is None else float(d_op)
    return d_me, d_op


def _est_opp_delta(hist, M, d_me, me):
    """Invert the opponent's own demands under Rubinstein play, EMA-smoothed."""
    est = None
    opp = other(me)
    for h in hist or []:
        if norm_player(h.get("proposer")) != opp:
            continue
        g = offer_gain(h.get("offer") or {}, opp)
        if g is None or M <= 0:
            continue
        z = min(max(g / M, 0.05), 0.98)
        if d_me <= 0:
            continue
        d = (z - 1 + d_me) / (z * d_me)
        if not (d == d):  # NaN guard
            continue
        d = min(max(d, 0.5), 1.0)
        est = d if est is None else 0.4 * est + 0.6 * d
    return est if est is not None else D_PRIOR


def _last_rejected_to_opp(hist, me):
    """Largest amount we offered the opponent that they rejected."""
    best = None
    opp = other(me)
    for h in hist or []:
        if norm_player(h.get("proposer")) != me:
            continue
        if str(h.get("decision") or "").lower() != "reject":
            continue
        g = offer_gain(h.get("offer") or {}, opp)
        if g is not None:
            best = g if best is None else max(best, g)
    return best


def _spe_keep(M, T, t, d_me, d_op):
    """SPE take (in round-t money) of the proposer at round t, where the
    proposer at round t is ME. Parity handled relative to t."""
    if T is None:
        den = 1.0 - d_me * d_op
        if den < 1e-9:
            return 0.52 * M
        return M * (1.0 - d_op) / den
    W = M
    for r in range(T - 1, t - 1, -1):
        i_propose_at_r = ((r - t) % 2 == 0)
        d_resp = d_op if i_propose_at_r else d_me
        W = M - d_resp * W
    return W


def bargaining_strategy(game: dict) -> dict:
    state = game["game_state"]
    va = game["valid_actions"]
    me = norm_player(game.get("your_player") or state.get("current_player"))
    M = float(state["money_to_divide"])
    t = int(state.get("round") or 1)
    T = state.get("max_rounds")
    T = int(T) if T else None
    hist = state.get("history") or []
    d_me, d_op_known = _deltas(state, me)
    d_op = d_op_known if d_op_known is not None else _est_opp_delta(hist, M, d_me, me)
    bucket = opp_bucket(game)

    d_me_e = d_me if T is not None else d_me * RHO
    d_op_e = d_op if T is not None else d_op * RHO
    ladder = targets.barg_ladder(me, d_me, d_op_known, T,
                                 bool(state.get("complete_information")),
                                 bool(state.get("messages_allowed")))

    if va.get("type") == "offer":
        return _offer(game, state, va, me, M, t, T, d_me, d_op, d_me_e, d_op_e,
                      bucket, hist, ladder)
    return _decide(game, state, me, M, t, T, d_me, d_op, d_op_e, d_me_e,
                   ladder)


def _offer(game, state, va, me, M, t, T, d_me, d_op, d_me_e, d_op_e, bucket,
           hist, ladder=None):
    keep = _spe_keep(M, T, t, d_me_e, d_op_e)
    # Pool-aware clamp: the reference pool settles near 50/50, so even when
    # backward induction says the endgame belongs to the opponent, opening
    # below a majority stance is percentile suicide. Floors below cover the
    # rejection risk on the other side.
    keep = min(max(keep, 0.52 * M), 0.70 * M)
    if ladder is not None:
        # Demand what realizes the scheduled pool percentile after my decay.
        lvl = _sched(t)
        if d_me <= 0.85 and d_op >= 0.95 and lvl > 75:
            lvl = 75      # impatient vs patient: maximize round-1 acceptance
        tgt = ladder.q(lvl)
        if tgt is not None:
            want = tgt * M / max(d_me ** (t - 1), 0.05)
            keep = min(max(keep, want), 0.80 * M)
    # Acceptance dead zone: asks with keep in (0.62, 0.70) get accepted ~9%
    # of the time vs 35-42% at 0.55-0.60; only a patient hand with a real
    # deadline can afford to fish there.
    if 0.62 * M < keep < 0.70 * M and not (d_me >= d_op and T is not None):
        keep = 0.60 * M
    # Opener probe in the hardened pool: every ask at 0.55-0.68 has been
    # rejected 775/775; the 0.48-0.54 band is untested and their counters
    # are rigid regardless of our ask, so probing 0.52 risks nothing.
    if t == 1 and d_me <= 0.95 and not state.get("complete_information"):
        keep = min(keep, 0.52 * M)
    if var_cfg.VARIANT == "A" and t == 1:
        keep = min(keep, 0.50 * M)     # variant A: even-split opener probe
    # Ported from the A-vs-B natural experiment (n=150-180/cell): an even
    # split opener wins exactly where we decay and the opponent does not
    # (CI vs patient: acceptance 37-58% at pct 64-84, vs 9-17% asking
    # more) and in hidden-info d 0.9-0.95. d=1.0 and impatient-opponent
    # openers keep the higher ask (B/current policy wins those).
    if t == 1 and ((state.get("complete_information") and d_me <= 0.95
                    and d_op >= 0.95)
                   or (0.88 <= d_me <= 0.95
                       and not state.get("complete_information"))):
        keep = min(keep, 0.50 * M)
    # After a rejected opener answered with a lowball counter, the pool's
    # accepted-ask mass sits near 0.55: stop asking above it.
    if t >= 3 and hist:
        got = [offer_gain(h.get("offer") or {}, me) for h in hist
               if norm_player(h.get("proposer")) == other(me)]
        got = [x for x in got if x is not None]
        if got and got[-1] <= 0.45 * M:
            keep = min(keep, 0.55 * M)
    y = M - keep                                 # offered to the opponent

    floor = FLOORS[bucket] * M
    rej = _last_rejected_to_opp(hist, me)
    if rej is not None:
        # Concession speed by patience: a truly patient hand (>=0.97) inches
        # and holds the 0.50 atom; a semi-impatient one (0.90-0.95) bleeds
        # ~26-40th percentile by crawling while its own decay bites, so it
        # concedes fast to the pool's 0.50 mass and closes.
        if d_me >= 0.97:
            step, cap_y = 0.025, (0.50 if t <= 5 else 0.55)
        elif d_me >= 0.9:
            step = 0.07
            cap_y = 0.50 if t <= 4 else 0.55
        else:
            step, cap_y = 0.05, 0.55
        floor = max(floor, min(rej + step * M, cap_y * M))

    give_cap = GIVE_CAP * M
    if d_me <= 0.85 and d_op >= 0.95 and t >= 3:
        give_cap = 0.70 * M          # badly impatient vs patient: close NOW
    elif t >= 5 and d_me < d_op:  # I bleed faster: open the cap gradually
        give_cap = min(0.68, GIVE_CAP + 0.03 * (t - 4)) * M
    elif t >= 7:                  # time-based opening against stonewalls
        give_cap = min(0.62, GIVE_CAP + 0.02 * (t - 6)) * M
    elif T is None and t >= 12:  # unlimited-horizon stalemate: inch open
        give_cap = min(0.62, GIVE_CAP + 0.01 * (t - 12)) * M

    y = min(max(y, floor), give_cap)
    action = split_action(M - y, M, me)

    if allow_message(state, va):
        decay_pct = (1.0 - d_op) * 100.0
        action["message"] = barg_offer_msg(
            action["alice_gain" if me == "player_2" else "bob_gain"],
            100.0 * y / M, decay_pct, t, M)
    return action


def _decide(game, state, me, M, t, T, d_me, d_op, d_op_e, d_me_e,
            ladder=None):
    offered = offer_gain(state.get("last_offer") or {}, me)
    if offered is None:
        return {"decision": "accept"}

    if T is not None and t >= T:                 # last chance: anything beats 0
        return {"decision": "accept" if offered > 0 else "reject"}

    # Pool-percentile accept: if taking this now already realizes the
    # scheduled percentile of the reference pool, take it.
    realized = offered * (d_me ** (t - 1))
    if ladder is not None:
        tgt = ladder.q(_sched(t))
        if tgt is not None and realized >= tgt * M:
            return {"decision": "accept"}
        # Semi-impatient responder (0.90-0.95) facing a same-or-more-patient
        # opponent on a finite horizon: chasing q90 early cost 18-31 points
        # against the accept-the-opener counterfactual; take the pool median.
        known = bool(state.get("complete_information"))
        if (d_me <= 0.95 and t <= 4
                and ((not known) or d_op >= d_me - 0.01)):
            # Pool hardened Aug 22: 0/775 of our asks accepted since, so
            # the opponent's early offer IS the deal. Take the median bar
            # (capped at 0.45) on any horizon, and at round 1, where the
            # offer is un-decayed, even the q25 bar beats waiting.
            q50 = ladder.q(50)
            if q50 is not None and realized >= min(q50, 0.45) * M:
                return {"decision": "accept"}
            if t == 1:
                q25 = ladder.q(25)
                if q25 is not None and realized >= q25 * M:
                    return {"decision": "accept"}
        if var_cfg.VARIANT == "A" and t <= 2:
            qa = ladder.q(25)          # variant A: take q25 immediately
            if qa is not None and realized >= qa * M:
                return {"decision": "accept"}
        if d_me <= 0.85 and d_op >= 0.95 and t >= 2:
            # Impatient vs patient: holding out is a guaranteed loss; take
            # anything at or above the pool's q25 for this config.
            q25 = ladder.q(25)
            if q25 is not None and realized >= q25 * M:
                return {"decision": "accept"}

    # Disclosed agents have accepted zero counters of ours since the pool
    # hardened; their opener is the best un-decayed value we will ever see.
    if t == 1 and d_me <= 0.95:
        prof = oppmodel.profile(game)
        floor_prof = prof.get("barg_accept_floor")
        if floor_prof is None and oppmodel.is_disclosed_agent(game):
            floor_prof = 0.42
        if floor_prof is not None and offered >= floor_prof * M:
            return {"decision": "accept"}

    keep_next = _spe_keep(M, T, t + 1, d_me_e, d_op_e)
    keep_next = min(max(keep_next, 0.40 * M), 0.70 * M)
    thr = 0.90 * d_me * keep_next                # deal-risk haircut
    cap_greed = CAP_GREED
    if ladder is not None:
        q90 = ladder.q(90)
        if q90 is not None:                      # hold out longer when the
            cap_greed = max(CAP_GREED, min(q90, 0.75))   # pool's top does
    thr = min(thr, cap_greed * M)

    # No-progress detector: if the opponent's last two offers to me barely
    # moved, take their standing demand early instead of bleeding decay
    # (55% of our agreements were late caves at realized 0.35 vs their
    # stagnant 0.43 standing offer).
    stagnant = False
    if t >= 3 and d_me <= 0.95:
        to_me = [offer_gain(h.get("offer") or {}, me)
                 for h in (state.get("history") or [])
                 if norm_player(h.get("proposer")) == other(me)]
        to_me = [x for x in to_me if x is not None]
        if len(to_me) >= 2 and (to_me[-1] - to_me[-2]) < 0.03 * M:
            stagnant = True
            thr = min(thr, 0.38 * M)
    # Tail rescue: kill the round-8-plus crumb spiral before decay turns a
    # standing 0.3M into a realized 0.05M at the opponent's final ultimatum.
    if stagnant and d_me <= 0.95 and T is not None and t >= 5:
        thr = min(thr, 0.30 * M)
    if t >= 8 and d_me <= 0.95:
        thr = min(thr, 0.25 * M)

    # Deep rounds: inflation makes stalemates worthless, take real money.
    if t >= 6:
        thr = min(thr, 0.40 * M)
    if t >= 9:
        thr = min(thr, 0.28 * M)
    if t >= 12:
        thr = min(thr, 0.15 * M)

    # Patient-side pool floor: with essentially no decay, never accept
    # below the pool's central mass (atom at 0.50) in early-mid rounds.
    # Gated at 0.97: applying it to 0.90-0.95 hands forced rejections whose
    # decayed eventual settlements landed 18-31 points below the opener.
    if (ladder is not None and d_me >= 0.97 and not stagnant
            and t <= (8 if d_me >= 0.99 else 4)
            and (T is None or t <= T - 2)):
        qf = ladder.q(50)
        if qf is not None:
            thr = max(thr, min(qf * M / max(d_me ** (t - 1), 0.05),
                               0.55 * M))

    # Stonewall salvage: a game this deep is already lost on percentile;
    # crumbs beat zero and the concurrency slot is worth more than pride.
    if t >= 25:
        thr = min(thr, 0.03 * M)
    if t >= 40:
        thr = min(thr, 1e-9)

    return {"decision": "accept" if offered >= thr else "reject"}
