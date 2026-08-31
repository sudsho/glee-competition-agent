"""Persuasion (repeated seller-buyer information game, T rounds).

Seller: Bayesian-persuasion mixing as the ceiling reference (always
recommend high quality; recommend low with rate q* = p(v-c)/((1-p)c)),
implemented as a trust-budget rule that front-loads honesty, plus an
endgame switch to all-recommend once the buyer can no longer punish
within the horizon. Buyer: beta-binomial posterior on recommendation
accuracy with optimism early, distrust margin late, and exponential
forgetting so a defecting seller is punished fast but not forever.
"""

import random

from norm import norm_player
from msgs import pers_recommend_msg, pers_skip_msg
import var_cfg

ALPHA_SLACK = 0.85   # fraction of q* the seller actually uses
M_S = 0.04           # seller trust safety margin
PHI = 0.8            # endgame hedge factor
PRIOR_W = 3.0        # buyer prior pseudo-count weight
Z0 = 1.0             # buyer initial optimism
GAMMA = 0.90         # forgetting: one bad delivery decays in ~6 rounds, not 18
NEG_WORDS = ("not ", "n't", "skip", "pass", "avoid", "low quality", "poor",
             "bad ", "don't", "do not", "save your", "hold off", "wouldn't",
             "no.", "no,", "honest note", "isn't worth")


def persuasion_strategy(game: dict) -> dict:
    state = game["game_state"]
    action_type = (game.get("valid_actions") or {}).get("type", "")
    if action_type in ("seller_message", "seller_recommendation"):
        return _seller(state, action_type)
    return _buyer(state)


# ---------------------------------------------------------------- seller

def _seller_ledger(hist):
    """What the buyer has verified: bought rounds and their quality. A lie
    the buyer never bought is invisible to them and costs no trust."""
    n = h = low_seen = lies = 0
    for r in hist or []:
        bought = r.get("bought")
        if bought is None:
            bought = str(r.get("buyer_decision", "")).lower() in ("yes", "true", "buy")
        q = str(r.get("quality") or r.get("round_quality") or "").lower()
        if q == "low":
            low_seen += 1
            if bought:
                lies += 1          # verified lie: bought a low on my word
        if bought and q in ("high", "low"):
            n += 1
            if q == "high":
                h += 1
    return n, h, low_seen, lies


def _seller(state, action_type):
    t = int(state.get("round") or 1)
    T = int(state.get("total_rounds") or 20)
    c = float(state.get("product_price") or 1.0)
    p = float(state.get("p") or 0.5)
    v = state.get("v")
    v = float(v) if v is not None else None
    quality = str(state.get("current_quality") or "high").lower()
    hist = state.get("history") or []
    n, h, low_seen, lies = _seller_ledger(hist)
    R = T - t + 1

    # (A never-buyer escape used to force recommendations here; season data
    # showed 0/144 conversions from it, while it poisoned the first purchase
    # of late-probing buyers. Removed.)
    recommend = _seller_recommend(quality, c, p, v, n, h, low_seen, lies, R,
                                  text_mode=(action_type == "seller_message"))

    if action_type == "seller_recommendation":
        return {"decision": "yes" if recommend else "no"}
    if recommend:
        return {"message": pers_recommend_msg(t, h, n)}
    return {"message": pers_skip_msg(t)}


def _seller_recommend(quality, c, p, v, n, h, low_seen, lies, R,
                      text_mode=False):
    if quality == "high":
        return True

    # Low quality from here on.
    if v is None:
        # Value unknown: q* is incomputable. At high p the pool's top
        # sellers sell nearly every round, so recommend everything there;
        # otherwise stay honest until trust no longer matters. Engaged
        # buyers show no measurable post-lie punishment, so the endgame
        # window widens to 4 rounds once they have bought.
        return p >= 0.75 or R <= (4 if n >= 1 else 2)

    if v <= c:
        return True          # rational buyers never buy; only upside remains

    if var_cfg.VARIANT == "C":
        return R <= 3        # variant C: pure honesty until the last 3 rounds

    # Always-recommend exactly where the buyer profits from obeying blind:
    # true blind-buy configs, plus the low-p knife edge (p*v == c) where our
    # low recommendations already convert at 0.75-0.90. High-p configs with
    # p*v < c looked similar but scored pct 26-36 under always-recommend;
    # they belong to the trust budget below.
    if p * v > c or (p < 0.6 and p * v >= c * 0.999):
        return True

    # Empirical pool finding: buyer TRUST, not the persuasion optimum, is
    # the binding constraint even in blind-buy configurations (pool buyers
    # under-buy and punish detected lies hard). So the trust budget governs
    # everywhere v > c, and the lying rate is hard-capped well below the
    # commitment-optimal q*.
    ratio = c / v
    a0 = (min(0.95, ratio + 0.05)) * PRIOR_W
    b0 = PRIOR_W - a0

    # Endgame: if the buyer cannot detect defection within the horizon, defect.
    if ratio > p:
        l_det = max(0.0, ((a0 + h) - ratio * (PRIOR_W + n)) / (ratio - p))
        if R <= PHI * (c / max(p * v, 1e-9)) * l_det:
            return True
    elif R <= 2:
        return True          # blind-buy configs: burn only the last rounds

    q_star = min(1.0, p * (v - c) / max((1.0 - p) * c, 1e-9))
    cap = 0.55 if p >= 0.45 else 0.35   # pool q75 at p=0.5 lies on ~half of lows
    margin = M_S
    if text_mode:
        # Text buyers punish detected lies far harder than binary ones
        # (our own text cells score 8-12 points under their binary twins).
        cap = min(cap, 0.25)
        margin = 2.0 * M_S
    q_eff = min(ALPHA_SLACK * q_star, cap)
    trust_ok = (a0 + h) / (PRIOR_W + n + 1.0) > ratio + margin
    budget_ok = lies < q_eff * max(low_seen, 1)
    return trust_ok and budget_ok and (random.random() < q_eff)


# ---------------------------------------------------------------- buyer

POS_PHRASES = ("recommend buying", "buy it", "buy this", "go for it",
               "worth buying", "great buy", "highly recommend",
               "don't miss", "do not miss")


def _rec_from_message(state) -> bool:
    """True = the seller is recommending a buy. Positive phrases win over
    incidental negative substrings ('you do not want to miss this')."""
    mtype = str(state.get("seller_message_type") or "").lower()
    msg = state.get("seller_message")
    if msg is None:
        return True
    s = str(msg).strip().lower()
    if mtype == "binary" or s in ("yes", "no"):
        return s.startswith("y")
    if any(ph in s for ph in POS_PHRASES):
        return True
    return not any(w in s for w in NEG_WORDS)


def _buyer_ledger(hist, t):
    """Decayed pseudo-counts from rounds where we bought (quality verified),
    the last round we bought in (0 if never), and the raw count of verified
    low-quality purchases."""
    a = b = 0.0
    b_raw = 0
    last_buy = 0
    for r in hist or []:
        bought = r.get("bought")
        if bought is None:
            bought = str(r.get("buyer_decision", "")).lower() in ("yes", "true", "buy")
        if not bought:
            continue
        rnd = int(r.get("round") or t)
        last_buy = max(last_buy, rnd)
        q = str(r.get("quality") or r.get("round_quality") or "").lower()
        if q not in ("high", "low"):
            continue
        w = GAMMA ** max(0, t - rnd)
        if q == "high":
            a += w
        else:
            b += w
            b_raw += 1
    return a, b, last_buy, b_raw


def _buyer(state):
    t = int(state.get("round") or 1)
    T = int(state.get("total_rounds") or 20)
    c = float(state.get("product_price") or 1.0)
    p = float(state.get("p") or 0.5)
    v = float(state.get("v") or 0.0)
    u = float(state.get("u") or 0.0)
    hist = state.get("history") or []

    if v <= c:
        return {"decision": "no"}          # even high quality loses money

    rec_buy = _rec_from_message(state)
    a_obs, b_obs, last_buy, b_raw = _buyer_ledger(hist, t)
    n_obs = a_obs + b_obs
    binary = str(state.get("seller_message_type") or "").lower() == "binary"

    # Knife-edge low-v configs must NOT bypass the trust machinery: blind
    # EV ~ 0 there means every extra low bought is pure loss (measured:
    # 8.3 lows/game and pool-median money left on the table).
    ev_blind = p * v + (1.0 - p) * u
    knife_lowv = abs(ev_blind - c) < 0.05 * c and v < 2.5 * c
    if ev_blind >= c * 0.98 and not knife_lowv:
        # Blind buying is profitable, so the seller's signal can only help
        # or hurt. Obey a PASS only while the seller's verified track record
        # is consistent with an informative recommendation; an adverse
        # seller (recommending lows, discouraging highs) gets ignored and
        # we harvest the positive blind EV every round.
        informative = n_obs < 4 or (a_obs / n_obs) >= p - 0.05
        if rec_buy or not informative:
            return {"decision": "yes"}
        return {"decision": "no"}

    if not rec_buy:
        # A strategic seller never discourages a profitable sale.
        return {"decision": "no"}

    # Stop-loss: one verified low purchase ends all buying in text games
    # (text sellers who lie once keep lying) and in low-value binary cells
    # where a single low already erases several highs of surplus.
    lowcell = v <= 1.3 * c and p <= 0.5
    if b_raw >= 1 and (not binary or lowcell or var_cfg.VARIANT == "C"):
        return {"decision": "no"}

    R = T - t + 1
    proven_liar = a_obs == 0 and b_obs >= 1.2   # decayed counts: ~2 recent lies
    if (t - last_buy) >= 4 and R > 3 and v >= 2 * c and not proven_liar:
        # Anti-lockout probe: a grim no-buy tail (observed 15-18 round
        # freezes) loses far more than an occasional test purchase. But a
        # seller whose every verified rec was a lie gets no more probes.
        return {"decision": "yes"}

    ratio = c / v
    # Low-value cells start SKEPTICAL: the seller must earn the first sale
    # (measured: optimistic priors there bought mostly lies).
    mu0 = max(0.05, ratio - 0.10) if lowcell else min(0.95, ratio + 0.05)
    a0 = mu0 * PRIOR_W
    b0 = PRIOR_W - a0
    a, b = a0 + a_obs, b0 + b_obs
    n = a + b
    mu = a / n

    t_explore = max(3.0, T / 4.0)
    z_floor = 0.0 if lowcell else (0.25 if t < T / 2 else 0.0)
    z = Z0 * max(z_floor, 1.0 - t / t_explore)
    ucb = mu + z * (mu * (1.0 - mu) / n) ** 0.5

    R = T - t + 1
    r_guard = max(2.0, 3.0 * c / max(p * v, 1e-9))
    m_end = 0.1 * c * max(0.0, 1.0 - R / r_guard)

    ev = ucb * v + (1.0 - ucb) * u
    return {"decision": "yes" if ev > c + m_end else "no"}
