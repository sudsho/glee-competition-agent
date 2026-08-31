"""Message templates. All content is strategic but stays credible; the
server replays every message into the opponent's prompt for the rest of
the game, so short, concrete, numerate lines work best. Hard cap 2000
chars server-side; we stay far under it."""


def _pick(variants, seed: int) -> str:
    return variants[seed % len(variants)]


def barg_offer_msg(opp_amount, opp_share_pct, opp_decay_pct, round_no, money) -> str:
    if opp_decay_pct >= 1:
        variants = [
            (f"Offering you {opp_amount} right now. Every round you wait, the pot loses "
             f"~{opp_decay_pct:.0f}% of its value to you, so a smaller slice later is worth "
             f"less than this today. Take the sure money."),
            (f"{opp_amount} guaranteed, this round. Delay costs you ~{opp_decay_pct:.0f}% per "
             f"round in real value; even a bigger nominal share next round nets you less. "
             f"This is the efficient deal."),
            (f"Let's not burn value. You get {opp_amount} now; with your ~{opp_decay_pct:.0f}% "
             f"per-round inflation, waiting for more is a losing trade. Deal?"),
        ]
    else:
        variants = [
            (f"Offering you {opp_amount} ({opp_share_pct:.0f}%). It's a clean, fair-minded "
             f"split we can both accept right away instead of grinding rounds."),
            (f"{opp_amount} to you, settled now. Long back-and-forth helps neither of us; "
             f"this is a solid share and I'm ready to close."),
        ]
    return _pick(variants, round_no)


def nego_seller_msg(price, round_no) -> str:
    variants = [
        (f"Asking {price}. I know exactly what this is worth and I have room to be patient, "
         f"but I'd rather close with you today at a serious price."),
        (f"{price} is my number. I'm not in a rush to sell below value; if that works for "
         f"you, we can shake on it now."),
        (f"I can do {price}. That already reflects a real concession on my side; meet me "
         f"there and it's yours."),
    ]
    return _pick(variants, round_no)


def nego_buyer_msg(price, round_no) -> str:
    variants = [
        (f"I can pay {price}. That's an honest number against my budget; above that the "
         f"deal stops making sense for me."),
        (f"{price} is where I am. I like the product but I have a hard ceiling, and this "
         f"offer is close to it."),
        (f"Offering {price}. I'd rather buy from you than walk, but I won't chase; "
         f"take it and we're done."),
    ]
    return _pick(variants, round_no)


def pers_recommend_msg(round_no, sold_high, sold_total) -> str:
    track = ""
    if sold_total >= 2 and sold_high == sold_total:
        track = (f" Check the history: all {sold_total} products you bought from me were "
                 f"high quality. I protect that record.")
    elif sold_total >= 3 and sold_high >= sold_total - 1:
        track = (f" My track record so far speaks for itself; I only push products I "
                 f"believe in.")
    variants = [
        f"This round's product is a genuinely good one. I recommend buying it.{track}",
        f"I'd buy this one myself. Strong recommendation.{track}",
        f"This is exactly the kind of product worth your money. Take it.{track}",
    ]
    return _pick(variants, round_no)


def pers_skip_msg(round_no) -> str:
    variants = [
        ("Honest note: I'd skip this one. I only recommend products that are worth your "
         "money, and this round's item doesn't clear my bar. Save your cash for the next."),
        ("Not this one. I want you to keep trusting my recommendations, so I'm telling "
         "you straight: pass this round."),
        ("Pass on this round's product. When I say buy, I want it to mean something."),
    ]
    return _pick(variants, round_no)
