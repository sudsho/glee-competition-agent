"""Name-conditioned opponent profiles, seeded from measured season behavior.

Sources (season analysis, n>=25 games per name): 'Quantile' never accepts
round-1 bargaining offers but reliably accepts keep <= 0.52-0.535 in rounds
3-5; the negotiation stonewaller set has never accepted a counter-price of
ours (all their deals close by us accepting their price); 'delta' behaves
like Quantile on the accept side. Profiles only sharpen play against named
opponents; hidden games keep the population policy.
"""

OPP_BOOK = {
    # Stonewaller set validated Aug 23 on 19k outcomes: names that accepted
    # OUR counter-price in at most 15% of their deals with us (n>=8 deals).
    # king and lab2a.ai were dropped: fuller data shows them conceding in
    # 34-37% of deals.
    "quantile": {"barg_accept_floor": 0.42, "nego_stonewaller": True},
    "delta":    {"barg_accept_floor": 0.42, "nego_stonewaller": True},
    "gamma":    {"nego_stonewaller": True},
    "theta":    {"nego_stonewaller": True},
    "tester1":  {"nego_stonewaller": True},
    "agenta":   {"nego_stonewaller": True},
    "test4":    {"nego_stonewaller": True},
    "test 1":   {"nego_stonewaller": True},
    "expert":   {"nego_stonewaller": True},
    "xx":       {"nego_stonewaller": True},
    "c-agent":  {"nego_stonewaller": True},
}


def profile(game: dict) -> dict:
    name = ((game.get("opponent") or {}).get("name") or "").strip().lower()
    return OPP_BOOK.get(name, {})


def is_disclosed_agent(game: dict) -> bool:
    return ((game.get("opponent") or {}).get("type") or "") == "agent"
