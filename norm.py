"""Shared helpers: player naming, offer parsing, exact-sum money pairs."""

P1 = {"player_1", "alice", "p1", "1", "seller"}
P2 = {"player_2", "bob", "p2", "2", "buyer"}


def norm_player(name) -> str:
    s = str(name or "").strip().lower()
    if s in P1:
        return "player_1"
    if s in P2:
        return "player_2"
    return "player_2" if ("2" in s or "bob" in s) else "player_1"


def other(p: str) -> str:
    return "player_2" if p == "player_1" else "player_1"


def pidx(p: str) -> int:
    return 1 if p == "player_1" else 2


def offer_gain(offer: dict, player: str):
    """Read a player's gain from a bargaining offer under any key style."""
    if not offer:
        return None
    label = "alice" if player == "player_1" else "bob"
    for k in (f"{player}_gain", f"{label}_gain"):
        if k in offer and offer[k] is not None:
            try:
                return float(offer[k])
            except (TypeError, ValueError):
                return None
    return None


def split_action(my_amount: float, money: float, me: str) -> dict:
    """Bargaining offer dict whose gains sum EXACTLY to money.

    Integer amounts when the pot is integral (it always is on the published
    grid) so float rounding can never invalidate the move.
    """
    if abs(money - round(money)) < 1e-9:
        total = int(round(money))
        mine = int(round(my_amount))
        mine = max(0, min(total, mine))
        theirs = total - mine
    else:
        mine = round(max(0.0, min(money, my_amount)), 2)
        theirs = round(money - mine, 2)
    if me == "player_1":
        return {"alice_gain": mine, "bob_gain": theirs}
    return {"alice_gain": theirs, "bob_gain": mine}


def opp_bucket(game: dict) -> str:
    """'human' | 'agent' | 'hidden' — how to treat the opponent."""
    o = ((game.get("opponent") or {}).get("type") or "hidden").lower()
    if o in ("human", "agent"):
        return o
    if o == "llm":
        return "agent"
    return "hidden"


def allow_message(state: dict, valid_actions: dict) -> bool:
    if not state.get("messages_allowed", True):
        return False
    fields = valid_actions.get("fields")
    if isinstance(fields, dict) and fields and "message" not in fields:
        return False
    return True
