"""Guaranteed-valid conservative moves, used when a strategy throws."""

from norm import norm_player, split_action


def safe_action(game: dict) -> dict:
    family = game.get("game_family", "")
    action_type = (game.get("valid_actions") or {}).get("type", "")
    state = game.get("game_state") or {}

    if action_type == "offer":
        if family == "bargaining":
            money = float(state.get("money_to_divide", 100))
            me = norm_player(game.get("your_player"))
            return split_action(money / 2, money, me)
        me = norm_player(state.get("current_player") or game.get("your_player"))
        val = state.get(f"{me}_value", state.get("player_1_value", 100))
        return {"product_price": float(val)}

    if action_type == "seller_message":
        return {"message": "I recommend this product."}

    if family == "bargaining":
        return {"decision": "accept"}
    if family == "negotiation":
        return {"decision": "AcceptOffer"}
    return {"decision": "yes"}
