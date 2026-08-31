"""Offline smoke tests: run every strategy branch on realistic payloads
and assert the returned actions are valid (shapes, sums, value ranges).
Run: python test_local.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from barg import bargaining_strategy
from nego import negotiation_strategy
from pers import persuasion_strategy
from safe import safe_action

FAIL = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAIL.append(name)


def barg_game(**kw):
    state = {
        "phase": "offer", "current_player": "player_1", "proposer": "player_1",
        "round": 1, "max_rounds": 12, "horizon_known": True,
        "money_to_divide": 10000, "delta_1": 0.9, "delta_2": 0.9,
        "last_offer": None, "history": [], "messages_allowed": True,
        "complete_information": True,
    }
    state.update(kw.pop("state", {}))
    g = {"game_id": "t", "game_family": "bargaining", "your_player": "player_1",
         "phase": state["phase"], "game_state": state,
         "valid_actions": {"type": "offer", "fields": {"alice_gain": {}, "bob_gain": {}, "message": {}}},
         "opponent": {"type": "hidden", "name": None}, "prompt": ""}
    g.update(kw)
    return g


print("bargaining:")
a = bargaining_strategy(barg_game())
check("offer sums to pot", a["alice_gain"] + a["bob_gain"] == 10000, a)
check("proposer keeps majority", a["alice_gain"] >= 5000, a)
check("opponent floor respected", a["bob_gain"] >= 2600, a)
check("message short", len(a.get("message", "")) < 1900)

# asymmetric power: I am patient (1.0), they bleed (0.8)
a = bargaining_strategy(barg_game(state={"delta_1": 1.0, "delta_2": 0.8}))
check("patient proposer demands more", a["alice_gain"] >= 6000, a)

# I'm weak: still never offer > 55% early
a = bargaining_strategy(barg_game(state={"delta_1": 0.8, "delta_2": 1.0}))
check("early give cap", a["bob_gain"] <= 5500, a)

# decision: lowball at t=2 -> reject
g = barg_game(state={"phase": "decision", "round": 2, "current_player": "player_1",
                     "last_offer": {"player_1_gain": 2000, "player_2_gain": 8000,
                                    "proposer": "player_2", "round": 2}})
g["valid_actions"] = {"type": "decision", "fields": {"decision": {}}}
a = bargaining_strategy(g)
check("reject lowball", a == {"decision": "reject"}, a)

# decision: good offer -> accept
g["game_state"]["last_offer"] = {"player_1_gain": 5900, "player_2_gain": 4100,
                                 "proposer": "player_2", "round": 2}
a = bargaining_strategy(g)
check("accept 59%", a == {"decision": "accept"}, a)

# final round: accept anything positive
g["game_state"].update({"round": 12,
                        "last_offer": {"player_1_gain": 400, "player_2_gain": 9600,
                                       "proposer": "player_2", "round": 12}})
a = bargaining_strategy(g)
check("final round accepts crumbs", a == {"decision": "accept"}, a)

# unlimited horizon, alice-style keys in offer
g = barg_game(state={"max_rounds": None, "horizon_known": False, "round": 3,
                     "phase": "decision", "delta_1": 0.95,
                     "last_offer": {"alice_gain": 4800, "bob_gain": 5200,
                                    "proposer": "player_2", "round": 3}})
g["game_state"].pop("delta_2")           # incomplete info: opp delta hidden
g["valid_actions"] = {"type": "decision", "fields": {"decision": {}}}
a = bargaining_strategy(g)
check("unlimited-horizon decision valid", a["decision"] in ("accept", "reject"), a)

print("negotiation:")


def nego_game(**kw):
    state = {
        "phase": "offer", "current_player": "player_1",
        "player_1_role": "seller", "player_2_role": "buyer",
        "player_1_value": 8000.0, "player_2_value": None,
        "round": 1, "max_rounds": 10, "horizon_known": True,
        "last_offer": None, "history": [], "messages_allowed": True,
        "complete_information": False,
    }
    state.update(kw.pop("state", {}))
    if state.get("player_2_value") is None:
        state.pop("player_2_value", None)
    g = {"game_id": "t", "game_family": "negotiation", "your_player": kw.pop("me", "player_1"),
         "phase": state["phase"], "game_state": state,
         "valid_actions": {"type": "offer", "fields": {"product_price": {}, "message": {}}},
         "opponent": {"type": "hidden", "name": None}, "prompt": ""}
    g.update(kw)
    return g


a = negotiation_strategy(nego_game())
check("seller anchors high", 8000 < a["product_price"] <= 15000, a)

a = negotiation_strategy(nego_game(state={"max_rounds": 1}))
check("TIOLI ask beats own value", a["product_price"] > 8000, a)

# seller with known buyer value: shave just under it
a = negotiation_strategy(nego_game(state={"player_2_value": 12000.0,
                                          "complete_information": True}))
check("seller prices under known b", 8000 < a["product_price"] < 12000, a)

# buyer decision: profitable price on final round -> accept
g = nego_game(me="player_2", state={
    "phase": "decision", "current_player": "player_2", "round": 10,
    "player_1_value": None, "player_2_value": 12000.0,
    "last_offer": {"price": 11000, "from_player": "player_1", "round": 10}})
g["game_state"].pop("player_1_value", None)
g["valid_actions"] = {"type": "decision",
                      "fields": {"decision": {}, "product_price": {}, "message": {}}}
a = negotiation_strategy(g)
check("buyer accepts on deadline", a["decision"] == "AcceptOffer", a)

# buyer decision: greedy price early -> reject with lower counter
g["game_state"].update({"round": 2,
                        "last_offer": {"price": 11800, "from_player": "player_1",
                                       "round": 2}})
a = negotiation_strategy(g)
check("buyer counters early gouge", a["decision"] == "RejectOffer"
      and a.get("product_price", 1e18) < 11800, a)

# buyer never counters above own value
check("counter below value", a.get("product_price", 0) <= 12000 * 0.99, a)

# unprofitable final-round price -> plain reject
g["game_state"].update({"round": 10,
                        "last_offer": {"price": 12500, "from_player": "player_1",
                                       "round": 10}})
a = negotiation_strategy(g)
check("buyer rejects bad deadline price", a["decision"] == "RejectOffer"
      and "product_price" not in a, a)

print("persuasion:")


def pers_game(action_type, **kw):
    state = {
        "phase": action_type, "current_player": "player_1",
        "product_price": 10000.0, "p": 0.5, "v": 12500.0, "u": 0.0,
        "round": 1, "total_rounds": 20, "current_quality": "high",
        "seller_message": None, "seller_message_type": "binary",
        "history": [],
    }
    state.update(kw.pop("state", {}))
    g = {"game_id": "t", "game_family": "persuasion",
         "your_player": kw.pop("me", "player_1"), "phase": state["phase"],
         "game_state": state,
         "valid_actions": {"type": action_type, "fields": {"decision": {}}},
         "opponent": {"type": "hidden", "name": None}, "prompt": ""}
    g.update(kw)
    return g


a = persuasion_strategy(pers_game("seller_recommendation"))
check("seller recommends high quality", a == {"decision": "yes"}, a)

a = persuasion_strategy(pers_game("seller_recommendation",
                                  state={"current_quality": "low", "round": 19}))
check("seller endgame burn on low", a["decision"] in ("yes", "no"), a)

a = persuasion_strategy(pers_game("seller_message",
                                  state={"seller_message_type": "text"}))
check("seller text message", isinstance(a.get("message"), str)
      and 0 < len(a["message"]) < 1900, a)

# buyer: v below price -> never buy
a = persuasion_strategy(pers_game("buyer_decision", me="player_2",
                                  state={"v": 9000.0, "current_quality": None,
                                         "seller_message": "yes"}))
check("buyer refuses when v<=price", a == {"decision": "no"}, a)

# buyer: blind-profitable -> buy on rec
a = persuasion_strategy(pers_game("buyer_decision", me="player_2",
                                  state={"p": 0.8, "v": 30000.0,
                                         "current_quality": None,
                                         "seller_message": "yes"}))
check("buyer takes blind-positive EV", a == {"decision": "yes"}, a)

# buyer: seller said no -> pass
a = persuasion_strategy(pers_game("buyer_decision", me="player_2",
                                  state={"current_quality": None,
                                         "seller_message": "no"}))
check("buyer obeys negative rec", a == {"decision": "no"}, a)

# buyer trust regime with poor track record -> pass
hist = [{"round": r, "seller_message": "yes", "buyer_decision": "yes",
         "bought": True, "quality": "low"} for r in range(1, 5)]
a = persuasion_strategy(pers_game("buyer_decision", me="player_2",
                                  state={"p": 1.0 / 3, "v": 20000.0, "round": 6,
                                         "current_quality": None,
                                         "seller_message": "yes",
                                         "history": hist}))
check("buyer punishes serial liar", a == {"decision": "no"}, a)

print("safe fallbacks:")
for fam, at in (("bargaining", "offer"), ("bargaining", "decision"),
                ("negotiation", "offer"), ("negotiation", "decision"),
                ("persuasion", "seller_message"), ("persuasion", "buyer_decision")):
    g = {"game_family": fam, "your_player": "player_1",
         "valid_actions": {"type": at},
         "game_state": {"money_to_divide": 100, "current_player": "player_1",
                        "player_1_value": 100}}
    a = safe_action(g)
    check(f"safe {fam}/{at}", isinstance(a, dict) and a, a)

print()
if FAIL:
    print(f"{len(FAIL)} FAILURES: {FAIL}")
    sys.exit(1)
print("all tests passed")
