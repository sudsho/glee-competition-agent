"""Extract per-game bargaining traces from the move logs for offline ablation.

One record per game: static configuration, both sides' proposals with the
responder's decision, and the outcome. Outcomes come from the runner's
'finished' lines (exact), from the API results file when the game belongs
to the primary agent, or from the harvest inference rule (our last
non-final offer, then silence, means the opponent accepted it).

Usage: python ablation_extract.py <tag>   (tag in main, vA, vB, vC, rep)
Writes analysis/ablation/traces_<tag>.pkl
"""

import ast
import json
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
OUT = ROOT / "analysis" / "ablation"
SOURCES = {
    "main": LOGS,
    "vA": LOGS / "variantA",
    "vB": LOGS / "variantB",
    "vC": LOGS / "variantC",
    "rep": LOGS / "variant",
}
FIN = re.compile(r"INFO Game ([0-9a-f-]{36}) .*finished! Result: (\{.*\})\s*$")


def _gain(offer, player):
    if not offer:
        return None
    label = "alice" if player == "player_1" else "bob"
    for k in (f"{player}_gain", f"{label}_gain"):
        if offer.get(k) is not None:
            try:
                return float(offer[k])
            except (TypeError, ValueError):
                return None
    return None


def _other(p):
    return "player_2" if p == "player_1" else "player_1"


def scan_moves(path):
    games = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if '"bargaining"' not in line or '"game_state"' not in line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("family") != "bargaining":
                continue
            st = e.get("game_state") or {}
            gid = e["game_id"]
            g = games.get(gid)
            if g is None:
                me = e.get("your_player")
                g = games[gid] = {
                    "gid": gid, "role": me, "opp": e.get("opponent") or {},
                    "M": float(st.get("money_to_divide") or 0),
                    "d_me": st.get(f"delta_{1 if me == 'player_1' else 2}"),
                    "d_op": st.get(f"delta_{2 if me == 'player_1' else 1}"),
                    "T": st.get("max_rounds"), "hk": bool(st.get("horizon_known")),
                    "ci": bool(st.get("complete_information")),
                    "ma": bool(st.get("messages_allowed")),
                    "ts0": e.get("ts"),
                }
            g["ts1"] = e.get("ts")
            me = g["role"]
            g["hist"] = [
                (int(h.get("round") or 0),
                 "me" if h.get("proposer") == me else "op",
                 _gain(h.get("offer") or {}, me),
                 _gain(h.get("offer") or {}, _other(me)),
                 (h.get("decision") or None))
                for h in (st.get("history") or [])
            ]
            lo = st.get("last_offer") or None
            g["last"] = {
                "round": int(e.get("round") or st.get("round") or 0),
                "phase": e.get("phase"),
                "action": e.get("action") or {},
                "lo": (None if not lo else
                       (_gain(lo, me), _gain(lo, _other(me)))),
            }
    return games


def scan_finish(path, wanted):
    fin = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "finished! Result" not in line:
                continue
            m = FIN.search(line)
            if not m or m.group(1) not in wanted:
                continue
            try:
                fin[m.group(1)] = ast.literal_eval(m.group(2))
            except (ValueError, SyntaxError):
                pass
    return fin


def load_api_results(wanted):
    res = {}
    p = LOGS / "results.jsonl"
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("game_id") in wanted and r.get("family") == "bargaining":
                prev = res.get(r["game_id"])
                # Prefer API truth, then log, then inferred.
                rank = {None: 0, "log": 1, "inferred": 2}.get(r.get("source"), 3)
                if prev is None or rank < prev[0]:
                    res[r["game_id"]] = (rank, r["result"])
    return {k: v[1] for k, v in res.items()}


def build(tag):
    d = SOURCES[tag]
    games = scan_moves(d / "moves.jsonl")
    fin = scan_finish(d / "runner.log", set(games))
    api = load_api_results(set(games)) if tag == "main" else {}
    out = []
    for gid, g in games.items():
        me, M = g["role"], g["M"]
        if M <= 0:
            continue
        ours, theirs = [], []
        for (t, who, g_me, g_op, dec) in g["hist"]:
            if g_me is None or g_op is None:
                continue
            if who == "me":
                ours.append([t, g_op / M, dec])
            else:
                theirs.append([t, g_me / M, dec])
        last = g["last"]
        t_last = last["round"]
        if last["phase"] == "offer":
            a = last["action"]
            mine = a.get("alice_gain") if me == "player_1" else a.get("bob_gain")
            theirs_amt = a.get("bob_gain") if me == "player_1" else a.get("alice_gain")
            if mine is not None and theirs_amt is not None:
                ours.append([t_last, float(theirs_amt) / M, None])
        elif last["lo"] is not None and last["lo"][0] is not None:
            theirs.append([t_last, last["lo"][0] / M,
                           (last["action"] or {}).get("decision")])

        res = api.get(gid) or fin.get(gid)
        src = "api" if gid in api else ("fin" if gid in fin else None)
        T = g["T"]
        d_me = g["d_me"]
        if res is None and last["phase"] == "offer" and ours \
                and ours[-1][2] is None and (T is None or t_last < int(T)) \
                and d_me is not None:
            keep = 1.0 - ours[-1][1]
            res = {f"{me}_payoff": keep * M * float(d_me) ** (t_last - 1),
                   "outcome": "agreement", "agreed_round": t_last}
            src = "inferred"
        if res is None:
            continue
        pay = res.get(f"{me}_payoff")
        if pay is None:
            continue
        outc = res.get("outcome")
        ar = res.get("agreed_round")
        # Resolve the opponent's decision on our final offer.
        if ours and ours[-1][2] is None:
            if outc == "agreement" and ar == ours[-1][0]:
                ours[-1][2] = "accept"
            elif outc != "agreement" and last["phase"] == "offer":
                ours[-1][2] = "reject"
        out.append({
            "gid": gid, "agent": tag, "role": me, "opp_type": (g["opp"].get("type") or "hidden"),
            "opp_name": g["opp"].get("name"), "M": M,
            "d_me": None if d_me is None else float(d_me),
            "d_op": None if g["d_op"] is None else float(g["d_op"]),
            "T": None if T is None else int(T), "hk": g["hk"], "ci": g["ci"], "ma": g["ma"],
            "ts0": g["ts0"], "ts1": g["ts1"], "ours": ours, "theirs": theirs,
            "pay": float(pay) / M, "outcome": outc, "agreed_round": ar, "src": src,
        })
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"traces_{tag}.pkl", "wb") as f:
        pickle.dump(out, f)
    n_src = {}
    for r in out:
        n_src[r["src"]] = n_src.get(r["src"], 0) + 1
    print(f"{tag}: {len(games)} bargaining games logged, {len(out)} with "
          f"outcomes {n_src}", flush=True)


if __name__ == "__main__":
    build(sys.argv[1])
