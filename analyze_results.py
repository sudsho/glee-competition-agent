"""Score our finished games against the reference-pool percentile tables.

Reads logs/results.jsonl, computes my payoff per game, locates it in the
per-configuration ladder, and reports estimated percentile by family and
role: exactly where rating is earned and leaked.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import targets  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QS = [5, 10, 25, 50, 75, 90, 95, 99]


def my_payoff(rec):
    res = rec.get("result") or {}
    who = rec.get("your_player")
    v = res.get(f"{who}_payoff")
    return None if v is None else float(v)


def pct_in_ladder(vec_norm, x_norm):
    """Rough percentile of x within a quantile ladder."""
    if vec_norm is None:
        return None
    below = sum(1 for q, val in zip(QS, vec_norm) if x_norm >= val)
    if below == 0:
        return 3
    if below >= len(QS):
        return 99
    lo_q = QS[below - 1]
    return lo_q


def main():
    path = ROOT / "logs" / "results.jsonl"
    fam_role = defaultdict(list)
    outcomes = defaultdict(lambda: defaultdict(int))
    pcts = defaultdict(list)

    lines = path.read_text(encoding="utf-8").splitlines()
    if "--tail" in sys.argv:
        lines = lines[-int(sys.argv[sys.argv.index("--tail") + 1]):]

    for line in lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        fam = rec.get("family")
        who = rec.get("your_player")
        cfg = rec.get("cfg") or {}
        pay = my_payoff(rec)
        outcomes[fam][rec.get("status")] += 1
        if pay is None:
            continue
        fam_role[(fam, who)].append(pay)

        norm = None
        vec = None
        if fam == "bargaining":
            M = float(cfg.get("money_to_divide") or 0) or 1
            d1 = cfg.get("delta_1")
            d2 = cfg.get("delta_2")
            d_me = float((d1 if who == "player_1" else d2) or 1.0)
            d_op = d2 if who == "player_1" else d1
            d_op = float(d_op) if d_op is not None else None
            lad = targets.barg_ladder(who, d_me, d_op, cfg.get("max_rounds"),
                                      bool(cfg.get("complete_information")),
                                      bool(cfg.get("messages_allowed")))
            vec = lad.vec if lad else None
            norm = pay / M
        elif fam == "negotiation":
            myv = float(cfg.get(f"{who}_value") or 0)
            if myv:
                for k in range(0, 13):
                    m = 10.0 ** k
                    if 0.75 <= myv / m <= 1.55:
                        break
                lad = targets.nego_ladder(who, myv / m, cfg.get("max_rounds"))
                vec = lad.vec if lad else None
                norm = pay / m
        if vec is not None and norm is not None:
            p = pct_in_ladder(vec, norm)
            if p is not None:
                pcts[(fam, who)].append(p)

    print("=== outcomes by family ===")
    for fam, cnt in outcomes.items():
        print(f"{fam:<12} {dict(cnt)}")

    print("\n=== my payoff stats (fam, role) ===")
    for k in sorted(fam_role):
        v = sorted(fam_role[k])
        n = len(v)
        zero = sum(1 for x in v if x <= 0)
        print(f"{k}: n={n} zero_or_neg={zero} "
              f"median={v[n // 2]:.1f} mean={sum(v) / n:.1f}")

    print("\n=== estimated pool percentile (fam, role) ===")
    for k in sorted(pcts):
        v = sorted(pcts[k])
        n = len(v)
        print(f"{k}: n={n} median_pct={v[n // 2]} mean_pct={sum(v) / n:.0f} "
              f"top_q90+={sum(1 for x in v if x >= 90)} bottom_q5={sum(1 for x in v if x <= 5)}")


if __name__ == "__main__":
    main()
