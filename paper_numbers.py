"""Every number the paper quotes from the ablations, rounded once from the
exact saved values (round half up). Writes analysis/ablation/paper_numbers.txt.
"""

import pickle
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ABL = Path(__file__).resolve().parent.parent / "analysis" / "ablation"
WIN = ["Aug 22-24", "Aug 25-27", "Aug 28-30"]


def r0(x, nd=0):
    q = Decimal(1).scaleb(-nd)
    return Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)


def pm(dm, se, nd=0):
    return f"{r0(dm, nd):+} +/- {r0(1.96 * se, nd)}"


def load(name):
    with open(ABL / name, "rb") as f:
        return pickle.load(f)


out = []
out.append("== Table 2 (versions)")
for w in WIN:
    rep = load(f"versions_{w.replace(' ', '_')}.pkl")[w]
    out.append(f"{w}: final {r0(rep['res']['final'])}, logged {r0(rep['realized'])}, "
               f"final-minus-logged {pm(*rep['realized_diff'], nd=1)}, "
               f"conflicts {rep['conflicts']}, F {rep['F']}")
    for v, (dm, se) in rep["diff"].items():
        out.append(f"   {v:18s} {pm(dm, se)}   pinned share {rep['pinned'][v]:.4f}")
    out.append(f"   final pinned share {rep['pinned']['final']:.4f}")

out.append("== Model dependence (total | log-pinned | modeled | modeled share | no-deal)")
for w in WIN:
    md = load(f"modeldep_{w.replace(' ', '_')}.pkl")
    out.append(f"{w}: final no-deal {md['base_nodeal']:.4f}")
    for v, (tot, pin, mod, share, nd) in md["rows"].items():
        out.append(f"   {v:18s} {r0(tot):+} | {r0(pin):+} | {r0(mod):+} | "
                   f"{share:.3f} | {nd:.3f}")
for tag in ["set_GIVE_CAP_0.49", "set_KEEP_HI_0.6", "set_FLOORS_1.2",
            "set_KEEP_LO_0.48", "set_KEEP_LO_0"]:
    md = load(f"modeldep_Aug_28-30_{tag}.pkl")
    for v, (tot, pin, mod, share, nd) in md["rows"].items():
        out.append(f"   {v:18s} {r0(tot):+} | {r0(pin):+} | {r0(mod):+} | "
                   f"{share:.3f} | {nd:.3f}")

out.append("== Table 3 (decomposition, Aug 28-30, n=3000)")
for p in ["targets", "bounds", "all", "lo", "hi", "cont"]:
    dm, se, n = load(f"decomp_{p}.pkl")
    out.append(f"   {p:8s} {pm(dm, se)}  (n={n})")

out.append("== Table 4 (sweeps, Aug 28-30, n=3000)")
for name in ["KEEP_LO", "KEEP_HI", "KEEP_MAX", "CONT_LO", "HAIRCUT",
             "CAP_GREED", "RHO", "GIVE_CAP", "FLOORS"]:
    d = load(f"sweep_{name}.pkl")
    cells = []
    for v, dm, se in d["row"]:
        cells.append(f"{v:g}: {pm(dm, se)} [{dm:+.3f} +/- {1.96 * se:.3f}]")
    out.append(f"   {name:9s} base {d['base']:.4f} | " + "; ".join(cells))

text = "\n".join(out)
(ABL / "paper_numbers.txt").write_text(text, encoding="utf-8")
print(text)
