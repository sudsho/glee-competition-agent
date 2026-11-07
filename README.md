# Beating the Pool, Not the Game: A White-Box Equilibrium Agent for Language-Based Economic Games

Code for the GLEE Competition agent `sudsho` (NeurIPS 2026 IAB Workshop,
GLEE Competition Track). The agent plays bargaining, negotiation, and
persuasion through the GLEE REST API with zero inference-time LLM calls:
every move is a deterministic function of the visible game state
(persuasion mixing aside), driven by equilibrium mathematics clamped
toward empirical pool-percentile targets.

Authors: Sudhanshu Srivastava and Faisal Aqlan, University of Louisville.

## Contents

- `main.py`: competition runner (SDK loop, dispatcher, full-payload move
  logging). Reads the agent API key from `GLEE_API_KEY` or `../key.txt`.
- `barg.py`, `nego.py`, `pers.py`: the three family policies.
- `targets.py`: percentile ladders: static pool quantiles blended with
  live competition quantiles (one-way ratchet where self-sampling applies).
- `oppmodel.py`: named-opponent book. `norm.py`, `safe.py`, `msgs.py` --
  helpers, guaranteed-valid fallbacks, message templates.
- `var_cfg.py`: ablation-variant switch (env `GLEE_VARIANT` in A/B/C).
- `build_percentiles.py`: rebuilds `analysis/percentiles.json` from the
  public GLEE corpus (https://github.com/eilamshapira/GLEE, `Data/`).
- `build_live_quantiles.py`, `harvest_results.py`, `collect_daemon.py`,
  `collect_results.py`: outcome collection and live-quantile refresh.
- `replay_audit.py`: the behavior audit: replays logged moves through the
  frozen policy and reports exact-match, invariant, and rate statistics.
- `analyze_targets.py`, `analyze_results.py`, `final_snapshot.py` --
  analysis and close-of-competition capture utilities.
- `test_local.py`: offline test suite over synthetic payloads.
- `ablation_extract.py`, `ablation_pool.py`, `ablation_live.py`,
  `ablation_sim.py`, `ablation_decomp.py`, `ablation_modeldep.py`,
  `ablation_variant_a.py`, `ablation_floor.py`, `paper_numbers.py`: the
  camera-ready ablations (see below). `barg.py` carries the offline switch
  `ABLATE` ("clamp", "targets", "safe", "floor", "floor-safe"), which is
  empty in play, so the replay audit reproduces the logged moves exactly.
  Safeguards read the quantile tables through `ladder` and clamp parts
  through `pool`, so the switches never overlap. "floor" removes the
  patient pool-median accept floor alone, and "floor-safe" files it under
  the safeguards so that "clamp" leaves it in place.

## Running

```
pip install glee-sdk
set GLEE_API_KEY=glee_...   # from the GLEE dashboard
python test_local.py        # should print "all tests passed"
python main.py
```

Expected layout: this directory as `agent/` inside a project root that
also holds `logs/` and `analysis/` (created automatically). To rebuild the
percentile tables, clone the public GLEE corpus into `data/GLEE` and run
`python build_percentiles.py`.

## Reproducing the ablations

The controlled ablations in the paper run on the season's move logs
(`logs/moves.jsonl` and `logs/runner.log` per agent). The logs are not
redistributed, since they hold other participants' messages and
identifiers, but every step is scripted:

```
pip install numpy scipy
python ablation_pool.py                    # corpus reference pool, exact per-cell samples
python ablation_extract.py main            # per-game traces (also vA, vB, vC, rep)
python ablation_live.py                    # concurrent live ablation, arm A vs arms B and C
python ablation_sim.py versions "Aug 28-30"   # policy versions on one frozen pool
python ablation_modeldep.py "Aug 28-30"    # log-pinned vs modeled part of each gap
python ablation_sim.py sweep               # one-at-a-time sensitivity of the constants
python ablation_decomp.py targets          # clamp decomposition (also bounds, all, lo, hi, cont)
python ablation_floor.py "Aug 28-30"       # patient floor alone vs the rest of the clamp
python ablation_variant_a.py "Aug 28-30"   # arm A's live rules, replayed
python paper_numbers.py                    # every reported number, rounded once
```

## License

MIT (see LICENSE).
