# Beating the Pool, Not the Game: A White-Box Equilibrium Agent for Language-Based Economic Games

Code for the GLEE Competition agent `sudsho` (NeurIPS 2026 IAB Workshop
competition-paper track). The agent plays bargaining, negotiation, and
persuasion through the GLEE REST API with zero inference-time LLM calls:
every move is a deterministic function of the visible game state
(persuasion mixing aside), driven by equilibrium mathematics clamped
toward empirical pool-percentile targets.

Authors: Sudhanshu Srivastava and Faisal Aqlan, University of Louisville.

## Contents

- `main.py` -- competition runner (SDK loop, dispatcher, full-payload move
  logging). Reads the agent API key from `GLEE_API_KEY` or `../key.txt`.
- `barg.py`, `nego.py`, `pers.py` -- the three family policies.
- `targets.py` -- percentile ladders: static pool quantiles blended with
  live competition quantiles (one-way ratchet where self-sampling applies).
- `oppmodel.py` -- named-opponent book. `norm.py`, `safe.py`, `msgs.py` --
  helpers, guaranteed-valid fallbacks, message templates.
- `var_cfg.py` -- ablation-variant switch (env `GLEE_VARIANT` in A/B/C).
- `build_percentiles.py` -- rebuilds `analysis/percentiles.json` from the
  public GLEE corpus (https://github.com/eilamshapira/GLEE, `Data/`).
- `build_live_quantiles.py`, `harvest_results.py`, `collect_daemon.py`,
  `collect_results.py` -- outcome collection and live-quantile refresh.
- `replay_audit.py` -- the behavior audit: replays logged moves through the
  frozen policy and reports exact-match, invariant, and rate statistics.
- `analyze_targets.py`, `analyze_results.py`, `final_snapshot.py` --
  analysis and close-of-competition capture utilities.
- `test_local.py` -- offline test suite over synthetic payloads.

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

## License

MIT (see LICENSE).
