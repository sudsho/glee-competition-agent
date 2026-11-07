"""Variant switch for A/B testing across the account's agent slots.

The account is ranked by its BEST agent, so variants are free options:
  ''  main (sudsho): the tuned baseline policy
  'A' Test 1: bargaining immediacy. Open at an even split, accept q25 in
      the first two rounds (full 'hardened pool' doctrine).
  'B' Test 2: negotiation aggression. Gambit more, concede deeper ZOPA
      share, cross viable buyer values earlier.
  'C' Test 3: persuasion honesty. The seller never lies before the last 3
      rounds, and the buyer stop-loss applies in every message mode.
Set via env GLEE_VARIANT at launch. Compare via each agent's /stats rating.
"""

import os

VARIANT = (os.environ.get("GLEE_VARIANT") or "").strip().upper()
