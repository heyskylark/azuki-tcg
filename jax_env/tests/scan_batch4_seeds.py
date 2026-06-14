"""Ad-hoc C-engine-only seed scan for batch4 coverage (not a pytest file).

Replays the batch4 driver protocol against the C engine alone and reports,
per (group, seed), which batch4 card abilities entered each ability phase
and how many portal / leader-activation actions were taken. Used to pick
SEEDS in test_l3_abilities_batch4.py.

Run:  .venv/bin/python jax_env/tests/scan_batch4_seeds.py [seeds...]
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for entry in (REPO / "build/python/src", REPO / "python/src", REPO / "jax_env",
              REPO / "jax_env/tests"):
  if str(entry) not in sys.path:
    sys.path.insert(0, str(entry))

import numpy as np  # noqa: E402

from conftest import CRef  # noqa: E402
from test_l3_abilities_batch4 import DECK_GROUPS, DRIVER_TYPES_4  # noqa: E402

BATCH4 = {
    "AZK01-059", "AZK01-061", "AZK01-062", "STT04-007", "STT04-009",
    "STT01-001", "STT02-001", "STT03-001", "STT04-001", "AZK01-119",
    "AZK01-121", "AZK01-123", "AZK01-125", "STT01-002", "STT02-002",
    "STT03-002", "STT04-002", "AZK01-120", "AZK01-122", "AZK01-124",
    "AZK01-126", "STT02-017",
}


def scan(group: str, seed: int, steps: int = 500) -> Counter:
  from azuki_jax import cards

  deck = DECK_GROUPS[group]
  cref = CRef(seed, deck_pool=None)
  cref.reset_with_decks(seed, deck, deck)
  cov = Counter()
  rng = np.random.default_rng(seed)
  last_ab = None
  for _ in range(steps):
    active = cref.active_player
    rows = cref.legal_actions(active)
    if not rows:
      break
    ctx = cref.raw(0).ability_context
    phase = int(ctx.phase)
    if phase != 0 and bool(ctx.has_source_card_def_id):
      code = cards.CARD_CODES[int(ctx.source_card_def_id)]
      key = f"ab{phase}:{code}"
      if key != last_ab:
        cov[key] += 1
      last_ab = key
    else:
      last_ab = None
    driver_rows = [r for r in rows if r[0] in DRIVER_TYPES_4]
    if not driver_rows:
      break
    action = driver_rows[int(rng.integers(0, len(driver_rows)))]
    if action[0] in (10, 11, 12):
      cov[f"act{action[0]}"] += 1
    cref.step(np.asarray(action, np.int32))
    term, trunc = cref.dones()
    if term or trunc:
      break
  cref.close()
  return cov


def main():
  seeds = [int(s) for s in sys.argv[1:]] or [7, 42, 99, 888, 1234, 2026, 31337]
  for group in sorted(DECK_GROUPS):
    print(f"=== group {group} ===")
    for seed in seeds:
      cov = scan(group, seed)
      b4 = {k: v for k, v in cov.items()
            if ":" in k and k.split(":")[1] in BATCH4}
      acts = {k: v for k, v in cov.items() if k.startswith("act")}
      print(f"  seed {seed}: acts={acts} batch4={b4}")


if __name__ == "__main__":
  main()
