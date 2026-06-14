"""Targeted diagnostic for the V3/1234 ab_phase divergence (step ~111).

Replays the batch4 V3 deck on seed 1234, printing the action and ability
context around the first ab_phase mismatch so the root cause is visible.
Run: XLA_FLAGS="--xla_gpu_enable_command_buffer=" \
     JAX_COMPILATION_CACHE_DIR=/tmp/jaxcache \
     .venv/bin/python jax_env/tests/diag_v3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for entry in (REPO / "build/python/src", REPO / "python/src", REPO / "jax_env"):
  if str(entry) not in sys.path:
    sys.path.insert(0, str(entry))

import jax  # noqa: E402

from conftest import CRef  # noqa: E402
from test_l2_vanilla import c_semantic_view, jax_semantic_view  # noqa: E402
from test_l3_abilities_batch3 import _selection_view_c, _selection_view_jax  # noqa: E402
from test_l3_abilities_batch4 import DECK_GROUPS, DRIVER_TYPES_4  # noqa: E402

from azuki_jax import cards  # noqa: E402
from azuki_jax.engine.step import engine_step, stabilize  # noqa: E402
from azuki_jax.env import init_state_with_decks  # noqa: E402
from azuki_jax.masks import build_mask  # noqa: E402
from azuki_jax.setup import deck_tables_from_card_lists  # noqa: E402


def cardname(defid):
  return cards.CARD_CODES[int(defid)] if int(defid) >= 0 else "-"


def main():
  seed = 1234
  deck = DECK_GROUPS["V3"]
  cref = CRef(seed, deck_pool=None)
  cref.reset_with_decks(seed, deck, deck)
  tables = deck_tables_from_card_lists(deck, deck)
  state = stabilize(init_state_with_decks(seed, tables))
  jit_step = jax.jit(engine_step)
  jit_mask = jax.jit(build_mask)
  rng = np.random.default_rng(seed)

  recent = []  # (step, action)
  for step_index in range(140):
    cview = c_semantic_view(cref); cview.update(_selection_view_c(cref))
    jview = jax_semantic_view(state); jview.update(_selection_view_jax(state))

    # report ab_phase + first diverging field BEFORE stepping
    diverged = [k for k in cview if k != "winner" and jview.get(k) != cview[k]]
    if diverged:
      print(f"\n*** DIVERGENCE at step {step_index} (before action) ***")
      print("  diverging keys:", diverged[:6])
      for k in diverged[:6]:
        print(f"    {k}: C={cview[k]}  JAX={jview[k]}")
      print("  C ab_phase:", cview.get("ab_phase"), " JAX ab_phase:", jview.get("ab_phase"))
      # C ability context
      raw = cref.raw(cref.active_player)
      ac = raw.ability_context
      print(f"  C ab ctx: phase={int(ac.phase)} src_def={int(ac.source_card_def_id)} "
            f"has_src={bool(ac.has_source_card_def_id)} cost_tt={int(ac.cost_target_type)} "
            f"eff_tt={int(ac.effect_target_type)}")
      print(f"  JAX ab: phase={int(state.ab_phase)} src={int(state.ab_source)} "
            f"owner={int(state.ab_owner)} "
            f"src_card={cardname(state.def_id[max(int(state.ab_owner),0), max(int(state.ab_source),0)])}")
      print("  last 6 actions:", recent[-6:])
      print(f"  C active={cref.active_player} phase={cview['phase']} "
            f"trig_count(jax)={int(state.trig_count)}")
      return

    active = cview["active"]
    c_rows = cref.legal_actions(active)
    if not c_rows:
      print(f"step {step_index}: no legal actions"); return
    driver = [r for r in c_rows if r[0] in DRIVER_TYPES_4]
    action = driver[int(rng.integers(0, len(driver)))]
    recent.append((step_index, tuple(action), f"ph{cview['phase']}/ab{cview['ab_phase']}"))

    cref.step(np.asarray(action, np.int32))
    state = jit_step(state, np.asarray(action, np.int32))
    if cref.dones()[0] or cref.dones()[1]:
      print(f"step {step_index}: episode ended"); return
  print("no divergence in 140 steps")


if __name__ == "__main__":
  main()
