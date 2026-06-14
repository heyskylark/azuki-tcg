"""Divergence diagnostic using the split-jit driver (fast compile, fast run).

Same output as diag_eager.py but uses split_step (jit apply + jit micro_tick,
Python-driven loop) instead of eager dispatch — fast per step once the two
small graphs compile. Use only if probe_split_compile shows the pieces compile
in minutes.

Usage: python jax_env/tests/diag_split.py batch4 <GROUP> <SEED> [max_steps]
       python jax_env/tests/diag_split.py pool <i> <j> <SEED> [max_steps]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for entry in (REPO / "build/python/src", REPO / "python/src", REPO / "jax_env"):
  if str(entry) not in sys.path:
    sys.path.insert(0, str(entry))

import jax  # noqa: E402

from conftest import CRef  # noqa: E402
from split_step import make_split_step  # noqa: E402
from test_l2_vanilla import c_semantic_view, jax_semantic_view  # noqa: E402
from test_l3_abilities_batch3 import _selection_view_c, _selection_view_jax  # noqa: E402
from test_l3_abilities_batch4 import DECK_GROUPS, DRIVER_TYPES_4  # noqa: E402

from azuki_jax import cards  # noqa: E402
from azuki_jax.env import init_state_with_decks  # noqa: E402
from azuki_jax.masks import build_mask  # noqa: E402
from azuki_jax.setup import deck_tables_from_card_lists  # noqa: E402


def cname(d):
  return cards.CARD_CODES[int(d)] if int(d) >= 0 else "-"


def run(deck0, deck1, seed, steps):
  cref = CRef(seed, deck_pool=None)
  cref.reset_with_decks(seed, deck0, deck1)
  tables = deck_tables_from_card_lists(deck0, deck1)
  split_step, split_stabilize, _, _ = make_split_step()
  jit_mask = jax.jit(build_mask)
  t0 = time.time()
  state = split_stabilize(init_state_with_decks(seed, tables))
  print(f"[{time.strftime('%H:%M:%S')}] stabilized in {time.time()-t0:.1f}s "
        f"(compiles done)", flush=True)
  rng = np.random.default_rng(seed)
  recent = []
  for si in range(steps):
    cv = c_semantic_view(cref); cv.update(_selection_view_c(cref))
    jv = jax_semantic_view(state); jv.update(_selection_view_jax(state))
    diverged = [k for k in cv if k != "winner" and jv.get(k) != cv[k]]
    active = cv["active"]
    c_rows = cref.legal_actions(active)
    legal, count, _ = jit_mask(state)
    j_rows = [tuple(int(x) for x in r) for r in np.asarray(legal)[: int(count)]]
    mask_div = c_rows != j_rows
    if diverged or mask_div:
      print(f"\n*** FIRST DIVERGENCE at step {si} ***", flush=True)
      print(f"  C phase={cv['phase']} ab_phase={cv['ab_phase']} active={active}")
      for k in diverged[:10]:
        print(f"  KEY {k}: C={cv[k]}  JAX={jv[k]}")
      if mask_div:
        print(f"  MASK only-in-C={[r for r in c_rows if r not in j_rows][:8]}")
        print(f"       only-in-JAX={[r for r in j_rows if r not in c_rows][:8]}")
      raw = cref.raw(active).ability_context
      sd = int(raw.source_card_def_id)
      print(f"  C ab ctx: phase={int(raw.phase)} src_def={sd} "
            f"({cards.CARD_CODES[sd] if sd>=0 else '-'}) "
            f"cost_tt={int(raw.cost_target_type)} eff_tt={int(raw.effect_target_type)} "
            f"sel_count={int(raw.selection_count)}")
      ow = max(int(state.ab_owner), 0); sr = max(int(state.ab_source), 0)
      print(f"  JAX ab: phase={int(state.ab_phase)} owner={int(state.ab_owner)} "
            f"src={int(state.ab_source)} src_card={cname(state.def_id[ow, sr])} "
            f"trig_count={int(state.trig_count)} unimpl_hits={int(state.ab_scratch[3])}")
      print(f"  recent actions: {recent[-8:]}")
      return
    driver = [r for r in c_rows if r[0] in DRIVER_TYPES_4]
    if not driver:
      print(f"step {si}: no driver actions"); return
    a = driver[int(rng.integers(0, len(driver)))]
    recent.append((si, tuple(int(x) for x in a), f"ph{cv['phase']}ab{cv['ab_phase']}"))
    cref.step(np.asarray(a, np.int32))
    state = split_step(state, np.asarray(a, np.int32))
    if si % 25 == 0:
      print(f"  ..step {si} ok", flush=True)
    if cref.dones()[0] or cref.dones()[1]:
      print(f"step {si}: episode ended cleanly, no divergence"); return
  print(f"no divergence in {steps} steps")


def main():
  mode = sys.argv[1]
  if mode == "batch4":
    group, seed = sys.argv[2], int(sys.argv[3])
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    print(f"=== SPLIT batch4 {group} seed {seed} (max {steps}) ===", flush=True)
    run(DECK_GROUPS[group], DECK_GROUPS[group], seed, steps)
  else:
    from training_deck_pool import load_training_deck_pool
    pool = [list(d) for d in load_training_deck_pool(
        str(REPO / ".codex/docs/azuki_tcg_decks_final.json"))]
    i, j, seed = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    print(f"=== SPLIT pool {i} vs {j} seed {seed} (max {steps}) ===", flush=True)
    run(pool[i], pool[j], seed, steps)


if __name__ == "__main__":
  main()
