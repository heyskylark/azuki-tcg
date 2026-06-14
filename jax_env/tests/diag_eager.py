"""Eager (no-jit) divergence diagnostic — avoids the ~2h engine_step compile.

Under jax.disable_jit(), lax.while_loop/cond/switch run as pure Python (only
the taken card branch evaluates), so there is NO XLA compilation. Slow per
step (~op-by-op dispatch) but it reaches a step-111 divergence in tens of
minutes instead of paying the giant fused compile, and runs fine alongside a
jit verification on the same GPU.

Usage:
  python jax_env/tests/diag_eager.py batch4 <GROUP> <SEED> [max_steps]
  python jax_env/tests/diag_eager.py pool <i> <j> <SEED> [max_steps]
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


def cname(d):
  return cards.CARD_CODES[int(d)] if int(d) >= 0 else "-"


def run(deck0, deck1, seed, steps):
  cref = CRef(seed, deck_pool=None)
  cref.reset_with_decks(seed, deck0, deck1)
  tables = deck_tables_from_card_lists(deck0, deck1)
  rng = np.random.default_rng(seed)
  recent = []
  with jax.disable_jit():
    state = stabilize(init_state_with_decks(seed, tables))
    for si in range(steps):
      cv = c_semantic_view(cref); cv.update(_selection_view_c(cref))
      jv = jax_semantic_view(state); jv.update(_selection_view_jax(state))
      diverged = [k for k in cv if k != "winner" and jv.get(k) != cv[k]]
      active = cv["active"]
      c_rows = cref.legal_actions(active)
      legal, count, _ = build_mask(state)
      j_rows = [tuple(int(x) for x in r) for r in np.asarray(legal)[: int(count)]]
      mask_div = c_rows != j_rows
      if diverged or mask_div:
        print(f"\n*** FIRST DIVERGENCE at step {si} ***", flush=True)
        print(f"  C phase={cv['phase']} ab_phase={cv['ab_phase']} active={active}")
        for k in diverged[:10]:
          print(f"  KEY {k}: C={cv[k]}  JAX={jv[k]}")
        if mask_div:
          only_c = [r for r in c_rows if r not in j_rows][:8]
          only_j = [r for r in j_rows if r not in c_rows][:8]
          print(f"  MASK only-in-C={only_c}\n       only-in-JAX={only_j}")
        raw = cref.raw(active).ability_context
        print(f"  C ab ctx: phase={int(raw.phase)} src_def={int(raw.source_card_def_id)} "
              f"({cards.CARD_CODES[int(raw.source_card_def_id)] if int(raw.source_card_def_id)>=0 else '-'}) "
              f"cost_tt={int(raw.cost_target_type)} eff_tt={int(raw.effect_target_type)} "
              f"sel_count={int(raw.selection_count)}")
        ow = max(int(state.ab_owner), 0); sr = max(int(state.ab_source), 0)
        print(f"  JAX ab: phase={int(state.ab_phase)} owner={int(state.ab_owner)} "
              f"src={int(state.ab_source)} src_card={cname(state.def_id[ow, sr])} "
              f"trig_count={int(state.trig_count)} redirect_count={int(state.redirect_count)} "
              f"unimpl_hits={int(state.ab_scratch[3])}")
        print(f"  recent actions: {recent[-8:]}")
        return
      driver = [r for r in c_rows if r[0] in DRIVER_TYPES_4]
      if not driver:
        print(f"step {si}: no driver actions"); return
      a = driver[int(rng.integers(0, len(driver)))]
      recent.append((si, tuple(int(x) for x in a), f"ph{cv['phase']}ab{cv['ab_phase']}"))
      cref.step(np.asarray(a, np.int32))
      state = engine_step(state, np.asarray(a, np.int32))
      if si % 10 == 0:
        print(f"  ..step {si} ok", flush=True)
      if cref.dones()[0] or cref.dones()[1]:
        print(f"step {si}: episode ended cleanly, no divergence"); return
  print(f"no divergence in {steps} steps")


def main():
  mode = sys.argv[1]
  if mode == "batch4":
    group, seed = sys.argv[2], int(sys.argv[3])
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    deck = DECK_GROUPS[group]
    print(f"=== EAGER batch4 {group} seed {seed} (max {steps}) ===", flush=True)
    run(deck, deck, seed, steps)
  else:
    from training_deck_pool import load_training_deck_pool
    pool = [list(d) for d in load_training_deck_pool(
        str(REPO / ".codex/docs/azuki_tcg_decks_final.json"))]
    i, j, seed = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    print(f"=== EAGER pool {i} vs {j} seed {seed} (max {steps}) ===", flush=True)
    run(pool[i], pool[j], seed, steps)


if __name__ == "__main__":
  main()
