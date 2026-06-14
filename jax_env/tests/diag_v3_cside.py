"""Resolve the V3/1234 contradiction by reading C's ground truth at the
divergent portal: drive C's gate confirmation forward and dump its
SELECT_COST_TARGET options (the entities C accepts as valid sacrifices), plus
both engines' full garden (def, cost, gate_pts, tapped). Run with DBG_GATE=1.
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
from azuki_jax.constants import Zone  # noqa: E402
from azuki_jax.engine.step import engine_step, stabilize  # noqa: E402
from azuki_jax.env import init_state_with_decks  # noqa: E402
from azuki_jax.setup import deck_tables_from_card_lists  # noqa: E402


def cn(d):
  d = int(d)
  return cards.CARD_CODES[d] if d >= 0 else "-"


def dump_c_garden(cref, p):
  raw = cref.raw(p).my_observation_data
  out = []
  for i in range(int(raw.garden_count) if hasattr(raw, "garden_count") else 5):
    c = raw.garden[i]
    d = int(c.card_def_id)
    if d >= 0:
      out.append((cn(d), int(cards.IKZ_COST[d]), int(cards.GATE_POINTS[d])))
  return out


def main():
  seed = 1234
  deck = DECK_GROUPS["V3"]
  cref = CRef(seed, deck_pool=None)
  cref.reset_with_decks(seed, deck, deck)
  tables = deck_tables_from_card_lists(deck, deck)
  rng = np.random.default_rng(seed)
  with jax.disable_jit():
    state = stabilize(init_state_with_decks(seed, tables))
    for si in range(120):
      cv = c_semantic_view(cref); cv.update(_selection_view_c(cref))
      jv = jax_semantic_view(state); jv.update(_selection_view_jax(state))
      diverged = [k for k in cv if k != "winner" and jv.get(k) != cv[k]]
      if diverged:
        print(f"\n*** DIVERGENCE at step {si}: {diverged[:6]} ***", flush=True)
        active = cv["active"]
        raw = cref.raw(active).ability_context
        print(f"  C ctx: phase={int(raw.phase)} src={cn(raw.source_card_def_id)} "
              f"cost_tt={int(raw.cost_target_type)}")
        # C garden (def, cost, gate_pts) via the observation
        print(f"  C garden view: {cv.get('garden'+str(active))}")
        print(f"  JAX garden view: {jv.get('garden'+str(active))}")
        # drive C's confirmation to see accepted cost targets
        if int(raw.phase) == 1:  # CONFIRMATION
          legal_before = cref.legal_actions(active)
          print(f"  C confirm-phase legal: {legal_before[:6]}")
          # find CONFIRM_ABILITY (type 16) and send it
          conf = [a for a in legal_before if a[0] == 16]
          if conf:
            cref.step(np.asarray(conf[0], np.int32))
            cost_opts = cref.legal_actions(active)
            raw2 = cref.raw(active).ability_context
            print(f"  After CONFIRM -> C ctx.phase={int(raw2.phase)} "
                  f"cost_tt={int(raw2.cost_target_type)} ALL legal: {cost_opts[:12]}")
            sel = [a for a in cost_opts if a[0] == 13]  # SELECT_COST_TARGET=13
            print(f"  SELECT_COST_TARGET(13) options: {sel[:8]} "
                  f"(cost_tt=4 => sub1 = friendly-garden slot index)")
        return
      active = cv["active"]
      c_rows = cref.legal_actions(active)
      driver = [r for r in c_rows if r[0] in DRIVER_TYPES_4]
      if not driver:
        print(f"step {si}: no actions"); return
      a = driver[int(rng.integers(0, len(driver)))]
      cref.step(np.asarray(a, np.int32))
      state = engine_step(state, np.asarray(a, np.int32))
      if si % 15 == 0:
        print(f"  ..step {si}", flush=True)
      if cref.dones()[0] or cref.dones()[1]:
        print(f"step {si}: ended"); return
  print("no divergence")


if __name__ == "__main__":
  main()
