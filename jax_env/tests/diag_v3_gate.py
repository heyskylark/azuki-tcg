"""V3/1234 gate-validate diagnostic (full-check + real apply_gate_portal).

Full semantic-view check each step (finds the TRUE first divergence with the
current code), and at every GATE_PORTAL: reconstruct the validate inputs and
print _azk01_124_validate DIRECTLY (bypassing the validate_card dispatch) plus
the per-candidate cost_row, AND the REAL apply_gate_portal post-state ab_phase.
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
import jax.numpy as jnp  # noqa: E402

from conftest import CRef  # noqa: E402
from test_l2_vanilla import c_semantic_view, jax_semantic_view  # noqa: E402
from test_l3_abilities_batch3 import _selection_view_c, _selection_view_jax  # noqa: E402
from test_l3_abilities_batch4 import DECK_GROUPS, DRIVER_TYPES_4  # noqa: E402

from azuki_jax import cards  # noqa: E402
from azuki_jax.abilities import cards_batch4 as b4  # noqa: E402
from azuki_jax.constants import Zone  # noqa: E402
from azuki_jax.engine import apply as apply_mod  # noqa: E402
from azuki_jax.engine.apply import _enter_board_slot, card_at_slot  # noqa: E402
from azuki_jax.engine.helpers import gate_instance  # noqa: E402
from azuki_jax.engine.step import engine_step, stabilize  # noqa: E402
from azuki_jax.env import init_state_with_decks  # noqa: E402
from azuki_jax.setup import deck_tables_from_card_lists  # noqa: E402


def cn(d):
  d = int(d)
  return cards.CARD_CODES[d] if d >= 0 else "-"


def inspect_gate(pre, p, alley_index, garden_index):
  alley_card = int(card_at_slot(pre, p, Zone.ALLEY, alley_index))
  safe = max(alley_card, 0)
  s2 = _enter_board_slot(pre, p, jnp.asarray(safe), Zone.GARDEN,
                         jnp.asarray(garden_index), do=jnp.asarray(alley_card >= 0))
  s2 = s2._replace(
      ab_scratch=s2.ab_scratch.at[0].set(jnp.int16(safe)).at[1]
      .set(jnp.int16(garden_index)).at[2].set(jnp.int16(1))
  )
  gate = int(gate_instance(s2, p))
  power = int(b4._gate_power(s2, p))
  portaled = int(b4._portaled_inst(s2))
  vdirect = bool(b4._azk01_124_validate(s2, jnp.asarray(p), jnp.asarray(max(gate, 0))))
  row = np.asarray(b4._azk01_124_cost_row(s2, p))
  print(f"   portaled inst={alley_card} def={cn(s2.def_id[p, safe])} | "
        f"power={power} portaled_inst={portaled} _azk01_124_validate(DIRECT)={vdirect}")
  zone = np.asarray(s2.zone[p]); dids = np.asarray(s2.def_id[p])
  tap = np.asarray(s2.tapped[p]); ic = np.asarray(cards.IKZ_COST)
  for inst in range(zone.shape[0]):
    if zone[inst] == int(Zone.GARDEN):
      d = int(dids[inst])
      print(f"      g[inst={inst}] {cn(d)} tapped={bool(tap[inst])} "
            f"cost={int(ic[max(d,0)])} cost_row={bool(row[inst])} portaled={inst==alley_card}")
  # REAL apply_gate_portal outcome
  post = apply_mod.apply_gate_portal(pre, jnp.asarray(alley_index),
                                     jnp.asarray(garden_index))
  print(f"   REAL apply_gate_portal -> ab_phase={int(post.ab_phase)} "
        f"scratch={[int(x) for x in np.asarray(post.ab_scratch[:3])]}")


def main():
  seed = 1234
  deck = DECK_GROUPS["V3"]
  cref = CRef(seed, deck_pool=None)
  cref.reset_with_decks(seed, deck, deck)
  tables = deck_tables_from_card_lists(deck, deck)
  rng = np.random.default_rng(seed)
  with jax.disable_jit():
    state = stabilize(init_state_with_decks(seed, tables))
    for si in range(60):
      cv = c_semantic_view(cref); cv.update(_selection_view_c(cref))
      jv = jax_semantic_view(state); jv.update(_selection_view_jax(state))
      diverged = [k for k in cv if k != "winner" and jv.get(k) != cv[k]]
      if diverged:
        print(f"\n*** TRUE FIRST DIVERGENCE at step {si} ***", flush=True)
        for k in diverged[:8]:
          print(f"   {k}: C={cv[k]} JAX={jv[k]}")
        return
      active = cv["active"]
      c_rows = cref.legal_actions(active)
      driver = [r for r in c_rows if r[0] in DRIVER_TYPES_4]
      if not driver:
        print(f"step {si}: no actions"); return
      a = driver[int(rng.integers(0, len(driver)))]
      if int(a[0]) == 10:
        print(f"\n=== step {si} GATE_PORTAL alley={int(a[1])} garden={int(a[2])} "
              f"active={active} ===", flush=True)
        inspect_gate(state, int(state.active_player), int(a[1]), int(a[2]))
      cref.step(np.asarray(a, np.int32))
      state = engine_step(state, np.asarray(a, np.int32))
      if int(a[0]) == 10:
        raw = cref.raw(active).ability_context
        print(f"   -> C ctx.phase={int(raw.phase)} JAX ab_phase={int(state.ab_phase)}",
              flush=True)
      if si % 15 == 0:
        print(f"  ..step {si}", flush=True)
      if cref.dones()[0] or cref.dones()[1]:
        print(f"step {si}: ended"); return
  print("no divergence in 60 steps")


if __name__ == "__main__":
  main()
