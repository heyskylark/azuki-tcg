"""Track STT02-012 latch/passive state across the m16 vector replay.

Finds the step where JAX's hidden stt02_012_latch/passive_atk diverges from
what C's sticky-buff semantics imply, before the visible stat divergence at
step 106.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for entry in (REPO / "build/python/src", REPO / "python/src", REPO / "jax_env", REPO / "jax_env/tests"):
  if str(entry) not in sys.path:
    sys.path.insert(0, str(entry))

from conftest import CRef  # noqa: E402
from test_l3_abilities_batch4 import DRIVER_TYPES_4  # noqa: E402
from training_deck_pool import load_training_deck_pool  # noqa: E402
from azk_puffer.jax_vector import JaxVecEnv, SEND  # noqa: E402
from azuki_jax import cards  # noqa: E402
from azuki_jax.env import init_state_with_decks  # noqa: E402
from azuki_jax.engine.step import stabilize  # noqa: E402
from azuki_jax.setup import deck_tables_from_card_lists  # noqa: E402

LEFT, RIGHT, SEED, STOP = 16, 16, 12361, 106
STT02_012 = cards.CODE_TO_ID["STT02-012"]


def main() -> None:
  pool = [
      list(deck)
      for deck in load_training_deck_pool(str(REPO / ".codex/docs/azuki_tcg_decks_final.json"))
  ]
  cref = CRef(SEED, deck_pool=None)
  cref.reset_with_decks(SEED, pool[LEFT], pool[RIGHT])
  rng = np.random.default_rng(SEED)
  tables = deck_tables_from_card_lists(pool[LEFT], pool[RIGHT])
  state = stabilize(init_state_with_decks(SEED, tables))

  batched = jax.tree.map(lambda x: jnp.stack([x]), state)
  env = JaxVecEnv(1, pool, seed=0)
  obs, legal, count = env._observe_fn(batched)
  env._states = batched
  env._terms = jnp.zeros((1, 2), jnp.bool_)
  env._truncs = jnp.zeros((1, 2), jnp.bool_)
  rewards = jnp.zeros((1, 2), jnp.float32)
  env._pending = (obs, rewards, env._terms, env._truncs, legal, count, batched.active_player)

  prev_summary = None
  for step in range(STOP):
    active = cref.active_player
    rows = cref.legal_actions(active)
    if not rows:
      print(f"ended at {step}")
      break
    driver = [r for r in rows if r[0] in DRIVER_TYPES_4]
    action = driver[int(rng.integers(0, len(driver)))]
    actions = np.zeros((1, 2, 4), np.int32)
    actions[0, 0] = action
    actions[0, 1] = action
    cref.step(actions[0, active])
    env.flag = SEND
    env.send(actions.reshape(2, 4))

    st = env._states
    def_host = np.asarray(st.def_id)[0]
    zone_host = np.asarray(st.zone)[0]
    latch = np.asarray(st.stt02_012_latch)[0]
    p_atk = np.asarray(st.passive_atk)[0]
    p_hp = np.asarray(st.passive_hp)[0]
    cur_atk = np.asarray(st.cur_atk)[0]
    cur_hp = np.asarray(st.cur_hp)[0]
    lines = []
    for p in range(2):
      for i in np.flatnonzero(def_host[p] == STT02_012):
        gcount = [int(np.sum((zone_host[q] == 4) )) for q in range(2)]
        lines.append(
            f"p{p}i{i} z={zone_host[p, i]} latch={bool(latch[p, i])}"
            f" patk={int(p_atk[p, i])} php={int(p_hp[p, i])}"
            f" cur={int(cur_atk[p, i])}/{int(cur_hp[p, i])}"
        )
    summary = "; ".join(lines)
    if summary != prev_summary:
      print(f"step {step} action={tuple(int(x) for x in action)}: {summary}", flush=True)
      prev_summary = summary
  cref.close()


if __name__ == "__main__":
  main()
