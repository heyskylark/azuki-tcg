"""Side-by-side C vs JAX state dump around m14 steps 180-182."""
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
from test_l2_vanilla import c_semantic_view, jax_semantic_view  # noqa: E402
from test_l3_abilities_batch3 import _selection_view_c, _selection_view_jax  # noqa: E402
from test_l3_abilities_batch4 import DRIVER_TYPES_4  # noqa: E402
from training_deck_pool import load_training_deck_pool  # noqa: E402
from azk_puffer.jax_vector import JaxVecEnv, SEND  # noqa: E402
from azuki_jax.env import init_state_with_decks  # noqa: E402
from azuki_jax.engine.step import stabilize  # noqa: E402
from azuki_jax.setup import deck_tables_from_card_lists  # noqa: E402

LEFT, RIGHT, SEED, STOP = 14, 14, 12359, 182


def dump(tag, cview, jview):
  keys = ("hand0", "hand1", "alley0", "alley1", "deck0")
  for k in keys:
    c = cview.get(k)
    j = jview.get(k)
    marker = "  " if c == j else "!!"
    print(f"{marker} {tag} {k}: C={c}")
    if c != j:
      print(f"{marker} {tag} {k}: J={j}")
  for k, v in cview.items():
    if k.startswith("ab_") or "selection" in k:
      j = jview.get(k)
      if v != j:
        print(f"!! {tag} {k}: C={v} J={j}")


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

  def jview():
    view = jax_semantic_view(jax.tree.map(lambda x: x[0], env._states))
    view.update(_selection_view_jax(jax.tree.map(lambda x: x[0], env._states)))
    return view

  def cview():
    view = c_semantic_view(cref)
    view.update(_selection_view_c(cref))
    return view

  for step in range(STOP):
    active = cref.active_player
    rows = cref.legal_actions(active)
    if not rows:
      print(f"ended at {step}")
      return
    driver = [r for r in rows if r[0] in DRIVER_TYPES_4]
    action = driver[int(rng.integers(0, len(driver)))]
    actions = np.zeros((1, 2, 4), np.int32)
    actions[0, 0] = action
    actions[0, 1] = action
    if step >= 179:
      print(f"=== step {step} action={action}")
    cref.step(actions[0, active])
    env.flag = SEND
    env.send(actions.reshape(2, 4))
    if step >= 179:
      dump(f"post{step}", cview(), jview())
    if step % 25 == 0:
      print(f"  step {step}", flush=True)

  cref.close()


if __name__ == "__main__":
  main()
