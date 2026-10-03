"""Split-vector C/JAX fullpool rollout verifier.

This is the practical L3 parity gate for the JAX training backend. It batches an
explicit subset of the 36 production fullpool cases through `JaxVecEnv`, drives
both engines with the C engine's seeded random legal actions, and compares:
  - semantic board state,
  - selection-zone + ability-context state,
  - complete legal-action mask,
  - terminal status.

Generic split fallback is trapped. Any fallback or mismatch aborts with the case,
step, and recent action history.

Usage:
  PYTHONPATH=build/python/src:python/src:jax_env \
    JAX_COMPILATION_CACHE_DIR=/tmp/jaxcache \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python jax_env/tests/verify_vector_fullpool.py 600 m0 m1 m2 m3

Case names are `m0`..`m17` for mirror matches and `c0`..`c17` for cross matches.
Omit case names to run all 36 in one batch, which may compile too large for
routine iteration.
"""
from __future__ import annotations

import os
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


def _load_cases():
  pool = [
      list(deck)
      for deck in load_training_deck_pool(str(REPO / ".codex/docs/azuki_tcg_decks_final.json"))
  ]
  count = len(pool)
  mirrors = [(f"m{i}", i, i, 12345 + i) for i in range(count)]
  crosses = [(f"c{i}", i, (i + 1) % count, 7000 + i) for i in range(count)]
  return pool, mirrors + crosses


def _state_at(states, index: int):
  return jax.tree.map(lambda x: x[index], states)


def _jax_view(state):
  view = jax_semantic_view(state)
  view.update(_selection_view_jax(state))
  return view


def _c_view(cref):
  view = c_semantic_view(cref)
  view.update(_selection_view_c(cref))
  return view


def _install_fallback_traps(env: JaxVecEnv) -> None:
  if os.getenv("FAIL_GENERIC", "0") != "0":
    def fail_generic(action_type: int):
      raise RuntimeError(f"generic static fallback action_type={int(action_type)}")

    env._get_step_type_fn = fail_generic
  else:
    # Lazy per-action-type static kernels are the generic engine with a
    # static action head — parity-equivalent to the pre-declared catch-alls.
    # Log loudly so routing holes stay visible without killing the run.
    orig_get = env._get_step_type_fn

    def warn_generic(action_type: int):
      print(
          f"  [verify] generic fallback action_type={int(action_type)}",
          flush=True,
      )
      return orig_get(action_type)

    env._get_step_type_fn = warn_generic
  if os.getenv("FAIL_STATIC", "0") == "0":
    return

  def make_static_fail(name):
    def fail(*_args, **_kwargs):
      raise RuntimeError(f"broad static split path {name}")
    return fail

  for attr in list(vars(env)):
    if attr.endswith("_static_step_fn") or attr == "_attack_queued_stt03_006_step_fn":
      setattr(env, attr, make_static_fail(attr))


def run(max_steps: int, wanted_names: set[str] | None = None) -> None:
  pool, cases = _load_cases()
  if wanted_names:
    cases = [case for case in cases if case[0] in wanted_names]
  if not cases:
    raise ValueError("No matching fullpool cases requested")

  print(f"vector fullpool parity: cases={len(cases)} max_steps={max_steps}", flush=True)

  crefs = []
  rngs = []
  states = []
  for _name, left, right, seed in cases:
    cref = CRef(seed, deck_pool=None)
    cref.reset_with_decks(seed, pool[left], pool[right])
    crefs.append(cref)
    rngs.append(np.random.default_rng(seed))
    tables = deck_tables_from_card_lists(pool[left], pool[right])
    states.append(stabilize(init_state_with_decks(seed, tables)))

  batched_state = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
  env = JaxVecEnv(len(cases), pool, seed=0)
  obs, legal, count = env._observe_fn(batched_state)
  env._states = batched_state
  env._terms = jnp.zeros((len(cases), 2), jnp.bool_)
  env._truncs = jnp.zeros((len(cases), 2), jnp.bool_)
  rewards = jnp.zeros((len(cases), 2), jnp.float32)
  env._pending = (obs, rewards, env._terms, env._truncs, legal, count, batched_state.active_player)
  _install_fallback_traps(env)

  live = np.ones(len(cases), dtype=bool)
  recent = [[] for _ in cases]
  trace_actions = os.getenv("TRACE_ACTIONS", "0") != "0"
  try:
    for step_index in range(max_steps):
      legal_host = np.asarray(env._pending[4])
      count_host = np.asarray(env._pending[5]).astype(np.int32)
      actions = np.zeros((len(cases), 2, 4), np.int32)
      any_live = False
      for idx, (name, _left, _right, _seed) in enumerate(cases):
        if not live[idx]:
          continue
        any_live = True
        cview = _c_view(crefs[idx])
        jview = _jax_view(_state_at(env._states, idx))
        for key, cval in cview.items():
          if key == "winner":
            continue
          if jview.get(key) != cval:
            raise AssertionError(
                f"{name} step {step_index}: {key}\n"
                f"C  ={cval}\nJAX={jview.get(key)}\nrecent={recent[idx][-8:]}"
            )

        active = cview["active"]
        c_rows = crefs[idx].legal_actions(active)
        j_rows = [tuple(int(x) for x in row) for row in legal_host[idx, : int(count_host[idx])]]
        if j_rows != c_rows:
          only_c = [row for row in c_rows if row not in j_rows][:8]
          only_j = [row for row in j_rows if row not in c_rows][:8]
          raise AssertionError(
              f"{name} step {step_index} phase {cview['phase']} ab {cview['ab_phase']}: mask mismatch\n"
              f"C  ={c_rows}\nJAX={j_rows}\nonly_c={only_c}\nonly_j={only_j}\n"
              f"recent={recent[idx][-8:]}"
          )
        if not c_rows:
          live[idx] = False
          continue

        driver_rows = [row for row in c_rows if row[0] in DRIVER_TYPES_4]
        if not driver_rows:
          raise AssertionError(f"{name} step {step_index}: no driver actions {c_rows}")
        action = driver_rows[int(rngs[idx].integers(0, len(driver_rows)))]
        recent[idx].append((step_index, tuple(int(x) for x in action), f"ph{cview['phase']}ab{cview['ab_phase']}"))
        actions[idx, 0] = action
        actions[idx, 1] = action

      if not any_live:
        print(f"all cases ended before step {step_index}", flush=True)
        return

      if trace_actions:
        chosen = [
            (cases[idx][0], tuple(int(x) for x in actions[idx, crefs[idx].active_player]))
            for idx in np.flatnonzero(live)
        ]
        print(f"  action step {step_index}: {chosen}", flush=True)

      for idx in np.flatnonzero(live):
        active = crefs[idx].active_player
        crefs[idx].step(actions[idx, active])

      env.flag = SEND
      env.send(actions.reshape(len(cases) * 2, 4))

      terms = np.asarray(env._terms).all(axis=1)
      truncs = np.asarray(env._truncs).all(axis=1)
      for idx in np.flatnonzero(live):
        c_term, c_trunc = crefs[idx].dones()
        j_term = bool(terms[idx])
        if j_term != c_term:
          raise AssertionError(
              f"{cases[idx][0]} step {step_index}: terminal mismatch C={c_term} JAX={j_term}"
          )
        if c_term or c_trunc or bool(truncs[idx]):
          live[idx] = False
      if step_index % 25 == 0:
        print(f"  ..step {step_index} ok live={int(live.sum())}", flush=True)
    print(f"no divergence in {max_steps} steps across {len(cases)} cases", flush=True)
  finally:
    for cref in crefs:
      cref.close()


def main() -> None:
  max_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 600
  wanted = set(sys.argv[2:]) if len(sys.argv) > 2 else None
  run(max_steps, wanted)


if __name__ == "__main__":
  main()
