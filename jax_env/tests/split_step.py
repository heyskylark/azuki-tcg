"""Split-jit driver: jit apply_user_action and micro_tick SEPARATELY and run
the auto_resolve loop in Python, avoiding the ~2h fused engine_step compile.

Semantically identical to engine.step.engine_step / stabilize: the fused
auto_resolve is `while ~requires_action & winner==-1 & tick_guard<64: micro_tick`
(cond checked before each body). The Python loop mirrors that exactly; the only
added cost is one device->host sync per tick for the cond. Use for fast
diagnostics and verification when the giant fused compile is not worth paying.
"""
from __future__ import annotations

import jax

from azuki_jax.engine.phases import requires_action
from azuki_jax.engine.step import MAX_AUTO_TICKS, apply_user_action, micro_tick


def make_split_step():
  jit_apply = jax.jit(apply_user_action)
  jit_tick = jax.jit(micro_tick)
  jit_req = jax.jit(requires_action)

  def _drive(state):
    for _ in range(MAX_AUTO_TICKS):
      if int(state.winner) != -1 or bool(jit_req(state)):
        break
      state = jit_tick(state)
    return state

  def split_step(state, action):
    state = jit_apply(state, action)
    return _drive(state)

  def split_stabilize(state):
    return _drive(state)

  return split_step, split_stabilize, jit_tick, jit_apply
