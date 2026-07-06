"""Native packed-obs path vs legacy emulated path: bit-exact policy equivalence.

Drives the native C vec env with random legal actions, converts each packed
observation into the legacy flat emulated layout via the original
observation_to_dict + emulate() code, and checks that two identically-seeded
policies (one per layout) produce matching encodings and logits.

Run: PYTHONPATH=build/python/src:python/src pytest python/src/test_native_obs_equivalence.py -v
"""

from __future__ import annotations

import ctypes

import numpy as np
import torch

import azk_puffer.emulation as emulation
import azk_puffer.pytorch as azk_pytorch
from azk_native import AzukiNativeEnv, NATIVE_OBS_DTYPE
from observation import OBSERVATION_CTYPE, observation_to_dict
from policy.v2 import tcg_sampler
from policy.v2.tcg_policy import TCGLSTM, build_policy_model
from tcg_parallel import AzukiTCGParallel
from training_deck_pool import load_training_deck_pool

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
POLICY_CONFIG = {
  "critic_head_type": "full_lstm_mlp",
  "win_prob_aux_enabled": True,
  "win_prob_aux_coef": 0.05,
}


def _build_policy(env_for_spec):
  torch.manual_seed(42)
  base = build_policy_model(env_for_spec, policy_config=POLICY_CONFIG)
  return TCGLSTM(env_for_spec, base).to(DEVICE).eval()


def test_native_obs_equivalence():
  pool = load_training_deck_pool()
  legacy_env = emulation.PettingZooPufferEnv(
    AzukiTCGParallel(seed=1, deck_pool=pool), seed=1
  )
  env = AzukiNativeEnv(num_envs=4, deck_pool=pool, seed=3)
  env.reset(seed=3)
  struct_view = env.observations.view(NATIVE_OBS_DTYPE).reshape(env.num_agents)

  rng = np.random.default_rng(0)
  for _ in range(80):
    for agent in range(env.num_agents):
      mask = struct_view[agent]["action_mask"]
      count = int(mask["legal_action_count"])
      if count > 0:
        pick = int(rng.integers(0, count))
        env.actions[agent] = (
          mask["legal_primary"][pick],
          mask["legal_sub1"][pick],
          mask["legal_sub2"][pick],
          mask["legal_sub3"][pick],
        )
      else:
        env.actions[agent] = 0
    env.step()

  packed = env.observations.copy()

  obs_structs = np.zeros(env.num_agents, dtype=legacy_env.obs_dtype)
  c_view = packed.ctypes.data_as(ctypes.POINTER(OBSERVATION_CTYPE))
  for agent in range(env.num_agents):
    emulation.emulate(obs_structs[agent : agent + 1], observation_to_dict(c_view[agent]))
  legacy_flat = obs_structs.view(np.uint8).reshape(env.num_agents, -1)

  pol_native = _build_policy(env)
  pol_legacy = _build_policy(legacy_env)
  sd_native = pol_native.state_dict()
  sd_legacy = pol_legacy.state_dict()
  assert set(sd_native) == set(sd_legacy)
  for key in sd_native:
    assert torch.equal(sd_native[key], sd_legacy[key]), key

  obs_native = torch.from_numpy(packed).to(DEVICE)
  obs_legacy = torch.from_numpy(legacy_flat).to(DEVICE)

  with torch.no_grad():
    enc_native, ctx_native = pol_native.policy.encode_observations(
      obs_native, state={"lstm_h": None, "lstm_c": None}
    )
    enc_legacy, ctx_legacy = pol_legacy.policy.encode_observations(
      obs_legacy, state={"lstm_h": None, "lstm_c": None}
    )

  scale = max(enc_legacy.abs().max().item(), 1.0)
  assert (enc_native - enc_legacy).abs().max().item() <= 1e-4 * scale
  for key, value in ctx_native.items():
    other = ctx_legacy.get(key)
    if torch.is_tensor(value) and torch.is_tensor(other):
      bound = max(other.float().abs().max().item(), 1.0)
      assert (value.float() - other.float()).abs().max().item() <= 1e-4 * bound, key

  tcg_sampler.set_fallback_sampler(azk_pytorch.sample_logits)
  with torch.no_grad():
    hidden = torch.zeros(env.num_agents, pol_native.hidden_size, device=DEVICE)
    state_native = {"lstm_h": hidden.clone(), "lstm_c": hidden.clone()}
    state_legacy = {"lstm_h": hidden.clone(), "lstm_c": hidden.clone()}
    logits_native, value_native = pol_native.forward_eval(obs_native, state_native)
    logits_legacy, value_legacy = pol_legacy.forward_eval(obs_legacy, state_legacy)

    assert (value_native - value_legacy).abs().max().item() <= 2e-3
    assert logits_native.legal_action_logits.shape == logits_legacy.legal_action_logits.shape
    assert (
      logits_native.legal_action_logits - logits_legacy.legal_action_logits
    ).abs().max().item() <= 2e-3
    assert torch.equal(logits_native.legal_actions, logits_legacy.legal_actions)
    assert torch.equal(logits_native.legal_action_count, logits_legacy.legal_action_count)

    actions, logprob_native, entropy_native = tcg_sampler.tcg_sample_logits(logits_native)
    _, logprob_legacy, entropy_legacy = tcg_sampler.tcg_sample_logits(
      logits_legacy, action=actions
    )
    assert (logprob_native - logprob_legacy).abs().max().item() <= 2e-3
    assert (entropy_native - entropy_legacy).abs().max().item() <= 2e-3

  env.close()
  legacy_env.close()


if __name__ == "__main__":
  test_native_obs_equivalence()
  print("EQUIVALENCE OK")
