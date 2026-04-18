import azk_puffer.pytorch as azk_pytorch
from azk_puffer.models import LSTMWrapper
from gymnasium.wrappers.normalize import RunningMeanStd

import numpy as np
import torch
from torch import nn

from observation import (
  ACTION_TYPE_COUNT,
  ALLEY_SIZE,
  GARDEN_SIZE,
  MAX_ATTACHED_WEAPONS,
  MAX_DECK_SIZE,
  MAX_HAND_SIZE,
  RECENT_ACTION_HISTORY_LEN,
  MAX_SELECTION_ZONE_SIZE,
)
from policy.card_metadata_table import load_policy_card_metadata_table
from policy.tcg_distribution import TCGActionDistribution, TCGLegalActionDistribution

MAX_PLAYERS_PER_MATCH = 2
CARD_TYPE_COUNT = 7
ELEMENT_COUNT = 5
GAME_PHASE_COUNT = 8
ABILITY_PHASE_COUNT = 6
PRIMARY_ACTION_COUNT = ACTION_TYPE_COUNT
PRIMARY_ACTION_ID_BATCH = tuple(range(PRIMARY_ACTION_COUNT))
MAX_INDEX_SIZE = 50
ACTION_COMPONENT_COUNT = 4
ACT_NOOP = 0
ACT_PLAY_ENTITY_TO_GARDEN = 1
ACT_PLAY_ENTITY_TO_ALLEY = 2
ACT_ATTACK = 6
ACT_ATTACH_WEAPON_FROM_HAND = 7
ACT_PLAY_SPELL_FROM_HAND = 8
ACT_DECLARE_DEFENDER = 9
ACT_GATE_PORTAL = 10
ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY = 11
ACT_ACTIVATE_ALLEY_ABILITY = 12
ACT_SELECT_COST_TARGET = 13
ACT_SELECT_EFFECT_TARGET = 14
ACT_CONFIRM_ABILITY = 16
ACT_SELECT_FROM_SELECTION = 18
ACT_BOTTOM_DECK_CARD = 19
ACT_BOTTOM_DECK_ALL = 20
ACT_SELECT_TO_ALLEY = 21
ACT_SELECT_TO_EQUIP = 22
ACT_SELECT_TO_GARDEN = 23
ACT_TOP_DECK_CARD = 24
ACT_MULLIGAN_SHUFFLE = 25

CARD_TYPE_ENC_OUTPUT_SIZE = 4
ELEMENT_ENC_OUTPUT_SIZE = 4
ABILITY_TIMING_ENC_OUTPUT_SIZE = 4
INDEX_ENC_OUTPUT_SIZE = 8
PHASE_ENC_OUTPUT_SIZE = 4
ABILITY_PHASE_ENC_OUTPUT_SIZE = 4
NAME_TEXT_ENC_OUTPUT_SIZE = 16
EFFECT_TEXT_ENC_OUTPUT_SIZE = 24
SUBTYPE_TEXT_ENC_OUTPUT_SIZE = 16
KEYWORD_FEATURE_ENC_OUTPUT_SIZE = 16
CARD_METADATA_EMBED_SIZE = 48
ACTION_HISTORY_PRIMARY_ENC_OUTPUT_SIZE = 8
ACTION_HISTORY_SUBACTION_ENC_OUTPUT_SIZE = 8
ACTION_HISTORY_STEP_EMBED_SIZE = 16
ACTION_HISTORY_PLAYER_EMBED_SIZE = 32
UNIT_EMBED_SIZE = 64
CRITIC_HEAD_TYPE_SHARED_PRIMARY = "shared_primary"
CRITIC_HEAD_TYPE_FULL_LSTM_MLP = "full_lstm_mlp"
CRITIC_MLP_HIDDEN_SIZE = 512
CRITIC_MLP_PROJECTION_SIZE = 128
WIN_PROB_AUX_COEF_DEFAULT = 0.1
SPLIT_VALUE_COMPONENT_COEF_DEFAULT = 0.5
PRIVILEGED_CRITIC_ENABLED_DEFAULT = False
PRIVILEGED_CRITIC_TOKEN_SCALAR_SIZE = 2
PRIVILEGED_CRITIC_EMBED_DIM = UNIT_EMBED_SIZE
PRIVILEGED_CRITIC_DECK_HEADS = 4
PRIVILEGED_CRITIC_DECK_LAYERS = 2
PRIVILEGED_CRITIC_DECK_FF_SIZE = 256
PRIVILEGED_CRITIC_FUSION_HIDDEN_SIZE = CRITIC_MLP_HIDDEN_SIZE
PRIVILEGED_CRITIC_FUSION_PROJECTION_SIZE = CRITIC_MLP_PROJECTION_SIZE
PRIVILEGED_CRITIC_FEATURE_SCALE_DEFAULT = 1.0

PROCESS_SET_HIDDEN_SIZE = 128
LSTM_HIDDEN_SIZE = 4096
POLICY_MODEL_VERSION_METADATA_V1 = "metadata_v1"
ACTOR_HEAD_TYPE_FACTORIZED = "factorized"
ACTOR_HEAD_TYPE_LEGAL_ACTION_SCORER = "legal_action_scorer"
LEGAL_ACTION_SCORER_USE_REFERENCES_DEFAULT = True
LEGAL_ACTION_ARG_KIND_UNUSED = 0
LEGAL_ACTION_ARG_KIND_HAND = 1
LEGAL_ACTION_ARG_KIND_SELF_GARDEN = 2
LEGAL_ACTION_ARG_KIND_SELF_ALLEY = 3
LEGAL_ACTION_ARG_KIND_SELF_GARDEN_OR_LEADER = 4
LEGAL_ACTION_ARG_KIND_OPP_DEFENDER = 5
LEGAL_ACTION_ARG_KIND_SELECTION = 6
LEGAL_ACTION_ARG_KIND_ABILITY_INDEX = 7
LEGAL_ACTION_ARG_KIND_BOOL = 8
LEGAL_ACTION_ARG_KIND_GENERIC_TARGET = 9
LEGAL_ACTION_ARG_KIND_COUNT = 10
LEGAL_ACTION_ARG_KIND_EMBED_SIZE = 8


def _numpy_dtype_to_torch(dtype: np.dtype) -> torch.dtype:
  dtype = np.dtype(dtype)
  if dtype == np.dtype(np.int8):
    return torch.int8
  if dtype == np.dtype(np.uint8):
    return torch.uint8
  if dtype == np.dtype(np.int16):
    return torch.int16
  if dtype == np.dtype(np.uint16):
    return torch.uint16
  if dtype == np.dtype(np.int32):
    return torch.int32
  if dtype == np.dtype(np.uint32):
    return torch.uint32
  if dtype == np.dtype(np.int64):
    return torch.int64
  if dtype == np.dtype(np.uint64):
    return torch.uint64
  if dtype == np.dtype(np.float32):
    return torch.float32
  if dtype == np.dtype(np.float64):
    return torch.float64
  if dtype == np.dtype(np.bool_):
    return torch.bool
  raise TypeError(f"Unsupported numpy dtype for nativize: {dtype}")


def _build_native_dtype_from_numpy(dtype: np.dtype, base_offset: int = 0):
  def _prefix_shape(meta, prefix_shape, delta):
    if isinstance(meta, tuple):
      torch_dtype, shape, offset, _ = meta
      return (torch_dtype, tuple(prefix_shape) + tuple(shape), offset, delta)
    return {
      key: _prefix_shape(value, prefix_shape, delta)
      for key, value in meta.items()
    }

  dtype = np.dtype(dtype)
  if dtype.fields:
    out = {}
    for name, (field_dtype, field_offset) in dtype.fields.items():
      out[name] = _build_native_dtype_from_numpy(
        np.dtype(field_dtype),
        base_offset + int(field_offset),
      )
    return out

  if dtype.subdtype is not None:
    scalar_dtype, subshape = dtype.subdtype
    base_meta = _build_native_dtype_from_numpy(
      np.dtype(scalar_dtype),
      base_offset,
    )
    return _prefix_shape(base_meta, tuple(int(x) for x in subshape), int(dtype.itemsize))

  torch_dtype = _numpy_dtype_to_torch(dtype)
  return (torch_dtype, (), int(base_offset), int(dtype.itemsize))


class ScalarRunningNorm(nn.Module):
  """Normalizes scalar/boolean tensors with running mean/std and clamps to [-clip, clip]."""

  def __init__(self, *, clip: float = 5.0, eps: float = 1e-8, rms_epsilon: float = 1e-4):
    super().__init__()
    self.clip = clip
    self.eps = eps
    self.rms_epsilon = rms_epsilon
    self._rms = {}

  def _buffer_names(self, key: str):
    return (
      f"_rms_{key}_mean",
      f"_rms_{key}_var",
      f"_rms_{key}_count",
    )

  def _ensure_buffers(self, key: str, feature_shape):
    mean_name, var_name, count_name = self._buffer_names(key)
    shape = torch.Size(feature_shape) if feature_shape else torch.Size([])
    if not hasattr(self, mean_name):
      self.register_buffer(mean_name, torch.zeros(shape, dtype=torch.float64))
      self.register_buffer(var_name, torch.ones(shape, dtype=torch.float64))
      self.register_buffer(count_name, torch.tensor(self.rms_epsilon, dtype=torch.float64))

  def _get_rms(self, key: str, feature_shape):
    if key in self._rms:
      return self._rms[key]

    mean_name, var_name, count_name = self._buffer_names(key)
    mean_buf = getattr(self, mean_name, None)
    if mean_buf is not None:
      var_buf = getattr(self, var_name)
      count_buf = getattr(self, count_name)
      rms = RunningMeanStd(shape=tuple(mean_buf.shape), epsilon=self.rms_epsilon)
      rms.mean = mean_buf.detach().cpu().numpy()
      rms.var = var_buf.detach().cpu().numpy()
      rms.count = float(count_buf.detach().cpu().item())
    else:
      rms = RunningMeanStd(shape=feature_shape, epsilon=self.rms_epsilon)
      self._ensure_buffers(key, feature_shape)
    self._rms[key] = rms
    return rms

  def _sync_buffers(self, key: str, rms: RunningMeanStd):
    mean_name, var_name, count_name = self._buffer_names(key)
    setattr(self, mean_name, torch.as_tensor(rms.mean, dtype=torch.float64))
    setattr(self, var_name, torch.as_tensor(rms.var, dtype=torch.float64))
    setattr(self, count_name, torch.tensor(rms.count, dtype=torch.float64))

  def _get_stats(self, key: str, device: torch.device, dtype: torch.dtype):
    mean_name, var_name, _ = self._buffer_names(key)
    mean = getattr(self, mean_name).to(device=device, dtype=dtype)
    var = getattr(self, var_name).to(device=device, dtype=dtype)
    return mean, var

  def forward(self, key: str, tensor: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if tensor is None:
      return tensor

    tensor = tensor.float()
    feature_shape = () if tensor.dim() == 1 else (tensor.shape[-1],)
    rms = self._get_rms(key, feature_shape)

    if self.training:
      with torch.no_grad():
        values_for_update = tensor
        if mask is not None:
          mask_update = mask
          if mask_update.dim() == tensor.dim():
            mask_update = mask_update.any(dim=-1)
          values_for_update = tensor[mask_update]
        np_values = values_for_update.detach().cpu().numpy()
        if np_values.size > 0:
          rms.update(np_values.reshape((-1, *feature_shape)) if feature_shape else np_values.reshape(-1))
          self._sync_buffers(key, rms)

    mean, var = self._get_stats(key, tensor.device, tensor.dtype)
    normalized = (tensor - mean) / torch.sqrt(var + self.eps)

    if mask is not None:
      mask_broadcast = mask
      while mask_broadcast.dim() < normalized.dim():
        mask_broadcast = mask_broadcast.unsqueeze(-1)
      normalized = torch.where(mask_broadcast, normalized, torch.zeros_like(normalized))

    return torch.clamp(normalized, -self.clip, self.clip)


class ProcessSetProcessor(nn.Module):
  def __init__(self, input_size, hidden_size=PROCESS_SET_HIDDEN_SIZE, output_size=UNIT_EMBED_SIZE):
    super().__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, output_size)
    self.act = nn.ReLU()
    self.output_size = output_size

  def forward(self, x, mask=None):
    hidden = self.act(self.fc1(x))
    set_embeddings = self.fc2(hidden)

    if mask is not None:
      if mask.dim() == set_embeddings.dim() - 1:
        mask = mask.unsqueeze(-1)
      mask = mask.to(dtype=torch.bool)
      expanded_mask = mask.expand_as(set_embeddings)
      masked_embeddings = set_embeddings.masked_fill(~expanded_mask, torch.finfo(set_embeddings.dtype).min)
      pooled, _ = torch.max(masked_embeddings, dim=-2)
      slot_valid = expanded_mask.any(dim=-2)
      pooled = torch.where(slot_valid, pooled, torch.zeros_like(pooled))
      pooled = torch.nan_to_num(pooled, nan=0.0, neginf=0.0, posinf=0.0)
      set_embeddings = set_embeddings * expanded_mask.to(dtype=set_embeddings.dtype)
      return set_embeddings, pooled

    pooled, _ = torch.max(set_embeddings, dim=-2)
    return set_embeddings, pooled


class SingleUnitProjection(nn.Module):
  def __init__(self, input_size, hidden_size=PROCESS_SET_HIDDEN_SIZE, output_size=UNIT_EMBED_SIZE):
    super().__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, output_size)
    self.act = nn.ReLU()

  def forward(self, x):
    return self.fc2(self.act(self.fc1(x)))


class MaskedTransformerDeckEncoder(nn.Module):
  def __init__(
    self,
    model_dim: int = UNIT_EMBED_SIZE,
    *,
    num_heads: int = PRIVILEGED_CRITIC_DECK_HEADS,
    num_layers: int = PRIVILEGED_CRITIC_DECK_LAYERS,
    ff_size: int = PRIVILEGED_CRITIC_DECK_FF_SIZE,
  ):
    super().__init__()
    self.model_dim = int(model_dim)
    encoder_layer = nn.TransformerEncoderLayer(
      d_model=model_dim,
      nhead=num_heads,
      dim_feedforward=ff_size,
      dropout=0.0,
      activation="gelu",
      batch_first=True,
      norm_first=True,
    )
    self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
    self.output_norm = nn.LayerNorm(model_dim)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    for module in self.encoder.modules():
      if module is self.encoder:
        continue
      reset_parameters = getattr(module, "reset_parameters", None)
      if callable(reset_parameters):
        reset_parameters()
    self.output_norm.reset_parameters()
    nn.init.normal_(self.cls_token, std=0.02)

  def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if tokens.dim() != 3:
      raise ValueError(f"Deck tokens must be rank 3, got shape {tuple(tokens.shape)}")

    batch_size = tokens.shape[0]
    cls = self.cls_token.expand(batch_size, -1, -1)
    sequence = torch.cat([cls, tokens], dim=1)

    if mask.dim() != 2:
      raise ValueError(f"Deck mask must be rank 2, got shape {tuple(mask.shape)}")
    cls_mask = torch.ones((batch_size, 1), device=mask.device, dtype=torch.bool)
    valid_mask = torch.cat([cls_mask, mask.to(dtype=torch.bool)], dim=1)
    key_padding_mask = ~valid_mask

    encoded = self.encoder(sequence, src_key_padding_mask=key_padding_mask)
    return self.output_norm(encoded[:, 0, :])


class TCGLSTM(LSTMWrapper):
  def __init__(self, env, policy, input_size=None, hidden_size=LSTM_HIDDEN_SIZE):
    if input_size is None:
      input_size = policy.lstm_input_size
    super().__init__(env, policy, input_size, hidden_size)

  def _split_encoded(self, encoded):
    if isinstance(encoded, tuple) and len(encoded) == 2:
      return encoded
    return encoded, None

  def forward_eval(self, observations, state):
    lstm_inputs, action_context = self._split_encoded(
      self.policy.encode_observations(observations, state=state)
    )
    batch_size = int(lstm_inputs.shape[0]) if torch.is_tensor(lstm_inputs) else None
    h = state.get("lstm_h")
    c = state.get("lstm_c")

    if h is not None:
      if batch_size is None:
        raise ValueError("Unable to infer batch size from observations")
      assert h.shape[0] == c.shape[0] == batch_size, "LSTM state must be (h, c)"
      lstm_state = (h, c)
    else:
      lstm_state = None

    hidden, c = self.cell(lstm_inputs, lstm_state)
    state["hidden"] = hidden
    state["lstm_h"] = hidden
    state["lstm_c"] = c
    logits, values = self.policy.decode_actions(hidden, action_context=action_context, state=state)
    win_prob_logits = state.get("_azk_win_prob_logits")
    if torch.is_tensor(win_prob_logits):
      state["_azk_win_prob_logits"] = win_prob_logits.reshape(batch_size)
    terminal_value = state.get("_azk_value_terminal")
    if torch.is_tensor(terminal_value):
      state["_azk_value_terminal"] = terminal_value.reshape(batch_size)
    shaped_value = state.get("_azk_value_shaped")
    if torch.is_tensor(shaped_value):
      state["_azk_value_shaped"] = shaped_value.reshape(batch_size)
    return logits, values

  def forward(self, observations, state):
    x = observations
    lstm_h = state.get("lstm_h")
    lstm_c = state.get("lstm_c")

    x_shape, space_shape = x.shape, self.obs_shape
    x_n, space_n = len(x_shape), len(space_shape)
    if x_shape[-space_n:] != space_shape:
      raise ValueError("Invalid input tensor shape", x.shape)

    if x_n == space_n + 1:
      B, TT = x_shape[0], 1
    elif x_n == space_n + 2:
      B, TT = x_shape[:2]
    else:
      raise ValueError("Invalid input tensor shape", x.shape)

    if lstm_h is not None:
      assert lstm_h.shape[1] == lstm_c.shape[1] == B, "LSTM state must be (h, c)"
      lstm_state = (lstm_h, lstm_c)
    else:
      lstm_state = None

    x = x.reshape(B * TT, *space_shape)
    lstm_inputs, action_context = self._split_encoded(self.policy.encode_observations(x, state))

    hidden = lstm_inputs.reshape(B, TT, self.input_size).transpose(0, 1)
    hidden, (lstm_h, lstm_c) = self.lstm.forward(hidden, lstm_state)
    hidden = hidden.float().transpose(0, 1)

    flat_hidden = hidden.reshape(B * TT, self.hidden_size)
    logits, values = self.policy.decode_actions(flat_hidden, action_context=action_context, state=state)
    values = values.reshape(B, TT)
    win_prob_logits = state.get("_azk_win_prob_logits")
    if torch.is_tensor(win_prob_logits):
      state["_azk_win_prob_logits"] = win_prob_logits.reshape(B, TT)
    terminal_value = state.get("_azk_value_terminal")
    if torch.is_tensor(terminal_value):
      state["_azk_value_terminal"] = terminal_value.reshape(B, TT)
    shaped_value = state.get("_azk_value_shaped")
    if torch.is_tensor(shaped_value):
      state["_azk_value_shaped"] = shaped_value.reshape(B, TT)
    state["hidden"] = hidden
    state["lstm_h"] = lstm_h.detach()
    state["lstm_c"] = lstm_c.detach()
    return logits, values


class TCG(nn.Module):
  def __init__(
    self,
    env,
    *,
    model_version: str = POLICY_MODEL_VERSION_METADATA_V1,
    actor_head_type: str = ACTOR_HEAD_TYPE_FACTORIZED,
    legal_action_scorer_use_references: bool = LEGAL_ACTION_SCORER_USE_REFERENCES_DEFAULT,
    critic_head_type: str = CRITIC_HEAD_TYPE_FULL_LSTM_MLP,
    privileged_critic_enabled: bool = PRIVILEGED_CRITIC_ENABLED_DEFAULT,
    privileged_critic_embed_dim: int = PRIVILEGED_CRITIC_EMBED_DIM,
    privileged_critic_deck_heads: int = PRIVILEGED_CRITIC_DECK_HEADS,
    privileged_critic_deck_layers: int = PRIVILEGED_CRITIC_DECK_LAYERS,
    privileged_critic_deck_ff_size: int = PRIVILEGED_CRITIC_DECK_FF_SIZE,
    privileged_critic_fusion_hidden_size: int = PRIVILEGED_CRITIC_FUSION_HIDDEN_SIZE,
    privileged_critic_fusion_projection_size: int = PRIVILEGED_CRITIC_FUSION_PROJECTION_SIZE,
    privileged_critic_feature_scale: float = PRIVILEGED_CRITIC_FEATURE_SCALE_DEFAULT,
    win_prob_aux_enabled: bool = False,
    win_prob_aux_coef: float = WIN_PROB_AUX_COEF_DEFAULT,
    split_value_heads_enabled: bool = False,
    split_value_component_coef: float = SPLIT_VALUE_COMPONENT_COEF_DEFAULT,
    **kwargs,
  ):
    super().__init__()

    self.is_continuous = False
    self.model_version = model_version
    self.actor_head_type = str(actor_head_type)
    self.legal_action_scorer_use_references = bool(legal_action_scorer_use_references)
    self.critic_head_type = str(critic_head_type)
    self.privileged_critic_enabled = bool(privileged_critic_enabled)
    self.privileged_critic_embed_dim = int(privileged_critic_embed_dim)
    self.privileged_critic_deck_heads = int(privileged_critic_deck_heads)
    self.privileged_critic_deck_layers = int(privileged_critic_deck_layers)
    self.privileged_critic_deck_ff_size = int(privileged_critic_deck_ff_size)
    self.privileged_critic_fusion_hidden_size = int(privileged_critic_fusion_hidden_size)
    self.privileged_critic_fusion_projection_size = int(privileged_critic_fusion_projection_size)
    self.privileged_critic_feature_scale = float(privileged_critic_feature_scale)
    self.win_prob_aux_enabled = bool(win_prob_aux_enabled)
    self.win_prob_aux_coef = float(win_prob_aux_coef)
    self.split_value_heads_enabled = bool(split_value_heads_enabled)
    self.split_value_component_coef = float(split_value_component_coef)
    self.scalar_normalizer = ScalarRunningNorm()
    if self.actor_head_type not in {
      ACTOR_HEAD_TYPE_FACTORIZED,
      ACTOR_HEAD_TYPE_LEGAL_ACTION_SCORER,
    }:
      raise ValueError(
        f"Unsupported actor_head_type '{self.actor_head_type}'. "
        f"Known values: {ACTOR_HEAD_TYPE_FACTORIZED}, {ACTOR_HEAD_TYPE_LEGAL_ACTION_SCORER}"
      )
    if self.privileged_critic_embed_dim <= 0:
      raise ValueError(
        f"privileged_critic_embed_dim must be positive, got {self.privileged_critic_embed_dim}"
      )
    if self.privileged_critic_deck_heads <= 0:
      raise ValueError(
        f"privileged_critic_deck_heads must be positive, got {self.privileged_critic_deck_heads}"
      )
    if self.privileged_critic_embed_dim % self.privileged_critic_deck_heads != 0:
      raise ValueError(
        "privileged_critic_embed_dim must be divisible by privileged_critic_deck_heads, "
        f"got embed_dim={self.privileged_critic_embed_dim} "
        f"heads={self.privileged_critic_deck_heads}"
      )
    if self.privileged_critic_deck_layers <= 0:
      raise ValueError(
        f"privileged_critic_deck_layers must be positive, got {self.privileged_critic_deck_layers}"
      )
    if self.privileged_critic_deck_ff_size <= 0:
      raise ValueError(
        f"privileged_critic_deck_ff_size must be positive, got {self.privileged_critic_deck_ff_size}"
      )
    if self.privileged_critic_fusion_hidden_size <= 0:
      raise ValueError(
        "privileged_critic_fusion_hidden_size must be positive, "
        f"got {self.privileged_critic_fusion_hidden_size}"
      )
    if self.privileged_critic_fusion_projection_size <= 0:
      raise ValueError(
        "privileged_critic_fusion_projection_size must be positive, "
        f"got {self.privileged_critic_fusion_projection_size}"
      )
    if self.privileged_critic_feature_scale < 0.0:
      raise ValueError(
        f"privileged_critic_feature_scale must be non-negative, got {self.privileged_critic_feature_scale}"
      )
    if self.win_prob_aux_coef < 0.0:
      raise ValueError(f"win_prob_aux_coef must be non-negative, got {self.win_prob_aux_coef}")
    if self.split_value_component_coef < 0.0:
      raise ValueError(
        f"split_value_component_coef must be non-negative, got {self.split_value_component_coef}"
      )

    if self.critic_head_type not in {
      CRITIC_HEAD_TYPE_SHARED_PRIMARY,
      CRITIC_HEAD_TYPE_FULL_LSTM_MLP,
    }:
      raise ValueError(
        f"Unsupported critic_head_type '{self.critic_head_type}'. "
        f"Known critic heads: {CRITIC_HEAD_TYPE_SHARED_PRIMARY}, {CRITIC_HEAD_TYPE_FULL_LSTM_MLP}"
      )

    static_table = load_policy_card_metadata_table()
    self.static_vocab_size = static_table.vocab_size
    self.metadata_embedding_dim = static_table.embedding_dim
    self.keyword_vocab_size = static_table.keyword_vocab_size
    self.register_buffer("static_card_present_mask", static_table.card_present_mask)
    self.register_buffer("static_card_type", static_table.card_type_ids)
    self.register_buffer("static_element", static_table.element_ids)
    self.register_buffer("static_base_ikz_cost", static_table.ikz_cost)
    self.register_buffer("static_base_attack", static_table.attack)
    self.register_buffer("static_base_health", static_table.health)
    self.register_buffer("static_base_gate_points", static_table.gate_points)
    self.register_buffer("static_has_ability", static_table.has_ability)
    self.register_buffer("static_ability_timing", static_table.ability_timing_ids)
    self.register_buffer("static_ability_optional", static_table.ability_is_optional)
    self.register_buffer("static_keyword_multi_hot", static_table.keyword_multi_hot)
    self.register_buffer("static_name_embeddings", static_table.name_embeddings)
    self.register_buffer("static_effect_embeddings", static_table.effect_embeddings)
    self.register_buffer("static_subtype_pooled_embeddings", static_table.subtype_pooled_embeddings)

    ability_timing_vocab = int(self.static_ability_timing.max().item()) + 1
    ability_timing_vocab = max(ability_timing_vocab, 1)

    self.card_type_encoder = nn.Embedding(CARD_TYPE_COUNT, CARD_TYPE_ENC_OUTPUT_SIZE)
    self.element_encoder = nn.Embedding(ELEMENT_COUNT, ELEMENT_ENC_OUTPUT_SIZE)
    self.ability_timing_encoder = nn.Embedding(ability_timing_vocab, ABILITY_TIMING_ENC_OUTPUT_SIZE)
    self.index_encoder = nn.Embedding(MAX_INDEX_SIZE, INDEX_ENC_OUTPUT_SIZE)
    self.game_phase_encoder = nn.Embedding(GAME_PHASE_COUNT, PHASE_ENC_OUTPUT_SIZE)
    self.ability_phase_encoder = nn.Embedding(ABILITY_PHASE_COUNT, ABILITY_PHASE_ENC_OUTPUT_SIZE)
    self.action_history_primary_encoder = nn.Embedding(
      PRIMARY_ACTION_COUNT + 1, ACTION_HISTORY_PRIMARY_ENC_OUTPUT_SIZE
    )
    self.action_history_subaction_encoder = nn.Embedding(
      MAX_INDEX_SIZE + 1, ACTION_HISTORY_SUBACTION_ENC_OUTPUT_SIZE
    )
    self.name_text_encoder = nn.Linear(
      self.metadata_embedding_dim, NAME_TEXT_ENC_OUTPUT_SIZE, bias=False
    )
    self.effect_text_encoder = nn.Linear(
      self.metadata_embedding_dim, EFFECT_TEXT_ENC_OUTPUT_SIZE, bias=False
    )
    self.subtype_text_encoder = nn.Linear(
      self.metadata_embedding_dim, SUBTYPE_TEXT_ENC_OUTPUT_SIZE, bias=False
    )
    self.keyword_feature_encoder = nn.Linear(
      self.keyword_vocab_size, KEYWORD_FEATURE_ENC_OUTPUT_SIZE, bias=False
    )
    card_metadata_input_size = (
      NAME_TEXT_ENC_OUTPUT_SIZE
      + EFFECT_TEXT_ENC_OUTPUT_SIZE
      + SUBTYPE_TEXT_ENC_OUTPUT_SIZE
      + KEYWORD_FEATURE_ENC_OUTPUT_SIZE
      + CARD_TYPE_ENC_OUTPUT_SIZE
      + ELEMENT_ENC_OUTPUT_SIZE
      + ABILITY_TIMING_ENC_OUTPUT_SIZE
      + 7
    )
    self.card_metadata_projector = SingleUnitProjection(
      card_metadata_input_size,
      hidden_size=PROCESS_SET_HIDDEN_SIZE,
      output_size=CARD_METADATA_EMBED_SIZE,
    )

    self.primary_action_encoder = nn.Sequential(
      nn.Embedding(PRIMARY_ACTION_COUNT, UNIT_EMBED_SIZE),
      nn.Flatten(),
    )
    self.register_buffer("primary_action_id_batch", torch.tensor(PRIMARY_ACTION_ID_BATCH, dtype=torch.long))
    self.legal_action_subaction_encoder = nn.Embedding(
      MAX_INDEX_SIZE + 1,
      INDEX_ENC_OUTPUT_SIZE,
    )
    self.legal_action_arg_kind_encoder = nn.Embedding(
      LEGAL_ACTION_ARG_KIND_COUNT,
      LEGAL_ACTION_ARG_KIND_EMBED_SIZE,
    )

    weapon_input_size = (
      CARD_METADATA_EMBED_SIZE
      + 5
    )
    self.weapon_set_processor = ProcessSetProcessor(weapon_input_size)

    hand_input_size = (
      CARD_METADATA_EMBED_SIZE
      + INDEX_ENC_OUTPUT_SIZE
      + 8
    )
    self.hand_set_processor = ProcessSetProcessor(hand_input_size)
    self.discard_set_processor = ProcessSetProcessor(hand_input_size)
    self.privileged_hand_set_processor = None
    self.privileged_deck_token_projector = None
    self.privileged_deck_encoder = None
    self.privileged_critic_fusion = None
    self._cached_privileged_critic_features = None

    if self.privileged_critic_enabled:
      privileged_zone_input_size = (
        CARD_METADATA_EMBED_SIZE
        + INDEX_ENC_OUTPUT_SIZE
        + PRIVILEGED_CRITIC_TOKEN_SCALAR_SIZE
      )
      self.privileged_hand_set_processor = ProcessSetProcessor(
        privileged_zone_input_size,
        hidden_size=PROCESS_SET_HIDDEN_SIZE,
        output_size=self.privileged_critic_embed_dim,
      )
      self.privileged_deck_token_projector = SingleUnitProjection(
        privileged_zone_input_size,
        hidden_size=PROCESS_SET_HIDDEN_SIZE,
        output_size=self.privileged_critic_embed_dim,
      )
      self.privileged_deck_encoder = MaskedTransformerDeckEncoder(
        self.privileged_critic_embed_dim,
        num_heads=self.privileged_critic_deck_heads,
        num_layers=self.privileged_critic_deck_layers,
        ff_size=self.privileged_critic_deck_ff_size,
      )

    ikz_input_size = (
      CARD_METADATA_EMBED_SIZE
      + INDEX_ENC_OUTPUT_SIZE
      + 2
    )
    self.ikz_set_processor = ProcessSetProcessor(ikz_input_size)

    board_input_size = (
      CARD_METADATA_EMBED_SIZE
      + INDEX_ENC_OUTPUT_SIZE
      + UNIT_EMBED_SIZE
      + 17
    )
    self.board_set_processor = ProcessSetProcessor(board_input_size)

    leader_input_size = (
      CARD_METADATA_EMBED_SIZE
      + UNIT_EMBED_SIZE
      + 12
    )
    gate_input_size = CARD_METADATA_EMBED_SIZE + 4

    self.leader_projector = SingleUnitProjection(leader_input_size)
    self.gate_projector = SingleUnitProjection(gate_input_size)

    context_input_size = (
      PHASE_ENC_OUTPUT_SIZE
      + ABILITY_PHASE_ENC_OUTPUT_SIZE
      + CARD_METADATA_EMBED_SIZE
      + 9
    )
    self.global_context_projector = SingleUnitProjection(context_input_size)
    self.global_counts_projector = SingleUnitProjection(9, output_size=UNIT_EMBED_SIZE)

    recent_action_step_input_size = (
      ACTION_HISTORY_PRIMARY_ENC_OUTPUT_SIZE
      + (ACTION_HISTORY_SUBACTION_ENC_OUTPUT_SIZE * 3)
      + 2
    )
    self.recent_action_step_projector = SingleUnitProjection(
      recent_action_step_input_size,
      hidden_size=64,
      output_size=ACTION_HISTORY_STEP_EMBED_SIZE,
    )
    self.self_recent_action_projector = SingleUnitProjection(
      RECENT_ACTION_HISTORY_LEN * ACTION_HISTORY_STEP_EMBED_SIZE,
      output_size=ACTION_HISTORY_PLAYER_EMBED_SIZE,
    )
    self.opp_recent_action_projector = SingleUnitProjection(
      RECENT_ACTION_HISTORY_LEN * ACTION_HISTORY_STEP_EMBED_SIZE,
      output_size=ACTION_HISTORY_PLAYER_EMBED_SIZE,
    )
    self.recent_action_history_projector = SingleUnitProjection(
      ACTION_HISTORY_PLAYER_EMBED_SIZE * 2,
      output_size=UNIT_EMBED_SIZE,
    )

    combat_context_input_size = (
      CARD_METADATA_EMBED_SIZE * 2
      + INDEX_ENC_OUTPUT_SIZE * 2
      + 13
    )
    self.combat_context_projector = SingleUnitProjection(
      combat_context_input_size,
      output_size=UNIT_EMBED_SIZE,
    )
    self.global_fusion_projector = SingleUnitProjection(
      UNIT_EMBED_SIZE * 4,
      output_size=UNIT_EMBED_SIZE,
    )

    self.zone_component_count = 14
    self.lstm_input_size = UNIT_EMBED_SIZE * (self.zone_component_count + 1)

    self.q_primary = nn.Linear(LSTM_HIDDEN_SIZE, UNIT_EMBED_SIZE)
    self.q_unit1 = nn.Linear(LSTM_HIDDEN_SIZE, UNIT_EMBED_SIZE)
    self.q_unit2 = nn.Linear(LSTM_HIDDEN_SIZE, UNIT_EMBED_SIZE)
    self.q_bins2 = nn.Linear(LSTM_HIDDEN_SIZE, MAX_INDEX_SIZE)
    self.q_bins3 = nn.Linear(LSTM_HIDDEN_SIZE, MAX_INDEX_SIZE)
    self.gate_1_embeder = nn.Embedding(PRIMARY_ACTION_COUNT, UNIT_EMBED_SIZE)
    self.gate_2_embeder = nn.Embedding(PRIMARY_ACTION_COUNT, UNIT_EMBED_SIZE)
    legal_action_candidate_input_size = (
      UNIT_EMBED_SIZE
      + (INDEX_ENC_OUTPUT_SIZE * 3)
      + (LEGAL_ACTION_ARG_KIND_EMBED_SIZE * 3)
      + (UNIT_EMBED_SIZE * 3)
      + 6
    )
    self.q_legal_action = nn.Linear(LSTM_HIDDEN_SIZE, UNIT_EMBED_SIZE)
    self.legal_action_candidate_projector = SingleUnitProjection(
      legal_action_candidate_input_size,
      hidden_size=PROCESS_SET_HIDDEN_SIZE,
      output_size=UNIT_EMBED_SIZE,
    )
    self.legal_action_candidate_bias = azk_pytorch.layer_init(
      nn.Linear(UNIT_EMBED_SIZE, 1),
      std=1,
    )

    if self.critic_head_type == CRITIC_HEAD_TYPE_SHARED_PRIMARY:
      self.critic_projector = None
      public_critic_feature_dim = UNIT_EMBED_SIZE
    else:
      self.critic_projector = nn.Sequential(
        azk_pytorch.layer_init(nn.Linear(LSTM_HIDDEN_SIZE, CRITIC_MLP_HIDDEN_SIZE)),
        nn.ReLU(),
        azk_pytorch.layer_init(nn.Linear(CRITIC_MLP_HIDDEN_SIZE, CRITIC_MLP_PROJECTION_SIZE)),
        nn.ReLU(),
      )
      public_critic_feature_dim = CRITIC_MLP_PROJECTION_SIZE

    value_feature_dim = public_critic_feature_dim
    if self.privileged_critic_enabled:
      self.privileged_critic_fusion = nn.Sequential(
        azk_pytorch.layer_init(
          nn.Linear(
            public_critic_feature_dim + (self.privileged_critic_embed_dim * 3),
            self.privileged_critic_fusion_hidden_size,
          )
        ),
        nn.ReLU(),
        azk_pytorch.layer_init(
          nn.Linear(
            self.privileged_critic_fusion_hidden_size,
            self.privileged_critic_fusion_projection_size,
          )
        ),
        nn.ReLU(),
      )
      value_feature_dim = self.privileged_critic_fusion_projection_size

    self.value_fn = azk_pytorch.layer_init(nn.Linear(value_feature_dim, 1), std=1)
    self.value_terminal_fn = None
    self.value_shaped_fn = None
    if self.split_value_heads_enabled:
      self.value_terminal_fn = azk_pytorch.layer_init(nn.Linear(value_feature_dim, 1), std=1)
      self.value_shaped_fn = azk_pytorch.layer_init(nn.Linear(value_feature_dim, 1), std=1)
    self.win_prob_projector = None
    self.win_prob_fn = None
    if self.win_prob_aux_enabled:
      self.win_prob_projector = nn.Sequential(
        azk_pytorch.layer_init(nn.Linear(LSTM_HIDDEN_SIZE, CRITIC_MLP_HIDDEN_SIZE)),
        nn.ReLU(),
        azk_pytorch.layer_init(nn.Linear(CRITIC_MLP_HIDDEN_SIZE, CRITIC_MLP_PROJECTION_SIZE)),
        nn.ReLU(),
      )
      self.win_prob_fn = azk_pytorch.layer_init(
        nn.Linear(CRITIC_MLP_PROJECTION_SIZE, 1),
        std=1,
      )

    emulated_spec = getattr(env, "emulated", None)
    if emulated_spec is None:
      raise AttributeError("env must expose emulated metadata for nativize")
    obs_dtype = emulated_spec.get("emulated_observation_dtype")
    if obs_dtype is None:
      raise AttributeError("env.emulated missing emulated_observation_dtype")
    self._obs_struct_dtype = _build_native_dtype_from_numpy(obs_dtype)
    self._cached_mask_observations = None

  def __policy_device(self) -> torch.device:
    sample_param = next(self.parameters(), None)
    if sample_param is not None:
      return sample_param.device
    return torch.device("cpu")

  def __tensorize_structured_observation(self, value, device: torch.device):
    if torch.is_tensor(value):
      return value.to(device=device)

    if isinstance(value, np.ndarray):
      tensor = torch.as_tensor(value, device=device)
      if tensor.dim() == 0:
        return tensor.reshape(1, 1)
      return tensor.unsqueeze(0)

    if isinstance(value, np.generic):
      return torch.as_tensor([[value.item()]], device=device)

    if isinstance(value, (bool, int, float)):
      return torch.as_tensor([[value]], device=device)

    if isinstance(value, dict):
      return {
        key: self.__tensorize_structured_observation(subvalue, device)
        for key, subvalue in value.items()
      }

    if isinstance(value, tuple):
      return tuple(self.__tensorize_structured_observation(subvalue, device) for subvalue in value)

    if isinstance(value, list):
      return [self.__tensorize_structured_observation(subvalue, device) for subvalue in value]

    raise TypeError(f"Unsupported structured observation value type: {type(value)}")

  def __detach_observation_tree(self, value):
    if torch.is_tensor(value):
      return value.detach()
    if isinstance(value, dict):
      return {
        key: self.__detach_observation_tree(subvalue)
        for key, subvalue in value.items()
      }
    if isinstance(value, tuple):
      return tuple(self.__detach_observation_tree(subvalue) for subvalue in value)
    if isinstance(value, list):
      return [self.__detach_observation_tree(subvalue) for subvalue in value]
    return value

  def __prepare_structured_observations(self, observations):
    if isinstance(observations, dict):
      structured_obs = self.__tensorize_structured_observation(
        observations,
        self.__policy_device(),
      )
      return structured_obs, False, structured_obs

    obs_tensor = observations if torch.is_tensor(observations) else torch.as_tensor(observations)
    squeeze_batch = obs_tensor.dim() == 1
    if squeeze_batch:
      obs_tensor = obs_tensor.unsqueeze(0)
    obs_tensor = obs_tensor.to(self.__policy_device())
    structured_obs = azk_pytorch.nativize_tensor(obs_tensor, self._obs_struct_dtype)
    return structured_obs, squeeze_batch, obs_tensor

  def __store_mask_observations(self, obs_tensor: torch.Tensor, state):
    detached = self.__detach_observation_tree(obs_tensor)
    if state is not None:
      try:
        state["_azk_mask_observations"] = detached
      except (TypeError, AttributeError):
        pass
    self._cached_mask_observations = detached

  def __store_privileged_critic_features(self, features: torch.Tensor | None, state):
    if state is not None:
      try:
        state["_azk_privileged_critic_features"] = features
      except (TypeError, AttributeError):
        pass
    self._cached_privileged_critic_features = features

  def __get_struct_field(self, container, *names):
    for name in names:
      if isinstance(container, dict):
        if name in container:
          return container[name]
        continue
      try:
        return container[name]
      except (KeyError, ValueError, TypeError, IndexError):
        continue
    raise KeyError(f"Missing expected field. Tried: {names}")

  def __normalize_zone_entries(self, zone_entries):
    if isinstance(zone_entries, dict):
      def _zone_key_sort_key(key):
        if isinstance(key, int):
          return (0, key)
        if isinstance(key, str) and key.isdigit():
          return (0, int(key))
        return (1, str(key))

      return [zone_entries[key] for key in sorted(zone_entries.keys(), key=_zone_key_sort_key)]
    return zone_entries

  def __stack_zone_field(self, slots, field_name):
    stacked = torch.stack([slot[field_name] for slot in slots], dim=1)
    if stacked.size(-1) == 1:
      stacked = stacked.squeeze(-1)
    return stacked

  def _squeeze_trailing_singleton(self, tensor: torch.Tensor):
    if tensor.dim() > 0 and tensor.size(-1) == 1:
      return tensor.squeeze(-1)
    return tensor

  def _card_index_and_mask(self, card_def_ids: torch.Tensor):
    card_def_ids = card_def_ids.long()
    valid_mask = card_def_ids >= 0
    idx = (card_def_ids + 1).clamp(0, self.static_vocab_size - 1)
    return idx, valid_mask

  def _lookup_static(self, idx: torch.Tensor):
    return {
      "present_mask": self.static_card_present_mask[idx],
      "card_type": self.static_card_type[idx],
      "element": self.static_element[idx],
      "base_ikz_cost": self.static_base_ikz_cost[idx],
      "base_attack": self.static_base_attack[idx],
      "base_health": self.static_base_health[idx],
      "base_gate_points": self.static_base_gate_points[idx],
      "has_ability": self.static_has_ability[idx],
      "ability_timing": self.static_ability_timing[idx],
      "ability_optional": self.static_ability_optional[idx],
      "keyword_multi_hot": self.static_keyword_multi_hot[idx],
      "name_embedding": self.static_name_embeddings[idx],
      "effect_embedding": self.static_effect_embeddings[idx],
      "subtype_pooled_embedding": self.static_subtype_pooled_embeddings[idx],
    }

  def _index_embedding(self, zone_indices: torch.Tensor):
    return self.index_encoder(zone_indices.long().clamp(0, MAX_INDEX_SIZE - 1))

  def _encode_card_metadata_from_index(self, idx: torch.Tensor, valid_mask: torch.Tensor | None = None):
    static = self._lookup_static(idx)

    present_mask = static["present_mask"] > 0.5
    if valid_mask is None:
      valid_mask = present_mask
    else:
      valid_mask = valid_mask.to(dtype=torch.bool) & present_mask

    card_type_emb = self.card_type_encoder(static["card_type"])
    element_emb = self.element_encoder(static["element"])
    ability_timing_emb = self.ability_timing_encoder(static["ability_timing"])
    name_emb = self.name_text_encoder(static["name_embedding"])
    effect_emb = self.effect_text_encoder(static["effect_embedding"])
    subtype_emb = self.subtype_text_encoder(static["subtype_pooled_embedding"])
    keyword_emb = self.keyword_feature_encoder(static["keyword_multi_hot"])

    scalar = torch.stack(
      [
        static["present_mask"],
        static["base_ikz_cost"],
        static["base_attack"],
        static["base_health"],
        static["base_gate_points"],
        static["has_ability"],
        static["ability_optional"],
      ],
      dim=-1,
    )
    scalar = self.scalar_normalizer("card_metadata_scalar", scalar, mask=valid_mask)

    metadata_input = torch.cat(
      [
        name_emb,
        effect_emb,
        subtype_emb,
        keyword_emb,
        card_type_emb,
        element_emb,
        ability_timing_emb,
        scalar,
      ],
      dim=-1,
    )
    metadata_emb = self.card_metadata_projector(metadata_input)

    if valid_mask.dim() < metadata_emb.dim():
      valid_mask = valid_mask.unsqueeze(-1)
    return metadata_emb * valid_mask.to(dtype=metadata_emb.dtype)

  def _encode_weapons(self, weapon_slots, weapon_count):
    weapon_slots = self.__normalize_zone_entries(weapon_slots)
    card_def_ids = self.__stack_zone_field(weapon_slots, "card_def_id")
    idx, valid_mask = self._card_index_and_mask(card_def_ids)
    static = self._lookup_static(idx)

    card_emb = self._encode_card_metadata_from_index(idx, valid_mask=valid_mask)

    cur_atk = self.__stack_zone_field(weapon_slots, "cur_atk").float()
    scalar = torch.stack(
      [
        cur_atk,
        static["base_attack"],
        static["base_ikz_cost"],
        static["has_ability"],
        static["ability_optional"],
      ],
      dim=-1,
    )

    max_slots = scalar.size(1)
    slot_indices = torch.arange(max_slots, device=scalar.device).unsqueeze(0)
    count_mask = slot_indices < weapon_count.long().view(-1, 1)
    mask = valid_mask & count_mask
    scalar = self.scalar_normalizer("weapon_scalar", scalar, mask=mask)

    weapon_input = torch.cat([card_emb, scalar], dim=-1)
    _, pooled = self.weapon_set_processor(weapon_input, mask=mask)
    return pooled

  def _encode_hand_or_discard(self, slots, *, key_prefix: str, processor: ProcessSetProcessor):
    slots = self.__normalize_zone_entries(slots)
    card_def_ids = self.__stack_zone_field(slots, "card_def_id")
    zone_indices = self.__stack_zone_field(slots, "zone_index")

    idx, mask = self._card_index_and_mask(card_def_ids)
    static = self._lookup_static(idx)

    card_emb = self._encode_card_metadata_from_index(idx, valid_mask=mask)
    zone_emb = self._index_embedding(zone_indices)

    scalar = torch.stack(
      [
        static["base_ikz_cost"],
        static["base_attack"],
        static["base_health"],
        static["base_gate_points"],
        static["present_mask"],
        static["element"].float(),
        static["has_ability"],
        static["ability_optional"],
      ],
      dim=-1,
    )
    scalar = self.scalar_normalizer(f"{key_prefix}_scalar", scalar, mask=mask)

    zone_input = torch.cat([card_emb, zone_emb, scalar], dim=-1)
    set_embeddings, pooled = processor(zone_input, mask=mask)
    return set_embeddings, pooled

  def _encode_ikz_area(self, slots):
    slots = self.__normalize_zone_entries(slots)
    card_def_ids = self.__stack_zone_field(slots, "card_def_id")
    zone_indices = self.__stack_zone_field(slots, "zone_index")
    tapped = self.__stack_zone_field(slots, "tapped").float()
    cooldown = self.__stack_zone_field(slots, "cooldown").float()

    idx, mask = self._card_index_and_mask(card_def_ids)
    card_emb = self._encode_card_metadata_from_index(idx, valid_mask=mask)
    zone_emb = self._index_embedding(zone_indices)

    scalar = torch.stack([tapped, cooldown], dim=-1)
    scalar = self.scalar_normalizer("ikz_scalar", scalar, mask=mask)

    zone_input = torch.cat([card_emb, zone_emb, scalar], dim=-1)
    set_embeddings, pooled = self.ikz_set_processor(zone_input, mask=mask)
    return set_embeddings, pooled

  def _encode_critic_privileged_zone_tokens(self, slots, *, key_prefix: str):
    slots = self.__normalize_zone_entries(slots)
    card_def_ids = self.__stack_zone_field(slots, "card_def_id")
    zone_indices = self.__stack_zone_field(slots, "zone_index")

    idx, mask = self._card_index_and_mask(card_def_ids)
    card_emb = self._encode_card_metadata_from_index(idx, valid_mask=mask)
    zone_emb = self._index_embedding(zone_indices)

    normalized_position = zone_indices.float() / float(max(MAX_INDEX_SIZE - 1, 1))
    scalar = torch.stack([normalized_position, mask.float()], dim=-1)
    scalar = self.scalar_normalizer(f"{key_prefix}_privileged_scalar", scalar, mask=mask)
    zone_input = torch.cat([card_emb, zone_emb, scalar], dim=-1)
    return zone_input, mask

  def _encode_critic_privileged_hand(self, slots):
    zone_input, mask = self._encode_critic_privileged_zone_tokens(
      slots,
      key_prefix="critic_privileged_hand",
    )
    _, pooled = self.privileged_hand_set_processor(zone_input, mask=mask)
    return pooled

  def _encode_critic_privileged_deck(self, slots, *, key_prefix: str):
    zone_input, mask = self._encode_critic_privileged_zone_tokens(slots, key_prefix=key_prefix)
    batch_size, slot_count, feature_dim = zone_input.shape
    tokens = self.privileged_deck_token_projector(
      zone_input.reshape(batch_size * slot_count, feature_dim)
    ).reshape(batch_size, slot_count, self.privileged_critic_embed_dim)
    tokens = tokens * mask.unsqueeze(-1).to(dtype=tokens.dtype)
    return self.privileged_deck_encoder(tokens, mask)

  def _encode_board_zone(self, slots, *, key_prefix: str):
    slots = self.__normalize_zone_entries(slots)
    card_def_ids = self.__stack_zone_field(slots, "card_def_id")
    zone_indices = self.__stack_zone_field(slots, "zone_index")
    tapped = self.__stack_zone_field(slots, "tapped").float()
    cooldown = self.__stack_zone_field(slots, "cooldown").float()
    has_cur_stats = self.__stack_zone_field(slots, "has_cur_stats").float()
    cur_atk = self.__stack_zone_field(slots, "cur_atk").float()
    cur_hp = self.__stack_zone_field(slots, "cur_hp").float()
    has_charge = self.__stack_zone_field(slots, "has_charge").float()
    has_defender = self.__stack_zone_field(slots, "has_defender").float()
    has_infiltrate = self.__stack_zone_field(slots, "has_infiltrate").float()
    is_frozen = self.__stack_zone_field(slots, "is_frozen").float()
    is_shocked = self.__stack_zone_field(slots, "is_shocked").float()
    is_effect_immune = self.__stack_zone_field(slots, "is_effect_immune").float()
    weapon_count = self.__stack_zone_field(slots, "weapon_count")

    idx, mask = self._card_index_and_mask(card_def_ids)
    static = self._lookup_static(idx)

    card_emb = self._encode_card_metadata_from_index(idx, valid_mask=mask)
    zone_emb = self._index_embedding(zone_indices)

    weapon_embeddings = []
    for slot in slots:
      weapon_embeddings.append(
        self._encode_weapons(slot["weapons"], slot["weapon_count"])
      )
    weapon_emb = torch.stack(weapon_embeddings, dim=1)

    scalar = torch.stack(
      [
        tapped,
        cooldown,
        has_cur_stats,
        cur_atk,
        cur_hp,
        has_charge,
        has_defender,
        has_infiltrate,
        is_frozen,
        is_shocked,
        is_effect_immune,
        static["base_ikz_cost"],
        static["base_attack"],
        static["base_health"],
        static["base_gate_points"],
        static["has_ability"],
        static["ability_optional"],
      ],
      dim=-1,
    )
    scalar = self.scalar_normalizer(f"{key_prefix}_scalar", scalar, mask=mask)

    zone_input = torch.cat([card_emb, zone_emb, weapon_emb, scalar], dim=-1)
    set_embeddings, pooled = self.board_set_processor(zone_input, mask=mask)
    return set_embeddings, pooled

  def _encode_leader(self, leader_obs, *, key_prefix: str):
    card_def_ids = self._squeeze_trailing_singleton(leader_obs["card_def_id"]).long()
    idx, valid_mask = self._card_index_and_mask(card_def_ids)
    static = self._lookup_static(idx)

    weapon_count = self._squeeze_trailing_singleton(leader_obs["weapon_count"])
    weapon_emb = self._encode_weapons(leader_obs["weapons"], weapon_count)

    card_emb = self._encode_card_metadata_from_index(idx, valid_mask=valid_mask)

    scalar = torch.stack(
      [
        self._squeeze_trailing_singleton(leader_obs["cur_atk"]).float(),
        self._squeeze_trailing_singleton(leader_obs["cur_hp"]).float(),
        self._squeeze_trailing_singleton(leader_obs["tapped"]).float(),
        self._squeeze_trailing_singleton(leader_obs["cooldown"]).float(),
        self._squeeze_trailing_singleton(leader_obs["has_charge"]).float(),
        self._squeeze_trailing_singleton(leader_obs["has_defender"]).float(),
        self._squeeze_trailing_singleton(leader_obs["has_infiltrate"]).float(),
        static["base_attack"],
        static["base_health"],
        static["base_ikz_cost"],
        static["base_gate_points"],
        static["has_ability"],
      ],
      dim=-1,
    )
    scalar = self.scalar_normalizer(f"{key_prefix}_leader_scalar", scalar)

    leader_input = torch.cat([card_emb, weapon_emb, scalar], dim=-1)
    return self.leader_projector(leader_input)

  def _encode_gate(self, gate_obs, *, key_prefix: str):
    card_def_ids = self._squeeze_trailing_singleton(gate_obs["card_def_id"]).long()
    idx, valid_mask = self._card_index_and_mask(card_def_ids)
    static = self._lookup_static(idx)

    card_emb = self._encode_card_metadata_from_index(idx, valid_mask=valid_mask)

    scalar = torch.stack(
      [
        self._squeeze_trailing_singleton(gate_obs["tapped"]).float(),
        self._squeeze_trailing_singleton(gate_obs["cooldown"]).float(),
        static["has_ability"],
        static["ability_optional"],
      ],
      dim=-1,
    )
    scalar = self.scalar_normalizer(f"{key_prefix}_gate_scalar", scalar)

    gate_input = torch.cat([card_emb, scalar], dim=-1)
    return self.gate_projector(gate_input)

  def _encode_global_context(self, structured_obs):
    ability = self.__get_struct_field(structured_obs, "ability_context")
    phase = self._squeeze_trailing_singleton(
      self.__get_struct_field(structured_obs, "phase")
    ).long().clamp(0, GAME_PHASE_COUNT - 1)
    ability_phase = self._squeeze_trailing_singleton(
      ability["phase"]
    ).long().clamp(0, ABILITY_PHASE_COUNT - 1)

    source_card = self._squeeze_trailing_singleton(ability["source_card_def_id"]).long()
    source_idx, source_valid = self._card_index_and_mask(source_card)

    phase_emb = self.game_phase_encoder(phase)
    ability_phase_emb = self.ability_phase_encoder(ability_phase)
    source_emb = self._encode_card_metadata_from_index(source_idx, valid_mask=source_valid)

    action_mask = self.__get_struct_field(structured_obs, "action_mask")
    is_active = (self._squeeze_trailing_singleton(action_mask["legal_action_count"]) > 0).float()

    scalar = torch.stack(
      [
        self._squeeze_trailing_singleton(ability["pending_confirmation_count"]).float(),
        self._squeeze_trailing_singleton(ability["has_source_card_def_id"]).float(),
        self._squeeze_trailing_singleton(ability["cost_target_type"]).float(),
        self._squeeze_trailing_singleton(ability["effect_target_type"]).float(),
        self._squeeze_trailing_singleton(ability["selection_count"]).float(),
        self._squeeze_trailing_singleton(ability["selection_picked"]).float(),
        self._squeeze_trailing_singleton(ability["selection_pick_max"]).float(),
        self._squeeze_trailing_singleton(ability["active_player_index"]).float(),
        is_active,
      ],
      dim=-1,
    )
    scalar = self.scalar_normalizer("global_context_scalar", scalar)

    context_input = torch.cat([phase_emb, ability_phase_emb, source_emb, scalar], dim=-1)
    return self.global_context_projector(context_input)

  def _encode_recent_action_sequence(self, recent_actions, *, key_prefix: str):
    recent_actions = self.__normalize_zone_entries(recent_actions)
    valid = self.__stack_zone_field(recent_actions, "valid").to(dtype=torch.bool)
    primary = self.__stack_zone_field(recent_actions, "primary").long()
    sub1 = self.__stack_zone_field(recent_actions, "sub1").long()
    sub2 = self.__stack_zone_field(recent_actions, "sub2").long()
    sub3 = self.__stack_zone_field(recent_actions, "sub3").long()
    was_noop = self.__stack_zone_field(recent_actions, "was_noop").float()

    zero_primary = torch.zeros_like(primary)
    zero_sub = torch.zeros_like(sub1)
    primary_idx = torch.where(
      valid,
      primary.clamp(0, PRIMARY_ACTION_COUNT - 1) + 1,
      zero_primary,
    )
    sub1_idx = torch.where(valid, sub1.clamp(0, MAX_INDEX_SIZE - 1) + 1, zero_sub)
    sub2_idx = torch.where(valid, sub2.clamp(0, MAX_INDEX_SIZE - 1) + 1, zero_sub)
    sub3_idx = torch.where(valid, sub3.clamp(0, MAX_INDEX_SIZE - 1) + 1, zero_sub)

    primary_emb = self.action_history_primary_encoder(primary_idx)
    sub1_emb = self.action_history_subaction_encoder(sub1_idx)
    sub2_emb = self.action_history_subaction_encoder(sub2_idx)
    sub3_emb = self.action_history_subaction_encoder(sub3_idx)

    scalar = torch.stack([valid.float(), was_noop], dim=-1)
    scalar = self.scalar_normalizer(f"{key_prefix}_recent_action_scalar", scalar, mask=valid)
    step_input = torch.cat([primary_emb, sub1_emb, sub2_emb, sub3_emb, scalar], dim=-1)

    batch_size, step_count, _ = step_input.shape
    projected_steps = self.recent_action_step_projector(
      step_input.reshape(batch_size * step_count, -1)
    ).reshape(batch_size, step_count, ACTION_HISTORY_STEP_EMBED_SIZE)
    projected_steps = projected_steps * valid.unsqueeze(-1).to(dtype=projected_steps.dtype)
    flattened = projected_steps.reshape(batch_size, step_count * ACTION_HISTORY_STEP_EMBED_SIZE)

    if key_prefix == "self":
      return self.self_recent_action_projector(flattened)
    if key_prefix == "opp":
      return self.opp_recent_action_projector(flattened)
    raise ValueError(f"Unsupported recent action history key_prefix: {key_prefix}")

  def _encode_recent_action_history(self, structured_obs):
    phase = self._squeeze_trailing_singleton(
      self.__get_struct_field(structured_obs, "phase")
    )
    if phase.dim() == 0:
      phase = phase.unsqueeze(0)

    try:
      self_recent_actions = self.__get_struct_field(structured_obs, "self_recent_actions")
      opp_recent_actions = self.__get_struct_field(structured_obs, "opp_recent_actions")
    except KeyError:
      return torch.zeros(
        (phase.shape[0], UNIT_EMBED_SIZE),
        device=phase.device,
        dtype=torch.float32,
      )

    self_history_vec = self._encode_recent_action_sequence(
      self_recent_actions,
      key_prefix="self",
    )
    opp_history_vec = self._encode_recent_action_sequence(
      opp_recent_actions,
      key_prefix="opp",
    )
    return self.recent_action_history_projector(
      torch.cat([self_history_vec, opp_history_vec], dim=-1)
    )

  def _encode_combat_context(self, structured_obs):
    phase = self._squeeze_trailing_singleton(
      self.__get_struct_field(structured_obs, "phase")
    )
    if phase.dim() == 0:
      phase = phase.unsqueeze(0)

    try:
      combat_context = self.__get_struct_field(structured_obs, "combat_context")
    except KeyError:
      return torch.zeros(
        (phase.shape[0], UNIT_EMBED_SIZE),
        device=phase.device,
        dtype=torch.float32,
      )

    attacker_card = self._squeeze_trailing_singleton(
      combat_context["attacker_card_def_id"]
    ).long()
    target_card = self._squeeze_trailing_singleton(
      combat_context["target_card_def_id"]
    ).long()
    attacker_idx, attacker_valid = self._card_index_and_mask(attacker_card)
    target_idx, target_valid = self._card_index_and_mask(target_card)
    attacker_card_emb = self._encode_card_metadata_from_index(
      attacker_idx, valid_mask=attacker_valid
    )
    target_card_emb = self._encode_card_metadata_from_index(
      target_idx, valid_mask=target_valid
    )

    attacker_slot_idx = self._squeeze_trailing_singleton(
      combat_context["attacker_slot_index"]
    )
    target_slot_idx = self._squeeze_trailing_singleton(
      combat_context["target_slot_index"]
    )
    attacker_slot_emb = self._index_embedding(attacker_slot_idx)
    target_slot_emb = self._index_embedding(target_slot_idx)
    attacker_slot_emb = attacker_slot_emb * attacker_valid.unsqueeze(-1).to(dtype=attacker_slot_emb.dtype)
    target_slot_emb = target_slot_emb * target_valid.unsqueeze(-1).to(dtype=target_slot_emb.dtype)

    combat_active = self._squeeze_trailing_singleton(
      combat_context["combat_active"]
    ).to(dtype=torch.bool)
    scalar = torch.stack(
      [
        combat_active.float(),
        self._squeeze_trailing_singleton(combat_context["response_window_active"]).float(),
        self._squeeze_trailing_singleton(combat_context["defender_intercepted"]).float(),
        self._squeeze_trailing_singleton(combat_context["attacker_is_self"]).float(),
        self._squeeze_trailing_singleton(combat_context["attacker_is_leader"]).float(),
        self._squeeze_trailing_singleton(combat_context["attacker_is_garden"]).float(),
        self._squeeze_trailing_singleton(combat_context["attacker_is_alley"]).float(),
        attacker_slot_idx.float(),
        self._squeeze_trailing_singleton(combat_context["target_is_self"]).float(),
        self._squeeze_trailing_singleton(combat_context["target_is_leader"]).float(),
        self._squeeze_trailing_singleton(combat_context["target_is_garden"]).float(),
        self._squeeze_trailing_singleton(combat_context["target_is_alley"]).float(),
        target_slot_idx.float(),
      ],
      dim=-1,
    )
    scalar = self.scalar_normalizer("combat_context_scalar", scalar, mask=combat_active)
    combat_input = torch.cat(
      [
        attacker_card_emb,
        target_card_emb,
        attacker_slot_emb,
        target_slot_emb,
        scalar,
      ],
      dim=-1,
    )
    return self.combat_context_projector(combat_input)

  def _encode_critic_privileged_features(self, structured_obs):
    if not self.privileged_critic_enabled:
      return None

    phase = self._squeeze_trailing_singleton(
      self.__get_struct_field(structured_obs, "phase")
    )
    if phase.dim() == 0:
      phase = phase.unsqueeze(0)

    try:
      critic_privileged = self.__get_struct_field(structured_obs, "critic_privileged")
    except KeyError:
      return torch.zeros(
        (phase.shape[0], self.privileged_critic_embed_dim * 3),
        device=phase.device,
        dtype=torch.float32,
      )

    opponent_hand_vec = self._encode_critic_privileged_hand(critic_privileged["opponent_hand"])
    self_deck_vec = self._encode_critic_privileged_deck(
      critic_privileged["self_deck"],
      key_prefix="critic_privileged_self_deck",
    )
    opponent_deck_vec = self._encode_critic_privileged_deck(
      critic_privileged["opponent_deck"],
      key_prefix="critic_privileged_opponent_deck",
    )
    return torch.cat([opponent_hand_vec, self_deck_vec, opponent_deck_vec], dim=-1)

  def _public_critic_features(self, flat_hidden: torch.Tensor, projected_hidden: torch.Tensor):
    if self.critic_head_type == CRITIC_HEAD_TYPE_SHARED_PRIMARY:
      return projected_hidden
    return self.critic_projector(flat_hidden)

  def _shared_critic_projection(self, flat_hidden: torch.Tensor) -> torch.Tensor:
    if self.actor_head_type == ACTOR_HEAD_TYPE_LEGAL_ACTION_SCORER:
      return self.q_legal_action(flat_hidden)
    return self.q_primary(flat_hidden)

  def _value_features(self, flat_hidden: torch.Tensor, projected_hidden: torch.Tensor, state=None):
    public_critic_features = self._public_critic_features(flat_hidden, projected_hidden)
    if not self.privileged_critic_enabled:
      return public_critic_features

    privileged_features = None
    if state is not None:
      privileged_features = state.get("_azk_privileged_critic_features")
    if privileged_features is None:
      privileged_features = self._cached_privileged_critic_features

    if not torch.is_tensor(privileged_features):
      raise ValueError("Privileged critic enabled but privileged features were not prepared")

    if self.privileged_critic_feature_scale != 1.0:
      privileged_features = privileged_features * self.privileged_critic_feature_scale

    value_features = self.privileged_critic_fusion(
      torch.cat([public_critic_features, privileged_features], dim=-1)
    )
    return value_features

  def _win_prob_features(self, flat_hidden: torch.Tensor):
    if self.win_prob_projector is None:
      raise ValueError("win_prob_features requested but win_prob_projector is not initialized")
    return self.win_prob_projector(flat_hidden)

  def reset_critic_head(self) -> None:
    if self.critic_projector is not None:
      for module in self.critic_projector.modules():
        if isinstance(module, nn.Linear):
          azk_pytorch.layer_init(module)
    if self.privileged_hand_set_processor is not None:
      for module in self.privileged_hand_set_processor.modules():
        if isinstance(module, nn.Linear):
          azk_pytorch.layer_init(module)
    if self.privileged_deck_token_projector is not None:
      for module in self.privileged_deck_token_projector.modules():
        if isinstance(module, nn.Linear):
          azk_pytorch.layer_init(module)
    if self.privileged_deck_encoder is not None:
      self.privileged_deck_encoder.reset_parameters()
    if self.privileged_critic_fusion is not None:
      for module in self.privileged_critic_fusion.modules():
        if isinstance(module, nn.Linear):
          azk_pytorch.layer_init(module)
    azk_pytorch.layer_init(self.value_fn, std=1)
    if self.value_terminal_fn is not None:
      azk_pytorch.layer_init(self.value_terminal_fn, std=1)
    if self.value_shaped_fn is not None:
      azk_pytorch.layer_init(self.value_shaped_fn, std=1)
    if self.win_prob_projector is not None:
      for module in self.win_prob_projector.modules():
        if isinstance(module, nn.Linear):
          azk_pytorch.layer_init(module)
    if self.win_prob_fn is not None:
      azk_pytorch.layer_init(self.win_prob_fn, std=1)

  def forward(self, x, state=None):
    target_vector, action_context = self.encode_observations(x, state=state)
    actions, value = self.decode_actions(target_vector, action_context=action_context, state=state)
    return actions, value

  def forward_train(self, x, state=None):
    return self.forward(x, state)

  def encode_observations(self, observations, state=None):
    structured_obs, squeeze_batch, obs_tensor = self.__prepare_structured_observations(observations)
    self.__store_mask_observations(obs_tensor, state)
    self.__store_privileged_critic_features(
      self._encode_critic_privileged_features(structured_obs),
      state,
    )

    player = self.__get_struct_field(structured_obs, "player", "my_observation_data")
    opponent = self.__get_struct_field(structured_obs, "opponent", "opponent_observation_data")

    hand_matrix, hand_vec = self._encode_hand_or_discard(player["hand"], key_prefix="hand", processor=self.hand_set_processor)
    player_discard_matrix, player_discard_vec = self._encode_hand_or_discard(player["discard"], key_prefix="player_discard", processor=self.discard_set_processor)
    opponent_discard_matrix, opponent_discard_vec = self._encode_hand_or_discard(opponent["discard"], key_prefix="opponent_discard", processor=self.discard_set_processor)

    player_garden_matrix, player_garden_vec = self._encode_board_zone(player["garden"], key_prefix="player_garden")
    player_alley_matrix, player_alley_vec = self._encode_board_zone(player["alley"], key_prefix="player_alley")
    player_selection_matrix, player_selection_vec = self._encode_board_zone(player["selection"], key_prefix="player_selection")
    opponent_garden_matrix, opponent_garden_vec = self._encode_board_zone(opponent["garden"], key_prefix="opponent_garden")
    opponent_alley_matrix, opponent_alley_vec = self._encode_board_zone(opponent["alley"], key_prefix="opponent_alley")

    player_ikz_matrix, player_ikz_vec = self._encode_ikz_area(player["ikz_area"])
    opponent_ikz_matrix, opponent_ikz_vec = self._encode_ikz_area(opponent["ikz_area"])

    player_leader_vec = self._encode_leader(player["leader"], key_prefix="player")
    opponent_leader_vec = self._encode_leader(opponent["leader"], key_prefix="opponent")
    player_gate_vec = self._encode_gate(player["gate"], key_prefix="player")
    opponent_gate_vec = self._encode_gate(opponent["gate"], key_prefix="opponent")

    global_counts = torch.stack(
      [
        self._squeeze_trailing_singleton(player["hand_count"]).float(),
        self._squeeze_trailing_singleton(player["deck_count"]).float(),
        self._squeeze_trailing_singleton(player["ikz_pile_count"]).float(),
        self._squeeze_trailing_singleton(player["selection_count"]).float(),
        self._squeeze_trailing_singleton(player["has_ikz_token"]).float(),
        self._squeeze_trailing_singleton(opponent["hand_count"]).float(),
        self._squeeze_trailing_singleton(opponent["deck_count"]).float(),
        self._squeeze_trailing_singleton(opponent["ikz_pile_count"]).float(),
        self._squeeze_trailing_singleton(opponent["has_ikz_token"]).float(),
      ],
      dim=-1,
    )
    global_counts = self.scalar_normalizer("global_counts", global_counts)
    global_counts_vec = self.global_counts_projector(global_counts)
    global_context_vec = self._encode_global_context(structured_obs)
    combat_context_vec = self._encode_combat_context(structured_obs)
    recent_action_history_vec = self._encode_recent_action_history(structured_obs)
    global_vec = self.global_fusion_projector(
      torch.cat(
        [
          global_counts_vec,
          global_context_vec,
          combat_context_vec,
          recent_action_history_vec,
        ],
        dim=-1,
      )
    )

    target_vector = torch.cat(
      [
        hand_vec,
        player_discard_vec,
        opponent_discard_vec,
        player_garden_vec,
        player_alley_vec,
        player_selection_vec,
        opponent_garden_vec,
        opponent_alley_vec,
        player_ikz_vec,
        opponent_ikz_vec,
        player_leader_vec,
        player_gate_vec,
        opponent_leader_vec,
        opponent_gate_vec,
        global_vec,
      ],
      dim=-1,
    )

    target_matrix = torch.cat(
      [
        hand_matrix,
        player_discard_matrix,
        opponent_discard_matrix,
        player_garden_matrix,
        player_alley_matrix,
        player_selection_matrix,
        opponent_garden_matrix,
        opponent_alley_matrix,
        player_ikz_matrix,
        opponent_ikz_matrix,
        player_leader_vec.unsqueeze(-2),
        player_gate_vec.unsqueeze(-2),
        opponent_leader_vec.unsqueeze(-2),
        opponent_gate_vec.unsqueeze(-2),
      ],
      dim=-2,
    )

    action_context = {
      "legacy_target_matrix": target_matrix,
      "hand_matrix": hand_matrix,
      "player_garden_matrix": player_garden_matrix,
      "player_alley_matrix": player_alley_matrix,
      "player_selection_matrix": player_selection_matrix,
      "opponent_garden_matrix": opponent_garden_matrix,
      "opponent_alley_matrix": opponent_alley_matrix,
      "player_leader_vec": player_leader_vec.unsqueeze(-2),
      "opponent_leader_vec": opponent_leader_vec.unsqueeze(-2),
    }

    if squeeze_batch:
      target_vector = target_vector.squeeze(0)
    return target_vector, action_context

  def build_primary_action_mask_tensor(self, observations):
    structured_obs, squeeze_batch, _ = self.__prepare_structured_observations(observations)
    action_mask = self.__get_struct_field(structured_obs, "action_mask")
    primary_mask = action_mask["primary_action_mask"].to(dtype=torch.bool)
    if squeeze_batch and primary_mask.dim() > 1 and primary_mask.size(0) == 1:
      return primary_mask.squeeze(0)
    return primary_mask

  def _prepare_action_context(self, action_context, *, device: torch.device):
    if action_context is None:
      raise ValueError("decode_actions requires action_context")
    if torch.is_tensor(action_context):
      target_matrix = action_context.to(device=device)
      if target_matrix.dim() == 2:
        target_matrix = target_matrix.unsqueeze(0)
      elif target_matrix.dim() == 1:
        target_matrix = target_matrix.unsqueeze(0).unsqueeze(0)
      return {"legacy_target_matrix": target_matrix}
    if not isinstance(action_context, dict):
      raise TypeError(f"Unsupported action_context type: {type(action_context)!r}")

    prepared = {}
    for key, value in action_context.items():
      if not torch.is_tensor(value):
        continue
      tensor = value.to(device=device)
      if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
      elif tensor.dim() == 2 and tensor.shape[-1] == UNIT_EMBED_SIZE:
        tensor = tensor.unsqueeze(0)
      prepared[key] = tensor
    return prepared

  def _lookup_action_context_tensor(self, action_context: dict, key: str) -> torch.Tensor:
    tensor = action_context.get(key)
    if not torch.is_tensor(tensor):
      raise KeyError(f"Missing action_context tensor '{key}'")
    return tensor

  def _gather_zone_rows(self, matrix: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if matrix.dim() != 3:
      raise ValueError(f"Zone matrix must be rank 3, got shape {tuple(matrix.shape)}")
    clamped = indices.clamp(0, matrix.size(1) - 1)
    gather_index = clamped.unsqueeze(-1).expand(-1, -1, matrix.size(-1))
    return matrix.gather(1, gather_index)

  def _gather_garden_or_leader_refs(
    self,
    garden_matrix: torch.Tensor,
    leader_matrix: torch.Tensor,
    indices: torch.Tensor,
  ) -> torch.Tensor:
    refs = torch.zeros(
      (*indices.shape, garden_matrix.size(-1)),
      device=garden_matrix.device,
      dtype=garden_matrix.dtype,
    )
    garden_mask = indices < GARDEN_SIZE
    leader_mask = indices == GARDEN_SIZE
    if garden_mask.any():
      garden_refs = self._gather_zone_rows(garden_matrix, indices.clamp(0, GARDEN_SIZE - 1))
      refs = torch.where(garden_mask.unsqueeze(-1), garden_refs, refs)
    if leader_mask.any():
      leader_refs = leader_matrix.expand(indices.shape[0], indices.shape[1], -1)
      refs = torch.where(leader_mask.unsqueeze(-1), leader_refs, refs)
    return refs

  def _gather_opponent_defender_refs(
    self,
    opponent_garden_matrix: torch.Tensor,
    opponent_leader_matrix: torch.Tensor,
    opponent_alley_matrix: torch.Tensor,
    indices: torch.Tensor,
  ) -> torch.Tensor:
    refs = torch.zeros(
      (*indices.shape, opponent_garden_matrix.size(-1)),
      device=opponent_garden_matrix.device,
      dtype=opponent_garden_matrix.dtype,
    )
    garden_mask = indices < GARDEN_SIZE
    leader_mask = indices == GARDEN_SIZE
    alley_mask = (indices > GARDEN_SIZE) & (indices <= (GARDEN_SIZE + ALLEY_SIZE))
    if garden_mask.any():
      garden_refs = self._gather_zone_rows(
        opponent_garden_matrix,
        indices.clamp(0, GARDEN_SIZE - 1),
      )
      refs = torch.where(garden_mask.unsqueeze(-1), garden_refs, refs)
    if leader_mask.any():
      leader_refs = opponent_leader_matrix.expand(indices.shape[0], indices.shape[1], -1)
      refs = torch.where(leader_mask.unsqueeze(-1), leader_refs, refs)
    if alley_mask.any():
      alley_index = (indices - (GARDEN_SIZE + 1)).clamp(0, ALLEY_SIZE - 1)
      alley_refs = self._gather_zone_rows(opponent_alley_matrix, alley_index)
      refs = torch.where(alley_mask.unsqueeze(-1), alley_refs, refs)
    return refs

  def _legal_action_arg_kinds(self, primary: torch.Tensor) -> torch.Tensor:
    sub1_kind = torch.full_like(primary, LEGAL_ACTION_ARG_KIND_UNUSED)
    sub2_kind = torch.full_like(primary, LEGAL_ACTION_ARG_KIND_UNUSED)
    sub3_kind = torch.full_like(primary, LEGAL_ACTION_ARG_KIND_UNUSED)

    hand_primary = (
      (primary == ACT_PLAY_ENTITY_TO_GARDEN)
      | (primary == ACT_PLAY_ENTITY_TO_ALLEY)
      | (primary == ACT_ATTACH_WEAPON_FROM_HAND)
      | (primary == ACT_PLAY_SPELL_FROM_HAND)
    )
    sub1_kind = torch.where(
      hand_primary,
      torch.full_like(sub1_kind, LEGAL_ACTION_ARG_KIND_HAND),
      sub1_kind,
    )
    sub1_kind = torch.where(
      primary == ACT_GATE_PORTAL,
      torch.full_like(sub1_kind, LEGAL_ACTION_ARG_KIND_SELF_ALLEY),
      sub1_kind,
    )
    sub1_kind = torch.where(
      primary == ACT_ACTIVATE_ALLEY_ABILITY,
      torch.full_like(sub1_kind, LEGAL_ACTION_ARG_KIND_ABILITY_INDEX),
      sub1_kind,
    )
    sub1_kind = torch.where(
      (primary == ACT_ATTACK) | (primary == ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY),
      torch.full_like(sub1_kind, LEGAL_ACTION_ARG_KIND_SELF_GARDEN_OR_LEADER),
      sub1_kind,
    )
    sub1_kind = torch.where(
      primary == ACT_DECLARE_DEFENDER,
      torch.full_like(sub1_kind, LEGAL_ACTION_ARG_KIND_SELF_GARDEN),
      sub1_kind,
    )
    selection_primary = (
      (primary == ACT_SELECT_FROM_SELECTION)
      | (primary == ACT_SELECT_TO_ALLEY)
      | (primary == ACT_SELECT_TO_EQUIP)
      | (primary == ACT_SELECT_TO_GARDEN)
      | (primary == ACT_TOP_DECK_CARD)
      | (primary == ACT_BOTTOM_DECK_CARD)
    )
    sub1_kind = torch.where(
      selection_primary,
      torch.full_like(sub1_kind, LEGAL_ACTION_ARG_KIND_SELECTION),
      sub1_kind,
    )
    sub1_kind = torch.where(
      (primary == ACT_SELECT_COST_TARGET) | (primary == ACT_SELECT_EFFECT_TARGET),
      torch.full_like(sub1_kind, LEGAL_ACTION_ARG_KIND_GENERIC_TARGET),
      sub1_kind,
    )

    sub2_kind = torch.where(
      primary == ACT_PLAY_ENTITY_TO_GARDEN,
      torch.full_like(sub2_kind, LEGAL_ACTION_ARG_KIND_SELF_GARDEN),
      sub2_kind,
    )
    sub2_kind = torch.where(
      primary == ACT_PLAY_ENTITY_TO_ALLEY,
      torch.full_like(sub2_kind, LEGAL_ACTION_ARG_KIND_SELF_ALLEY),
      sub2_kind,
    )
    sub2_kind = torch.where(
      primary == ACT_ATTACH_WEAPON_FROM_HAND,
      torch.full_like(sub2_kind, LEGAL_ACTION_ARG_KIND_SELF_GARDEN_OR_LEADER),
      sub2_kind,
    )
    sub2_kind = torch.where(
      primary == ACT_GATE_PORTAL,
      torch.full_like(sub2_kind, LEGAL_ACTION_ARG_KIND_SELF_GARDEN),
      sub2_kind,
    )
    sub2_kind = torch.where(
      primary == ACT_ATTACK,
      torch.full_like(sub2_kind, LEGAL_ACTION_ARG_KIND_OPP_DEFENDER),
      sub2_kind,
    )
    sub2_kind = torch.where(
      primary == ACT_PLAY_SPELL_FROM_HAND,
      torch.full_like(sub2_kind, LEGAL_ACTION_ARG_KIND_ABILITY_INDEX),
      sub2_kind,
    )
    sub2_kind = torch.where(
      primary == ACT_ACTIVATE_ALLEY_ABILITY,
      torch.full_like(sub2_kind, LEGAL_ACTION_ARG_KIND_SELF_ALLEY),
      sub2_kind,
    )
    sub2_kind = torch.where(
      primary == ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY,
      torch.full_like(sub2_kind, LEGAL_ACTION_ARG_KIND_ABILITY_INDEX),
      sub2_kind,
    )
    sub2_kind = torch.where(
      primary == ACT_SELECT_TO_ALLEY,
      torch.full_like(sub2_kind, LEGAL_ACTION_ARG_KIND_SELF_ALLEY),
      sub2_kind,
    )
    sub2_kind = torch.where(
      primary == ACT_SELECT_TO_EQUIP,
      torch.full_like(sub2_kind, LEGAL_ACTION_ARG_KIND_SELF_GARDEN_OR_LEADER),
      sub2_kind,
    )
    sub2_kind = torch.where(
      primary == ACT_SELECT_TO_GARDEN,
      torch.full_like(sub2_kind, LEGAL_ACTION_ARG_KIND_SELF_GARDEN),
      sub2_kind,
    )

    bool_primary = (
      (primary == ACT_PLAY_ENTITY_TO_GARDEN)
      | (primary == ACT_PLAY_ENTITY_TO_ALLEY)
      | (primary == ACT_ATTACH_WEAPON_FROM_HAND)
      | (primary == ACT_PLAY_SPELL_FROM_HAND)
      | (primary == ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY)
    )
    sub3_kind = torch.where(
      bool_primary,
      torch.full_like(sub3_kind, LEGAL_ACTION_ARG_KIND_BOOL),
      sub3_kind,
    )
    return torch.stack((sub1_kind, sub2_kind, sub3_kind), dim=-1)

  def _gather_legal_action_refs(self, action_context: dict, arg_kinds: torch.Tensor, subactions: torch.Tensor):
    hand_matrix = self._lookup_action_context_tensor(action_context, "hand_matrix")
    player_garden_matrix = self._lookup_action_context_tensor(action_context, "player_garden_matrix")
    player_alley_matrix = self._lookup_action_context_tensor(action_context, "player_alley_matrix")
    player_selection_matrix = self._lookup_action_context_tensor(action_context, "player_selection_matrix")
    opponent_garden_matrix = self._lookup_action_context_tensor(action_context, "opponent_garden_matrix")
    opponent_alley_matrix = self._lookup_action_context_tensor(action_context, "opponent_alley_matrix")
    player_leader_vec = self._lookup_action_context_tensor(action_context, "player_leader_vec")
    opponent_leader_vec = self._lookup_action_context_tensor(action_context, "opponent_leader_vec")

    refs = []
    ref_valid = []
    for component_index in range(3):
      component_kind = arg_kinds[..., component_index]
      component_subaction = subactions[..., component_index]
      component_ref = torch.zeros(
        (*component_subaction.shape, UNIT_EMBED_SIZE),
        device=component_subaction.device,
        dtype=hand_matrix.dtype,
      )
      component_has_ref = torch.zeros_like(component_subaction, dtype=torch.bool)

      hand_mask = component_kind == LEGAL_ACTION_ARG_KIND_HAND
      if hand_mask.any():
        hand_refs = self._gather_zone_rows(hand_matrix, component_subaction)
        component_ref = torch.where(hand_mask.unsqueeze(-1), hand_refs, component_ref)
        component_has_ref |= hand_mask

      self_garden_mask = component_kind == LEGAL_ACTION_ARG_KIND_SELF_GARDEN
      if self_garden_mask.any():
        garden_refs = self._gather_zone_rows(player_garden_matrix, component_subaction)
        component_ref = torch.where(self_garden_mask.unsqueeze(-1), garden_refs, component_ref)
        component_has_ref |= self_garden_mask

      self_alley_mask = component_kind == LEGAL_ACTION_ARG_KIND_SELF_ALLEY
      if self_alley_mask.any():
        alley_refs = self._gather_zone_rows(player_alley_matrix, component_subaction)
        component_ref = torch.where(self_alley_mask.unsqueeze(-1), alley_refs, component_ref)
        component_has_ref |= self_alley_mask

      selection_mask = component_kind == LEGAL_ACTION_ARG_KIND_SELECTION
      if selection_mask.any():
        selection_refs = self._gather_zone_rows(player_selection_matrix, component_subaction)
        component_ref = torch.where(selection_mask.unsqueeze(-1), selection_refs, component_ref)
        component_has_ref |= selection_mask

      self_garden_or_leader_mask = component_kind == LEGAL_ACTION_ARG_KIND_SELF_GARDEN_OR_LEADER
      if self_garden_or_leader_mask.any():
        garden_or_leader_refs = self._gather_garden_or_leader_refs(
          player_garden_matrix,
          player_leader_vec,
          component_subaction,
        )
        component_ref = torch.where(
          self_garden_or_leader_mask.unsqueeze(-1),
          garden_or_leader_refs,
          component_ref,
        )
        component_has_ref |= self_garden_or_leader_mask & (component_subaction <= GARDEN_SIZE)

      opp_defender_mask = component_kind == LEGAL_ACTION_ARG_KIND_OPP_DEFENDER
      if opp_defender_mask.any():
        defender_refs = self._gather_opponent_defender_refs(
          opponent_garden_matrix,
          opponent_leader_vec,
          opponent_alley_matrix,
          component_subaction,
        )
        component_ref = torch.where(opp_defender_mask.unsqueeze(-1), defender_refs, component_ref)
        component_has_ref |= opp_defender_mask & (component_subaction <= (GARDEN_SIZE + ALLEY_SIZE))

      refs.append(component_ref)
      ref_valid.append(component_has_ref)

    return torch.stack(refs, dim=-2), torch.stack(ref_valid, dim=-1)

  def _build_legal_action_candidate_embeddings(
    self,
    flat_hidden: torch.Tensor,
    action_context: dict,
    legal_actions: torch.Tensor,
  ) -> torch.Tensor:
    primary = legal_actions[..., 0]
    subactions = legal_actions[..., 1:]
    arg_kinds = self._legal_action_arg_kinds(primary)

    primary_emb = self.primary_action_encoder[0](primary.clamp(0, PRIMARY_ACTION_COUNT - 1))
    subaction_idx = subactions.clamp(0, MAX_INDEX_SIZE - 1) + 1
    subaction_emb = self.legal_action_subaction_encoder(subaction_idx)
    arg_kind_emb = self.legal_action_arg_kind_encoder(arg_kinds)

    if self.legal_action_scorer_use_references:
      semantic_refs, semantic_ref_valid = self._gather_legal_action_refs(
        action_context,
        arg_kinds,
        subactions,
      )
    else:
      semantic_refs = torch.zeros(
        (*subactions.shape, UNIT_EMBED_SIZE),
        device=flat_hidden.device,
        dtype=flat_hidden.dtype,
      )
      semantic_ref_valid = torch.zeros(
        subactions.shape,
        device=flat_hidden.device,
        dtype=torch.bool,
      )

    normalized_subactions = subactions.float() / float(max(MAX_INDEX_SIZE - 1, 1))
    scalar = torch.cat(
      [
        normalized_subactions,
        semantic_ref_valid.float(),
      ],
      dim=-1,
    )

    batch_size, candidate_count, _ = legal_actions.shape
    candidate_input = torch.cat(
      [
        primary_emb,
        subaction_emb.reshape(batch_size, candidate_count, -1),
        arg_kind_emb.reshape(batch_size, candidate_count, -1),
        semantic_refs.reshape(batch_size, candidate_count, -1),
        scalar,
      ],
      dim=-1,
    )
    flattened = candidate_input.reshape(batch_size * candidate_count, -1)
    return self.legal_action_candidate_projector(flattened).reshape(
      batch_size,
      candidate_count,
      UNIT_EMBED_SIZE,
    )

  def _build_factorized_action_distribution(
    self,
    flat_hidden: torch.Tensor,
    target_matrix: torch.Tensor,
    legal_actions: torch.Tensor,
    legal_action_count: torch.Tensor,
    primary_action_mask: torch.Tensor,
  ) -> TCGActionDistribution:
    projected_hidden = self.q_primary(flat_hidden)
    unit1_projected_hidden = self.q_unit1(flat_hidden)
    unit2_projected_hidden = self.q_unit2(flat_hidden)
    bins2_projected_hidden = self.q_bins2(flat_hidden)
    bins3_projected_hidden = self.q_bins3(flat_hidden)

    primary_action_embeddings = self.primary_action_encoder[0](self.primary_action_id_batch)
    primary_action_logits = projected_hidden @ primary_action_embeddings.T

    gate_1_table = torch.sigmoid(self.gate_1_embeder(self.primary_action_id_batch))
    gate_2_table = torch.sigmoid(self.gate_2_embeder(self.primary_action_id_batch))

    return TCGActionDistribution(
      primary_logits=primary_action_logits,
      primary_action_mask=primary_action_mask,
      legal_actions=legal_actions,
      legal_action_count=legal_action_count,
      target_matrix=target_matrix,
      unit1_projection=unit1_projected_hidden,
      unit2_projection=unit2_projected_hidden,
      bins2_logits=bins2_projected_hidden,
      bins3_logits=bins3_projected_hidden,
      gate1_table=gate_1_table,
      gate2_table=gate_2_table,
    )

  def _build_legal_action_distribution(
    self,
    flat_hidden: torch.Tensor,
    action_context: dict,
    legal_actions: torch.Tensor,
    legal_action_count: torch.Tensor,
  ) -> TCGLegalActionDistribution:
    legal_action_query = self.q_legal_action(flat_hidden)
    candidate_embeddings = self._build_legal_action_candidate_embeddings(
      flat_hidden,
      action_context,
      legal_actions,
    )
    logits = (candidate_embeddings * legal_action_query.unsqueeze(1)).sum(dim=-1)
    logits = logits + self.legal_action_candidate_bias(candidate_embeddings).squeeze(-1)
    return TCGLegalActionDistribution(
      legal_action_logits=logits,
      legal_actions=legal_actions,
      legal_action_count=legal_action_count,
    )

  def decode_actions(self, flat_hidden, action_context=None, state=None):
    if flat_hidden.dim() == 1:
      flat_hidden = flat_hidden.unsqueeze(0)

    mask_observations = state.get("_azk_mask_observations") if state is not None else self._cached_mask_observations
    if mask_observations is None:
      raise ValueError("mask_observations is None when decode_actions is called")

    structured_mask_obs, _, _ = self.__prepare_structured_observations(mask_observations)
    action_mask_struct = self.__get_struct_field(structured_mask_obs, "action_mask")

    device = flat_hidden.device
    action_context = self._prepare_action_context(action_context, device=device)
    target_matrix = self._lookup_action_context_tensor(action_context, "legacy_target_matrix")
    primary_action_mask = action_mask_struct["primary_action_mask"].to(dtype=torch.bool, device=device)

    # Support both observation layouts:
    # - emulated: action_mask.legal_actions.{legal_primary,legal_sub1,legal_sub2,legal_sub3}
    # - packed native: action_mask.{legal_primary,legal_sub1,legal_sub2,legal_sub3}
    try:
      legal_actions_struct = self.__get_struct_field(action_mask_struct, "legal_actions")
      legal_primary = legal_actions_struct["legal_primary"]
      legal_sub1 = legal_actions_struct["legal_sub1"]
      legal_sub2 = legal_actions_struct["legal_sub2"]
      legal_sub3 = legal_actions_struct["legal_sub3"]
    except KeyError:
      legal_primary = self.__get_struct_field(action_mask_struct, "legal_primary")
      legal_sub1 = self.__get_struct_field(action_mask_struct, "legal_sub1")
      legal_sub2 = self.__get_struct_field(action_mask_struct, "legal_sub2")
      legal_sub3 = self.__get_struct_field(action_mask_struct, "legal_sub3")

    legal_actions = torch.stack(
      (
        legal_primary,
        legal_sub1,
        legal_sub2,
        legal_sub3,
      ),
      dim=-1,
    ).to(device=device, dtype=torch.long)
    legal_action_count = action_mask_struct["legal_action_count"].to(device=device, dtype=torch.long).view(-1)

    if self.actor_head_type == ACTOR_HEAD_TYPE_LEGAL_ACTION_SCORER:
      distribution = self._build_legal_action_distribution(
        flat_hidden,
        action_context,
        legal_actions,
        legal_action_count=legal_action_count,
      )
    else:
      distribution = self._build_factorized_action_distribution(
        flat_hidden,
        target_matrix,
        legal_actions,
        legal_action_count,
        primary_action_mask,
      )

    value_features = self._value_features(
      flat_hidden,
      self._shared_critic_projection(flat_hidden),
      state=state,
    )
    if self.split_value_heads_enabled:
      terminal_values = self.value_terminal_fn(value_features)
      shaped_values = self.value_shaped_fn(value_features)
      values = terminal_values + shaped_values
      if state is not None:
        state["_azk_value_terminal"] = terminal_values
        state["_azk_value_shaped"] = shaped_values
    else:
      values = self.value_fn(value_features)
    if state is not None and self.win_prob_fn is not None:
      state["_azk_win_prob_logits"] = self.win_prob_fn(self._win_prob_features(flat_hidden))
    return distribution, values


def build_policy_model(env, policy_config: dict | None = None, **kwargs) -> TCG:
  policy_config = policy_config or {}
  model_version = str(
    policy_config.get("model_version", POLICY_MODEL_VERSION_METADATA_V1)
  )
  actor_head_type = str(
    policy_config.get("actor_head_type", ACTOR_HEAD_TYPE_FACTORIZED)
  )
  legal_action_scorer_use_references = bool(
    policy_config.get(
      "legal_action_scorer_use_references",
      LEGAL_ACTION_SCORER_USE_REFERENCES_DEFAULT,
    )
  )
  critic_head_type = str(
    policy_config.get("critic_head_type", CRITIC_HEAD_TYPE_FULL_LSTM_MLP)
  )
  privileged_critic_enabled = bool(
    policy_config.get("privileged_critic_enabled", PRIVILEGED_CRITIC_ENABLED_DEFAULT)
  )
  privileged_critic_embed_dim = int(
    policy_config.get("privileged_critic_embed_dim", PRIVILEGED_CRITIC_EMBED_DIM)
  )
  privileged_critic_deck_heads = int(
    policy_config.get("privileged_critic_deck_heads", PRIVILEGED_CRITIC_DECK_HEADS)
  )
  privileged_critic_deck_layers = int(
    policy_config.get("privileged_critic_deck_layers", PRIVILEGED_CRITIC_DECK_LAYERS)
  )
  privileged_critic_deck_ff_size = int(
    policy_config.get("privileged_critic_deck_ff_size", PRIVILEGED_CRITIC_DECK_FF_SIZE)
  )
  privileged_critic_fusion_hidden_size = int(
    policy_config.get(
      "privileged_critic_fusion_hidden_size",
      PRIVILEGED_CRITIC_FUSION_HIDDEN_SIZE,
    )
  )
  privileged_critic_fusion_projection_size = int(
    policy_config.get(
      "privileged_critic_fusion_projection_size",
      PRIVILEGED_CRITIC_FUSION_PROJECTION_SIZE,
    )
  )
  privileged_critic_feature_scale = float(
    policy_config.get(
      "privileged_critic_feature_scale",
      PRIVILEGED_CRITIC_FEATURE_SCALE_DEFAULT,
    )
  )
  win_prob_aux_enabled = bool(policy_config.get("win_prob_aux_enabled", False))
  win_prob_aux_coef = float(policy_config.get("win_prob_aux_coef", WIN_PROB_AUX_COEF_DEFAULT))
  split_value_heads_enabled = bool(policy_config.get("split_value_heads_enabled", False))
  split_value_component_coef = float(
    policy_config.get("split_value_component_coef", SPLIT_VALUE_COMPONENT_COEF_DEFAULT)
  )

  if model_version == POLICY_MODEL_VERSION_METADATA_V1:
    return TCG(
      env,
      model_version=model_version,
      actor_head_type=actor_head_type,
      legal_action_scorer_use_references=legal_action_scorer_use_references,
      critic_head_type=critic_head_type,
      privileged_critic_enabled=privileged_critic_enabled,
      privileged_critic_embed_dim=privileged_critic_embed_dim,
      privileged_critic_deck_heads=privileged_critic_deck_heads,
      privileged_critic_deck_layers=privileged_critic_deck_layers,
      privileged_critic_deck_ff_size=privileged_critic_deck_ff_size,
      privileged_critic_fusion_hidden_size=privileged_critic_fusion_hidden_size,
      privileged_critic_fusion_projection_size=privileged_critic_fusion_projection_size,
      privileged_critic_feature_scale=privileged_critic_feature_scale,
      win_prob_aux_enabled=win_prob_aux_enabled,
      win_prob_aux_coef=win_prob_aux_coef,
      split_value_heads_enabled=split_value_heads_enabled,
      split_value_component_coef=split_value_component_coef,
      **kwargs,
    )

  raise ValueError(
    f"Unsupported v2 policy model_version '{model_version}'. "
    f"Known versions: {POLICY_MODEL_VERSION_METADATA_V1}"
  )
