import contextlib

import azk_puffer.pytorch as azk_pytorch
from azk_puffer.models import LSTMWrapper
from gymnasium.wrappers.normalize import RunningMeanStd

import numpy as np
import torch
from torch import nn

from observation import (
  ACTION_TYPE_COUNT,
  ALLEY_SIZE,
  DECK_CONTEXT_MODE_COUNT,
  GARDEN_SIZE,
  MAX_DECK_BUILD_CANDIDATES,
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
ACT_DECK_PICK_CARD = 3
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
GATE_ID_EMBED_SIZE = 16
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
LEGAL_ACTION_ARG_KIND_CARD_CANDIDATE = 10
LEGAL_ACTION_ARG_KIND_COUNT = 11
LEGAL_ACTION_ARG_KIND_EMBED_SIZE = 8


def _build_legal_action_arg_kind_table() -> torch.Tensor:
  table = torch.full(
    (PRIMARY_ACTION_COUNT, 3),
    LEGAL_ACTION_ARG_KIND_UNUSED,
    dtype=torch.long,
  )

  table[
    [
      ACT_PLAY_ENTITY_TO_GARDEN,
      ACT_PLAY_ENTITY_TO_ALLEY,
      ACT_ATTACH_WEAPON_FROM_HAND,
      ACT_PLAY_SPELL_FROM_HAND,
    ],
    0,
  ] = LEGAL_ACTION_ARG_KIND_HAND
  table[ACT_GATE_PORTAL, 0] = LEGAL_ACTION_ARG_KIND_SELF_ALLEY
  table[ACT_ACTIVATE_ALLEY_ABILITY, 0] = LEGAL_ACTION_ARG_KIND_ABILITY_INDEX
  table[
    [ACT_ATTACK, ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY],
    0,
  ] = LEGAL_ACTION_ARG_KIND_SELF_GARDEN_OR_LEADER
  table[ACT_DECLARE_DEFENDER, 0] = LEGAL_ACTION_ARG_KIND_SELF_GARDEN
  table[
    [
      ACT_SELECT_FROM_SELECTION,
      ACT_SELECT_TO_ALLEY,
      ACT_SELECT_TO_EQUIP,
      ACT_SELECT_TO_GARDEN,
      ACT_TOP_DECK_CARD,
      ACT_BOTTOM_DECK_CARD,
    ],
    0,
  ] = LEGAL_ACTION_ARG_KIND_SELECTION
  table[
    [ACT_SELECT_COST_TARGET, ACT_SELECT_EFFECT_TARGET],
    0,
  ] = LEGAL_ACTION_ARG_KIND_GENERIC_TARGET
  table[ACT_DECK_PICK_CARD, 0] = LEGAL_ACTION_ARG_KIND_CARD_CANDIDATE

  table[ACT_PLAY_ENTITY_TO_GARDEN, 1] = LEGAL_ACTION_ARG_KIND_SELF_GARDEN
  table[ACT_PLAY_ENTITY_TO_ALLEY, 1] = LEGAL_ACTION_ARG_KIND_SELF_ALLEY
  table[ACT_ATTACH_WEAPON_FROM_HAND, 1] = LEGAL_ACTION_ARG_KIND_SELF_GARDEN_OR_LEADER
  table[ACT_GATE_PORTAL, 1] = LEGAL_ACTION_ARG_KIND_SELF_GARDEN
  table[ACT_ATTACK, 1] = LEGAL_ACTION_ARG_KIND_OPP_DEFENDER
  table[ACT_PLAY_SPELL_FROM_HAND, 1] = LEGAL_ACTION_ARG_KIND_ABILITY_INDEX
  table[ACT_ACTIVATE_ALLEY_ABILITY, 1] = LEGAL_ACTION_ARG_KIND_SELF_ALLEY
  table[ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY, 1] = LEGAL_ACTION_ARG_KIND_ABILITY_INDEX
  table[ACT_SELECT_TO_ALLEY, 1] = LEGAL_ACTION_ARG_KIND_SELF_ALLEY
  table[ACT_SELECT_TO_EQUIP, 1] = LEGAL_ACTION_ARG_KIND_SELF_GARDEN_OR_LEADER
  table[ACT_SELECT_TO_GARDEN, 1] = LEGAL_ACTION_ARG_KIND_SELF_GARDEN

  table[
    [
      ACT_PLAY_ENTITY_TO_GARDEN,
      ACT_PLAY_ENTITY_TO_ALLEY,
      ACT_ATTACH_WEAPON_FROM_HAND,
      ACT_PLAY_SPELL_FROM_HAND,
      ACT_ACTIVATE_GARDEN_OR_LEADER_ABILITY,
    ],
    2,
  ] = LEGAL_ACTION_ARG_KIND_BOOL
  return table


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


class _PackedField:
  """Leaf accessor for one field of the packed C observation struct.

  Extraction is a zero-copy torch.as_strided view over the flat uint8
  observation rows: (B, offset..)-strided at element granularity, with one
  size/stride pair per enclosing struct-array level (zone slots, weapons).
  """

  __slots__ = ("torch_dtype", "view_dtype", "offset", "elem_bytes", "dims")

  def __init__(self, torch_dtype, offset, elem_bytes, dims):
    self.torch_dtype = torch_dtype
    self.offset = int(offset)
    self.elem_bytes = int(elem_bytes)
    self.dims = tuple(dims)  # ((count, stride_bytes), ...) outermost first
    if torch_dtype in (torch.bool, torch.uint8):
      self.view_dtype = torch.uint8  # bools are read as raw bytes (consumers cast)
    elif torch_dtype is torch.int8:
      self.view_dtype = torch.int8
    else:
      self.view_dtype = torch_dtype
    if self.offset % self.elem_bytes != 0:
      raise ValueError(f"Packed field offset {offset} not aligned to {elem_bytes}")
    for _, stride in self.dims:
      if stride % self.elem_bytes != 0:
        raise ValueError(f"Packed field stride {stride} not aligned to {elem_bytes}")

  def extract(self, obs_u8: torch.Tensor) -> torch.Tensor:
    batch, row_bytes = obs_u8.shape
    esz = self.elem_bytes
    if self.view_dtype is torch.uint8:
      base = obs_u8
    else:
      base = obs_u8.view(self.view_dtype)
    sizes = (batch, *(count for count, _ in self.dims))
    strides = (row_bytes // esz, *(stride // esz for _, stride in self.dims))
    return base.as_strided(sizes, strides, storage_offset=base.storage_offset() + self.offset // esz)


def _build_packed_specs(dtype: np.dtype, base_offset: int = 0, dims=()):
  """Walk a numpy structured dtype (mirroring the C struct) into _PackedField specs."""
  dtype = np.dtype(dtype)
  if dtype.fields:
    out = {}
    for name, (field_dtype, field_offset) in dtype.fields.items():
      out[name] = _build_packed_specs(field_dtype, base_offset + int(field_offset), dims)
    return out
  if dtype.subdtype is not None:
    scalar_dtype, subshape = dtype.subdtype
    if len(subshape) != 1:
      raise ValueError(f"Unsupported packed subarray shape {subshape}")
    inner = np.dtype(scalar_dtype)
    return _build_packed_specs(
      inner, base_offset, dims + ((int(subshape[0]), int(inner.itemsize)),)
    )
  return _PackedField(_numpy_dtype_to_torch(dtype), base_offset, dtype.itemsize, dims)


def _is_packed_native_dtype(obs_dtype: np.dtype) -> bool:
  names = getattr(np.dtype(obs_dtype), "names", None) or ()
  return "my_observation_data" in names


class ScalarRunningNorm(nn.Module):
  """Normalizes scalar/boolean tensors with running mean/std and clamps to [-clip, clip].

  Fully GPU-resident: running moments are float64 device buffers updated with
  a Chan parallel combine (same math as gymnasium's RunningMeanStd, which the
  previous implementation round-tripped through numpy on every call). No
  host<->device syncs occur in forward. Buffer names/dtypes are unchanged for
  checkpoint compatibility.
  """

  def __init__(self, *, clip: float = 5.0, eps: float = 1e-8, rms_epsilon: float = 1e-4):
    super().__init__()
    self.clip = clip
    self.eps = eps
    self.rms_epsilon = rms_epsilon

  def _buffer_names(self, key: str):
    return (
      f"_rms_{key}_mean",
      f"_rms_{key}_var",
      f"_rms_{key}_count",
    )

  def _ensure_buffers(self, key: str, feature_shape, device, ref_dtype=torch.float64):
    mean_name, var_name, count_name = self._buffer_names(key)
    if not hasattr(self, mean_name):
      shape = torch.Size(feature_shape) if feature_shape else torch.Size([])
      self.register_buffer(mean_name, torch.zeros(shape, dtype=torch.float64, device=device))
      self.register_buffer(var_name, torch.ones(shape, dtype=torch.float64, device=device))
      self.register_buffer(
        count_name, torch.tensor(self.rms_epsilon, dtype=torch.float64, device=device)
      )
    else:
      mean_buf = getattr(self, mean_name)
      if mean_buf.device != device:
        setattr(self, mean_name, mean_buf.to(device))
        setattr(self, var_name, getattr(self, var_name).to(device))
        setattr(self, count_name, getattr(self, count_name).to(device))
    return (
      getattr(self, mean_name),
      getattr(self, var_name),
      getattr(self, count_name),
    )

  def _update_stats(self, key: str, tensor: torch.Tensor, mask: torch.Tensor | None) -> None:
    feature_shape = () if tensor.dim() == 1 else (tensor.shape[-1],)
    mean_buf, var_buf, count_buf = self._ensure_buffers(key, feature_shape, tensor.device)
    with torch.no_grad():
      x = tensor.detach().double()
      if feature_shape:
        x = x.reshape(-1, feature_shape[0])
      else:
        x = x.reshape(-1, 1)
      if mask is not None:
        mask_update = mask
        if mask_update.dim() == tensor.dim():
          mask_update = mask_update.any(dim=-1)
        m = mask_update.reshape(-1, 1).double()
        n = m.sum()
        denom = n.clamp(min=1.0)
        batch_mean = (x * m).sum(dim=0) / denom
        batch_var = (((x - batch_mean) ** 2) * m).sum(dim=0) / denom
      else:
        n = torch.full((), float(x.shape[0]), dtype=torch.float64, device=x.device)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
      if not feature_shape:
        batch_mean = batch_mean.squeeze(0)
        batch_var = batch_var.squeeze(0)
      # Chan parallel combine; with n == 0 all update terms vanish, so no
      # host-side branch on the (device-resident) count is needed.
      total = count_buf + n
      delta = batch_mean - mean_buf
      new_mean = mean_buf + delta * (n / total)
      m_a = var_buf * count_buf
      m_b = batch_var * n
      new_var = (m_a + m_b + (delta ** 2) * (count_buf * n / total)) / total
      mean_buf.copy_(new_mean)
      var_buf.copy_(new_var)
      count_buf.copy_(total)

  def forward(self, key: str, tensor: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if tensor is None:
      return tensor

    tensor = tensor.float()
    feature_shape = () if tensor.dim() == 1 else (tensor.shape[-1],)
    mean_buf, var_buf, _ = self._ensure_buffers(key, feature_shape, tensor.device)

    if self.training:
      self._update_stats(key, tensor, mask)

    mean = mean_buf.to(dtype=tensor.dtype)
    var = var_buf.to(dtype=tensor.dtype)
    normalized = (tensor - mean) / torch.sqrt(var + self.eps)

    if mask is not None:
      mask_broadcast = mask
      while mask_broadcast.dim() < normalized.dim():
        mask_broadcast = mask_broadcast.unsqueeze(-1)
      normalized = normalized * mask_broadcast.to(dtype=normalized.dtype)

    return torch.clamp(normalized, -self.clip, self.clip)

  def update_only(self, key: str, tensor: torch.Tensor, mask: torch.Tensor | None = None) -> None:
    """Update running moments without producing a normalized output."""
    if tensor is None or not self.training:
      return
    self._update_stats(key, tensor.float(), mask)

  def normalize_only(self, key: str, tensor: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Normalize with current stats without updating them."""
    if tensor is None:
      return tensor
    tensor = tensor.float()
    feature_shape = () if tensor.dim() == 1 else (tensor.shape[-1],)
    mean_buf, var_buf, _ = self._ensure_buffers(key, feature_shape, tensor.device)
    mean = mean_buf.to(dtype=tensor.dtype)
    var = var_buf.to(dtype=tensor.dtype)
    normalized = (tensor - mean) / torch.sqrt(var + self.eps)
    if mask is not None:
      mask_broadcast = mask
      while mask_broadcast.dim() < normalized.dim():
        mask_broadcast = mask_broadcast.unsqueeze(-1)
      normalized = normalized * mask_broadcast.to(dtype=normalized.dtype)
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


class SumSetProcessor(nn.Module):
  def __init__(self, input_size, hidden_size=PROCESS_SET_HIDDEN_SIZE, output_size=UNIT_EMBED_SIZE):
    super().__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, output_size)
    self.act = nn.ReLU()
    self.output_size = output_size

  def forward(self, x, mask=None):
    set_embeddings = self.fc2(self.act(self.fc1(x)))
    if mask is None:
      return set_embeddings, set_embeddings.sum(dim=-2)

    if mask.dim() == set_embeddings.dim() - 1:
      mask = mask.unsqueeze(-1)
    mask = mask.to(dtype=torch.bool)
    masked = set_embeddings * mask.to(dtype=set_embeddings.dtype)
    count = mask.any(dim=-1).sum(dim=-1, keepdim=True).clamp(min=1)
    pooled = masked.sum(dim=-2) / torch.sqrt(count.to(dtype=masked.dtype))
    return masked, pooled


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

  # --- CUDA-graph rollout ---------------------------------------------------
  # The rollout forward is launch-bound (~1500 tiny kernels at batch ~1k, far
  # above its FLOP cost). Shapes are static except the legal-action trim
  # bucket (<=6 power-of-two variants), so we capture one graph per bucket and
  # replay it in a single launch. Sampling stays eager so multinomial RNG is
  # never captured. Capture runs lazily inside the trainer's no_grad+autocast
  # context, so the recorded kernels match the eager execution exactly.

  _CG_STATE_KEYS = (
    "hidden",
    "lstm_h",
    "lstm_c",
    "_azk_win_prob_logits",
    "_azk_value_terminal",
    "_azk_value_shaped",
  )

  def enable_rollout_cuda_graphs(self):
    if getattr(self, "_cg_enabled", False):
      return
    if not bool(getattr(self.policy, "_native_layout", False)):
      print("[cuda-graphs] rollout graphs require the native packed obs layout; keeping eager forward_eval")
      return
    self._cg_enabled = True
    self._cg_broken = False
    self._cg_graphs = {}
    self._cg_static = None
    self._cg_eager_forward_eval = self.forward_eval
    self.forward_eval = self._forward_eval_cuda_graph

  def _forward_eval_cuda_graph(self, observations, state):
    h = state.get("lstm_h")
    c = state.get("lstm_c")
    if (
      self._cg_broken
      or h is None
      or c is None
      or not observations.is_cuda
    ):
      return self._cg_eager_forward_eval(observations, state)

    if self._cg_static is None:
      self._cg_static = (
        torch.empty_like(observations),
        torch.empty_like(h),
        torch.empty_like(c),
      )
    s_obs, s_h, s_c = self._cg_static
    if s_obs.shape != observations.shape or s_h.shape != h.shape:
      return self._cg_eager_forward_eval(observations, state)
    s_obs.copy_(observations, non_blocking=True)
    s_h.copy_(h, non_blocking=True)
    s_c.copy_(c, non_blocking=True)

    counts = self.policy._packed_specs["action_mask"]["legal_action_count"].extract(s_obs)
    bucket = self.policy._compute_legal_action_trim_bucket(counts)

    entry = self._cg_graphs.get(bucket)
    if entry is None:
      try:
        entry = self._cg_capture(bucket)
      except Exception as exc:
        print(f"[cuda-graphs] capture failed for bucket {bucket}: {exc!r}; falling back to eager rollout")
        self._cg_broken = True
        return self._cg_eager_forward_eval(observations, state)
    graph, out_state, out_logits, out_values = entry
    graph.replay()
    for key in self._CG_STATE_KEYS:
      if key in out_state:
        state[key] = out_state[key]
    return out_logits, out_values

  def _cg_capture(self, bucket: int):
    base = self.policy
    s_obs, s_h, s_c = self._cg_static
    base._trim_bucket_override = bucket
    # If the trainer torch.compile'd encode/decode (train path), capture the
    # eager originals instead: dynamo's guard machinery reads the CUDA RNG
    # seed, which is illegal during stream capture. Replays never re-enter
    # Python, so the train path keeps its compiled versions untouched.
    swapped_methods = None
    if hasattr(base, "_eager_encode_observations"):
      swapped_methods = (base.encode_observations, base.decode_actions)
      base.encode_observations = base._eager_encode_observations
      base.decode_actions = base._eager_decode_actions
    # The trainer calls this under an ambient autocast whose weight cache is
    # freed when that context exits. A capture must never record pointers to
    # those cached casts (freed memory on replay + silently stale weights
    # after optimizer steps), so warmup+capture run with the cache disabled:
    # casts become graph ops that re-read the live fp32 weights every replay.
    if torch.is_autocast_enabled("cuda"):
      autocast_ctx = torch.autocast(
        "cuda",
        dtype=torch.get_autocast_dtype("cuda"),
        cache_enabled=False,
      )
    else:
      autocast_ctx = contextlib.nullcontext()
    from policy.v2 import tcg_sampler  # runtime import; sampler imports us

    try:
      with autocast_ctx:
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
          for _ in range(3):
            warm_state = {"lstm_h": s_h, "lstm_c": s_c}
            warm_logits, _ = self._cg_eager_forward_eval(s_obs, warm_state)
            tcg_sampler.tcg_sample_logits(warm_logits, action=None)
        torch.cuda.current_stream().wait_stream(side_stream)

        graph = torch.cuda.CUDAGraph()
        capture_state = {"lstm_h": s_h, "lstm_c": s_c}
        # Each bucket gets its own private memory pool: sharing a pool across
        # graphs is only safe when they replay in capture order, and buckets
        # replay in data-dependent order (sharing produced illegal accesses).
        # Sampling is captured too (multinomial RNG is graph-safe: the default
        # generator's offset advances per replay); the sampler returns the
        # presampled static tensors when the distribution carries them.
        with torch.cuda.graph(graph):
          logits, values = self._cg_eager_forward_eval(s_obs, capture_state)
          presampled = tcg_sampler.tcg_sample_logits(logits, action=None)
      object.__setattr__(logits, "_azk_presampled", presampled)
      object.__setattr__(logits, "_azk_presampled_params", tcg_sampler.get_sampling_params())
      entry = (graph, capture_state, logits, values)
      self._cg_graphs[bucket] = entry
      print(f"[cuda-graphs] captured rollout graph for trim bucket {bucket}")
      return entry
    finally:
      base._trim_bucket_override = None
      if swapped_methods is not None:
        base.encode_observations, base.decode_actions = swapped_methods


class TCG(nn.Module):
  def __init__(
    self,
    env,
    *,
    model_version: str = POLICY_MODEL_VERSION_METADATA_V1,
    actor_head_type: str = ACTOR_HEAD_TYPE_LEGAL_ACTION_SCORER,
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
    gate_id_embedding_enabled: bool = False,
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
    # The projected card-metadata embeddings of same-element gate cards are
    # near-identical (cos ~0.97-0.9995 measured on trained checkpoints): the
    # text-effect features that distinguish them do not survive projection, so
    # the policy cannot condition strategy on WHICH gate it was assigned. This
    # learned per-card-id channel restores distinguishability for the gate
    # slots only (deck_context gate + gate zone encoders).
    self.gate_id_embedding_enabled = bool(gate_id_embedding_enabled)
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

    emulated_spec = getattr(env, "emulated", None)
    if emulated_spec is None:
      raise AttributeError("env must expose emulated metadata for nativize")
    obs_dtype = emulated_spec.get("emulated_observation_dtype")
    if obs_dtype is None:
      raise AttributeError("env.emulated missing emulated_observation_dtype")
    self._native_layout = bool(emulated_spec.get("native_layout", False)) or _is_packed_native_dtype(obs_dtype)
    if self._native_layout:
      self._packed_specs = _build_packed_specs(obs_dtype)
      self._obs_struct_dtype = None
      dtype_names = getattr(np.dtype(obs_dtype), "names", None) or ()
      self.deck_context_enabled = "deck_context" in dtype_names
    else:
      self._packed_specs = None
      self._obs_struct_dtype = _build_native_dtype_from_numpy(obs_dtype)
      self.deck_context_enabled = (
        isinstance(self._obs_struct_dtype, dict) and "deck_context" in self._obs_struct_dtype
      )
    if self.deck_context_enabled and self.actor_head_type != ACTOR_HEAD_TYPE_LEGAL_ACTION_SCORER:
      raise ValueError("deck_building_enabled requires actor_head_type='legal_action_scorer'")

    static_table = load_policy_card_metadata_table()
    self.static_vocab_size = static_table.vocab_size
    self.metadata_embedding_dim = static_table.embedding_dim
    self.keyword_vocab_size = static_table.keyword_vocab_size
    self.register_buffer("static_card_present_mask", static_table.card_present_mask, persistent=False)
    self.register_buffer("static_card_type", static_table.card_type_ids, persistent=False)
    self.register_buffer("static_element", static_table.element_ids, persistent=False)
    self.register_buffer("static_base_ikz_cost", static_table.ikz_cost, persistent=False)
    self.register_buffer("static_base_attack", static_table.attack, persistent=False)
    self.register_buffer("static_base_health", static_table.health, persistent=False)
    self.register_buffer("static_base_gate_points", static_table.gate_points, persistent=False)
    self.register_buffer("static_has_ability", static_table.has_ability, persistent=False)
    self.register_buffer("static_ability_timing", static_table.ability_timing_ids, persistent=False)
    self.register_buffer("static_ability_optional", static_table.ability_is_optional, persistent=False)
    self.register_buffer(
      "static_card_scalar",
      torch.stack(
        [
          static_table.card_present_mask,
          static_table.ikz_cost,
          static_table.attack,
          static_table.health,
          static_table.gate_points,
          static_table.has_ability,
          static_table.ability_is_optional,
        ],
        dim=-1,
      ),
      persistent=False,
    )
    self.register_buffer("static_keyword_multi_hot", static_table.keyword_multi_hot, persistent=False)
    self.register_buffer("static_name_embeddings", static_table.name_embeddings, persistent=False)
    self.register_buffer("static_effect_embeddings", static_table.effect_embeddings, persistent=False)
    self.register_buffer(
      "static_subtype_pooled_embeddings",
      static_table.subtype_pooled_embeddings,
      persistent=False,
    )

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
    self.register_buffer("legal_action_arg_kind_table", _build_legal_action_arg_kind_table())
    self.legal_action_subaction_encoder = nn.Embedding(
      MAX_INDEX_SIZE + 1,
      INDEX_ENC_OUTPUT_SIZE,
    )
    legal_action_arg_kind_count = (
      LEGAL_ACTION_ARG_KIND_COUNT
      if self.deck_context_enabled
      else LEGAL_ACTION_ARG_KIND_CARD_CANDIDATE
    )
    self.legal_action_arg_kind_encoder = nn.Embedding(
      legal_action_arg_kind_count,
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
    self.gate_id_embedding = None
    if self.gate_id_embedding_enabled:
      self.gate_id_embedding = nn.Embedding(self.static_vocab_size, GATE_ID_EMBED_SIZE)
      # Start near-neutral relative to the ~2.5-norm metadata embeddings so the
      # identity channel informs without dominating early training.
      nn.init.normal_(self.gate_id_embedding.weight, std=0.25)
      gate_input_size += GATE_ID_EMBED_SIZE

    self.leader_projector = SingleUnitProjection(leader_input_size)
    self.gate_projector = SingleUnitProjection(gate_input_size)
    self.deck_mode_encoder = None
    self.deck_context_card_processor = None
    self.deck_context_projector = None
    self.deck_candidate_projector = None
    if self.deck_context_enabled:
      self.deck_mode_encoder = nn.Embedding(DECK_CONTEXT_MODE_COUNT, PHASE_ENC_OUTPUT_SIZE)
      deck_card_input_size = CARD_METADATA_EMBED_SIZE + 2
      self.deck_context_card_processor = SumSetProcessor(deck_card_input_size)
      deck_context_input_size = (
        PHASE_ENC_OUTPUT_SIZE
        + (CARD_METADATA_EMBED_SIZE * 2)
        + UNIT_EMBED_SIZE
        + 5
      )
      if self.gate_id_embedding_enabled:
        deck_context_input_size += GATE_ID_EMBED_SIZE
      self.deck_context_projector = SingleUnitProjection(deck_context_input_size)
      self.deck_candidate_projector = SingleUnitProjection(
        CARD_METADATA_EMBED_SIZE + 3,
        output_size=UNIT_EMBED_SIZE,
      )

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

    self.zone_component_count = 15 if self.deck_context_enabled else 14
    self.lstm_input_size = UNIT_EMBED_SIZE * (self.zone_component_count + 1)

    self._text_table_cache: torch.Tensor | None = None
    self._text_table_cache_version = -1
    self._text_table_version = 0
    self._metadata_table_cache: torch.Tensor | None = None
    self._metadata_table_cache_version = -1

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

    self._cached_mask_observations = None
    self._cached_cobs = None
    self._cached_trim_bucket = None
    self._trim_bucket_override = None

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

  def __store_mask_observations(self, obs_tensor, state):
    if torch.is_tensor(obs_tensor):
      obs_tensor = obs_tensor.detach()
    if state is not None:
      try:
        state["_azk_mask_observations"] = obs_tensor
      except (TypeError, AttributeError):
        pass
    self._cached_mask_observations = obs_tensor

  def __store_privileged_critic_features(self, features: torch.Tensor | None, state):
    if state is not None:
      try:
        state["_azk_privileged_critic_features"] = features
      except (TypeError, AttributeError):
        pass
    self._cached_privileged_critic_features = features

  # --- canonical observation tree -------------------------------------------
  # Every observation layout (packed C struct, legacy emulated dtype, raw
  # structured dict) is normalized into one canonical tree: zones are dicts of
  # whole-zone field tensors (B, S) (weapons: (B, S, W)), struct scalars are
  # (B,) tensors. All encoders consume this form, so the per-slot Python
  # loops (and their thousands of tiny autograd nodes) are gone.

  def _canonical_observations(self, observations):
    if isinstance(observations, dict):
      structured = self.__tensorize_structured_observation(
        observations, self.__policy_device()
      )
      return self._canonical_from_structured(structured), False, None

    obs_tensor = observations if torch.is_tensor(observations) else torch.as_tensor(observations)
    squeeze_batch = obs_tensor.dim() == 1
    if squeeze_batch:
      obs_tensor = obs_tensor.unsqueeze(0)
    obs_tensor = obs_tensor.to(self.__policy_device())
    if not obs_tensor.is_contiguous():
      obs_tensor = obs_tensor.contiguous()
    if self._native_layout:
      cobs = self._canonical_from_packed(obs_tensor)
    else:
      structured = azk_pytorch.nativize_tensor(obs_tensor, self._obs_struct_dtype)
      cobs = self._canonical_from_structured(structured)
    return cobs, squeeze_batch, obs_tensor

  def _canonical_from_packed(self, obs_u8: torch.Tensor):
    sp = self._packed_specs
    ex = lambda spec: spec.extract(obs_u8)  # noqa: E731

    def tap(node):
      return {
        "tapped": ex(node["tap_state"]["tapped"]),
        "cooldown": ex(node["tap_state"]["cooldown"]),
      }

    def weapons(node):
      return {
        "card_def_id": ex(node["weapons"]["card_def_id"]),
        "cur_atk": ex(node["weapons"]["cur_atk"]),
      }

    def leader(node):
      return {
        "card_def_id": ex(node["card_def_id"]),
        **tap(node),
        "cur_atk": ex(node["cur_stats"]["cur_atk"]),
        "cur_hp": ex(node["cur_stats"]["cur_hp"]),
        "weapon_count": ex(node["weapon_count"]),
        "weapons": weapons(node),
        "has_charge": ex(node["has_charge"]),
        "has_defender": ex(node["has_defender"]),
        "has_infiltrate": ex(node["has_infiltrate"]),
      }

    def gate(node):
      return {"card_def_id": ex(node["card_def_id"]), **tap(node)}

    def card_zone(node):
      return {"card_def_id": ex(node["card_def_id"]), "zone_index": ex(node["zone_index"])}

    def board_zone(node):
      return {
        "card_def_id": ex(node["card_def_id"]),
        "zone_index": ex(node["zone_index"]),
        **tap(node),
        "has_cur_stats": ex(node["has_cur_stats"]),
        "cur_atk": ex(node["cur_stats"]["cur_atk"]),
        "cur_hp": ex(node["cur_stats"]["cur_hp"]),
        "weapon_count": ex(node["weapon_count"]),
        "weapons": weapons(node),
        "has_charge": ex(node["has_charge"]),
        "has_defender": ex(node["has_defender"]),
        "has_infiltrate": ex(node["has_infiltrate"]),
        "is_frozen": ex(node["is_frozen"]),
        "is_shocked": ex(node["is_shocked"]),
        "is_effect_immune": ex(node["is_effect_immune"]),
      }

    def ikz_zone(node):
      return {
        "card_def_id": ex(node["card_def_id"]),
        "zone_index": ex(node["zone_index"]),
        **tap(node),
      }

    my = sp["my_observation_data"]
    opp = sp["opponent_observation_data"]
    player = {
      "leader": leader(my["leader"]),
      "gate": gate(my["gate"]),
      "hand": card_zone(my["hand"]),
      "alley": board_zone(my["alley"]),
      "garden": board_zone(my["garden"]),
      "discard": card_zone(my["discard"]),
      "selection": board_zone(my["selection"]),
      "ikz_area": ikz_zone(my["ikz_area"]),
      "hand_count": ex(my["hand_count"]),
      "deck_count": ex(my["deck_count"]),
      "ikz_pile_count": ex(my["ikz_pile_count"]),
      "selection_count": ex(my["selection_count"]),
      "has_ikz_token": ex(my["has_ikz_token"]),
    }
    opponent = {
      "leader": leader(opp["leader"]),
      "gate": gate(opp["gate"]),
      "alley": board_zone(opp["alley"]),
      "garden": board_zone(opp["garden"]),
      "discard": card_zone(opp["discard"]),
      "ikz_area": ikz_zone(opp["ikz_area"]),
      "hand_count": ex(opp["hand_count"]),
      "deck_count": ex(opp["deck_count"]),
      "ikz_pile_count": ex(opp["ikz_pile_count"]),
      "has_ikz_token": ex(opp["has_ikz_token"]),
    }
    ability = sp["ability_context"]
    combat = sp["combat_context"]
    ra_fields = ("valid", "primary", "sub1", "sub2", "sub3", "was_noop")
    cp = sp["critic_privileged"]
    am = sp["action_mask"]
    cobs = {
      "player": player,
      "opponent": opponent,
      "phase": ex(sp["phase"]),
      "ability_context": {k: ex(ability[k]) for k in (
        "phase", "pending_confirmation_count", "has_source_card_def_id",
        "source_card_def_id", "cost_target_type", "effect_target_type",
        "selection_count", "selection_picked", "selection_pick_max",
        "active_player_index",
      )},
      "combat_context": {k: ex(combat[k]) for k in (
        "combat_active", "response_window_active", "defender_intercepted",
        "attacker_is_self", "attacker_is_leader", "attacker_is_garden",
        "attacker_is_alley", "attacker_card_def_id", "attacker_slot_index",
        "target_is_self", "target_is_leader", "target_is_garden",
        "target_is_alley", "target_card_def_id", "target_slot_index",
      )},
      "self_recent_actions": {k: ex(sp["self_recent_actions"][k]) for k in ra_fields},
      "opp_recent_actions": {k: ex(sp["opp_recent_actions"][k]) for k in ra_fields},
      "critic_privileged": {
        "opponent_hand": card_zone(cp["opponent_hand"]),
        "self_deck": card_zone(cp["self_deck"]),
        "opponent_deck": card_zone(cp["opponent_deck"]),
      },
      "action_mask": {
        "primary_action_mask": ex(am["primary_action_mask"]),
        "legal_action_count": ex(am["legal_action_count"]),
        "legal_primary": ex(am["legal_primary"]),
        "legal_sub1": ex(am["legal_sub1"]),
        "legal_sub2": ex(am["legal_sub2"]),
        "legal_sub3": ex(am["legal_sub3"]),
      },
    }
    if self.deck_context_enabled and "deck_context" in sp:
      dc = sp["deck_context"]
      cobs["deck_context"] = {
        "mode": ex(dc["mode"]),
        "gate_card_def_id": ex(dc["gate_card_def_id"]),
        "leader_card_def_id": ex(dc["leader_card_def_id"]),
        "main_card_def_ids": ex(dc["main_card_def_ids"]),
        "main_count": ex(dc["main_count"]),
        "candidate_card_def_ids": ex(dc["candidate_card_def_ids"]),
        "candidate_copy_counts": ex(dc["candidate_copy_counts"]),
        "candidate_count": ex(dc["candidate_count"]),
      }
    return cobs

  @staticmethod
  def _struct_get(container, *names):
    for name in names:
      if isinstance(container, dict) and name in container:
        return container[name]
    raise KeyError(f"Missing expected field. Tried: {names}")

  def _canonical_from_structured(self, structured):
    """Canonicalize the legacy emulated layout (nativize output or raw dict tree).

    Zone slots arrive as ordered containers (dict field order == dtype
    declaration order, or tuples); each field is stacked once per zone.
    """

    def sq(value):
      t = value if torch.is_tensor(value) else torch.as_tensor(value, device=self.__policy_device())
      while t.dim() > 1 and t.size(-1) == 1:
        t = t.squeeze(-1)
      if t.dim() == 0:
        t = t.reshape(1)
      return t

    def entries(zone):
      if isinstance(zone, dict):
        return list(zone.values())
      return list(zone)

    def stack_field(slots, name):
      t = torch.stack([torch.as_tensor(s[name]) for s in slots], dim=1)
      while t.dim() > 2 and t.size(-1) == 1:
        t = t.squeeze(-1)
      return t

    def weapons_of(node):
      w = entries(self._struct_get(node, "weapons"))
      return {
        "card_def_id": stack_field(w, "card_def_id"),
        "cur_atk": stack_field(w, "cur_atk"),
      }

    def zone_weapons(slots):
      per_slot = [weapons_of(s) for s in slots]
      return {
        "card_def_id": torch.stack([p["card_def_id"] for p in per_slot], dim=1),
        "cur_atk": torch.stack([p["cur_atk"] for p in per_slot], dim=1),
      }

    def leader(node):
      out = {k: sq(node[k]) for k in (
        "card_def_id", "tapped", "cooldown", "cur_atk", "cur_hp",
        "weapon_count", "has_charge", "has_defender", "has_infiltrate",
      )}
      out["weapons"] = weapons_of(node)
      return out

    def gate(node):
      return {k: sq(node[k]) for k in ("card_def_id", "tapped", "cooldown")}

    def card_zone(node):
      slots = entries(node)
      return {
        "card_def_id": stack_field(slots, "card_def_id"),
        "zone_index": stack_field(slots, "zone_index"),
      }

    def board_zone(node):
      slots = entries(node)
      out = {k: stack_field(slots, k) for k in (
        "card_def_id", "zone_index", "tapped", "cooldown", "has_cur_stats",
        "cur_atk", "cur_hp", "weapon_count", "has_charge", "has_defender",
        "has_infiltrate", "is_frozen", "is_shocked", "is_effect_immune",
      )}
      out["weapons"] = zone_weapons(slots)
      return out

    def ikz_zone(node):
      slots = entries(node)
      return {k: stack_field(slots, k) for k in (
        "card_def_id", "zone_index", "tapped", "cooldown",
      )}

    def recent(node):
      slots = entries(node)
      return {k: stack_field(slots, k) for k in (
        "valid", "primary", "sub1", "sub2", "sub3", "was_noop",
      )}

    my = self._struct_get(structured, "player", "my_observation_data")
    opp = self._struct_get(structured, "opponent", "opponent_observation_data")
    player = {
      "leader": leader(my["leader"]),
      "gate": gate(my["gate"]),
      "hand": card_zone(my["hand"]),
      "alley": board_zone(my["alley"]),
      "garden": board_zone(my["garden"]),
      "discard": card_zone(my["discard"]),
      "selection": board_zone(my["selection"]),
      "ikz_area": ikz_zone(my["ikz_area"]),
      **{k: sq(my[k]) for k in (
        "hand_count", "deck_count", "ikz_pile_count", "selection_count", "has_ikz_token",
      )},
    }
    opponent = {
      "leader": leader(opp["leader"]),
      "gate": gate(opp["gate"]),
      "alley": board_zone(opp["alley"]),
      "garden": board_zone(opp["garden"]),
      "discard": card_zone(opp["discard"]),
      "ikz_area": ikz_zone(opp["ikz_area"]),
      **{k: sq(opp[k]) for k in (
        "hand_count", "deck_count", "ikz_pile_count", "has_ikz_token",
      )},
    }
    ability = self._struct_get(structured, "ability_context")
    combat = self._struct_get(structured, "combat_context")
    cp = self._struct_get(structured, "critic_privileged")
    am = self._struct_get(structured, "action_mask")
    legal = self._struct_get(am, "legal_actions")
    cobs = {
      "player": player,
      "opponent": opponent,
      "phase": sq(self._struct_get(structured, "phase")),
      "ability_context": {k: sq(ability[k]) for k in (
        "phase", "pending_confirmation_count", "has_source_card_def_id",
        "source_card_def_id", "cost_target_type", "effect_target_type",
        "selection_count", "selection_picked", "selection_pick_max",
        "active_player_index",
      )},
      "combat_context": {k: sq(combat[k]) for k in (
        "combat_active", "response_window_active", "defender_intercepted",
        "attacker_is_self", "attacker_is_leader", "attacker_is_garden",
        "attacker_is_alley", "attacker_card_def_id", "attacker_slot_index",
        "target_is_self", "target_is_leader", "target_is_garden",
        "target_is_alley", "target_card_def_id", "target_slot_index",
      )},
      "self_recent_actions": recent(self._struct_get(structured, "self_recent_actions")),
      "opp_recent_actions": recent(self._struct_get(structured, "opp_recent_actions")),
      "critic_privileged": {
        "opponent_hand": card_zone(cp["opponent_hand"]),
        "self_deck": card_zone(cp["self_deck"]),
        "opponent_deck": card_zone(cp["opponent_deck"]),
      },
      "action_mask": {
        "primary_action_mask": am["primary_action_mask"]
        if torch.is_tensor(am["primary_action_mask"])
        else torch.as_tensor(am["primary_action_mask"]),
        "legal_action_count": sq(am["legal_action_count"]),
        "legal_primary": legal["legal_primary"],
        "legal_sub1": legal["legal_sub1"],
        "legal_sub2": legal["legal_sub2"],
        "legal_sub3": legal["legal_sub3"],
      },
    }
    if self.deck_context_enabled and isinstance(structured, dict) and "deck_context" in structured:
      dc = structured["deck_context"]
      cobs["deck_context"] = {
        "mode": sq(dc["mode"]),
        "gate_card_def_id": sq(dc["gate_card_def_id"]),
        "leader_card_def_id": sq(dc["leader_card_def_id"]),
        "main_card_def_ids": dc["main_card_def_ids"],
        "main_count": sq(dc["main_count"]),
        "candidate_card_def_ids": dc["candidate_card_def_ids"],
        "candidate_copy_counts": dc["candidate_copy_counts"],
        "candidate_count": sq(dc["candidate_count"]),
      }
    return cobs

  def _squeeze_trailing_singleton(self, tensor: torch.Tensor):
    # Packed scalar fields can be (B, 1), while canonical native fields are
    # already (B,). Never collapse the batch dimension when B == 1.
    if tensor.dim() > 1 and tensor.size(-1) == 1:
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
    }

  def _text_feature_table(self) -> torch.Tensor:
    """Per-card projected text features [vocab, name+effect+subtype dims].

    The text encoders are per-card functions of static tables, so encoding the
    whole vocab once per forward and gathering small rows is exactly
    equivalent to encoding per slot, but avoids materializing
    [batch, slots, 1536] activations for every zone.
    """
    if (
      self._text_table_cache is None
      or self._text_table_cache_version != self._text_table_version
    ):
      self._text_table_cache = torch.cat(
        [
          self.name_text_encoder(self.static_name_embeddings),
          self.effect_text_encoder(self.static_effect_embeddings),
          self.subtype_text_encoder(self.static_subtype_pooled_embeddings),
        ],
        dim=-1,
      )
      self._text_table_cache_version = self._text_table_version
    return self._text_table_cache

  def _invalidate_text_feature_table(self) -> None:
    self._text_table_version += 1
    self._text_table_cache = None

  def _index_embedding(self, zone_indices: torch.Tensor):
    return self.index_encoder(zone_indices.long().clamp(0, MAX_INDEX_SIZE - 1))

  def _metadata_embedding_table(self) -> torch.Tensor:
    """Per-card metadata embeddings [vocab, CARD_METADATA_EMBED_SIZE].

    Computed once per forward over the (tiny) card vocab; per-occurrence
    lookups then use nn.functional.embedding, whose backward is an optimized
    scatter. The previous per-occurrence formulation advanced-indexed a
    grad-requiring table with millions of indices per minibatch, and its
    index_put backward dominated the entire training step.
    """
    if (
      self._metadata_table_cache is not None
      and self._metadata_table_cache_version == self._text_table_version
    ):
      return self._metadata_table_cache

    # This path always builds the complete vocabulary in its canonical order.
    # Use the registered static tables directly instead of identity-gathering
    # every field through arange(static_vocab_size).
    static = {
      "present_mask": self.static_card_present_mask,
      "card_type": self.static_card_type,
      "element": self.static_element,
      "base_ikz_cost": self.static_base_ikz_cost,
      "base_attack": self.static_base_attack,
      "base_health": self.static_base_health,
      "base_gate_points": self.static_base_gate_points,
      "has_ability": self.static_has_ability,
      "ability_timing": self.static_ability_timing,
      "ability_optional": self.static_ability_optional,
      "keyword_multi_hot": self.static_keyword_multi_hot,
    }
    card_type_emb = self.card_type_encoder(static["card_type"])
    element_emb = self.element_encoder(static["element"])
    ability_timing_emb = self.ability_timing_encoder(static["ability_timing"])
    text_emb = self._text_feature_table()
    keyword_emb = self.keyword_feature_encoder(static["keyword_multi_hot"])

    scalar = self.static_card_scalar
    scalar = self.scalar_normalizer.normalize_only("card_metadata_scalar", scalar)

    metadata_input = torch.cat(
      [
        text_emb,
        keyword_emb,
        card_type_emb,
        element_emb,
        ability_timing_emb,
        scalar,
      ],
      dim=-1,
    )
    table = self.card_metadata_projector(metadata_input)
    self._metadata_table_cache = table
    self._metadata_table_cache_version = self._text_table_version
    return table

  def _encode_card_metadata_from_index(self, idx: torch.Tensor, valid_mask: torch.Tensor | None = None):
    present_mask = self.static_card_present_mask[idx] > 0.5
    if valid_mask is None:
      valid_mask = present_mask
    else:
      valid_mask = valid_mask.to(dtype=torch.bool) & present_mask

    if self.training:
      # Preserve the occurrence-weighted running-norm statistics of the
      # per-occurrence formulation (values are pure buffer gathers, no grad).
      with torch.no_grad():
        scalar = nn.functional.embedding(
          idx.reshape(-1), self.static_card_scalar
        ).view(*idx.shape, self.static_card_scalar.shape[-1])
        self.scalar_normalizer.update_only("card_metadata_scalar", scalar, mask=valid_mask)

    table = self._metadata_embedding_table()
    metadata_emb = nn.functional.embedding(idx.reshape(-1), table).view(
      *idx.shape, table.shape[-1]
    )

    if valid_mask.dim() < metadata_emb.dim():
      valid_mask = valid_mask.unsqueeze(-1)
    return metadata_emb * valid_mask.to(dtype=metadata_emb.dtype)

  def _encode_weapons(self, weapons, weapon_count):
    """Encode a weapons block of shape (..., W); leading dims are flattened.

    Called once per zone with all slots batched: (B, W) for leaders,
    (B, S, W) for board zones.
    """
    card_def_ids = weapons["card_def_id"]
    lead_shape = card_def_ids.shape[:-1]
    max_slots = card_def_ids.shape[-1]
    card_def_ids = card_def_ids.reshape(-1, max_slots)
    cur_atk = weapons["cur_atk"].reshape(-1, max_slots).float()
    weapon_count = weapon_count.reshape(-1)

    idx, valid_mask = self._card_index_and_mask(card_def_ids)
    static = self._lookup_static(idx)

    card_emb = self._encode_card_metadata_from_index(idx, valid_mask=valid_mask)

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

    slot_indices = torch.arange(max_slots, device=scalar.device).unsqueeze(0)
    count_mask = slot_indices < weapon_count.long().view(-1, 1)
    mask = valid_mask & count_mask
    scalar = self.scalar_normalizer("weapon_scalar", scalar, mask=mask)

    weapon_input = torch.cat([card_emb, scalar], dim=-1)
    _, pooled = self.weapon_set_processor(weapon_input, mask=mask)
    return pooled.reshape(*lead_shape, pooled.shape[-1])

  def _encode_hand_or_discard(self, zone, *, key_prefix: str, processor: ProcessSetProcessor):
    card_def_ids = zone["card_def_id"]
    zone_indices = zone["zone_index"]

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

  def _encode_ikz_area(self, zone):
    card_def_ids = zone["card_def_id"]
    zone_indices = zone["zone_index"]
    tapped = zone["tapped"].float()
    cooldown = zone["cooldown"].float()

    idx, mask = self._card_index_and_mask(card_def_ids)
    card_emb = self._encode_card_metadata_from_index(idx, valid_mask=mask)
    zone_emb = self._index_embedding(zone_indices)

    scalar = torch.stack([tapped, cooldown], dim=-1)
    scalar = self.scalar_normalizer("ikz_scalar", scalar, mask=mask)

    zone_input = torch.cat([card_emb, zone_emb, scalar], dim=-1)
    set_embeddings, pooled = self.ikz_set_processor(zone_input, mask=mask)
    return set_embeddings, pooled

  def _encode_critic_privileged_zone_tokens(self, zone, *, key_prefix: str):
    card_def_ids = zone["card_def_id"]
    zone_indices = zone["zone_index"]

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

  def _encode_board_zone(self, zone, *, key_prefix: str):
    card_def_ids = zone["card_def_id"]
    zone_indices = zone["zone_index"]
    tapped = zone["tapped"].float()
    cooldown = zone["cooldown"].float()
    has_cur_stats = zone["has_cur_stats"].float()
    # The legacy dict path zeroed cur stats for cards without them; replicate.
    cur_atk = zone["cur_atk"].float() * has_cur_stats
    cur_hp = zone["cur_hp"].float() * has_cur_stats
    has_charge = zone["has_charge"].float()
    has_defender = zone["has_defender"].float()
    has_infiltrate = zone["has_infiltrate"].float()
    is_frozen = zone["is_frozen"].float()
    is_shocked = zone["is_shocked"].float()
    is_effect_immune = zone["is_effect_immune"].float()
    weapon_count = zone["weapon_count"]

    idx, mask = self._card_index_and_mask(card_def_ids)
    static = self._lookup_static(idx)

    card_emb = self._encode_card_metadata_from_index(idx, valid_mask=mask)
    zone_emb = self._index_embedding(zone_indices)

    weapon_emb = self._encode_weapons(zone["weapons"], weapon_count)

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
    card_def_ids = leader_obs["card_def_id"].long()
    idx, valid_mask = self._card_index_and_mask(card_def_ids)
    static = self._lookup_static(idx)

    weapon_emb = self._encode_weapons(leader_obs["weapons"], leader_obs["weapon_count"])

    card_emb = self._encode_card_metadata_from_index(idx, valid_mask=valid_mask)

    scalar = torch.stack(
      [
        leader_obs["cur_atk"].float(),
        leader_obs["cur_hp"].float(),
        leader_obs["tapped"].float(),
        leader_obs["cooldown"].float(),
        leader_obs["has_charge"].float(),
        leader_obs["has_defender"].float(),
        leader_obs["has_infiltrate"].float(),
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
    card_def_ids = gate_obs["card_def_id"].long()
    idx, valid_mask = self._card_index_and_mask(card_def_ids)
    static = self._lookup_static(idx)

    card_emb = self._encode_card_metadata_from_index(idx, valid_mask=valid_mask)

    scalar = torch.stack(
      [
        gate_obs["tapped"].float(),
        gate_obs["cooldown"].float(),
        static["has_ability"],
        static["ability_optional"],
      ],
      dim=-1,
    )
    scalar = self.scalar_normalizer(f"{key_prefix}_gate_scalar", scalar)

    parts = [card_emb, scalar]
    if self.gate_id_embedding is not None:
      id_emb = self.gate_id_embedding(idx) * valid_mask.to(card_emb.dtype).unsqueeze(-1)
      parts.append(id_emb)
    gate_input = torch.cat(parts, dim=-1)
    return self.gate_projector(gate_input)

  def _count_matching_cards_per_slot(self, card_def_ids: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    same_card = card_def_ids.unsqueeze(1) == card_def_ids.unsqueeze(2)
    same_card = same_card & valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
    return same_card.sum(dim=-1).float()

  def _encode_deck_context(self, cobs):
    phase = cobs["phase"]
    batch_size = phase.shape[0]
    device = phase.device

    if not self.deck_context_enabled:
      return (
        torch.zeros((batch_size, UNIT_EMBED_SIZE), device=device, dtype=torch.float32),
        torch.zeros(
          (batch_size, MAX_DECK_BUILD_CANDIDATES, UNIT_EMBED_SIZE),
          device=device,
          dtype=torch.float32,
        ),
        torch.zeros((batch_size,), device=device, dtype=torch.long),
      )

    deck_context = cobs.get("deck_context")
    if deck_context is None:
      return (
        torch.zeros((batch_size, UNIT_EMBED_SIZE), device=device, dtype=torch.float32),
        torch.zeros(
          (batch_size, MAX_DECK_BUILD_CANDIDATES, UNIT_EMBED_SIZE),
          device=device,
          dtype=torch.float32,
        ),
        torch.zeros((batch_size,), device=device, dtype=torch.long),
      )

    mode = self._squeeze_trailing_singleton(deck_context["mode"]).long().clamp(0, DECK_CONTEXT_MODE_COUNT - 1)
    gate_card = self._squeeze_trailing_singleton(deck_context["gate_card_def_id"]).long()
    leader_card = self._squeeze_trailing_singleton(deck_context["leader_card_def_id"]).long()
    main_count = self._squeeze_trailing_singleton(deck_context["main_count"]).long().clamp(0, MAX_DECK_SIZE)
    candidate_count = self._squeeze_trailing_singleton(deck_context["candidate_count"]).long().clamp(
      0,
      MAX_DECK_BUILD_CANDIDATES,
    )

    gate_idx, gate_valid = self._card_index_and_mask(gate_card)
    leader_idx, leader_valid = self._card_index_and_mask(leader_card)
    gate_emb = self._encode_card_metadata_from_index(gate_idx, valid_mask=gate_valid)
    leader_emb = self._encode_card_metadata_from_index(leader_idx, valid_mask=leader_valid)

    main_card_ids = deck_context["main_card_def_ids"].long()
    if main_card_ids.dim() == 1:
      main_card_ids = main_card_ids.unsqueeze(0)
    main_idx, main_valid_by_id = self._card_index_and_mask(main_card_ids)
    main_positions = torch.arange(main_card_ids.size(1), device=device).unsqueeze(0)
    main_valid = main_valid_by_id & (main_positions < main_count.view(-1, 1))
    main_emb = self._encode_card_metadata_from_index(main_idx, valid_mask=main_valid)
    main_copy_counts = self._count_matching_cards_per_slot(main_card_ids, main_valid)
    main_scalar = torch.stack(
      [
        main_copy_counts / 4.0,
        main_valid.float(),
      ],
      dim=-1,
    )
    main_scalar = self.scalar_normalizer("deck_context_main_scalar", main_scalar, mask=main_valid)
    _, main_vec = self.deck_context_card_processor(
      torch.cat([main_emb, main_scalar], dim=-1),
      mask=main_valid,
    )

    candidate_card_ids = deck_context["candidate_card_def_ids"].long()
    if candidate_card_ids.dim() == 1:
      candidate_card_ids = candidate_card_ids.unsqueeze(0)
    candidate_idx, candidate_valid_by_id = self._card_index_and_mask(candidate_card_ids)
    candidate_positions = torch.arange(candidate_card_ids.size(1), device=device).unsqueeze(0)
    candidate_valid = candidate_valid_by_id & (candidate_positions < candidate_count.view(-1, 1))
    candidate_emb = self._encode_card_metadata_from_index(candidate_idx, valid_mask=candidate_valid)
    candidate_copy_counts = deck_context["candidate_copy_counts"].float()
    if candidate_copy_counts.dim() == 1:
      candidate_copy_counts = candidate_copy_counts.unsqueeze(0)
    candidate_scalar = torch.stack(
      [
        candidate_copy_counts / 4.0,
        candidate_valid.float(),
        (mode.view(-1, 1).expand_as(candidate_copy_counts) > 0).float(),
      ],
      dim=-1,
    )
    candidate_scalar = self.scalar_normalizer(
      "deck_context_candidate_scalar",
      candidate_scalar,
      mask=candidate_valid,
    )
    batch_size, candidate_slots, _ = candidate_emb.shape
    candidate_matrix = self.deck_candidate_projector(
      torch.cat([candidate_emb, candidate_scalar], dim=-1).reshape(batch_size * candidate_slots, -1)
    ).reshape(batch_size, candidate_slots, UNIT_EMBED_SIZE)
    candidate_matrix = candidate_matrix * candidate_valid.unsqueeze(-1).to(dtype=candidate_matrix.dtype)

    mode_emb = self.deck_mode_encoder(mode)
    scalar = torch.stack(
      [
        main_count.float(),
        candidate_count.float(),
        gate_valid.float(),
        leader_valid.float(),
        mode.float(),
      ],
      dim=-1,
    )
    scalar = self.scalar_normalizer("deck_context_scalar", scalar)
    context_parts = [mode_emb, gate_emb, leader_emb, main_vec, scalar]
    if self.gate_id_embedding is not None:
      gate_id_emb = self.gate_id_embedding(gate_idx) * gate_valid.to(gate_emb.dtype).unsqueeze(-1)
      context_parts.insert(2, gate_id_emb)
    deck_context_vec = self.deck_context_projector(torch.cat(context_parts, dim=-1))
    return deck_context_vec, candidate_matrix, candidate_count

  def _encode_global_context(self, cobs):
    ability = cobs["ability_context"]
    phase = cobs["phase"].long().clamp(0, GAME_PHASE_COUNT - 1)
    ability_phase = ability["phase"].long().clamp(0, ABILITY_PHASE_COUNT - 1)

    source_card = ability["source_card_def_id"].long()
    source_idx, source_valid = self._card_index_and_mask(source_card)

    phase_emb = self.game_phase_encoder(phase)
    ability_phase_emb = self.ability_phase_encoder(ability_phase)
    source_emb = self._encode_card_metadata_from_index(source_idx, valid_mask=source_valid)

    is_active = (cobs["action_mask"]["legal_action_count"].long() > 0).float()

    scalar = torch.stack(
      [
        ability["pending_confirmation_count"].float(),
        ability["has_source_card_def_id"].float(),
        ability["cost_target_type"].float(),
        ability["effect_target_type"].float(),
        ability["selection_count"].float(),
        ability["selection_picked"].float(),
        ability["selection_pick_max"].float(),
        ability["active_player_index"].float(),
        is_active,
      ],
      dim=-1,
    )
    scalar = self.scalar_normalizer("global_context_scalar", scalar)

    context_input = torch.cat([phase_emb, ability_phase_emb, source_emb, scalar], dim=-1)
    return self.global_context_projector(context_input)

  def _encode_recent_action_sequence(self, recent_actions, *, key_prefix: str):
    valid = recent_actions["valid"].to(dtype=torch.bool)
    primary = recent_actions["primary"].long()
    sub1 = recent_actions["sub1"].long()
    sub2 = recent_actions["sub2"].long()
    sub3 = recent_actions["sub3"].long()
    was_noop = recent_actions["was_noop"].float()

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

  def _encode_recent_action_history(self, cobs):
    phase = cobs["phase"]

    self_recent_actions = cobs.get("self_recent_actions")
    opp_recent_actions = cobs.get("opp_recent_actions")
    if self_recent_actions is None or opp_recent_actions is None:
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

  def _encode_combat_context(self, cobs):
    phase = cobs["phase"]

    combat_context = cobs.get("combat_context")
    if combat_context is None:
      return torch.zeros(
        (phase.shape[0], UNIT_EMBED_SIZE),
        device=phase.device,
        dtype=torch.float32,
      )

    attacker_card = combat_context["attacker_card_def_id"].long()
    target_card = combat_context["target_card_def_id"].long()
    attacker_idx, attacker_valid = self._card_index_and_mask(attacker_card)
    target_idx, target_valid = self._card_index_and_mask(target_card)
    attacker_card_emb = self._encode_card_metadata_from_index(
      attacker_idx, valid_mask=attacker_valid
    )
    target_card_emb = self._encode_card_metadata_from_index(
      target_idx, valid_mask=target_valid
    )

    attacker_slot_idx = combat_context["attacker_slot_index"]
    target_slot_idx = combat_context["target_slot_index"]
    attacker_slot_emb = self._index_embedding(attacker_slot_idx)
    target_slot_emb = self._index_embedding(target_slot_idx)
    attacker_slot_emb = attacker_slot_emb * attacker_valid.unsqueeze(-1).to(dtype=attacker_slot_emb.dtype)
    target_slot_emb = target_slot_emb * target_valid.unsqueeze(-1).to(dtype=target_slot_emb.dtype)

    combat_active = combat_context["combat_active"].to(dtype=torch.bool)
    scalar = torch.stack(
      [
        combat_active.float(),
        combat_context["response_window_active"].float(),
        combat_context["defender_intercepted"].float(),
        combat_context["attacker_is_self"].float(),
        combat_context["attacker_is_leader"].float(),
        combat_context["attacker_is_garden"].float(),
        combat_context["attacker_is_alley"].float(),
        attacker_slot_idx.float(),
        combat_context["target_is_self"].float(),
        combat_context["target_is_leader"].float(),
        combat_context["target_is_garden"].float(),
        combat_context["target_is_alley"].float(),
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

  def _encode_critic_privileged_features(self, cobs):
    if not self.privileged_critic_enabled:
      return None

    phase = cobs["phase"]

    critic_privileged = cobs.get("critic_privileged")
    if critic_privileged is None:
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
    if self.training:
      self._invalidate_text_feature_table()
    cobs, squeeze_batch, obs_tensor = self._canonical_observations(observations)
    self.__store_mask_observations(
      obs_tensor if obs_tensor is not None else observations, state
    )
    self._cached_cobs = cobs
    # Cache the legal-action trim bucket for decode_actions. The .item() sync
    # is nearly free here (only H2D copies are enqueued); in decode_actions it
    # would stall the CPU behind the whole encode+LSTM launch queue. The
    # override is set during CUDA-graph capture/replay, where a sync inside
    # the captured region is illegal and the bucket is chosen by the caller.
    if self._trim_bucket_override is not None:
      self._cached_trim_bucket = self._trim_bucket_override
    else:
      self._cached_trim_bucket = self._compute_legal_action_trim_bucket(
        cobs["action_mask"]["legal_action_count"]
      )
    if state is not None:
      try:
        state["_azk_cobs"] = cobs
      except (TypeError, AttributeError):
        pass
    self.__store_privileged_critic_features(
      self._encode_critic_privileged_features(cobs),
      state,
    )

    player = cobs["player"]
    opponent = cobs["opponent"]

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
    deck_context_vec, deck_candidate_matrix, deck_candidate_count = self._encode_deck_context(cobs)

    global_counts = torch.stack(
      [
        player["hand_count"].float(),
        player["deck_count"].float(),
        player["ikz_pile_count"].float(),
        player["selection_count"].float(),
        player["has_ikz_token"].float(),
        opponent["hand_count"].float(),
        opponent["deck_count"].float(),
        opponent["ikz_pile_count"].float(),
        opponent["has_ikz_token"].float(),
      ],
      dim=-1,
    )
    global_counts = self.scalar_normalizer("global_counts", global_counts)
    global_counts_vec = self.global_counts_projector(global_counts)
    global_context_vec = self._encode_global_context(cobs)
    combat_context_vec = self._encode_combat_context(cobs)
    recent_action_history_vec = self._encode_recent_action_history(cobs)
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

    target_vector_components = [
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
    ]
    if self.deck_context_enabled:
      target_vector_components.append(deck_context_vec)
    target_vector_components.append(global_vec)
    target_vector = torch.cat(target_vector_components, dim=-1)

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
      "player_garden_or_leader_matrix": torch.cat(
        [player_garden_matrix, player_leader_vec.unsqueeze(-2)],
        dim=-2,
      ),
      "opponent_defender_matrix": torch.cat(
        [
          opponent_garden_matrix,
          opponent_leader_vec.unsqueeze(-2),
          opponent_alley_matrix,
        ],
        dim=-2,
      ),
    }
    if self.deck_context_enabled:
      action_context["deck_candidate_matrix"] = deck_candidate_matrix
      action_context["deck_candidate_count"] = deck_candidate_count

    if squeeze_batch:
      target_vector = target_vector.squeeze(0)
    return target_vector, action_context

  def build_primary_action_mask_tensor(self, observations):
    cobs, squeeze_batch, _ = self._canonical_observations(observations)
    primary_mask = cobs["action_mask"]["primary_action_mask"].to(dtype=torch.bool)
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
      if key == "deck_candidate_count":
        if tensor.dim() == 0:
          tensor = tensor.unsqueeze(0)
        prepared[key] = tensor.reshape(-1)
        continue
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
    combined_matrix: torch.Tensor,
    indices: torch.Tensor,
  ) -> torch.Tensor:
    refs = self._gather_zone_rows(combined_matrix, indices.clamp(0, GARDEN_SIZE))
    valid = (indices <= GARDEN_SIZE).unsqueeze(-1)
    return refs * valid.to(dtype=refs.dtype)

  def _gather_opponent_defender_refs(
    self,
    combined_matrix: torch.Tensor,
    indices: torch.Tensor,
  ) -> torch.Tensor:
    refs = self._gather_zone_rows(
      combined_matrix,
      indices.clamp(0, GARDEN_SIZE + ALLEY_SIZE),
    )
    valid = (indices <= (GARDEN_SIZE + ALLEY_SIZE)).unsqueeze(-1)
    return refs * valid.to(dtype=refs.dtype)

  def _legal_action_arg_kinds(self, primary: torch.Tensor) -> torch.Tensor:
    clamped_primary = primary.clamp(0, PRIMARY_ACTION_COUNT - 1)
    return self.legal_action_arg_kind_table[clamped_primary]

  def _gather_legal_action_refs(self, action_context: dict, arg_kinds: torch.Tensor, subactions: torch.Tensor):
    hand_matrix = self._lookup_action_context_tensor(action_context, "hand_matrix")
    player_garden_matrix = self._lookup_action_context_tensor(action_context, "player_garden_matrix")
    player_alley_matrix = self._lookup_action_context_tensor(action_context, "player_alley_matrix")
    player_selection_matrix = self._lookup_action_context_tensor(action_context, "player_selection_matrix")
    player_garden_or_leader_matrix = self._lookup_action_context_tensor(
      action_context,
      "player_garden_or_leader_matrix",
    )
    opponent_defender_matrix = self._lookup_action_context_tensor(
      action_context,
      "opponent_defender_matrix",
    )
    deck_candidate_matrix = action_context.get("deck_candidate_matrix")
    deck_candidate_count = action_context.get("deck_candidate_count")
    has_deck_candidates = torch.is_tensor(deck_candidate_matrix) and torch.is_tensor(deck_candidate_count)

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

      # Each kind is gathered unconditionally: gating on `mask.any()` costs a
      # GPU->CPU sync per branch (up to 21 per forward), which stalls the launch
      # pipeline and breaks CUDA-graph/compile capture. The gathers are cheap.
      hand_mask = component_kind == LEGAL_ACTION_ARG_KIND_HAND
      hand_refs = self._gather_zone_rows(hand_matrix, component_subaction)
      component_ref = torch.where(hand_mask.unsqueeze(-1), hand_refs, component_ref)
      component_has_ref |= hand_mask

      self_garden_mask = component_kind == LEGAL_ACTION_ARG_KIND_SELF_GARDEN
      garden_refs = self._gather_zone_rows(player_garden_matrix, component_subaction)
      component_ref = torch.where(self_garden_mask.unsqueeze(-1), garden_refs, component_ref)
      component_has_ref |= self_garden_mask

      self_alley_mask = component_kind == LEGAL_ACTION_ARG_KIND_SELF_ALLEY
      alley_refs = self._gather_zone_rows(player_alley_matrix, component_subaction)
      component_ref = torch.where(self_alley_mask.unsqueeze(-1), alley_refs, component_ref)
      component_has_ref |= self_alley_mask

      selection_mask = component_kind == LEGAL_ACTION_ARG_KIND_SELECTION
      selection_refs = self._gather_zone_rows(player_selection_matrix, component_subaction)
      component_ref = torch.where(selection_mask.unsqueeze(-1), selection_refs, component_ref)
      component_has_ref |= selection_mask

      if has_deck_candidates:
        candidate_mask = component_kind == LEGAL_ACTION_ARG_KIND_CARD_CANDIDATE
        candidate_refs = self._gather_zone_rows(deck_candidate_matrix, component_subaction)
        component_ref = torch.where(candidate_mask.unsqueeze(-1), candidate_refs, component_ref)
        candidate_valid = component_subaction < deck_candidate_count.view(-1, 1)
        component_has_ref |= candidate_mask & candidate_valid

      self_garden_or_leader_mask = component_kind == LEGAL_ACTION_ARG_KIND_SELF_GARDEN_OR_LEADER
      garden_or_leader_refs = self._gather_garden_or_leader_refs(
        player_garden_or_leader_matrix,
        component_subaction,
      )
      component_ref = torch.where(
        self_garden_or_leader_mask.unsqueeze(-1),
        garden_or_leader_refs,
        component_ref,
      )
      component_has_ref |= self_garden_or_leader_mask & (component_subaction <= GARDEN_SIZE)

      opp_defender_mask = component_kind == LEGAL_ACTION_ARG_KIND_OPP_DEFENDER
      defender_refs = self._gather_opponent_defender_refs(
        opponent_defender_matrix,
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
    candidate_component = arg_kinds == LEGAL_ACTION_ARG_KIND_CARD_CANDIDATE
    subaction_idx = subactions.clamp(0, MAX_INDEX_SIZE - 1) + 1
    subaction_idx = torch.where(candidate_component, torch.zeros_like(subaction_idx), subaction_idx)
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
    normalized_subactions = torch.where(
      candidate_component,
      torch.zeros_like(normalized_subactions),
      normalized_subactions,
    )
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

  def _compute_legal_action_trim_bucket(self, legal_action_count: torch.Tensor) -> int:
    # Round the trim up to a power of two so torch.compile / CUDA graphs see a
    # bounded set of shapes (<=6 variants) instead of one per legal count.
    max_active_rows = int(legal_action_count.long().max().item()) if legal_action_count.numel() > 0 else 0
    active_candidate_count = max(1, max_active_rows)
    bucket = 32
    while bucket < active_candidate_count:
      bucket *= 2
    return bucket

  def _trim_active_legal_action_candidates(
    self,
    legal_actions: torch.Tensor,
    legal_action_count: torch.Tensor,
  ) -> torch.Tensor:
    # Legal rows are packed at the front of the padded 1024-row table. Trim the
    # batch to the active prefix so we do not embed/project rows that cannot be sampled.
    # The bucket is normally computed (and synced) once in encode_observations,
    # where the GPU queue is still short; syncing here would drain the whole
    # enqueued encode+LSTM graph.
    bucket = getattr(self, "_cached_trim_bucket", None)
    if bucket is None:
      bucket = self._compute_legal_action_trim_bucket(legal_action_count)
    active_candidate_count = min(bucket, legal_actions.size(1))
    return legal_actions[:, :active_candidate_count]

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
    active_legal_actions = self._trim_active_legal_action_candidates(
      legal_actions,
      legal_action_count,
    )
    candidate_embeddings = self._build_legal_action_candidate_embeddings(
      flat_hidden,
      action_context,
      active_legal_actions,
    )
    logits = (candidate_embeddings * legal_action_query.unsqueeze(1)).sum(dim=-1)
    logits = logits + self.legal_action_candidate_bias(candidate_embeddings).squeeze(-1)
    return TCGLegalActionDistribution(
      legal_action_logits=logits,
      legal_actions=active_legal_actions,
      legal_action_count=legal_action_count,
    )

  def decode_actions(self, flat_hidden, action_context=None, state=None):
    if flat_hidden.dim() == 1:
      flat_hidden = flat_hidden.unsqueeze(0)

    cobs = state.get("_azk_cobs") if state is not None else None
    if cobs is None:
      cobs = getattr(self, "_cached_cobs", None)
    if cobs is None:
      mask_observations = (
        state.get("_azk_mask_observations") if state is not None else self._cached_mask_observations
      )
      if mask_observations is None:
        raise ValueError("mask_observations is None when decode_actions is called")
      cobs, _, _ = self._canonical_observations(mask_observations)

    action_mask_struct = cobs["action_mask"]

    device = flat_hidden.device
    action_context = self._prepare_action_context(action_context, device=device)
    target_matrix = self._lookup_action_context_tensor(action_context, "legacy_target_matrix")
    primary_action_mask = action_mask_struct["primary_action_mask"].to(dtype=torch.bool, device=device)

    legal_actions = torch.stack(
      (
        action_mask_struct["legal_primary"].long(),
        action_mask_struct["legal_sub1"].long(),
        action_mask_struct["legal_sub2"].long(),
        action_mask_struct["legal_sub3"].long(),
      ),
      dim=-1,
    ).to(device=device)
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
    policy_config.get("actor_head_type", ACTOR_HEAD_TYPE_LEGAL_ACTION_SCORER)
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
  gate_id_embedding_enabled = bool(policy_config.get("gate_id_embedding_enabled", False))

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
      gate_id_embedding_enabled=gate_id_embedding_enabled,
      **kwargs,
    )

  raise ValueError(
    f"Unsupported v2 policy model_version '{model_version}'. "
    f"Known versions: {POLICY_MODEL_VERSION_METADATA_V1}"
  )
