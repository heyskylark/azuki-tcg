import pufferlib.models
import pufferlib.pytorch
from gymnasium.wrappers.normalize import RunningMeanStd

import torch
from torch import nn

from observation import (
  ACTION_TYPE_COUNT,
  MAX_ATTACHED_WEAPONS,
  MAX_DECK_SIZE,
  MAX_HAND_SIZE,
  MAX_SELECTION_ZONE_SIZE,
)
from policy.static_card_table import load_policy_static_card_table
from policy.tcg_distribution import TCGActionDistribution

MAX_PLAYERS_PER_MATCH = 2
CARD_TYPE_COUNT = 7
GAME_PHASE_COUNT = 8
ABILITY_PHASE_COUNT = 6
PRIMARY_ACTION_COUNT = ACTION_TYPE_COUNT
PRIMARY_ACTION_ID_BATCH = tuple(range(PRIMARY_ACTION_COUNT))
MAX_INDEX_SIZE = 50
ACTION_COMPONENT_COUNT = 4
ACT_NOOP = 0
ACT_ATTACK = 6
ACT_ATTACH_WEAPON_FROM_HAND = 7

CARD_DEF_ENC_OUTPUT_SIZE = 16
CARD_TYPE_ENC_OUTPUT_SIZE = 4
ABILITY_TIMING_ENC_OUTPUT_SIZE = 4
INDEX_ENC_OUTPUT_SIZE = 8
PHASE_ENC_OUTPUT_SIZE = 4
ABILITY_PHASE_ENC_OUTPUT_SIZE = 4
UNIT_EMBED_SIZE = 64

PROCESS_SET_HIDDEN_SIZE = 128
LSTM_HIDDEN_SIZE = 4096


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


class TCGLSTM(pufferlib.models.LSTMWrapper):
  def __init__(self, env, policy, input_size=None, hidden_size=LSTM_HIDDEN_SIZE):
    if input_size is None:
      input_size = policy.lstm_input_size
    super().__init__(env, policy, input_size, hidden_size)

  def _split_encoded(self, encoded):
    if isinstance(encoded, tuple) and len(encoded) == 2:
      return encoded
    return encoded, None

  def forward_eval(self, observations, state):
    lstm_inputs, target_matrix = self._split_encoded(
      self.policy.encode_observations(observations, state=state)
    )
    h = state.get("lstm_h")
    c = state.get("lstm_c")

    if h is not None:
      assert h.shape[0] == c.shape[0] == observations.shape[0], "LSTM state must be (h, c)"
      lstm_state = (h, c)
    else:
      lstm_state = None

    hidden, c = self.cell(lstm_inputs, lstm_state)
    state["hidden"] = hidden
    state["lstm_h"] = hidden
    state["lstm_c"] = c
    logits, values = self.policy.decode_actions(hidden, target_matrix=target_matrix, state=state)
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
    lstm_inputs, target_matrix = self._split_encoded(self.policy.encode_observations(x, state))

    hidden = lstm_inputs.reshape(B, TT, self.input_size).transpose(0, 1)
    hidden, (lstm_h, lstm_c) = self.lstm.forward(hidden, lstm_state)
    hidden = hidden.float().transpose(0, 1)

    flat_hidden = hidden.reshape(B * TT, self.hidden_size)
    logits, values = self.policy.decode_actions(flat_hidden, target_matrix=target_matrix, state=state)
    values = values.reshape(B, TT)
    state["hidden"] = hidden
    state["lstm_h"] = lstm_h.detach()
    state["lstm_c"] = lstm_c.detach()
    return logits, values


class TCG(nn.Module):
  def __init__(self, env, **kwargs):
    super().__init__()

    self.is_continuous = False
    self.scalar_normalizer = ScalarRunningNorm()

    static_table = load_policy_static_card_table()
    self.static_vocab_size = static_table.vocab_size
    self.register_buffer("static_card_type", static_table.card_type)
    self.register_buffer("static_base_ikz_cost", static_table.base_ikz_cost)
    self.register_buffer("static_base_attack", static_table.base_attack)
    self.register_buffer("static_base_health", static_table.base_health)
    self.register_buffer("static_base_gate_points", static_table.base_gate_points)
    self.register_buffer("static_innate_charge", static_table.innate_has_charge)
    self.register_buffer("static_innate_defender", static_table.innate_has_defender)
    self.register_buffer("static_innate_infiltrate", static_table.innate_has_infiltrate)
    self.register_buffer("static_has_ability", static_table.has_ability)
    self.register_buffer("static_ability_timing", static_table.ability_timing)
    self.register_buffer("static_ability_optional", static_table.ability_is_optional)

    ability_timing_vocab = int(self.static_ability_timing.max().item()) + 1
    ability_timing_vocab = max(ability_timing_vocab, 1)

    self.card_def_encoder = nn.Embedding(self.static_vocab_size, CARD_DEF_ENC_OUTPUT_SIZE)
    self.card_type_encoder = nn.Embedding(CARD_TYPE_COUNT, CARD_TYPE_ENC_OUTPUT_SIZE)
    self.ability_timing_encoder = nn.Embedding(ability_timing_vocab, ABILITY_TIMING_ENC_OUTPUT_SIZE)
    self.index_encoder = nn.Embedding(MAX_INDEX_SIZE, INDEX_ENC_OUTPUT_SIZE)
    self.game_phase_encoder = nn.Embedding(GAME_PHASE_COUNT, PHASE_ENC_OUTPUT_SIZE)
    self.ability_phase_encoder = nn.Embedding(ABILITY_PHASE_COUNT, ABILITY_PHASE_ENC_OUTPUT_SIZE)

    self.primary_action_encoder = nn.Sequential(
      nn.Embedding(PRIMARY_ACTION_COUNT, UNIT_EMBED_SIZE),
      nn.Flatten(),
    )
    self.register_buffer("primary_action_id_batch", torch.tensor(PRIMARY_ACTION_ID_BATCH, dtype=torch.long))

    weapon_input_size = (
      CARD_DEF_ENC_OUTPUT_SIZE
      + CARD_TYPE_ENC_OUTPUT_SIZE
      + ABILITY_TIMING_ENC_OUTPUT_SIZE
      + 5
    )
    self.weapon_set_processor = ProcessSetProcessor(weapon_input_size)

    hand_input_size = (
      CARD_DEF_ENC_OUTPUT_SIZE
      + CARD_TYPE_ENC_OUTPUT_SIZE
      + ABILITY_TIMING_ENC_OUTPUT_SIZE
      + INDEX_ENC_OUTPUT_SIZE
      + 9
    )
    self.hand_set_processor = ProcessSetProcessor(hand_input_size)
    self.discard_set_processor = ProcessSetProcessor(hand_input_size)

    ikz_input_size = (
      CARD_DEF_ENC_OUTPUT_SIZE
      + CARD_TYPE_ENC_OUTPUT_SIZE
      + ABILITY_TIMING_ENC_OUTPUT_SIZE
      + INDEX_ENC_OUTPUT_SIZE
      + 2
    )
    self.ikz_set_processor = ProcessSetProcessor(ikz_input_size)

    board_input_size = (
      CARD_DEF_ENC_OUTPUT_SIZE
      + CARD_TYPE_ENC_OUTPUT_SIZE
      + ABILITY_TIMING_ENC_OUTPUT_SIZE
      + INDEX_ENC_OUTPUT_SIZE
      + UNIT_EMBED_SIZE
      + 17
    )
    self.board_set_processor = ProcessSetProcessor(board_input_size)

    leader_input_size = (
      CARD_DEF_ENC_OUTPUT_SIZE
      + CARD_TYPE_ENC_OUTPUT_SIZE
      + ABILITY_TIMING_ENC_OUTPUT_SIZE
      + UNIT_EMBED_SIZE
      + 12
    )
    gate_input_size = CARD_DEF_ENC_OUTPUT_SIZE + CARD_TYPE_ENC_OUTPUT_SIZE + ABILITY_TIMING_ENC_OUTPUT_SIZE + 4

    self.leader_projector = SingleUnitProjection(leader_input_size)
    self.gate_projector = SingleUnitProjection(gate_input_size)

    context_input_size = (
      PHASE_ENC_OUTPUT_SIZE
      + ABILITY_PHASE_ENC_OUTPUT_SIZE
      + CARD_DEF_ENC_OUTPUT_SIZE
      + 9
    )
    self.global_context_projector = SingleUnitProjection(context_input_size)

    self.zone_component_count = 14
    self.lstm_input_size = UNIT_EMBED_SIZE * (self.zone_component_count + 1)

    self.q_primary = nn.Linear(LSTM_HIDDEN_SIZE, UNIT_EMBED_SIZE)
    self.q_unit1 = nn.Linear(LSTM_HIDDEN_SIZE, UNIT_EMBED_SIZE)
    self.q_unit2 = nn.Linear(LSTM_HIDDEN_SIZE, UNIT_EMBED_SIZE)
    self.q_bins2 = nn.Linear(LSTM_HIDDEN_SIZE, MAX_INDEX_SIZE)
    self.q_bins3 = nn.Linear(LSTM_HIDDEN_SIZE, MAX_INDEX_SIZE)
    self.gate_1_embeder = nn.Embedding(PRIMARY_ACTION_COUNT, UNIT_EMBED_SIZE)
    self.gate_2_embeder = nn.Embedding(PRIMARY_ACTION_COUNT, UNIT_EMBED_SIZE)
    self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(UNIT_EMBED_SIZE, 1), std=1)

    emulated_spec = getattr(env, "emulated", None)
    if emulated_spec is None:
      raise AttributeError("env must expose emulated metadata for nativize")
    self._obs_struct_dtype = pufferlib.pytorch.nativize_dtype(emulated_spec)
    self._cached_mask_observations = None

  def __policy_device(self) -> torch.device:
    sample_param = next(self.parameters(), None)
    if sample_param is not None:
      return sample_param.device
    return torch.device("cpu")

  def __prepare_structured_observations(self, observations):
    obs_tensor = observations if torch.is_tensor(observations) else torch.as_tensor(observations)
    squeeze_batch = obs_tensor.dim() == 1
    if squeeze_batch:
      obs_tensor = obs_tensor.unsqueeze(0)
    obs_tensor = obs_tensor.to(self.__policy_device())
    structured_obs = pufferlib.pytorch.nativize_tensor(obs_tensor, self._obs_struct_dtype)
    return structured_obs, squeeze_batch, obs_tensor

  def __store_mask_observations(self, obs_tensor: torch.Tensor, state):
    detached = obs_tensor.detach()
    if state is not None:
      try:
        state["_azk_mask_observations"] = detached
      except (TypeError, AttributeError):
        pass
    self._cached_mask_observations = detached

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

  def _card_index_and_mask(self, card_def_ids: torch.Tensor):
    card_def_ids = card_def_ids.long()
    valid_mask = card_def_ids >= 0
    idx = (card_def_ids + 1).clamp(0, self.static_vocab_size - 1)
    return idx, valid_mask

  def _lookup_static(self, idx: torch.Tensor):
    return {
      "card_type": self.static_card_type[idx],
      "base_ikz_cost": self.static_base_ikz_cost[idx],
      "base_attack": self.static_base_attack[idx],
      "base_health": self.static_base_health[idx],
      "base_gate_points": self.static_base_gate_points[idx],
      "innate_charge": self.static_innate_charge[idx],
      "innate_defender": self.static_innate_defender[idx],
      "innate_infiltrate": self.static_innate_infiltrate[idx],
      "has_ability": self.static_has_ability[idx],
      "ability_timing": self.static_ability_timing[idx],
      "ability_optional": self.static_ability_optional[idx],
    }

  def _index_embedding(self, zone_indices: torch.Tensor):
    return self.index_encoder(zone_indices.long().clamp(0, MAX_INDEX_SIZE - 1))

  def _encode_weapons(self, weapon_slots, weapon_count):
    weapon_slots = self.__normalize_zone_entries(weapon_slots)
    card_def_ids = self.__stack_zone_field(weapon_slots, "card_def_id")
    idx, valid_mask = self._card_index_and_mask(card_def_ids)
    static = self._lookup_static(idx)

    card_emb = self.card_def_encoder(idx)
    type_emb = self.card_type_encoder(static["card_type"])
    timing_emb = self.ability_timing_encoder(static["ability_timing"])

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

    weapon_input = torch.cat([card_emb, type_emb, timing_emb, scalar], dim=-1)
    _, pooled = self.weapon_set_processor(weapon_input, mask=mask)
    return pooled

  def _encode_hand_or_discard(self, slots, *, key_prefix: str, processor: ProcessSetProcessor):
    slots = self.__normalize_zone_entries(slots)
    card_def_ids = self.__stack_zone_field(slots, "card_def_id")
    zone_indices = self.__stack_zone_field(slots, "zone_index")

    idx, mask = self._card_index_and_mask(card_def_ids)
    static = self._lookup_static(idx)

    card_emb = self.card_def_encoder(idx)
    type_emb = self.card_type_encoder(static["card_type"])
    timing_emb = self.ability_timing_encoder(static["ability_timing"])
    zone_emb = self._index_embedding(zone_indices)

    scalar = torch.stack(
      [
        static["base_ikz_cost"],
        static["base_attack"],
        static["base_health"],
        static["base_gate_points"],
        static["innate_charge"],
        static["innate_defender"],
        static["innate_infiltrate"],
        static["has_ability"],
        static["ability_optional"],
      ],
      dim=-1,
    )
    scalar = self.scalar_normalizer(f"{key_prefix}_scalar", scalar, mask=mask)

    zone_input = torch.cat([card_emb, type_emb, timing_emb, zone_emb, scalar], dim=-1)
    set_embeddings, pooled = processor(zone_input, mask=mask)
    return set_embeddings, pooled

  def _encode_ikz_area(self, slots):
    slots = self.__normalize_zone_entries(slots)
    card_def_ids = self.__stack_zone_field(slots, "card_def_id")
    zone_indices = self.__stack_zone_field(slots, "zone_index")
    tapped = self.__stack_zone_field(slots, "tapped").float()
    cooldown = self.__stack_zone_field(slots, "cooldown").float()

    idx, mask = self._card_index_and_mask(card_def_ids)
    static = self._lookup_static(idx)

    card_emb = self.card_def_encoder(idx)
    type_emb = self.card_type_encoder(static["card_type"])
    timing_emb = self.ability_timing_encoder(static["ability_timing"])
    zone_emb = self._index_embedding(zone_indices)

    scalar = torch.stack([tapped, cooldown], dim=-1)
    scalar = self.scalar_normalizer("ikz_scalar", scalar, mask=mask)

    zone_input = torch.cat([card_emb, type_emb, timing_emb, zone_emb, scalar], dim=-1)
    set_embeddings, pooled = self.ikz_set_processor(zone_input, mask=mask)
    return set_embeddings, pooled

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

    card_emb = self.card_def_encoder(idx)
    type_emb = self.card_type_encoder(static["card_type"])
    timing_emb = self.ability_timing_encoder(static["ability_timing"])
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

    zone_input = torch.cat([card_emb, type_emb, timing_emb, zone_emb, weapon_emb, scalar], dim=-1)
    set_embeddings, pooled = self.board_set_processor(zone_input, mask=mask)
    return set_embeddings, pooled

  def _encode_leader(self, leader_obs, *, key_prefix: str):
    card_def_ids = leader_obs["card_def_id"].long()
    idx, _ = self._card_index_and_mask(card_def_ids)
    static = self._lookup_static(idx)

    weapon_emb = self._encode_weapons(leader_obs["weapons"], leader_obs["weapon_count"])

    card_emb = self.card_def_encoder(idx)
    type_emb = self.card_type_encoder(static["card_type"])
    timing_emb = self.ability_timing_encoder(static["ability_timing"])

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
        static["innate_charge"],
        static["innate_defender"],
        static["innate_infiltrate"],
      ],
      dim=-1,
    )
    scalar = self.scalar_normalizer(f"{key_prefix}_leader_scalar", scalar)

    leader_input = torch.cat([card_emb, type_emb, timing_emb, weapon_emb, scalar], dim=-1)
    return self.leader_projector(leader_input)

  def _encode_gate(self, gate_obs, *, key_prefix: str):
    card_def_ids = gate_obs["card_def_id"].long()
    idx, _ = self._card_index_and_mask(card_def_ids)
    static = self._lookup_static(idx)

    card_emb = self.card_def_encoder(idx)
    type_emb = self.card_type_encoder(static["card_type"])
    timing_emb = self.ability_timing_encoder(static["ability_timing"])

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

    gate_input = torch.cat([card_emb, type_emb, timing_emb, scalar], dim=-1)
    return self.gate_projector(gate_input)

  def _encode_global_context(self, structured_obs):
    ability = self.__get_struct_field(structured_obs, "ability_context")
    phase = self.__get_struct_field(structured_obs, "phase").long().clamp(0, GAME_PHASE_COUNT - 1)
    ability_phase = ability["phase"].long().clamp(0, ABILITY_PHASE_COUNT - 1)

    source_card = ability["source_card_def_id"].long()
    source_idx, _ = self._card_index_and_mask(source_card)

    phase_emb = self.game_phase_encoder(phase)
    ability_phase_emb = self.ability_phase_encoder(ability_phase)
    source_emb = self.card_def_encoder(source_idx)

    action_mask = self.__get_struct_field(structured_obs, "action_mask")
    is_active = (action_mask["legal_action_count"] > 0).float()

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

  def forward(self, x, state=None):
    target_vector, target_matrix = self.encode_observations(x, state=state)
    actions, value = self.decode_actions(target_vector, target_matrix=target_matrix, state=state)
    return actions, value

  def forward_train(self, x, state=None):
    return self.forward(x, state)

  def encode_observations(self, observations, state=None):
    structured_obs, squeeze_batch, obs_tensor = self.__prepare_structured_observations(observations)
    self.__store_mask_observations(obs_tensor, state)

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
    global_context_vec = self._encode_global_context(structured_obs)
    global_vec = global_context_vec + nn.functional.pad(global_counts, (0, UNIT_EMBED_SIZE - global_counts.size(-1)))

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

    if squeeze_batch:
      target_vector = target_vector.squeeze(0)
      target_matrix = target_matrix.squeeze(0)

    return target_vector, target_matrix

  def build_primary_action_mask_tensor(self, observations):
    structured_obs, squeeze_batch, _ = self.__prepare_structured_observations(observations)
    action_mask = self.__get_struct_field(structured_obs, "action_mask")
    primary_mask = action_mask["primary_action_mask"].to(dtype=torch.bool)
    if squeeze_batch and primary_mask.dim() > 1 and primary_mask.size(0) == 1:
      return primary_mask.squeeze(0)
    return primary_mask

  def decode_actions(self, flat_hidden, target_matrix=None, state=None):
    if target_matrix is None:
      raise ValueError("decode_actions requires target_matrix for unit selections")

    if target_matrix.dim() == 2:
      target_matrix = target_matrix.unsqueeze(0)
    elif target_matrix.dim() == 1:
      target_matrix = target_matrix.unsqueeze(0).unsqueeze(0)

    mask_observations = state.get("_azk_mask_observations") if state is not None else self._cached_mask_observations
    if mask_observations is None:
      raise ValueError("mask_observations is None when decode_actions is called")

    structured_mask_obs, _, _ = self.__prepare_structured_observations(mask_observations)
    action_mask_struct = self.__get_struct_field(structured_mask_obs, "action_mask")

    device = flat_hidden.device
    primary_action_mask = action_mask_struct["primary_action_mask"].to(dtype=torch.bool, device=device)
    legal_actions = action_mask_struct["legal_actions"]
    legal_actions = torch.stack(
      (
        legal_actions["legal_primary"],
        legal_actions["legal_sub1"],
        legal_actions["legal_sub2"],
        legal_actions["legal_sub3"],
      ),
      dim=-1,
    ).to(device=device, dtype=torch.long)
    legal_action_count = action_mask_struct["legal_action_count"].to(device=device, dtype=torch.long).view(-1)

    projected_hidden = self.q_primary(flat_hidden)
    unit1_projected_hidden = self.q_unit1(flat_hidden)
    unit2_projected_hidden = self.q_unit2(flat_hidden)
    bins2_projected_hidden = self.q_bins2(flat_hidden)
    bins3_projected_hidden = self.q_bins3(flat_hidden)

    if projected_hidden.dim() == 1:
      projected_hidden = projected_hidden.unsqueeze(0)
    if unit1_projected_hidden.dim() == 1:
      unit1_projected_hidden = unit1_projected_hidden.unsqueeze(0)
    if unit2_projected_hidden.dim() == 1:
      unit2_projected_hidden = unit2_projected_hidden.unsqueeze(0)
    if bins2_projected_hidden.dim() == 1:
      bins2_projected_hidden = bins2_projected_hidden.unsqueeze(0)
    if bins3_projected_hidden.dim() == 1:
      bins3_projected_hidden = bins3_projected_hidden.unsqueeze(0)

    primary_action_embeddings = self.primary_action_encoder[0](self.primary_action_id_batch)
    primary_action_logits = projected_hidden @ primary_action_embeddings.T

    gate_1_table = torch.sigmoid(self.gate_1_embeder(self.primary_action_id_batch))
    gate_2_table = torch.sigmoid(self.gate_2_embeder(self.primary_action_id_batch))

    distribution = TCGActionDistribution(
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

    values = self.value_fn(projected_hidden)
    return distribution, values
