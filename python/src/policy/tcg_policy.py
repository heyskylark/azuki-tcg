import os
import pufferlib.models
import pufferlib.pytorch
from gymnasium.wrappers.normalize import RunningMeanStd

from python.src.observation import MAX_ATTACHED_WEAPONS
from torch import nn
import torch

from policy.tcg_distribution import TCGActionDistribution

MAX_PLAYERS_PER_MATCH = 2
CARD_TYPE_COUNT = 7
CARD_ID_COUNT = 37
GAME_PHASE_COUNT = 8
PRIMARY_ACTION_COUNT = 13
PRIMARY_ACTION_ID_BATCH = tuple(range(PRIMARY_ACTION_COUNT))
MAX_INDEX_SIZE = 50
ACTION_COMPONENT_COUNT = 4
ACT_NOOP = 0
ACT_ATTACK = 6
ACT_ATTACH_WEAPON_FROM_HAND = 7
GARDEN_ZONE_ID = 0
ALLEY_ZONE_ID = 1

OWNER_ENC_OUTPUT_SIZE = 4
GARDEN_OR_ALLEY_ENC_OUTPUT_SIZE = 4
CARD_TYPE_ENC_OUTPUT_SIZE = 4
CARD_ID_ENC_OUTPUT_SIZE = 8
GAME_PHASE_ENC_OUTPUT_SIZE = 4
PRIMARY_ACTION_ENC_OUTPUT_SIZE = 4
INDEX_ENC_OUTPUT_SIZE = 16
WEAPON_SET_INPUT_SIZE = CARD_TYPE_ENC_OUTPUT_SIZE + CARD_ID_ENC_OUTPUT_SIZE + 2

PROCESS_SET_HIDDEN_SIZE = 128 
PROCESS_SET_OUTPUT_SIZE = 32
TARGET_VECTOR_COMPONENT_COUNT = 13
GLOBAL_SCALAR_COMPONENT_COUNT = 7
LSTM_INPUT_SIZE = PROCESS_SET_OUTPUT_SIZE * TARGET_VECTOR_COMPONENT_COUNT + GLOBAL_SCALAR_COMPONENT_COUNT
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

class TCGLSTM(pufferlib.models.LSTMWrapper):
  def __init__(self, env, policy, input_size=LSTM_INPUT_SIZE, hidden_size=LSTM_HIDDEN_SIZE):
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
    assert lstm_inputs.shape == (B * TT, self.input_size)

    hidden = lstm_inputs.reshape(B, TT, self.input_size)
    hidden = hidden.transpose(0, 1)
    hidden, (lstm_h, lstm_c) = self.lstm.forward(hidden, lstm_state)
    hidden = hidden.float()
    hidden = hidden.transpose(0, 1)

    flat_hidden = hidden.reshape(B * TT, self.hidden_size)
    logits, values = self.policy.decode_actions(flat_hidden, target_matrix=target_matrix, state=state)
    values = values.reshape(B, TT)
    state["hidden"] = hidden
    state["lstm_h"] = lstm_h.detach()
    state["lstm_c"] = lstm_c.detach()
    return logits, values

# Processes an unordered set of feature vectors with a shared 2-layer MLP and max pooling.
# Returns both the per-element embeddings (shape: [..., N, S]) and the pooled vector (shape: [..., S]).
class ProcessSetProcessor(nn.Module):
  def __init__(self, input_size, hidden_size = PROCESS_SET_HIDDEN_SIZE, output_size = PROCESS_SET_OUTPUT_SIZE, activation=nn.ReLU):
    super().__init__()
    self.activation = activation()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, output_size)
    self.output_size = output_size

  def forward(self, x, mask=None):
    """Process a set/matrix shaped (*, N, K) and return (element_embeddings, pooled_vector)."""
    if x.dim() == 1:
      # Handles single-element inputs by adding a fake dimension (1,K)
      x = x.unsqueeze(-2)
    elif x.dim() < 2:
      raise ValueError("ProcessSetProcessor expects tensors with shape (*, N, K)")

    hidden = self.activation(self.fc1(x))
    set_embeddings = self.fc2(hidden)
    if set_embeddings.size(-2) == 0:
      pooled = torch.zeros(
        *set_embeddings.shape[:-2],
        self.output_size,
        device=set_embeddings.device,
        dtype=set_embeddings.dtype,
      )
    else:
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
      else:
        pooled, _ = torch.max(set_embeddings, dim=-2)
    return set_embeddings, pooled

class SingleUnitProjection(nn.Module):
  def __init__(self, input_size, hidden_size = PROCESS_SET_HIDDEN_SIZE, output_size = PROCESS_SET_OUTPUT_SIZE, activation=nn.ReLU):
    super().__init__()
    self.activation = activation()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    hidden = self.activation(self.fc1(x))
    return self.fc2(hidden)

class TCG(nn.Module):
  def __init__(self, env, **kwargs):
    super().__init__()

    self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(PROCESS_SET_OUTPUT_SIZE, 1), std=1)
    self.is_continuous = False
    self.scalar_normalizer = ScalarRunningNorm()

    # Categorical encodings
    self.owner_encoder = nn.Sequential(
      nn.Embedding(MAX_PLAYERS_PER_MATCH, OWNER_ENC_OUTPUT_SIZE),
      nn.Flatten(),
    )
    self.garden_or_alley_encoder = nn.Sequential(
      nn.Embedding(2, GARDEN_OR_ALLEY_ENC_OUTPUT_SIZE),
      nn.Flatten(),
    )
    self.card_type_encoder = nn.Sequential(
      nn.Embedding(CARD_TYPE_COUNT, CARD_TYPE_ENC_OUTPUT_SIZE),
      nn.Flatten(),
    )
    self.index_encoder = nn.Sequential(
      nn.Embedding(MAX_INDEX_SIZE, INDEX_ENC_OUTPUT_SIZE),
      nn.Flatten(),
    )
    self.card_id_encoder = nn.Sequential(
      nn.Embedding(CARD_ID_COUNT, CARD_ID_ENC_OUTPUT_SIZE),
      nn.Flatten(),
    )
    self.game_state_encoder = nn.Sequential(
      nn.Embedding(GAME_PHASE_COUNT, GAME_PHASE_ENC_OUTPUT_SIZE),
      nn.Flatten(),
    )
    self.primary_action_encoder = nn.Sequential(
      nn.Embedding(PRIMARY_ACTION_COUNT, PROCESS_SET_OUTPUT_SIZE),
      nn.Flatten(),
    )
    self.register_buffer(
      "primary_action_id_batch",
      torch.tensor(PRIMARY_ACTION_ID_BATCH, dtype=torch.long),
    )
    self._primary_action_embedding_batch = None
    self._cached_mask_observations = None

    #### Single Unit Projectors ####
    # Input: type emb, card id emb, weapons set emb, attack, health, is_tapped
    LEADER_FEATURES_INPUT_SIZE = CARD_TYPE_ENC_OUTPUT_SIZE + CARD_ID_ENC_OUTPUT_SIZE + PROCESS_SET_OUTPUT_SIZE + 3
    self.ally_leader_projector = SingleUnitProjection(
      LEADER_FEATURES_INPUT_SIZE
    )
    self.opponent_leader_projector = SingleUnitProjection(
      LEADER_FEATURES_INPUT_SIZE
    )

    # Input: type emb, card id emb, is_tapped
    GATE_FEATURES_INPUT_SIZE = CARD_TYPE_ENC_OUTPUT_SIZE + CARD_ID_ENC_OUTPUT_SIZE + 1
    self.ally_gate_projector = SingleUnitProjection(
      GATE_FEATURES_INPUT_SIZE
    )
    self.opponent_gate_projector = SingleUnitProjection(
      GATE_FEATURES_INPUT_SIZE
    )

    self.gate_1_embeder = nn.Embedding(PRIMARY_ACTION_COUNT, PROCESS_SET_OUTPUT_SIZE)
    self.gate_2_embeder = nn.Embedding(PRIMARY_ACTION_COUNT, PROCESS_SET_OUTPUT_SIZE)
    self.q_primary = nn.Linear(LSTM_HIDDEN_SIZE, PROCESS_SET_OUTPUT_SIZE)
    self.q_unit1 = nn.Linear(LSTM_HIDDEN_SIZE, PROCESS_SET_OUTPUT_SIZE)
    self.q_unit2 = nn.Linear(LSTM_HIDDEN_SIZE, PROCESS_SET_OUTPUT_SIZE)
    self.q_bins2 = nn.Linear(LSTM_HIDDEN_SIZE, MAX_INDEX_SIZE)
    self.q_bins3 = nn.Linear(LSTM_HIDDEN_SIZE, MAX_INDEX_SIZE)

    #### Process Set Processors ####
    self.weapon_set_processor = ProcessSetProcessor(WEAPON_SET_INPUT_SIZE)
    # Input: type emb, card id emb, zone index emb, tapped, cooldown
    IKZ_AREA_SET_INPUT_SIZE = CARD_TYPE_ENC_OUTPUT_SIZE + CARD_ID_ENC_OUTPUT_SIZE + INDEX_ENC_OUTPUT_SIZE + 2
    self.ally_ikz_area_set_processor = ProcessSetProcessor(IKZ_AREA_SET_INPUT_SIZE)
    self.opponent_ikz_area_set_processor = ProcessSetProcessor(IKZ_AREA_SET_INPUT_SIZE)

    # Input: type emb, card id emb, zone index emb, garden or alley emb, weapons set emb, tapped, cooldown, ikz cost, attack, health, gate points
    ALLEY_AND_GARDEN_SET_INPUT_SIZE = CARD_TYPE_ENC_OUTPUT_SIZE + CARD_ID_ENC_OUTPUT_SIZE + INDEX_ENC_OUTPUT_SIZE + GARDEN_OR_ALLEY_ENC_OUTPUT_SIZE + PROCESS_SET_OUTPUT_SIZE + 6
    self.ally_alley_and_garden_set_processor = ProcessSetProcessor(ALLEY_AND_GARDEN_SET_INPUT_SIZE)
    self.opponent_alley_and_garden_set_processor = ProcessSetProcessor(ALLEY_AND_GARDEN_SET_INPUT_SIZE)
    
    # Input: card id emb, type emb, hand index emb, ikz cost, attack, health, gate points
    HAND_AND_DISCARD_SET_INPUT_SIZE = CARD_ID_ENC_OUTPUT_SIZE + CARD_TYPE_ENC_OUTPUT_SIZE + INDEX_ENC_OUTPUT_SIZE + 4
    self.hand_set_processor = ProcessSetProcessor(HAND_AND_DISCARD_SET_INPUT_SIZE)
    self.ally_discard_set_processor = ProcessSetProcessor(HAND_AND_DISCARD_SET_INPUT_SIZE)
    self.opponent_discard_set_processor = ProcessSetProcessor(HAND_AND_DISCARD_SET_INPUT_SIZE)

    #### Observation structure dtype ####
    emulated_spec = getattr(env, "emulated", None)
    if emulated_spec is None:
      raise AttributeError("env must expose emulated metadata for nativize")
    self._obs_struct_dtype = pufferlib.pytorch.nativize_dtype(emulated_spec)

  def __batch_embed_primary_actions(self):
    embeddings = self.primary_action_encoder[0](self.primary_action_id_batch)
    self._primary_action_embedding_batch = embeddings
    return embeddings

  def __process_discard(self, discard_features, is_ally: bool):
    card_ids = discard_features["card_id"].long()
    type_ids = discard_features["type_id"].long()
    discard_indices = discard_features["zone_index"].long()
    # Empty slots default to type_id == 0; real cards (entities/weapons/spells/ikz) are > 0.
    card_mask = type_ids > 0

    type_embeddings = self.card_type_encoder[0](type_ids)
    card_embeddings = self.card_id_encoder[0](card_ids)
    discard_index_embeddings = self.index_encoder[0](discard_indices)

    scalar_stack = torch.stack(
      [
        discard_features["ikz_cost"].float(),
        discard_features["attack"].float(),
        discard_features["health"].float(),
        discard_features["gate_points"].float(),
      ],
      dim=-1,
    )
    scalar_stack = self.scalar_normalizer("discard_scalar", scalar_stack, mask=card_mask)

    discard_input = torch.cat([type_embeddings, card_embeddings, discard_index_embeddings, scalar_stack], dim=-1)

    processor = self.ally_discard_set_processor if is_ally else self.opponent_discard_set_processor

    set_embeddings, pooled = processor(discard_input, mask=card_mask)
    set_embeddings = set_embeddings * card_mask.unsqueeze(-1).to(dtype=set_embeddings.dtype)
    return set_embeddings, pooled

  def __process_hand(self, hand_features):
    card_ids = hand_features["card_id"].long()
    type_ids = hand_features["type_id"].long()
    hand_indices = hand_features["zone_index"].long()
    # Empty slots default to type_id == 0; real cards (entities/weapons/spells/ikz) are > 0.
    card_mask = type_ids > 0

    type_embeddings = self.card_type_encoder[0](type_ids)
    card_embeddings = self.card_id_encoder[0](card_ids)
    hand_index_embeddings = self.index_encoder[0](hand_indices)

    scalar_stack = torch.stack(
      [
        hand_features["ikz_cost"].float(),
        hand_features["attack"].float(),
        hand_features["health"].float(),
        hand_features["gate_points"].float(),
      ],
      dim=-1,
    )
    scalar_stack = self.scalar_normalizer("hand_scalar", scalar_stack, mask=card_mask)

    hand_input = torch.cat([type_embeddings, card_embeddings, hand_index_embeddings, scalar_stack], dim=-1)
    return self.hand_set_processor(hand_input)

  def __process_leader(self, leader_features, is_ally: bool):
    if is_ally:
      return self.ally_leader_projector(leader_features)
    else:
      return self.opponent_leader_projector(leader_features)

  def __process_gate(self, gate_features, is_ally: bool):
    if is_ally:
      return self.ally_gate_projector(gate_features)
    else:
      return self.opponent_gate_projector(gate_features)

  def __process_ikz_area(self, ikz_area_features, is_ally: bool):
    type_ids = ikz_area_features["type_id"].long()
    card_ids = ikz_area_features["card_id"].long()
    zone_indices = ikz_area_features["zone_index"].long()
    tapped = ikz_area_features["tapped"].float()
    cooldown = ikz_area_features["cooldown"].float()
    # Empty slots default to type_id == 0.
    card_mask = type_ids > 0

    type_embeddings = self.card_type_encoder[0](type_ids)
    card_embeddings = self.card_id_encoder[0](card_ids)
    zone_index_embeddings = self.index_encoder[0](zone_indices)

    scalar_stack = torch.stack(
      [
        tapped,
        cooldown,
      ],
      dim=-1,
    )

    processor = self.ally_ikz_area_set_processor if is_ally else self.opponent_ikz_area_set_processor
    scalar_stack = self.scalar_normalizer("ikz_area_scalar", scalar_stack, mask=card_mask)

    zone_input = torch.cat(
      [
        type_embeddings,
        card_embeddings,
        zone_index_embeddings,
        scalar_stack,
      ],
      dim=-1,
    )

    set_embeddings, pooled = processor(zone_input, mask=card_mask)
    set_embeddings = set_embeddings * card_mask.unsqueeze(-1).to(dtype=set_embeddings.dtype)
    return set_embeddings, pooled

  def __process_alley_or_garden(self, zone_slots, *, is_garden: bool, is_ally: bool):
    zone_slots = self.__normalize_zone_entries(zone_slots)

    type_ids = self.__stack_zone_field(zone_slots, "type_id").long()
    card_ids = self.__stack_zone_field(zone_slots, "card_id").long()
    zone_indices = self.__stack_zone_field(zone_slots, "zone_index").long()
    # Empty slots default to type_id == 0.
    card_mask = type_ids > 0

    tapped = self.__stack_zone_field(zone_slots, "tapped").float()
    cooldown = self.__stack_zone_field(zone_slots, "cooldown").float()
    ikz_cost = self.__stack_zone_field(zone_slots, "ikz_cost").float()
    attack = self.__stack_zone_field(zone_slots, "attack").float()
    health = self.__stack_zone_field(zone_slots, "health").float()
    gate_points = self.__stack_zone_field(zone_slots, "gate_points").float()

    type_embeddings = self.card_type_encoder[0](type_ids)
    card_embeddings = self.card_id_encoder[0](card_ids)
    zone_index_embeddings = self.index_encoder[0](zone_indices)

    zone_value = GARDEN_ZONE_ID if is_garden else ALLEY_ZONE_ID
    zone_tensor = torch.full_like(card_ids, zone_value, dtype=torch.long)
    garden_or_alley_embeddings = self.garden_or_alley_encoder[0](zone_tensor)

    weapon_embeddings = []
    for slot in zone_slots:
      slot_weapons = self.__normalize_zone_entries(slot["weapons"])
      weapon_embedding = self.__process_weapons(
        self.__build_weapon_features(slot_weapons),
        self.__flatten_scalar(slot["weapon_count"]),
      )
      weapon_embeddings.append(weapon_embedding)
    weapon_embeddings = torch.stack(weapon_embeddings, dim=1)

    scalar_stack = torch.stack(
      [
        tapped,
        cooldown,
        ikz_cost,
        attack,
        health,
        gate_points,
      ],
      dim=-1,
    )

    processor = self.ally_alley_and_garden_set_processor if is_ally else self.opponent_alley_and_garden_set_processor

    scalar_stack = self.scalar_normalizer("alley_garden_scalar", scalar_stack, mask=card_mask)

    zone_input = torch.cat(
      [
        type_embeddings,
        card_embeddings,
        zone_index_embeddings,
        garden_or_alley_embeddings,
        weapon_embeddings,
        scalar_stack,
      ],
      dim=-1,
    )

    set_embeddings, pooled = processor(zone_input, mask=card_mask)
    set_embeddings = set_embeddings * card_mask.unsqueeze(-1).to(dtype=set_embeddings.dtype)
    return set_embeddings, pooled

  def __process_alley(self, alley_features, is_ally: bool):
    return self.__process_alley_or_garden(alley_features, is_garden=False, is_ally=is_ally)

  def __process_garden(self, garden_features, is_ally: bool):
    return self.__process_alley_or_garden(garden_features, is_garden=True, is_ally=is_ally)

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

  def __maybe_squeeze_batch(self, tensor: torch.Tensor, squeeze_batch: bool) -> torch.Tensor:
    if squeeze_batch and tensor.dim() > 0 and tensor.size(0) == 1:
      return tensor.squeeze(0)
    return tensor

  def __flatten_scalar(self, tensor: torch.Tensor) -> torch.Tensor:
    """Ensure scalar-like tensors drop trailing unit dims so embeddings stay 2D."""
    if tensor.dim() > 1 and tensor.size(-1) == 1:
      return tensor.squeeze(-1)
    return tensor

  def __action_mask_struct(self, observations):
    structured_obs, squeeze_batch, _ = self.__prepare_structured_observations(observations)
    return structured_obs["action_mask"], squeeze_batch

  def __store_mask_observations(self, obs_tensor: torch.Tensor, state):
    if obs_tensor is None:
      return
    detached = obs_tensor.detach()
    if state is not None:
      try:
        state["_azk_mask_observations"] = detached
      except (TypeError, AttributeError):
        pass
    self._cached_mask_observations = detached

  def __normalize_zone_entries(self, zone_entries):
    if isinstance(zone_entries, dict):
      return [zone_entries[key] for key in sorted(zone_entries.keys())]
    return zone_entries

  def __stack_zone_field(self, slots, field_name):
    stacked = torch.stack([slot[field_name] for slot in slots], dim=1)
    if stacked.size(-1) == 1:
      stacked = stacked.squeeze(-1)
    return stacked

  def __build_hand_features(self, hand_slots):
    hand_type_tensor = self.__stack_zone_field(hand_slots, "type_id")
    hand_card_tensor = self.__stack_zone_field(hand_slots, "card_id")
    hand_ikz_tensor = self.__stack_zone_field(hand_slots, "ikz_cost")
    hand_attack_tensor = self.__stack_zone_field(hand_slots, "attack")
    hand_health_tensor = self.__stack_zone_field(hand_slots, "health")
    hand_gate_tensor = self.__stack_zone_field(hand_slots, "gate_points")
    hand_zone_tensor = self.__stack_zone_field(hand_slots, "zone_index")

    return {
      "type_id": hand_type_tensor,
      "card_id": hand_card_tensor,
      "ikz_cost": hand_ikz_tensor,
      "attack": hand_attack_tensor,
      "health": hand_health_tensor,
      "gate_points": hand_gate_tensor,
      "zone_index": hand_zone_tensor,
    }
  
  def __build_discard_features(self, discard_slots):
    discard_type_tensor = self.__stack_zone_field(discard_slots, "type_id")
    discard_card_tensor = self.__stack_zone_field(discard_slots, "card_id")
    discard_zone_tensor = self.__stack_zone_field(discard_slots, "zone_index")
    discard_ikz_cost_tensor = self.__stack_zone_field(discard_slots, "ikz_cost")
    discard_attack_tensor = self.__stack_zone_field(discard_slots, "attack")
    discard_health_tensor = self.__stack_zone_field(discard_slots, "health")
    discard_gate_points_tensor = self.__stack_zone_field(discard_slots, "gate_points")

    return {
      "type_id": discard_type_tensor,
      "card_id": discard_card_tensor,
      "zone_index": discard_zone_tensor,
      "ikz_cost": discard_ikz_cost_tensor,
      "attack": discard_attack_tensor,
      "health": discard_health_tensor,
      "gate_points": discard_gate_points_tensor,
    }

  def __build_ikz_area_features(self, ikz_slots):
    ikz_type_tensor = self.__stack_zone_field(ikz_slots, "type_id")
    ikz_card_tensor = self.__stack_zone_field(ikz_slots, "card_id")
    ikz_zone_tensor = self.__stack_zone_field(ikz_slots, "zone_index")
    ikz_tapped_tensor = self.__stack_zone_field(ikz_slots, "tapped")
    ikz_cooldown_tensor = self.__stack_zone_field(ikz_slots, "cooldown")

    return {
      "type_id": ikz_type_tensor,
      "card_id": ikz_card_tensor,
      "zone_index": ikz_zone_tensor,
      "tapped": ikz_tapped_tensor,
      "cooldown": ikz_cooldown_tensor,
    }

  def __build_weapon_features(self, weapon_slots):
    def stack_weapon_field(field_name, default_zero=False):
      try:
        return self.__stack_zone_field(weapon_slots, field_name)
      except (KeyError, AttributeError):
        if not default_zero:
          raise
        reference = self.__stack_zone_field(weapon_slots, "attack")
        return torch.zeros_like(reference)

    weapon_type_tensor = stack_weapon_field("type_id")
    weapon_card_tensor = stack_weapon_field("card_id")
    weapon_attack_tensor = stack_weapon_field("attack")
    weapon_ikz_tensor = stack_weapon_field("ikz_cost")

    return {
      "type_id": weapon_type_tensor,
      "card_id": weapon_card_tensor,
      "attack": weapon_attack_tensor,
      "ikz_cost": weapon_ikz_tensor,
    }

  def __get_organized_obs_data(self, structured_obs):
    player_obs = structured_obs["player"]
    opponent_obs = structured_obs["opponent"]
    
    hand_slots = self.__normalize_zone_entries(player_obs["hand"])
    player_alley_slots = self.__normalize_zone_entries(player_obs["alley"])
    player_garden_slots = self.__normalize_zone_entries(player_obs["garden"])
    opponent_alley_slots = self.__normalize_zone_entries(opponent_obs["alley"])
    opponent_garden_slots = self.__normalize_zone_entries(opponent_obs["garden"])
    player_ikz_slots = self.__normalize_zone_entries(player_obs["ikz_area"])
    opponent_ikz_slots = self.__normalize_zone_entries(opponent_obs["ikz_area"])
    player_discard_slots = self.__normalize_zone_entries(player_obs["discard"])
    opponent_discard_slots = self.__normalize_zone_entries(opponent_obs["discard"])

    player_weapon_slots = self.__normalize_zone_entries(player_obs["leader"]["weapons"])
    opponent_weapon_slots = self.__normalize_zone_entries(opponent_obs["leader"]["weapons"])

    return {
      "hand": self.__build_hand_features(hand_slots),
      "player_discard": self.__build_discard_features(player_discard_slots),
      "player_leader": player_obs["leader"],
      "player_leader_weapons": self.__build_weapon_features(player_weapon_slots),
      "player_gate": player_obs["gate"],
      "player_alley": player_alley_slots,
      "player_garden": player_garden_slots,
      "player_ikz_area": self.__build_ikz_area_features(player_ikz_slots),
      "opponent_discard": self.__build_discard_features(opponent_discard_slots),
      "opponent_leader": opponent_obs["leader"],
      "opponent_leader_weapons": self.__build_weapon_features(opponent_weapon_slots),
      "opponent_gate": opponent_obs["gate"],
      "opponent_alley": opponent_alley_slots,
      "opponent_garden": opponent_garden_slots,
      "opponent_ikz_area": self.__build_ikz_area_features(opponent_ikz_slots),
      "player_deck_count": player_obs["deck_count"],
      "opponent_deck_count": opponent_obs["deck_count"],
      "player_ikz_pile_count": player_obs["ikz_pile_count"],
      "opponent_ikz_pile_count": opponent_obs["ikz_pile_count"],
      "player_has_ikz_token": player_obs["has_ikz_token"],
      "opponent_has_ikz_token": opponent_obs["has_ikz_token"],
      "opponent_hand_count": opponent_obs["hand_count"],
    }

  def __process_weapons(self, weapon_features, weapon_count):
    type_ids = self.__flatten_scalar(weapon_features["type_id"]).long()
    card_ids = self.__flatten_scalar(weapon_features["card_id"]).long()
    attack = self.__flatten_scalar(weapon_features["attack"]).float()
    ikz_cost = self.__flatten_scalar(weapon_features["ikz_cost"]).float()

    if not torch.all((type_ids >= 0) & (type_ids < CARD_TYPE_COUNT)):
      out_of_bounds = type_ids[(type_ids < 0) | (type_ids >= CARD_TYPE_COUNT)]
      raise AssertionError(f"Some type_ids out of bounds: {out_of_bounds}")
    if not torch.all((card_ids >= 0) & (card_ids < CARD_ID_COUNT)):
      out_of_bounds = card_ids[(card_ids < 0) | (card_ids >= CARD_ID_COUNT)]
      raise AssertionError(f"Some card_ids out of bounds: {out_of_bounds}")
    assert torch.all(weapon_count < MAX_ATTACHED_WEAPONS), f"weapon_count must be < {MAX_ATTACHED_WEAPONS}, got: {weapon_count}"

    type_embeddings = self.card_type_encoder[0](type_ids)
    card_embeddings = self.card_id_encoder[0](card_ids)

    scalar_stack = torch.stack(
      [
        attack,
        ikz_cost,
      ],
      dim=-1,
    )

    if scalar_stack.dim() == 2:
      scalar_stack = scalar_stack.unsqueeze(-2)
      type_embeddings = type_embeddings.unsqueeze(-2)
      card_embeddings = card_embeddings.unsqueeze(-2)

    max_slots = scalar_stack.size(-2)
    device = scalar_stack.device
    counts = self.__flatten_scalar(weapon_count).long().view(-1, 1)
    slot_indices = torch.arange(max_slots, device=device).unsqueeze(0)
    weapon_mask = slot_indices < counts
    scalar_stack = self.scalar_normalizer("weapon_scalar", scalar_stack, mask=weapon_mask)

    weapon_input = torch.cat([type_embeddings, card_embeddings, scalar_stack], dim=-1)

    _, weapon_pooled = self.weapon_set_processor(weapon_input, mask=weapon_mask)
    return weapon_pooled

  def __encode_leader_obs(self, leader_obs, weapon_features, is_ally: bool):
    weapon_count = self.__flatten_scalar(leader_obs["weapon_count"]).long()
    weapon_embedding = self.__process_weapons(weapon_features, weapon_count)

    type_ids = self.__flatten_scalar(leader_obs["type_id"]).long()
    card_ids = self.__flatten_scalar(leader_obs["card_id"]).long()
    type_embedding = self.card_type_encoder[0](type_ids)
    card_embedding = self.card_id_encoder[0](card_ids)

    scalar_stack = torch.stack(
      [
        self.__flatten_scalar(leader_obs["attack"]).float(),
        self.__flatten_scalar(leader_obs["health"]).float(),
        self.__flatten_scalar(leader_obs["tapped"]).float(),
      ],
      dim=-1,
    )
    scalar_stack = self.scalar_normalizer("leader_scalar", scalar_stack)

    leader_input = torch.cat([type_embedding, card_embedding, weapon_embedding, scalar_stack], dim=-1)
    return self.__process_leader(leader_input, is_ally)

  def __encode_gate_obs(self, gate_obs, is_ally: bool):
    type_ids = self.__flatten_scalar(gate_obs["type_id"]).long()
    card_ids = self.__flatten_scalar(gate_obs["card_id"]).long()
    type_embedding = self.card_type_encoder[0](type_ids)
    card_embedding = self.card_id_encoder[0](card_ids)

    tap_tensor = self.__flatten_scalar(gate_obs["tapped"]).float().view(-1, 1)
    tap_tensor = self.scalar_normalizer("gate_scalar", tap_tensor)

    gate_input = torch.cat([type_embedding, card_embedding, tap_tensor], dim=-1)
    return self.__process_gate(gate_input, is_ally)
  
  def forward(self, x, state=None):
    target_vector, target_matrix = self.encode_observations(x, state=state)
    actions, value = self.decode_actions(target_vector, target_matrix=target_matrix, state=state)
    return actions, value

  def forward_train(self, x, state=None):
    return self.forward(x, state)
  
  def encode_observations(self, observations, state=None):
    structured_obs, squeeze_batch, obs_tensor = self.__prepare_structured_observations(observations)
    self.__store_mask_observations(obs_tensor, state)

    obs_data = self.__get_organized_obs_data(structured_obs)
    
    hand_matrix, hand_vector = self.__process_hand(obs_data["hand"])
    player_discard_matrix, player_discard_vector = self.__process_discard(obs_data["player_discard"], is_ally=True)
    _, opponent_discard_vector = self.__process_discard(obs_data["opponent_discard"], is_ally=False)
    player_garden_matrix, player_garden_vector = self.__process_garden(obs_data["player_garden"], is_ally=True)
    player_alley_matrix, player_alley_vector = self.__process_alley(obs_data["player_alley"], is_ally=True)
    opponent_garden_matrix, opponent_garden_vector = self.__process_garden(obs_data["opponent_garden"], is_ally=False)
    opponent_alley_matrix, opponent_alley_vector = self.__process_alley(obs_data["opponent_alley"], is_ally=False)
    player_ikz_matrix, player_ikz_vector = self.__process_ikz_area(obs_data["player_ikz_area"], is_ally=True)
    opponent_ikz_matrix, opponent_ikz_vector = self.__process_ikz_area(obs_data["opponent_ikz_area"], is_ally=False)
    player_leader_embedding = self.__encode_leader_obs(obs_data["player_leader"], obs_data["player_leader_weapons"], is_ally=True)
    opponent_leader_embedding = self.__encode_leader_obs(obs_data["opponent_leader"], obs_data["opponent_leader_weapons"], is_ally=False)
    player_gate_embedding = self.__encode_gate_obs(obs_data["player_gate"], is_ally=True)
    opponent_gate_embedding = self.__encode_gate_obs(obs_data["opponent_gate"], is_ally=False)

    global_scalar_features = torch.stack(
      [
        self.__flatten_scalar(obs_data["player_deck_count"]).float(),
        self.__flatten_scalar(obs_data["opponent_deck_count"]).float(),
        self.__flatten_scalar(obs_data["player_ikz_pile_count"]).float(),
        self.__flatten_scalar(obs_data["opponent_ikz_pile_count"]).float(),
        self.__flatten_scalar(obs_data["player_has_ikz_token"]).float(),
        self.__flatten_scalar(obs_data["opponent_has_ikz_token"]).float(),
        self.__flatten_scalar(obs_data["opponent_hand_count"]).float(),
      ],
      dim=-1,
    )
    global_scalar_features = self.scalar_normalizer("global_scalar", global_scalar_features)

    target_vector = torch.cat(
      [
        hand_vector,
        player_discard_vector,
        opponent_discard_vector,
        player_garden_vector,
        player_alley_vector,
        opponent_garden_vector,
        opponent_alley_vector,
        player_leader_embedding,
        player_gate_embedding,
        opponent_leader_embedding,
        opponent_gate_embedding,
        player_ikz_vector,
        opponent_ikz_vector,
        global_scalar_features,
      ],
      dim=-1,
    )

    target_matrix = torch.cat(
      [
        hand_matrix,
        player_discard_matrix,
        player_garden_matrix,
        player_alley_matrix,
        opponent_garden_matrix,
        opponent_alley_matrix,
        player_leader_embedding.unsqueeze(-2),
        player_gate_embedding.unsqueeze(-2),
        opponent_leader_embedding.unsqueeze(-2),
        opponent_gate_embedding.unsqueeze(-2),
        player_ikz_matrix,
        opponent_ikz_matrix,
      ],
      dim=-2,
    )

    if squeeze_batch:
      target_vector = target_vector.squeeze(0)
      target_matrix = target_matrix.squeeze(0)

    return target_vector, target_matrix

  def build_primary_action_mask_tensor(self, observations):
    """Return a bool tensor indicating which primary actions are legal."""
    action_mask, squeeze_batch = self.__action_mask_struct(observations)
    primary_mask = action_mask["primary_action_mask"].to(dtype=torch.bool)
    return self.__maybe_squeeze_batch(primary_mask, squeeze_batch)

  def decode_actions(self, flat_hidden, target_matrix=None, state=None):
    if target_matrix is None:
      raise ValueError("decode_actions requires target_matrix for unit selections")

    if target_matrix.dim() == 2:
      target_matrix = target_matrix.unsqueeze(0)
    elif target_matrix.dim() == 1:
      target_matrix = target_matrix.unsqueeze(0).unsqueeze(0)

    B = target_matrix.size(0)

    mask_observations = state["_azk_mask_observations"]
    if mask_observations is None:
      raise ValueError("mask_observations is None when decode_actions is called")

    structured_mask_obs, _, _ = self.__prepare_structured_observations(mask_observations)
    action_mask_struct = structured_mask_obs["action_mask"]

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

    primary_action_embeddings = self.__batch_embed_primary_actions()
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
