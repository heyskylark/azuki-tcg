import pufferlib.models
import pufferlib.pytorch

from torch import nn
import torch

MAX_PLAYERS_PER_MATCH = 2
CARD_TYPE_COUNT = 7
CARD_ID_COUNT = 37
GAME_PHASE_COUNT = 8
PRIMARY_ACTION_COUNT = 14
PRIMARY_ACTION_ID_BATCH = tuple(range(PRIMARY_ACTION_COUNT))
MAX_INDEX_SIZE = 50
ACT_ATTACK = 6
ACT_ATTACH_WEAPON_FROM_HAND = 7

OWNER_ENC_OUTPUT_SIZE = 4
GARDEN_OR_ALLEY_ENC_OUTPUT_SIZE = 4
CARD_TYPE_ENC_OUTPUT_SIZE = 4
CARD_ID_ENC_OUTPUT_SIZE = 8
GAME_PHASE_ENC_OUTPUT_SIZE = 4
PRIMARY_ACTION_ENC_OUTPUT_SIZE = 4
INDEX_ENC_OUTPUT_SIZE = 16

LSTM_INPUT_SIZE = 64
LSTM_HIDDEN_SIZE = 4096
PROCESS_SET_HIDDEN_SIZE = 256
PROCESS_SET_OUTPUT_SIZE = 64

MASK_MIN_VALUE = -1e9

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

  def forward(self, x):
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
    # Input: type emb, card id emb, zone index emb, is tapped
    IKZ_AREA_SET_INPUT_SIZE = CARD_TYPE_ENC_OUTPUT_SIZE + CARD_ID_ENC_OUTPUT_SIZE + INDEX_ENC_OUTPUT_SIZE + 1
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

  def __process_hand(self, hand_features):
    card_ids = hand_features["card_id"].long()
    type_ids = hand_features["type_id"].long()
    hand_indices = hand_features["zone_index"].long()

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

  def __process_ikz_area(self, ikz_area_features):
    raise NotImplementedError("TCG process ikz area data not implemented")

  def __process_discard(self, discard_features):
    raise NotImplementedError("TCG process discard data not implemented")

  def __process_alley(self, alley_features):
    raise NotImplementedError("TCG process alley data not implemented")

  def __process_garden(self, garden_features):
    raise NotImplementedError("TCG process garden data not implemented")

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

  def __mask_observations_from_state(self, state):
    if state is not None:
      try:
        cached = state.get("_azk_mask_observations")
      except AttributeError:
        cached = None
      if cached is not None:
        return cached
    return getattr(self, "_cached_mask_observations", None)

  def __mask_logits(self, logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
      raise ValueError("mask is None when mask_logits is called")
    
    bool_mask = mask.to(device=logits.device, dtype=torch.bool)
    if bool_mask.shape != logits.shape:
      bool_mask = bool_mask.expand_as(logits)
    return logits.masked_fill(~bool_mask, MASK_MIN_VALUE)

  def __submask_entry(self, container, index: int):
    if isinstance(container, dict):
      for key in (f"f{index}", str(index), index):
        if key in container:
          return container[key]
    try:
      return container[index]
    except (TypeError, IndexError, KeyError):
      return None

  def __first_submask_entry(self, container):
    if isinstance(container, dict):
      for idx in range(PRIMARY_ACTION_COUNT):
        entry = self.__submask_entry(container, idx)
        if entry is not None:
          return entry
      for value in container.values():
        if value is not None:
          return value
    elif isinstance(container, (list, tuple)):
      for entry in container:
        if entry is not None:
          return entry
    return None

  def __normalize_primary_action_index(self, primary_action_index, batch_size: int) -> torch.Tensor:
    device = self.__policy_device()
    if torch.is_tensor(primary_action_index):
      index_tensor = primary_action_index.to(device=device, dtype=torch.long)
    else:
      index_tensor = torch.as_tensor(primary_action_index, device=device, dtype=torch.long)

    if index_tensor.dim() == 0:
      index_tensor = index_tensor.view(1).expand(batch_size)
    elif index_tensor.dim() == 1:
      if index_tensor.numel() == 1 and batch_size > 1:
        index_tensor = index_tensor.expand(batch_size)
      elif index_tensor.numel() != batch_size:
        raise ValueError(
          f"primary_action_index batch ({index_tensor.numel()}) does not match observations ({batch_size})"
        )
    else:
      index_tensor = index_tensor.reshape(batch_size)

    index_tensor = index_tensor.contiguous()
    if torch.any((index_tensor < 0) | (index_tensor >= PRIMARY_ACTION_COUNT)):
      raise ValueError("primary_action_index contains values outside the valid range")
    return index_tensor

  def __get_organized_obs_data(self, structured_obs):
    player_obs = structured_obs["player"]
    hand = player_obs["hand"]
    if isinstance(hand, dict):
      hand_slots = [hand[key] for key in sorted(hand.keys())]
    else:
      hand_slots = hand

    def stack_hand_field(field_name):
      stacked = torch.stack([slot[field_name] for slot in hand_slots], dim=1)
      if stacked.size(-1) == 1:
        stacked = stacked.squeeze(-1)
      return stacked

    hand_type_tensor = stack_hand_field("type_id")
    hand_card_tensor = stack_hand_field("card_id")
    hand_ikz_tensor = stack_hand_field("ikz_cost")
    hand_attack_tensor = stack_hand_field("attack")
    hand_health_tensor = stack_hand_field("health")
    hand_gate_tensor = stack_hand_field("gate_points")
    hand_zone_tensor = stack_hand_field("zone_index")

    hand_features = {
      "type_id": hand_type_tensor,
      "card_id": hand_card_tensor,
      "ikz_cost": hand_ikz_tensor,
      "attack": hand_attack_tensor,
      "health": hand_health_tensor,
      "gate_points": hand_gate_tensor,
      "zone_index": hand_zone_tensor,
    }

    return hand_features
  
  def forward(self, x, state=None):
    target_vector, target_matrix = self.encode_observations(x, state=state)
    actions, value = self.decode_actions(target_vector, target_matrix=target_matrix, state=state)
    return actions, value

  def forward_train(self, x, state=None):
    return self.forward(x, state)
  
  def encode_observations(self, observations, state=None):
    structured_obs, squeeze_batch, obs_tensor = self.__prepare_structured_observations(observations)
    self.__store_mask_observations(obs_tensor, state)

    primary_action_embeddings = self.__batch_embed_primary_actions()
    if state is not None:
      state["primary_action_embeddings"] = primary_action_embeddings

    hand_features = self.__get_organized_obs_data(structured_obs)
    
    target_matrix, target_vector = self.__process_hand(hand_features)

    if squeeze_batch:
      target_vector = target_vector.squeeze(0)
      target_matrix = target_matrix.squeeze(0)

    return target_vector, target_matrix

  def build_primary_action_mask_tensor(self, observations):
    """Return a bool tensor indicating which primary actions are legal."""
    structured_obs, squeeze_batch, _ = self.__prepare_structured_observations(observations)
    action_mask = structured_obs["action_mask"]
    primary_mask = action_mask["primary_action_mask"].to(dtype=torch.bool)
    return self.__maybe_squeeze_batch(primary_mask, squeeze_batch)

  def build_subaction_mask_tensors(self, primary_action_index, observations):
    """Return bool tensors of valid subaction choices for a chosen primary action."""
    structured_obs, squeeze_batch, _ = self.__prepare_structured_observations(observations)
    sub_masks_container = structured_obs["action_mask"]["subaction_masks"]

    template_entry = self.__first_submask_entry(sub_masks_container)
    if template_entry is None:
      raise ValueError("Observation does not contain subaction mask data")

    zero_sub1 = torch.zeros_like(template_entry["subaction_1"], dtype=torch.bool)
    zero_sub2 = torch.zeros_like(template_entry["subaction_2"], dtype=torch.bool)
    zero_sub3 = torch.zeros_like(template_entry["subaction_3"], dtype=torch.bool)

    sub1_stack, sub2_stack, sub3_stack = [], [], []
    for idx in range(PRIMARY_ACTION_COUNT):
      entry = self.__submask_entry(sub_masks_container, idx)
      if entry is None:
        sub1_stack.append(zero_sub1)
        sub2_stack.append(zero_sub2)
        sub3_stack.append(zero_sub3)
        continue

      sub1_stack.append(entry["subaction_1"].to(dtype=torch.bool))
      sub2_stack.append(entry["subaction_2"].to(dtype=torch.bool))
      sub3_stack.append(entry["subaction_3"].to(dtype=torch.bool))

    stacked_sub1 = torch.stack(sub1_stack, dim=1)
    stacked_sub2 = torch.stack(sub2_stack, dim=1)
    stacked_sub3 = torch.stack(sub3_stack, dim=1)

    batch_size = stacked_sub1.size(0)
    index_tensor = self.__normalize_primary_action_index(primary_action_index, batch_size)
    mask_width = stacked_sub1.size(-1)
    gather_idx = index_tensor.view(batch_size, 1, 1).expand(-1, 1, mask_width)

    sub1_mask = torch.gather(stacked_sub1, 1, gather_idx).squeeze(1)
    sub2_mask = torch.gather(stacked_sub2, 1, gather_idx).squeeze(1)
    sub3_mask = torch.gather(stacked_sub3, 1, gather_idx).squeeze(1)

    return (
      self.__maybe_squeeze_batch(sub1_mask, squeeze_batch),
      self.__maybe_squeeze_batch(sub2_mask, squeeze_batch),
      self.__maybe_squeeze_batch(sub3_mask, squeeze_batch),
    )

  def decode_actions(self, flat_hidden, target_matrix=None, state=None):
    if target_matrix is None:
      raise ValueError("decode_actions requires target_matrix for unit selections")

    if target_matrix.dim() == 2:
      target_matrix = target_matrix.unsqueeze(0)
    elif target_matrix.dim() == 1:
      target_matrix = target_matrix.unsqueeze(0).unsqueeze(0)

    primary_action_embeddings = self.__batch_embed_primary_actions()
    mask_observations = self.__mask_observations_from_state(state)
    primary_action_mask = None
    subaction_masks = None

    if mask_observations is None:
      raise ValueError("mask_observations is None when decode_actions is called")

    primary_action_mask = self.build_primary_action_mask_tensor(mask_observations)

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

    def _pad_or_trim_to_index_dim(tensor: torch.Tensor) -> torch.Tensor:
      current = tensor.size(-1)
      target = MAX_INDEX_SIZE
      if current == target:
        return tensor
      if current > target:
        return tensor[..., :target]
      pad_size = target - current
      pad_shape = (*tensor.shape[:-1], pad_size)
      pad_value = torch.finfo(tensor.dtype).min
      pad_tensor = torch.full(pad_shape, pad_value, device=tensor.device, dtype=tensor.dtype)
      return torch.cat([tensor, pad_tensor], dim=-1)

    primary_action_logits = projected_hidden @ primary_action_embeddings.T
    primary_action_logits = self.__mask_logits(primary_action_logits, primary_action_mask)
    primary_action_probs = torch.softmax(primary_action_logits, dim=-1)
    chosen_action = primary_action_probs.argmax(dim=-1)

    subaction_masks = self.build_subaction_mask_tensors(chosen_action, mask_observations)

    gate_1_weights = torch.sigmoid(self.gate_1_embeder(chosen_action)).unsqueeze(1)
    gate_2_weights = torch.sigmoid(self.gate_2_embeder(chosen_action)).unsqueeze(1)

    unit1_logits = torch.sum(
      (target_matrix * gate_1_weights) * unit1_projected_hidden.unsqueeze(1),
      dim=-1,
    )
    unit2_logits = torch.sum(
      (target_matrix * gate_2_weights) * unit2_projected_hidden.unsqueeze(1),
      dim=-1,
    )

    unit1_logits = _pad_or_trim_to_index_dim(unit1_logits)
    unit2_logits = _pad_or_trim_to_index_dim(unit2_logits)
    unit1_logits = self.__mask_logits(unit1_logits, subaction_masks[0])

    bins2_logits = bins2_projected_hidden
    bins3_logits = bins3_projected_hidden

    requires_unit_subaction2 = (chosen_action == ACT_ATTACK) | (chosen_action == ACT_ATTACH_WEAPON_FROM_HAND)
    requires_unit_subaction2 = requires_unit_subaction2.unsqueeze(-1)
    subaction2_logits = torch.where(
      requires_unit_subaction2,
      unit2_logits,
      bins2_logits,
    )

    subaction2_logits = self.__mask_logits(subaction2_logits, subaction_masks[1])
    bins3_logits = self.__mask_logits(bins3_logits, subaction_masks[2])

    logits = (
      primary_action_logits,
      unit1_logits,
      subaction2_logits,
      bins3_logits,
    )

    values = self.value_fn(projected_hidden)
    
    return logits, values
