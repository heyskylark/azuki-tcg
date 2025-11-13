import pufferlib.models
import pufferlib.pytorch

from torch import nn
import torch

MAX_PLAYERS_PER_MATCH = 2
CARD_TYPE_COUNT = 7
CARD_ID_COUNT = 37
GAME_PHASE_COUNT = 8
PRIMARY_ACTION_COUNT = 15
MAX_INDEX_SIZE = 50

OWNER_ENC_OUTPUT_SIZE = 4
GARDEN_OR_ALLEY_ENC_OUTPUT_SIZE = 4
CARD_TYPE_ENC_OUTPUT_SIZE = 4
CARD_ID_ENC_OUTPUT_SIZE = 8
GAME_PHASE_ENC_OUTPUT_SIZE = 4
PRIMARY_ACTION_ENC_OUTPUT_SIZE = 4
INDEX_ENC_OUTPUT_SIZE = 16

PROCESS_SET_HIDDEN_SIZE = 256
PROCESS_SET_OUTPUT_SIZE = 64

class TCGLSTM(pufferlib.models.LSTMWrapper):
  def __init__(self, env, policy, input_size=512, hidden_size=4096):
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
    logits, values = self.policy.decode_actions(hidden, target_matrix=target_matrix)
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
    logits, values = self.policy.decode_actions(flat_hidden, target_matrix=target_matrix)
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
      nn.Embedding(PRIMARY_ACTION_COUNT, PRIMARY_ACTION_ENC_OUTPUT_SIZE),
      nn.Flatten(),
    )

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
    self._obs_struct_dtype = getattr(env, "obs_dtype", None)
    if self._obs_struct_dtype is None:
      raise AttributeError("env must expose obs_dtype for nativize")

  def __process_hand(self, hand_features):
    card_ids = hand_features["card_id"].long()
    type_ids = hand_features["type_id"].long()
    hand_indices = hand_features["zone_index"].long()

    type_embeddings = self.card_type_encoder[0](type_ids)
    card_embeddings = self.card_id_encoder[0](card_ids)
    hand_index_embeddings = self.hand_index_encoder[0](hand_indices)

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

  def __get_organized_obs_data(self, obs_tensor):
    sample_param = next(self.parameters(), None)
    target_device = sample_param.device if sample_param is not None else obs_tensor.device
    obs_tensor = obs_tensor.to(target_device)

    structured_obs = pufferlib.pytorch.nativize_tensor(obs_tensor, self._obs_struct_dtype)
    player_obs = structured_obs["player"]
    hand = player_obs["hand"]

    def stack_hand_field(field_name):
      return torch.stack([slot[field_name] for slot in hand], dim=1)

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
    target_vector, target_matrix = self.encode_observations(x)
    actions, value = self.decode_actions(target_vector, target_matrix=target_matrix)
    return actions, value

  def forward_train(self, x, state=None):
    return self.forward(x, state)
  
  def encode_observations(self, observations, state=None):
    obs_tensor = observations if torch.is_tensor(observations) else torch.as_tensor(observations)
    squeeze_batch = obs_tensor.dim() == 1
    if squeeze_batch:
      obs_tensor = obs_tensor.unsqueeze(0)

    hand_features = self.__get_organized_obs_data(obs_tensor)
    
    target_matrix, target_vector = self.__process_hand(hand_features)

    if squeeze_batch:
      target_vector = target_vector.squeeze(0)
      target_matrix = target_matrix.squeeze(0)

    return target_vector, target_matrix

  def decode_actions(self, flat_hidden, target_matrix=None):
    raise NotImplementedError("TCG decode actions not implemented")
