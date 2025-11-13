import pufferlib.models
import pufferlib.pytorch

from torch import nn
import torch

ALLEY_AND_GARDEN_ZONE_COUNT = 5
MAX_PLAYERS_PER_MATCH = 2
CARD_TYPE_COUNT = 7
CARD_ID_COUNT = 37
GAME_PHASE_COUNT = 8
PRIMARY_ACTION_COUNT = 15

OWNER_ENC_OUTPUT_SIZE = 4
GARDEN_OR_ALLEY_ENC_OUTPUT_SIZE = 4
CARD_TYPE_ENC_OUTPUT_SIZE = 4
ALLEY_AND_GARDEN_ZONE_ENC_OUTPUT_SIZE = 4
CARD_ID_ENC_OUTPUT_SIZE = 8
GAME_PHASE_ENC_OUTPUT_SIZE = 4
PRIMARY_ACTION_ENC_OUTPUT_SIZE = 4

PROCESS_SET_HIDDEN_SIZE = 256
PROCESS_SET_OUTPUT_SIZE = 64

class TCGLSTM(pufferlib.models.LSTMWrapper):
  def __init__(self, env, policy, input_size=512, hidden_size=4096):
    super().__init__(env, policy, input_size, hidden_size)

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
    set_embeddings = self.activation(self.fc2(hidden))
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
    self.alley_garden_zone_encoder = nn.Sequential(
      nn.Embedding(ALLEY_AND_GARDEN_ZONE_COUNT, ALLEY_AND_GARDEN_ZONE_ENC_OUTPUT_SIZE),
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

    # Process Set Processors
    self.hand_set_processor = ProcessSetProcessor(CARD_ID_ENC_OUTPUT_SIZE + CARD_TYPE_ENC_OUTPUT_SIZE + 5)

    # Observation structure dtype
    self._obs_struct_dtype = getattr(env, "obs_dtype", None)
    if self._obs_struct_dtype is None:
      raise AttributeError("env must expose obs_dtype for nativize")

  def __process_hand(self, hand_features):
    card_ids = hand_features["card_id"].long().clamp(min=0, max=CARD_ID_COUNT - 1)
    type_ids = hand_features["type_id"].long().clamp(min=0, max=CARD_TYPE_COUNT - 1)

    type_embeddings = self.card_type_encoder[0](type_ids)
    card_embeddings = self.card_id_encoder[0](card_ids)

    scalar_stack = torch.stack(
      [
        hand_features["ikz_cost"].float(),
        hand_features["attack"].float(),
        hand_features["health"].float(),
        hand_features["gate_points"].float(),
        hand_features["zone_index"].float(),
      ],
      dim=-1,
    )

    hand_input = torch.cat([type_embeddings, card_embeddings, scalar_stack], dim=-1)
    return self.hand_set_processor(hand_input)

  def __process_leader(self, leader_vector):
    raise NotImplementedError("TCG process leader data not implemented")

  def __process_gate(self, gate_vector):
    raise NotImplementedError("TCG process gate data not implemented")

  def __process_ikz_area(self, ikz_area_matrix):
    raise NotImplementedError("TCG process ikz area data not implemented")

  def __process_discard(self, discard_matrix):
    raise NotImplementedError("TCG process discard data not implemented")

  def __process_alley(self, alley_matrix):
    raise NotImplementedError("TCG process alley data not implemented")

  def __process_garden(self, garden_matrix):
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
    hidden = self.encode_observations(x)
    actions, value = self.decode_actions(hidden)
    return actions, value

  def forward_train(self, x, state=None):
    return self.forward(x, state)
  
  def encode_observations(self, observations, state=None):
    obs_tensor = observations if torch.is_tensor(observations) else torch.as_tensor(observations)
    squeeze_batch = obs_tensor.dim() == 1
    if squeeze_batch:
      obs_tensor = obs_tensor.unsqueeze(0)

    hand_features = self.__get_organized_obs_data(obs_tensor)
    
    hand_embeddings, hand_pooled = self.__process_hand(hand_features)

    if squeeze_batch:
      hand_pooled = hand_pooled.squeeze(0)

    return hand_pooled

  def decode_actions(self, flat_hidden):
    raise NotImplementedError("TCG decode actions not implemented")
