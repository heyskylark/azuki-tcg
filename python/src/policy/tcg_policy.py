import pufferlib
import pufferlib.models

from torch import nn
import torch
import torch.nn.functional as F

ALLEY_AND_GARDEN_ZONE_COUNT = 5
MAX_PLAYERS_PER_MATCH = 2
CARD_TYPE_COUNT = 7
CARD_ID_COUNT = 37
GAME_PHASE_COUNT = 8
PRIMARY_ACTION_COUNT = 15

class TCGLSTM(pufferlib.models.LSTMWrapper):
  def __init__(self, env, policy, input_size=512, hidden_size=4096):
    super().__init__(env, policy, input_size, hidden_size)

class TCG(nn.Module):
  def __init__(self, env, **kwargs):
    super().__init__()

    # Categorical encodings
    self.owner_encoder = nn.Sequential(
      nn.Embedding(MAX_PLAYERS_PER_MATCH, 4),
      nn.Flatten(),
    )
    self.garden_or_alley_encoder = nn.Sequential(
      nn.Embedding(2, 4),
      nn.Flatten(),
    )
    self.card_type_encoder = nn.Sequential(
      nn.Embedding(CARD_TYPE_COUNT, 4),
      nn.Flatten(),
    )
    self.alley_garden_zone_encoder = nn.Sequential(
      nn.Embedding(ALLEY_AND_GARDEN_ZONE_COUNT, 4),
      nn.Flatten(),
    )
    self.card_id_encoder = nn.Sequential(
      nn.Embedding(CARD_ID_COUNT, 8),
      nn.Flatten(),
    )
    self.game_state_encoder = nn.Sequential(
      nn.Embedding(GAME_PHASE_COUNT, 4),
      nn.Flatten(),
    )
    self.primary_action_encoder = nn.Sequential(
      nn.Embedding(PRIMARY_ACTION_COUNT, 4),
      nn.Flatten(),
    )

  def forward(self, x, state=None):
    raise NotImplementedError("TCG forward pass not implemented")

  def forward_train(self, x, state=None):
    return self.forward(x, state)

  def encode_observations(self, observations, state=None):
    raise NotImplementedError("TCG encode observations not implemented")

  def decode_actions(self, flat_hidden):
    raise NotImplementedError("TCG decode actions not implemented")
