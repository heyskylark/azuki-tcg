import torch

primary_action_mask = torch.tensor([[True, True, True, False, False, False, False, False, False, False, False, False, False]])
chosen_actions = torch.tensor([1,2])

batch_indexes = torch.arange(2).unsqueeze(1)
batch_indexes = batch_indexes.expand(2, 1024)

legal_primary_values = torch.zeros(2, 1024)
legal_primary_values[0, 1] = 1
legal_primary_values[0, 2] = 1
legal_primary_values[0, 3] = 1
legal_primary_values[1, 1] = 1
legal_primary_values[1, 2] = 1
legal_primary_values[1, 3] = 1

legal_subaction1_matches = (legal_primary_values == chosen_actions.view(-1, 1))

legal_subaction1_indexes = torch.zeros(2, 1024, dtype=torch.long)
legal_subaction1_indexes[0, 1] = 1
legal_subaction1_indexes[0, 2] = 1
legal_subaction1_indexes[0, 3] = 1
legal_subaction1_indexes[1, 1] = 1
legal_subaction1_indexes[1, 2] = 1
legal_subaction1_indexes[1, 3] = 1

legal_subaction_mask = torch.zeros(2, 7, dtype=torch.bool)
legal_subaction_mask[batch_indexes[legal_subaction1_matches], legal_subaction1_indexes[legal_subaction1_matches]] = True

unit1_logits = torch.tensor([
  [1, 2, 3, 4, 5, 6, 7],
  [1, 2, 3, 4, 5, 6, 7],
], dtype=torch.float32)
unit1_logits = unit1_logits.masked_fill(~legal_subaction_mask, -1e9)
subaction1_probs = torch.softmax(unit1_logits, dim=-1)
chosen_subaction1_indexes = subaction1_probs.argmax(dim=-1)

legal_subaction2_indexes = torch.zeros(2, 1024, dtype=torch.long)
legal_subaction2_indexes[0, 1] = 1
legal_subaction2_indexes[0, 2] = 2
legal_subaction2_indexes[0, 3] = 4
legal_subaction2_indexes[1, 1] = 1
legal_subaction2_indexes[1, 2] = 2
legal_subaction2_indexes[1, 3] = 5

legal_subaction2_matches = legal_subaction1_matches & (legal_subaction1_indexes == chosen_subaction1_indexes.view(-1, 1))
legal_subaction2_mask = torch.zeros(2, 7, dtype=torch.bool)
legal_subaction2_mask[batch_indexes[legal_subaction2_matches], legal_subaction2_indexes[legal_subaction2_matches]] = True
print(f"legal subaction 2 mask: {legal_subaction2_mask}")
unit2_logits = torch.tensor([
  [6, 7, 8, 9, 10, 11, 12],
  [8, 9, 10, 11, 12, 13, 14],
], dtype=torch.float32)
unit2_logits = unit2_logits.masked_fill(~legal_subaction2_mask, -1e9)
subaction2_probs = torch.softmax(unit2_logits, dim=-1)
chosen_subaction2s = subaction2_probs.argmax(dim=-1)
print(f"chosen subaction 2s: {chosen_subaction2s}")