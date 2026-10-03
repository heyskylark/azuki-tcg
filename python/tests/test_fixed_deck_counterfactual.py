import unittest

from run_fixed_deck_counterfactual import _balanced_shard_tasks, _task_schedule


class FixedDeckCounterfactualShardTests(unittest.TestCase):
  def setUp(self):
    gates = [f"gate-{index:02d}" for index in range(16)]
    self.payload = {
      "arms": {
        arm: {gate: {} for gate in gates}
        for arm in ("native", "sibling_gate", "sibling_leader", "sibling_both")
      }
    }
    self.tasks = _task_schedule(self.payload, (1, 9, 17))

  def test_balanced_shards_are_deterministic_complete_and_pair_preserving(self):
    first = [
      _balanced_shard_tasks(self.tasks, 4, 12, shard)
      for shard in range(12)
    ]
    second = [
      _balanced_shard_tasks(self.tasks, 4, 12, shard)
      for shard in range(12)
    ]
    self.assertEqual(first, second)
    self.assertTrue(all(len(shard) == 32 for shard in first))

    assignments = {}
    seen_indices = set()
    for shard_index, shard in enumerate(first):
      for global_index, task in shard:
        self.assertNotIn(global_index, seen_indices)
        seen_indices.add(global_index)
        group = (task["block_id"], task["candidate_seat"])
        assignments.setdefault(group, set()).add(shard_index)

    self.assertEqual(seen_indices, set(range(len(self.tasks))))
    self.assertTrue(all(len(shards) == 1 for shards in assignments.values()))


if __name__ == "__main__":
  unittest.main()
