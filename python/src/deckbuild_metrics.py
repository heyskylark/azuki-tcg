"""Python-side deckbuild metrics/snapshots for the native deck-building path.

The C env exports one record per completed episode (drafted decks, win,
behavior rates). This module rebuilds the exact metric key surface the legacy
DeckBuildingParallelEnv emitted (deckbuild/*, azk_step_deckbuild/*,
deckbuild_gatecard/*, deckbuild_result/*) as per-window MEANS merged into the
vec_log dict, and writes the same snapshot JSONL records.

It also flattens the wrapper's deck-build catalog into the integer arrays the
C draft state machine consumes, so candidate ordering matches the legacy
wrapper by construction.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from deck_building import (
  DeckBuildCatalog,
  MAX_MAIN_COPIES,
  build_deck_build_catalog,
)
from training_deck_pool import load_training_deck_pool

MAX_DECK_SIZE = 50
DEFAULT_SNAPSHOT_EVERY = 25
_BEHAVIOR_KEYS = (
  "attack_rate",
  "spell_rate",
  "weapon_rate",
  "portal_rate",
  "play_entity_rate",
  "noop_rate",
  "episode_length",
  "leader_health",
  "gate_ability_outcomes",
  "leader_ability_outcomes",
)

_REWARD_COMPONENT_NAMES = (
  "terminal_outcome",
  "truncation_timeout",
  "truncation_leader_edge",
  "truncation_board_edge",
  "potential_leader_health",
  "potential_garden_attack",
  "potential_untapped_garden",
  "potential_untapped_ikz",
  "direct_leader_edge",
  "direct_board_edge",
  "noop_penalty",
  "portal_gp",
  "portal_outcome",
  "early_tempo",
  "damage_mitigation",
  "temporary_charge",
  "temporary_attack",
  "entity_damage_exchange",
  "generated_ikz_conversion",
  "response_reserve",
  "gate_ability_outcome",
  "leader_ability_outcome",
)
_REWARD_STAT_NAMES = (
  "raw_sum",
  "raw_abs_sum",
  "raw_discounted_sum",
  "raw_max_abs",
  "raw_positive_count",
  "raw_negative_count",
  "scaled_sum",
  "scaled_abs_sum",
  "scaled_discounted_sum",
  "scaled_max_abs",
  "scaled_positive_count",
  "scaled_negative_count",
)
_ACTION_TYPE_NAMES = (
  "noop",
  "play_entity_to_garden",
  "play_entity_to_alley",
  "unused_3",
  "unused_4",
  "unused_5",
  "attack",
  "attach_weapon_from_hand",
  "play_spell_from_hand",
  "declare_defender",
  "gate_portal",
  "activate_garden_or_leader_ability",
  "activate_alley_ability",
  "select_cost_target",
  "select_effect_target",
  "unused_15",
  "confirm_ability",
  "unused_17",
  "select_from_selection",
  "bottom_deck_card",
  "bottom_deck_all",
  "select_to_alley",
  "select_to_equip",
  "select_to_garden",
  "top_deck_card",
  "mulligan_shuffle",
)
_REWARD_TURN_BUCKET_NAMES = ("turn_1_2", "turn_3_4", "turn_5_8", "turn_9_16", "turn_17_plus")


def _cost_bucket(cost: int) -> str:
  if cost <= 1:
    return "0_1"
  if cost <= 3:
    return "2_3"
  if cost <= 5:
    return "4_5"
  return "6_plus"


class NativeDeckbuildHelper:
  def __init__(
    self,
    *,
    deck_pool=None,
    snapshot_dir=None,
    snapshot_every=None,
    uniform_assignment: bool = False,
  ):
    pool = tuple(deck_pool) if deck_pool is not None else load_training_deck_pool()
    self.catalog: DeckBuildCatalog = build_deck_build_catalog(pool)
    records = self.catalog.records_by_def_id
    self._gate_elements = tuple(
      sorted({records[g].element for g in self.catalog.gate_def_id_population})
    )
    self._gate_pairs = tuple(
      f"{a}_vs_{b}" for a in self._gate_elements for b in self._gate_elements
    )
    self._gate_pairs_unordered = tuple(
      sorted({"_vs_".join(sorted((a, b))) for a in self._gate_elements for b in self._gate_elements})
    )
    self._main_elements = tuple(
      sorted(
        {
          records[def_id].element
          for def_ids in self.catalog.main_def_ids_by_element.values()
          for def_id in def_ids
        }
      )
    )
    self._main_types = ("ENTITY", "SPELL", "WEAPON")
    self._uniform_assignment = bool(uniform_assignment)

    self._snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
    try:
      self._snapshot_every = max(1, int(snapshot_every)) if snapshot_every else DEFAULT_SNAPSHOT_EVERY
    except (TypeError, ValueError):
      self._snapshot_every = DEFAULT_SNAPSHOT_EVERY
    self._snapshot_path = None
    self._completed_episodes = 0

  # ---- catalog flattening for the C draft state machine ---------------------
  def catalog_kwargs(self) -> dict:
    records = self.catalog.records_by_def_id
    gate_ids = sorted({int(g) for g in self.catalog.gate_def_id_population})
    leader_flat: list[int] = []
    leader_offsets = [0]
    main_flat: list[int] = []
    main_offsets = [0]
    for gate in gate_ids:
      element = records[gate].element
      leaders = self.catalog.leader_def_ids_by_element[element]
      mains = self.catalog.main_def_ids_by_element[element]
      leader_flat.extend(int(x) for x in leaders)
      leader_offsets.append(len(leader_flat))
      main_flat.extend(int(x) for x in mains)
      main_offsets.append(len(main_flat))
    # Same-element partner per gate (cyclic next among that element's gates,
    # -1 if the element has a single gate) for sibling-matchup oversampling.
    gates_by_element: dict[str, list[int]] = {}
    for gate in gate_ids:
      gates_by_element.setdefault(records[gate].element, []).append(gate)
    sibling_by_gate: dict[int, int] = {}
    for element_gates in gates_by_element.values():
      for index, gate in enumerate(element_gates):
        sibling_by_gate[gate] = (
          element_gates[(index + 1) % len(element_gates)] if len(element_gates) > 1 else -1
        )
    return {
      "draft_gate_def_ids": gate_ids,
      "draft_gate_sibling_def_ids": [sibling_by_gate[g] for g in gate_ids],
      "draft_gate_population": [int(g) for g in self.catalog.gate_def_id_population],
      "draft_leader_flat": leader_flat,
      "draft_leader_offsets": leader_offsets,
      "draft_main_flat": main_flat,
      "draft_main_offsets": main_offsets,
      "draft_ikz_def_id": int(self.catalog.ikz_record.card_def_id),
    }

  # ---- per-episode metric computation ---------------------------------------
  def _player_metrics(self, player: dict, opponent: dict) -> dict:
    records = self.catalog.records_by_def_id
    gate = records.get(int(player["gate"]))
    leader = records.get(int(player["leader"]))
    opp_gate = records.get(int(opponent["gate"]))
    main_ids = [int(c) for c in player["main"] if int(c) >= 0]
    counts = Counter(main_ids)
    total = len(main_ids)
    type_counts: Counter = Counter()
    element_counts: Counter = Counter()
    cost_sum = 0
    bucket_counts: Counter = Counter()
    for def_id in main_ids:
      rec = records.get(def_id)
      if rec is None:
        continue
      type_counts[rec.card_type] += 1
      element_counts[rec.element] += 1
      cost_sum += rec.ikz_cost
      bucket_counts[_cost_bucket(rec.ikz_cost)] += 1
    copy_hist = Counter(counts.values())
    unique = len(counts)
    probs = [qty / total for qty in counts.values()] if total else []
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    max_entropy = math.log(unique) if unique > 1 else 0.0

    m: dict[str, float] = {}
    m["deckbuild/completed"] = 1.0
    m["deckbuild/picks"] = float(
      MAX_DECK_SIZE if self._uniform_assignment else 1 + MAX_DECK_SIZE
    )
    m["deckbuild/main_count"] = float(total)
    m["deckbuild/main_unique"] = float(unique)
    m["deckbuild/main_unique_share"] = unique / MAX_DECK_SIZE
    m["deckbuild/main_avg_copies_per_unique"] = total / max(unique, 1)
    m["deckbuild/main_max_copy_count"] = float(max(counts.values()) if counts else 0)
    m["deckbuild/main_copy_entropy"] = entropy
    m["deckbuild/main_copy_entropy_norm"] = entropy / max_entropy if max_entropy > 0 else 0.0
    m["deckbuild/main_singleton_count"] = float(copy_hist.get(1, 0))
    m["deckbuild/main_pair_count"] = float(copy_hist.get(2, 0))
    m["deckbuild/main_triplet_count"] = float(copy_hist.get(3, 0))
    m["deckbuild/main_quad_count"] = float(copy_hist.get(4, 0))
    m["deckbuild/main_singleton_slot_share"] = copy_hist.get(1, 0) * 1 / MAX_DECK_SIZE
    m["deckbuild/main_pair_slot_share"] = copy_hist.get(2, 0) * 2 / MAX_DECK_SIZE
    m["deckbuild/main_triplet_slot_share"] = copy_hist.get(3, 0) * 3 / MAX_DECK_SIZE
    m["deckbuild/main_quad_slot_share"] = copy_hist.get(4, 0) * 4 / MAX_DECK_SIZE
    m["deckbuild/main_normal_share"] = element_counts.get("NORMAL", 0) / MAX_DECK_SIZE
    gate_element = gate.element if gate else "?"
    m["deckbuild/main_gate_element_share"] = element_counts.get(gate_element, 0) / MAX_DECK_SIZE
    opp_element = opp_gate.element if opp_gate else "?"
    gate_match = float(gate_element == opp_element)
    m["deckbuild/gate_match"] = gate_match
    for element in self._gate_elements:
      m[f"deckbuild/gate/{element}"] = float(gate_element == element)
      m[f"deckbuild/opponent_gate/{element}"] = float(opp_element == element)
      leader_element = leader.element if leader else "?"
      m[f"deckbuild/leader/{element}"] = float(leader_element == element)
    for card_type in self._main_types:
      m[f"deckbuild/main_type_count/{card_type}"] = float(type_counts.get(card_type, 0))
      m[f"deckbuild/main_type_share/{card_type}"] = type_counts.get(card_type, 0) / MAX_DECK_SIZE
    for element in self._main_elements:
      m[f"deckbuild/main_element_count/{element}"] = float(element_counts.get(element, 0))
      m[f"deckbuild/main_element_share/{element}"] = element_counts.get(element, 0) / MAX_DECK_SIZE
    avg_cost = cost_sum / MAX_DECK_SIZE
    m["deckbuild/main_avg_cost"] = avg_cost
    for bucket in ("0_1", "2_3", "4_5", "6_plus"):
      m[f"deckbuild/main_cost_share/{bucket}"] = bucket_counts.get(bucket, 0) / MAX_DECK_SIZE
    if leader is not None:
      m[f"deckbuild/leader_card/{leader.card_code}"] = 1.0

    if gate is not None:
      prefix = f"deckbuild_gatecard/{gate.card_code}"
      m[f"{prefix}/game"] = 1.0
      m[f"{prefix}/avg_cost"] = avg_cost
      m[f"{prefix}/main_unique"] = float(unique)
      m[f"{prefix}/copy_entropy_norm"] = m["deckbuild/main_copy_entropy_norm"]
      m[f"{prefix}/gate_element_share"] = m["deckbuild/main_gate_element_share"]
      for card_type in self._main_types:
        m[f"{prefix}/type_share/{card_type}"] = m[f"deckbuild/main_type_share/{card_type}"]
      for bucket in ("0_1", "2_3", "4_5", "6_plus"):
        m[f"{prefix}/cost_share/{bucket}"] = m[f"deckbuild/main_cost_share/{bucket}"]
      if leader is not None:
        m[f"{prefix}/leader/{leader.card_code}"] = 1.0

    # deckbuild_result/* — episode-end result metrics
    win = float(player.get("win", 0.0) or 0.0)
    r: dict[str, float] = {}
    r["deckbuild_result/game"] = 1.0
    r["deckbuild_result/win"] = win
    r["deckbuild_result/gate_match"] = gate_match
    r["deckbuild_result/gate_match_win_joint"] = win if gate_match else 0.0
    r["deckbuild_result/gate_mismatch"] = 1.0 - gate_match
    r["deckbuild_result/gate_mismatch_win_joint"] = 0.0 if gate_match else win
    r[f"deckbuild_result/gate/{gate_element}/game"] = 1.0
    r[f"deckbuild_result/gate/{gate_element}/win"] = win
    pair = f"{gate_element}_vs_{opp_element}"
    unordered = "_vs_".join(sorted((gate_element, opp_element)))
    r[f"deckbuild_result/gate_pair/{pair}/game"] = 1.0
    r[f"deckbuild_result/gate_pair/{pair}/win"] = win
    r[f"deckbuild_result/gate_pair_unordered/{unordered}/game"] = 1.0
    r[f"deckbuild_result/gate_pair_unordered/{unordered}/win"] = win
    branch = "gate_match" if gate_match else "gate_mismatch"
    r[f"deckbuild_result/{branch}/game"] = 1.0
    r[f"deckbuild_result/{branch}/win"] = win
    for element in self._gate_elements:
      one_hot = float(gate_element == element)
      r[f"deckbuild_result/gate_freq/{element}"] = one_hot
      r[f"deckbuild_result/gate_win_joint/{element}"] = win * one_hot
    for candidate in self._gate_pairs:
      one_hot = float(candidate == pair)
      r[f"deckbuild_result/gate_pair_freq/{candidate}"] = one_hot
      r[f"deckbuild_result/gate_pair_win_joint/{candidate}"] = win * one_hot
    for candidate in self._gate_pairs_unordered:
      one_hot = float(candidate == unordered)
      r[f"deckbuild_result/gate_pair_unordered_freq/{candidate}"] = one_hot
      r[f"deckbuild_result/gate_pair_unordered_win_joint/{candidate}"] = win * one_hot
    if gate is not None:
      gc = f"deckbuild_result/gatecard/{gate.card_code}"
      r[f"{gc}/game"] = 1.0
      r[f"{gc}/win"] = win
      for beh in _BEHAVIOR_KEYS:
        value = player.get(beh)
        if isinstance(value, (int, float)):
          r[f"{gc}/{beh}"] = float(value)
      ability = player.get("ability_rate")
      if isinstance(ability, (int, float)):
        r[f"{gc}/ability_rate"] = float(ability)

    out = {}
    for key, value in m.items():
      out[key] = value
      # Legacy wrapper mirrored every battle-start metric under azk_step_*.
      out[f"azk_step_{key}"] = value
    out.update(r)
    return out

  @staticmethod
  def _reward_length_bucket(episode_length: float) -> str:
    if episode_length <= 20:
      return "length_0_20"
    if episode_length <= 50:
      return "length_21_50"
    if episode_length <= 100:
      return "length_51_100"
    return "length_101_plus"

  @staticmethod
  def _accumulate_reward_entries(
    entries,
    prefixes: tuple[str, ...],
    sums: dict[str, float],
    counts: dict[str, int],
    maxima: dict[str, float],
    *,
    include_zeros: bool = False,
  ) -> None:
    parsed: dict[int, tuple | list] = {}
    if isinstance(entries, (list, tuple)):
      for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) != 13:
          continue
        component_id = int(entry[0])
        if 0 <= component_id < len(_REWARD_COMPONENT_NAMES):
          parsed[component_id] = entry
    component_ids = (
      range(len(_REWARD_COMPONENT_NAMES)) if include_zeros else parsed
    )
    for component_id in component_ids:
      component = _REWARD_COMPONENT_NAMES[component_id]
      entry = parsed.get(component_id)
      for stat_index, stat in enumerate(_REWARD_STAT_NAMES, start=1):
        value = float(entry[stat_index]) if entry is not None else 0.0
        for prefix in prefixes:
          key = f"{prefix}/{component}/{stat}"
          if stat.endswith("max_abs"):
            maxima[key] = max(maxima.get(key, 0.0), value)
          else:
            sums[key] += value
            counts[key] += 1

  def process_records(self, records: list[dict]) -> dict:
    """Aggregate drained per-episode records into mean metrics + snapshots."""
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    maxima: dict[str, float] = {}
    minima: dict[str, float] = {}
    telemetry_seen = False
    reference_counts: dict[str, int] = defaultdict(int)
    reference_total = 0

    def add_mean(key: str, value: float) -> None:
      sums[key] += float(value)
      counts[key] += 1

    for record in records:
      players = record.get("players") or []
      if len(players) != 2:
        continue
      prebuilt = bool(record.get("prebuilt", False))
      episode_length = float(record.get("episode_length", 0.0) or 0.0)
      add_mean("curriculum/prebuilt_game_fraction", float(prebuilt))
      add_mean("curriculum/drafted_game_fraction", float(not prebuilt))
      add_mean("curriculum/battle_length", episode_length)
      add_mean(
        "curriculum/prebuilt_battle_length",
        episode_length if prebuilt else 0.0,
      )
      add_mean(
        "curriculum/drafted_battle_length",
        0.0 if prebuilt else episode_length,
      )
      record_telemetry = record.get("reward_telemetry")
      if isinstance(record_telemetry, dict):
        telemetry_seen = True
        for field in (
          "raw_reconstruction_max_abs_error",
          "scaled_reconstruction_max_abs_error",
          "shaping_scale_max",
        ):
          key = f"reward_telemetry/{field}"
          maxima[key] = max(maxima.get(key, 0.0), float(record_telemetry.get(field, 0.0)))
        scale_min = float(record_telemetry.get("shaping_scale_min", 0.0))
        minima["reward_telemetry/shaping_scale_min"] = min(
          minima.get("reward_telemetry/shaping_scale_min", scale_min),
          scale_min,
        )
        step_count = int(record_telemetry.get("shaping_step_count", 0))
        add_mean("reward_telemetry/shaping_step_count", step_count)
        add_mean("reward_telemetry/gamma", float(record_telemetry.get("gamma", 0.0)))
        if step_count > 0:
          add_mean(
            "reward_telemetry/shaping_scale_mean",
            float(record_telemetry.get("shaping_scale_sum", 0.0)) / step_count,
          )

      for idx, player in enumerate(players):
        player = dict(player)
        player["episode_length"] = episode_length
        opponent = players[1 - idx]
        if not prebuilt:
          for key, value in self._player_metrics(player, opponent).items():
            add_mean(key, value)

        telemetry = player.get("reward_telemetry")
        if not isinstance(telemetry, dict):
          continue
        telemetry_seen = True
        gate_record = self.catalog.records_by_def_id.get(int(player.get("gate", -1)))
        leader_record = self.catalog.records_by_def_id.get(int(player.get("leader", -1)))
        win = float(player.get("win", 0.5) or 0.0)
        outcome = "winner" if win > 0.5 else "loser" if win < 0.5 else "draw"
        prefixes = [
          "reward_component/all",
          f"reward_component/outcome/{outcome}",
          f"reward_component/episode_length/{self._reward_length_bucket(episode_length)}",
        ]
        if gate_record is not None:
          prefixes.append(f"reward_component/gate/{gate_record.card_code}")
        if leader_record is not None:
          prefixes.append(f"reward_component/leader/{leader_record.card_code}")
        self._accumulate_reward_entries(
          telemetry.get("overall"),
          tuple(prefixes),
          sums,
          counts,
          maxima,
          include_zeros=True,
        )

        return_prefixes = [
          "reward_telemetry/return/all",
          f"reward_telemetry/return/outcome/{outcome}",
        ]
        for field in ("raw_shaping_return", "scaled_shaping_return", "terminal_return"):
          for prefix in return_prefixes:
            add_mean(f"{prefix}/{field}", float(telemetry.get(field, 0.0)))

        for action_slice in telemetry.get("by_action") or ():
          if not isinstance(action_slice, (list, tuple)) or len(action_slice) != 3:
            continue
          action_id = int(action_slice[0])
          if not 0 <= action_id < len(_ACTION_TYPE_NAMES):
            continue
          action_name = _ACTION_TYPE_NAMES[action_id]
          add_mean(
            f"reward_telemetry/action/{action_name}/step_count",
            float(action_slice[1]),
          )
          self._accumulate_reward_entries(
            action_slice[2],
            (f"reward_component/action/{action_name}",),
            sums,
            counts,
            maxima,
            include_zeros=True,
          )
        for turn_slice in telemetry.get("by_turn_bucket") or ():
          if not isinstance(turn_slice, (list, tuple)) or len(turn_slice) != 3:
            continue
          bucket_id = int(turn_slice[0])
          if not 0 <= bucket_id < len(_REWARD_TURN_BUCKET_NAMES):
            continue
          bucket_name = _REWARD_TURN_BUCKET_NAMES[bucket_id]
          add_mean(
            f"reward_telemetry/turn/{bucket_name}/step_count",
            float(turn_slice[1]),
          )
          self._accumulate_reward_entries(
            turn_slice[2],
            (f"reward_component/turn/{bucket_name}",),
            sums,
            counts,
            maxima,
            include_zeros=True,
          )

      # S4 reference-seat anchor: winrate of the drafting seat against fixed
      # reference decks — the external promotion yardstick.
      ref_seat = int(record.get("ref_seat", -1))
      if 0 <= ref_seat < len(players):
        reference_total += 1
        drafter = players[1 - ref_seat]
        fixed = players[ref_seat]
        add_mean("ref_anchor_winrate", float(drafter.get("win", 0.0) or 0.0))
        reference_counts[f"deck_index/{int(record.get('ref_deck_index', -1))}"] += 1
        reference_counts[f"seat/{ref_seat}"] += 1
        reference_counts[
          f"starting_player/{int(record.get('starting_player', -1))}"
        ] += 1
        for prefix, player in (("fixed", fixed), ("opponent", drafter)):
          gate = self.catalog.records_by_def_id.get(int(player.get("gate", -1)))
          leader = self.catalog.records_by_def_id.get(int(player.get("leader", -1)))
          if gate is not None:
            reference_counts[f"{prefix}_gate/{gate.card_code}"] += 1
          if leader is not None:
            reference_counts[f"{prefix}_leader/{leader.card_code}"] += 1
      add_mean("ref_seat_rate", 1.0 if 0 <= ref_seat < len(players) else 0.0)
      if not prebuilt:
        self._maybe_write_snapshot(record)
        self._completed_episodes += 1

    if telemetry_seen:
      for component in _REWARD_COMPONENT_NAMES:
        for stat in _REWARD_STAT_NAMES:
          key = f"reward_component/all/{component}/{stat}"
          if stat.endswith("max_abs"):
            maxima.setdefault(key, 0.0)
          elif key not in counts:
            sums[key] = 0.0
            counts[key] = 1
    if reference_total:
      for key, count in reference_counts.items():
        sums[f"reference_balance/{key}_share"] = float(count) / reference_total
        counts[f"reference_balance/{key}_share"] = 1
    metrics = {key: sums[key] / counts[key] for key in sums}
    metrics.update(maxima)
    metrics.update(minima)
    return metrics

  # ---- snapshots -------------------------------------------------------------
  def _maybe_write_snapshot(self, record: dict) -> None:
    if self._snapshot_dir is None:
      return
    if self._completed_episodes % self._snapshot_every != 0:
      return
    records_by_def_id = self.catalog.records_by_def_id
    players_out = []
    for player in record.get("players") or []:
      gate = records_by_def_id.get(int(player["gate"]))
      leader = records_by_def_id.get(int(player["leader"]))
      main_ids = [int(c) for c in player["main"] if int(c) >= 0]
      main_counts = Counter(main_ids)
      main = {}
      cost_sum = 0
      for def_id in sorted(main_counts):
        rec = records_by_def_id.get(def_id)
        code = rec.card_code if rec else str(def_id)
        main[code] = int(main_counts[def_id])
        cost_sum += (rec.ikz_cost if rec else 0) * main_counts[def_id]
      summary = {
        "gate": gate.card_code if gate else None,
        "leader": leader.card_code if leader else None,
        "main": main,
        "avg_cost": round(cost_sum / MAX_DECK_SIZE, 3),
        "win": float(player.get("win", 0.0) or 0.0),
      }
      for beh in ("attack_rate", "spell_rate", "weapon_rate", "portal_rate",
                  "play_entity_rate", "noop_rate", "leader_health"):
        value = player.get(beh)
        if isinstance(value, (int, float)):
          summary[beh] = round(float(value), 4)
      eplen = record.get("episode_length")
      if isinstance(eplen, (int, float)):
        summary["episode_length"] = round(float(eplen), 4)
      reward_telemetry = player.get("reward_telemetry")
      if isinstance(reward_telemetry, dict):
        summary["reward_telemetry"] = reward_telemetry
      players_out.append(summary)

    payload = {
      "ts": round(time.time(), 1),
      "episode": self._completed_episodes,
      "seed": int(record.get("seed", 0)),
      "ref_seat": int(record.get("ref_seat", -1)),
      "players": players_out,
    }
    reward_telemetry = record.get("reward_telemetry")
    if isinstance(reward_telemetry, dict):
      payload["reward_telemetry"] = reward_telemetry
    try:
      if self._snapshot_path is None:
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_path = self._snapshot_dir / f"decks_pid{os.getpid()}.jsonl"
      with self._snapshot_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except OSError:
      pass
