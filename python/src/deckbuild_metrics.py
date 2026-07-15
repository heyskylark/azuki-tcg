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
)


def _cost_bucket(cost: int) -> str:
  if cost <= 1:
    return "0_1"
  if cost <= 3:
    return "2_3"
  if cost <= 5:
    return "4_5"
  return "6_plus"


class NativeDeckbuildHelper:
  def __init__(self, *, deck_pool=None, snapshot_dir=None, snapshot_every=None):
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
    m["deckbuild/picks"] = float(1 + MAX_DECK_SIZE)
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

  def process_records(self, records: list[dict]) -> dict:
    """Aggregate drained per-episode records into mean metrics + snapshots."""
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for record in records:
      players = record.get("players") or []
      if len(players) != 2:
        continue
      for idx, player in enumerate(players):
        player = dict(player)
        player["episode_length"] = record.get("episode_length", 0.0)
        opponent = players[1 - idx]
        for key, value in self._player_metrics(player, opponent).items():
          sums[key] += value
          counts[key] += 1
      # S4 reference-seat anchor: winrate of the drafting seat against fixed
      # reference decks — the external promotion yardstick.
      ref_seat = int(record.get("ref_seat", -1))
      if 0 <= ref_seat < len(players):
        drafter = players[1 - ref_seat]
        sums["ref_anchor_winrate"] += float(drafter.get("win", 0.0) or 0.0)
        counts["ref_anchor_winrate"] += 1
      sums["ref_seat_rate"] += 1.0 if 0 <= ref_seat < len(players) else 0.0
      counts["ref_seat_rate"] += 1
      self._maybe_write_snapshot(record)
      self._completed_episodes += 1
    return {key: sums[key] / counts[key] for key in sums}

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
      players_out.append(summary)

    payload = {
      "ts": round(time.time(), 1),
      "episode": self._completed_episodes,
      "seed": int(record.get("seed", 0)),
      "ref_seat": int(record.get("ref_seat", -1)),
      "players": players_out,
    }
    try:
      if self._snapshot_path is None:
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_path = self._snapshot_dir / f"decks_pid{os.getpid()}.jsonl"
      with self._snapshot_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except OSError:
      pass
