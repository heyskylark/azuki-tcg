from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from action import ACTION_COMPONENT_COUNT, ActionType, build_action_space
from observation import (
  ACTION_TYPE_COUNT,
  ALLEY_SIZE,
  DECK_CONTEXT_MODE_BATTLE,
  DECK_CONTEXT_MODE_PICK_LEADER,
  DECK_CONTEXT_MODE_PICK_MAIN,
  GARDEN_SIZE,
  IKZ_AREA_SIZE,
  LEGAL_ACTION_UNUSED,
  MAX_ATTACHED_WEAPONS,
  MAX_DECK_BUILD_CANDIDATES,
  MAX_DECK_SIZE,
  MAX_HAND_SIZE,
  MAX_LEGAL_ACTIONS_COUNT,
  MAX_SELECTION_ZONE_SIZE,
  RECENT_ACTION_HISTORY_LEN,
  build_deck_context_space,
)
from training_deck_pool import NativeDeck, NativeDeckPool

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_CARD_METADATA_PATH = REPO_ROOT / "python" / "config" / "policy_card_metadata_v1.json"

IKZ_CARD_CODE = "IKZ-001"
IKZ_CARD_COUNT = 10
MAX_MAIN_COPIES = 4
MAIN_CARD_TYPES = frozenset(("ENTITY", "SPELL", "WEAPON"))
LEADER_CARD_TYPE = "LEADER"
GATE_CARD_TYPE = "GATE"
NORMAL_ELEMENT = "NORMAL"

# (metric name, native terminal-log info key) pairs surfaced per gate card.
_BEHAVIOR_INFO_KEYS = (
  ("attack_rate", "azk_attack_selected_rate"),
  ("spell_rate", "azk_play_spell_from_hand_selected_rate"),
  ("weapon_rate", "azk_attach_weapon_from_hand_selected_rate"),
  ("portal_rate", "azk_gate_portal_selected_rate"),
  ("play_entity_rate", "azk_play_selected_rate"),
  ("noop_rate", "azk_noop_selected_rate"),
  ("episode_length", "azk_episode_length"),
  ("leader_health", "leader_health"),
)

SNAPSHOT_DIR_ENV = "AZK_DECKBUILD_SNAPSHOT_DIR"
SNAPSHOT_EVERY_ENV = "AZK_DECKBUILD_SNAPSHOT_EVERY"
DEFAULT_SNAPSHOT_EVERY = 25


@dataclass(frozen=True)
class DeckCardRecord:
  card_code: str
  card_def_id: int
  card_type: str
  element: str
  ikz_cost: int = 0


@dataclass(frozen=True)
class DeckBuildCatalog:
  records_by_def_id: dict[int, DeckCardRecord]
  records_by_code: dict[str, DeckCardRecord]
  leader_def_ids_by_element: dict[str, tuple[int, ...]]
  main_def_ids_by_element: dict[str, tuple[int, ...]]
  gate_def_id_population: tuple[int, ...]
  ikz_record: DeckCardRecord


@dataclass
class PlayerDeckBuildState:
  gate_card_def_id: int
  leader_card_def_id: int
  main_card_def_ids: list[int]
  main_count: int

  @classmethod
  def create(cls, gate_card_def_id: int) -> "PlayerDeckBuildState":
    return cls(
      gate_card_def_id=int(gate_card_def_id),
      leader_card_def_id=-1,
      main_card_def_ids=[-1 for _ in range(MAX_DECK_SIZE)],
      main_count=0,
    )

  @property
  def is_complete(self) -> bool:
    return self.leader_card_def_id >= 0 and self.main_count == MAX_DECK_SIZE

  @property
  def mode(self) -> int:
    if self.is_complete:
      return DECK_CONTEXT_MODE_BATTLE
    if self.leader_card_def_id < 0:
      return DECK_CONTEXT_MODE_PICK_LEADER
    return DECK_CONTEXT_MODE_PICK_MAIN

  def main_copy_counts(self) -> Counter[int]:
    return Counter(card_id for card_id in self.main_card_def_ids[: self.main_count] if card_id >= 0)


def _load_policy_card_records(path: Path = POLICY_CARD_METADATA_PATH) -> tuple[DeckCardRecord, ...]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  records = payload.get("records")
  if not isinstance(records, list) or not records:
    raise ValueError(f"Policy card metadata must contain non-empty records: {path}")

  out: list[DeckCardRecord] = []
  for index, record in enumerate(records):
    if not isinstance(record, dict):
      raise ValueError(f"Policy metadata record {index} must be an object")
    card_code = record.get("card_code")
    card_def_id = record.get("card_def_id")
    card_type = record.get("card_type")
    element = record.get("element")
    if not isinstance(card_code, str) or not card_code:
      raise ValueError(f"Policy metadata record {index} has invalid card_code")
    if not isinstance(card_def_id, int) or isinstance(card_def_id, bool) or card_def_id < 0:
      raise ValueError(f"Policy metadata record {index} has invalid card_def_id")
    if not isinstance(card_type, str) or not card_type:
      raise ValueError(f"Policy metadata record {index} has invalid card_type")
    if not isinstance(element, str) or not element:
      raise ValueError(f"Policy metadata record {index} has invalid element")
    ikz_cost = record.get("ikz_cost", 0)
    if isinstance(ikz_cost, bool) or not isinstance(ikz_cost, int) or ikz_cost < 0:
      ikz_cost = 0
    out.append(
      DeckCardRecord(
        card_code=card_code,
        card_def_id=int(card_def_id),
        card_type=card_type.upper(),
        element=element.upper(),
        ikz_cost=int(ikz_cost),
      )
    )
  return tuple(sorted(out, key=lambda card: card.card_def_id))


def _find_single_card_of_type(
  deck: NativeDeck,
  records_by_code: dict[str, DeckCardRecord],
  card_type: str,
  *,
  deck_index: int,
) -> DeckCardRecord:
  matches: list[DeckCardRecord] = []
  for card_code, quantity in deck:
    record = records_by_code.get(card_code)
    if record is None:
      raise ValueError(f"Deck pool card '{card_code}' is missing from policy card metadata")
    if record.card_type == card_type:
      if int(quantity) != 1:
        raise ValueError(
          f"Deck pool deck {deck_index} has {quantity} copies of {card_type} {card_code}; expected 1"
        )
      matches.append(record)
  if len(matches) != 1:
    raise ValueError(f"Deck pool deck {deck_index} has {len(matches)} {card_type} cards; expected 1")
  return matches[0]


def _validate_deck_shape(
  deck: NativeDeck,
  records_by_code: dict[str, DeckCardRecord],
  *,
  deck_index: int,
) -> tuple[DeckCardRecord, DeckCardRecord]:
  leader = _find_single_card_of_type(deck, records_by_code, LEADER_CARD_TYPE, deck_index=deck_index)
  gate = _find_single_card_of_type(deck, records_by_code, GATE_CARD_TYPE, deck_index=deck_index)
  main_total = 0
  ikz_total = 0
  main_counts: Counter[str] = Counter()
  for card_code, quantity in deck:
    record = records_by_code[card_code]
    quantity = int(quantity)
    if record.card_type in MAIN_CARD_TYPES:
      main_total += quantity
      main_counts[card_code] += quantity
      if quantity > MAX_MAIN_COPIES:
        raise ValueError(
          f"Deck pool deck {deck_index} has {quantity} copies of {card_code}; max {MAX_MAIN_COPIES}"
        )
    elif card_code == IKZ_CARD_CODE:
      ikz_total += quantity
  for card_code, quantity in main_counts.items():
    if quantity > MAX_MAIN_COPIES:
      raise ValueError(
        f"Deck pool deck {deck_index} has {quantity} total copies of {card_code}; max {MAX_MAIN_COPIES}"
      )
  if main_total != MAX_DECK_SIZE:
    raise ValueError(f"Deck pool deck {deck_index} has {main_total} main cards; expected {MAX_DECK_SIZE}")
  if ikz_total != IKZ_CARD_COUNT:
    raise ValueError(f"Deck pool deck {deck_index} has {ikz_total} {IKZ_CARD_CODE}; expected {IKZ_CARD_COUNT}")
  if leader.element != gate.element:
    raise ValueError(
      f"Deck pool deck {deck_index} leader element {leader.element} does not match gate element {gate.element}"
    )
  return leader, gate


def build_deck_build_catalog(deck_pool: NativeDeckPool) -> DeckBuildCatalog:
  records = _load_policy_card_records()
  records_by_def_id = {record.card_def_id: record for record in records}
  records_by_code = {record.card_code: record for record in records}
  if len(records_by_def_id) != len(records) or len(records_by_code) != len(records):
    raise ValueError("Policy card metadata contains duplicate card ids or codes")

  ikz_record = records_by_code.get(IKZ_CARD_CODE)
  if ikz_record is None:
    raise ValueError(f"Policy card metadata is missing required {IKZ_CARD_CODE}")

  gate_population: list[int] = []
  observed_elements: set[str] = set()
  for deck_index, deck in enumerate(deck_pool):
    _, gate = _validate_deck_shape(deck, records_by_code, deck_index=deck_index)
    gate_population.append(gate.card_def_id)
    observed_elements.add(gate.element)
  if not gate_population:
    raise ValueError("Deck pool must contain at least one gate for deck building")

  leader_by_element: dict[str, list[int]] = {element: [] for element in observed_elements}
  main_by_element: dict[str, list[int]] = {element: [] for element in observed_elements}
  for record in records:
    if record.card_type == LEADER_CARD_TYPE and record.element in leader_by_element:
      leader_by_element[record.element].append(record.card_def_id)
    if record.card_type in MAIN_CARD_TYPES:
      for element in observed_elements:
        if record.element in (NORMAL_ELEMENT, element):
          main_by_element[element].append(record.card_def_id)

  for element in sorted(observed_elements):
    if not leader_by_element[element]:
      raise ValueError(f"No leader candidates found for gate element {element}")
    if not main_by_element[element]:
      raise ValueError(f"No main deck candidates found for gate element {element}")

  return DeckBuildCatalog(
    records_by_def_id=records_by_def_id,
    records_by_code=records_by_code,
    leader_def_ids_by_element={
      element: tuple(sorted(def_ids)) for element, def_ids in leader_by_element.items()
    },
    main_def_ids_by_element={
      element: tuple(sorted(def_ids)) for element, def_ids in main_by_element.items()
    },
    gate_def_id_population=tuple(gate_population),
    ikz_record=ikz_record,
  )


def _empty_weapon() -> dict[str, int]:
  return {"card_def_id": -1, "cur_atk": 0}


def _empty_leader() -> dict[str, Any]:
  return {
    "card_def_id": -1,
    "tapped": 0,
    "cooldown": 0,
    "cur_atk": 0,
    "cur_hp": 0,
    "weapon_count": 0,
    "weapons": tuple(_empty_weapon() for _ in range(MAX_ATTACHED_WEAPONS)),
    "has_charge": 0,
    "has_defender": 0,
    "has_infiltrate": 0,
  }


def _empty_gate() -> dict[str, int]:
  return {"card_def_id": -1, "tapped": 0, "cooldown": 0}


def _empty_zone_card(zone_index: int = 0) -> dict[str, int]:
  return {"card_def_id": -1, "zone_index": int(zone_index)}


def _empty_board_card(zone_index: int = 0) -> dict[str, Any]:
  return {
    "card_def_id": -1,
    "zone_index": int(zone_index),
    "tapped": 0,
    "cooldown": 0,
    "has_cur_stats": 0,
    "cur_atk": 0,
    "cur_hp": 0,
    "weapon_count": 0,
    "weapons": tuple(_empty_weapon() for _ in range(MAX_ATTACHED_WEAPONS)),
    "has_charge": 0,
    "has_defender": 0,
    "has_infiltrate": 0,
    "is_frozen": 0,
    "is_shocked": 0,
    "is_effect_immune": 0,
  }


def _empty_ikz_card(zone_index: int = 0) -> dict[str, int]:
  return {"card_def_id": -1, "zone_index": int(zone_index), "tapped": 0, "cooldown": 0}


def _empty_recent_action() -> dict[str, int]:
  return {"valid": 0, "primary": 0, "sub1": 0, "sub2": 0, "sub3": 0, "was_noop": 0}


def _empty_critic_privileged() -> dict[str, Any]:
  return {
    "opponent_hand": tuple(_empty_zone_card(i) for i in range(MAX_HAND_SIZE)),
    "self_deck": tuple(_empty_zone_card(i) for i in range(MAX_DECK_SIZE)),
    "opponent_deck": tuple(_empty_zone_card(i) for i in range(MAX_DECK_SIZE)),
  }


def _empty_action_mask() -> dict[str, Any]:
  return {
    "primary_action_mask": np.zeros(ACTION_TYPE_COUNT, dtype=np.bool_),
    "legal_action_count": 0,
    "legal_actions": {
      "legal_primary": np.full(MAX_LEGAL_ACTIONS_COUNT, LEGAL_ACTION_UNUSED, dtype=np.int16),
      "legal_sub1": np.full(MAX_LEGAL_ACTIONS_COUNT, LEGAL_ACTION_UNUSED, dtype=np.int16),
      "legal_sub2": np.full(MAX_LEGAL_ACTIONS_COUNT, LEGAL_ACTION_UNUSED, dtype=np.int16),
      "legal_sub3": np.full(MAX_LEGAL_ACTIONS_COUNT, LEGAL_ACTION_UNUSED, dtype=np.int16),
    },
  }


def empty_deck_context() -> dict[str, Any]:
  return {
    "mode": DECK_CONTEXT_MODE_BATTLE,
    "gate_card_def_id": -1,
    "leader_card_def_id": -1,
    "main_card_def_ids": np.full(MAX_DECK_SIZE, -1, dtype=np.int16),
    "main_count": 0,
    "candidate_card_def_ids": np.full(MAX_DECK_BUILD_CANDIDATES, -1, dtype=np.int16),
    "candidate_copy_counts": np.zeros(MAX_DECK_BUILD_CANDIDATES, dtype=np.uint8),
    "candidate_count": 0,
  }


def empty_training_observation() -> dict[str, Any]:
  player = {
    "leader": _empty_leader(),
    "gate": _empty_gate(),
    "hand": tuple(_empty_zone_card(i) for i in range(MAX_HAND_SIZE)),
    "alley": tuple(_empty_board_card(i) for i in range(ALLEY_SIZE)),
    "garden": tuple(_empty_board_card(i) for i in range(GARDEN_SIZE)),
    "discard": tuple(_empty_zone_card(i) for i in range(MAX_DECK_SIZE)),
    "selection": tuple(_empty_board_card(i) for i in range(MAX_SELECTION_ZONE_SIZE)),
    "ikz_area": tuple(_empty_ikz_card(i) for i in range(IKZ_AREA_SIZE)),
    "hand_count": 0,
    "deck_count": 0,
    "ikz_pile_count": 0,
    "selection_count": 0,
    "has_ikz_token": 0,
  }
  opponent = {
    "leader": _empty_leader(),
    "gate": _empty_gate(),
    "alley": tuple(_empty_board_card(i) for i in range(ALLEY_SIZE)),
    "garden": tuple(_empty_board_card(i) for i in range(GARDEN_SIZE)),
    "discard": tuple(_empty_zone_card(i) for i in range(MAX_DECK_SIZE)),
    "ikz_area": tuple(_empty_ikz_card(i) for i in range(IKZ_AREA_SIZE)),
    "hand_count": 0,
    "deck_count": 0,
    "ikz_pile_count": 0,
    "has_ikz_token": 0,
  }
  return {
    "phase": 0,
    "ability_context": {
      "phase": 0,
      "pending_confirmation_count": 0,
      "has_source_card_def_id": 0,
      "source_card_def_id": -1,
      "cost_target_type": 0,
      "effect_target_type": 0,
      "selection_count": 0,
      "selection_picked": 0,
      "selection_pick_max": 0,
      "active_player_index": -1,
    },
    "combat_context": {
      "combat_active": 0,
      "response_window_active": 0,
      "defender_intercepted": 0,
      "attacker_is_self": 0,
      "attacker_is_leader": 0,
      "attacker_is_garden": 0,
      "attacker_is_alley": 0,
      "attacker_card_def_id": -1,
      "attacker_slot_index": 0,
      "target_is_self": 0,
      "target_is_leader": 0,
      "target_is_garden": 0,
      "target_is_alley": 0,
      "target_card_def_id": -1,
      "target_slot_index": 0,
    },
    "self_recent_actions": tuple(_empty_recent_action() for _ in range(RECENT_ACTION_HISTORY_LEN)),
    "opp_recent_actions": tuple(_empty_recent_action() for _ in range(RECENT_ACTION_HISTORY_LEN)),
    "critic_privileged": _empty_critic_privileged(),
    "player": player,
    "opponent": opponent,
    "action_mask": _empty_action_mask(),
  }


def _copy_with_sanitized_privileged_decks(observation: dict[str, Any]) -> dict[str, Any]:
  out = dict(observation)
  critic = dict(out.get("critic_privileged", {}))
  critic["self_deck"] = tuple(_empty_zone_card(i) for i in range(MAX_DECK_SIZE))
  critic["opponent_deck"] = tuple(_empty_zone_card(i) for i in range(MAX_DECK_SIZE))
  out["critic_privileged"] = critic
  return out


class DeckBuildingParallelEnv(ParallelEnv):
  """ParallelEnv wrapper that learns deck construction before starting battle."""

  metadata = {"render_modes": ["human", "ansi"], "name": "azuki_tcg_deck_building_v0"}
  is_deck_building_wrapper = True

  def __init__(
    self,
    env: ParallelEnv,
    *,
    deck_pool: NativeDeckPool,
    seed: int | None = None,
    catalog: DeckBuildCatalog | None = None,
    fixed_deck_seats: tuple[int, ...] = (),
    snapshot_dir: str | Path | None = None,
    snapshot_every: int | None = None,
  ) -> None:
    super().__init__()
    self.env = env
    self._deck_pool = tuple(deck_pool)
    self._fixed_deck_seats = tuple(sorted(set(int(seat) for seat in fixed_deck_seats)))
    self._snapshot_dir_arg = snapshot_dir
    self._snapshot_every_arg = snapshot_every
    self.render_mode = getattr(env, "render_mode", "ansi")
    self.possible_agents = list(env.possible_agents)
    self.agents = self.possible_agents[:]
    self._agent_count = len(self.possible_agents)
    if self._agent_count != 2:
      raise ValueError(f"Deck building expects 2 agents, got {self._agent_count}")
    self._catalog = catalog or build_deck_build_catalog(deck_pool)
    self._metric_gate_elements = tuple(
      sorted(
        {
          self._catalog.records_by_def_id[gate_def_id].element
          for gate_def_id in self._catalog.gate_def_id_population
        }
      )
    )
    self._metric_gate_pairs = tuple(
      f"{element}_vs_{opponent_element}"
      for element in self._metric_gate_elements
      for opponent_element in self._metric_gate_elements
    )
    self._metric_gate_pairs_unordered = tuple(
      sorted(
        {
          "_vs_".join(sorted((element, opponent_element)))
          for element in self._metric_gate_elements
          for opponent_element in self._metric_gate_elements
        }
      )
    )
    self._metric_main_elements = tuple(
      sorted(
        {
          record.element
          for def_ids in self._catalog.main_def_ids_by_element.values()
          for record in (self._catalog.records_by_def_id[card_def_id] for card_def_id in def_ids)
        }
      )
    )
    self._metric_main_types = tuple(sorted(MAIN_CARD_TYPES))
    self._action_space = build_action_space()
    base_space = env.observation_space(self.possible_agents[0])
    if not isinstance(base_space, spaces.Dict):
      raise TypeError("DeckBuildingParallelEnv requires a Dict observation space")
    self._observation_space = spaces.Dict(dict(base_space.spaces, deck_context=build_deck_context_space()))
    self._rng = np.random.default_rng(seed)
    self._episode_seed = int(seed or 0)
    self._building = True
    self._active_player_index = 0
    self._states = [PlayerDeckBuildState.create(-1) for _ in self.possible_agents]
    self.rewards = {agent: 0.0 for agent in self.possible_agents}
    self.terminations = {agent: False for agent in self.possible_agents}
    self.truncations = {agent: False for agent in self.possible_agents}
    self.infos = {agent: {} for agent in self.possible_agents}
    # Vec workers do not inherit launcher env vars; prefer constructor args
    # (plumbed from env config) with env vars as an in-process fallback.
    resolved_dir = self._snapshot_dir_arg or os.getenv(SNAPSHOT_DIR_ENV, "").strip()
    self._snapshot_dir = Path(resolved_dir) if resolved_dir else None
    if self._snapshot_every_arg is not None:
      self._snapshot_every = max(1, int(self._snapshot_every_arg))
    else:
      try:
        self._snapshot_every = max(1, int(os.getenv(SNAPSHOT_EVERY_ENV, str(DEFAULT_SNAPSHOT_EVERY))))
      except ValueError:
        self._snapshot_every = DEFAULT_SNAPSHOT_EVERY
    self._snapshot_path: Path | None = None
    self._completed_episode_count = 0

  def observation_space(self, agent):
    return self._observation_space

  def action_space(self, agent):
    return self._action_space

  def _sample_gate_def_id(self) -> int:
    population = self._catalog.gate_def_id_population
    index = int(self._rng.integers(0, len(population)))
    return int(population[index])

  def _fixed_state_from_deck(self, deck: NativeDeck) -> PlayerDeckBuildState:
    records_by_code = self._catalog.records_by_code
    gate_def_id = -1
    leader_def_id = -1
    main_ids: list[int] = []
    for card_code, quantity in deck:
      record = records_by_code.get(card_code)
      if record is None:
        raise ValueError(f"Fixed deck card '{card_code}' missing from policy card metadata")
      if record.card_type == GATE_CARD_TYPE:
        gate_def_id = record.card_def_id
      elif record.card_type == LEADER_CARD_TYPE:
        leader_def_id = record.card_def_id
      elif record.card_type in MAIN_CARD_TYPES:
        main_ids.extend([record.card_def_id] * int(quantity))
    if gate_def_id < 0 or leader_def_id < 0 or len(main_ids) != MAX_DECK_SIZE:
      raise ValueError(
        f"Fixed deck must contain gate, leader, and {MAX_DECK_SIZE} main cards "
        f"(got gate={gate_def_id}, leader={leader_def_id}, main={len(main_ids)})"
      )
    state = PlayerDeckBuildState.create(gate_def_id)
    state.leader_card_def_id = int(leader_def_id)
    for index, card_def_id in enumerate(main_ids):
      state.main_card_def_ids[index] = int(card_def_id)
    state.main_count = len(main_ids)
    return state

  def _initial_states(self) -> list[PlayerDeckBuildState]:
    states: list[PlayerDeckBuildState] = []
    for player_index in range(self._agent_count):
      if player_index in self._fixed_deck_seats:
        deck_index = int(self._rng.integers(0, len(self._deck_pool)))
        states.append(self._fixed_state_from_deck(self._deck_pool[deck_index]))
      else:
        states.append(PlayerDeckBuildState.create(self._sample_gate_def_id()))
    return states

  def _gate_element(self, state: PlayerDeckBuildState) -> str:
    gate = self._catalog.records_by_def_id.get(state.gate_card_def_id)
    if gate is None:
      raise ValueError(f"Unknown gate card_def_id {state.gate_card_def_id}")
    return gate.element

  def _candidate_def_ids(self, player_index: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    state = self._states[player_index]
    if state.is_complete:
      return (), ()
    gate_element = self._gate_element(state)
    if state.leader_card_def_id < 0:
      candidates = self._catalog.leader_def_ids_by_element[gate_element]
      return candidates, tuple(0 for _ in candidates)
    if state.main_count >= MAX_DECK_SIZE:
      return (), ()

    copy_counts = state.main_copy_counts()
    candidates: list[int] = []
    candidate_copy_counts: list[int] = []
    for card_def_id in self._catalog.main_def_ids_by_element[gate_element]:
      current_count = int(copy_counts.get(card_def_id, 0))
      if current_count >= MAX_MAIN_COPIES:
        continue
      candidates.append(card_def_id)
      candidate_copy_counts.append(current_count)
    return tuple(candidates), tuple(candidate_copy_counts)

  def _deck_context_for_player(self, player_index: int, *, include_candidates: bool) -> dict[str, Any]:
    state = self._states[player_index]
    context = empty_deck_context()
    context["mode"] = state.mode if self._building else DECK_CONTEXT_MODE_BATTLE
    context["gate_card_def_id"] = int(state.gate_card_def_id)
    context["leader_card_def_id"] = int(state.leader_card_def_id)
    context["main_card_def_ids"] = np.asarray(state.main_card_def_ids, dtype=np.int16)
    context["main_count"] = int(state.main_count)
    if not include_candidates:
      return context

    candidates, copy_counts = self._candidate_def_ids(player_index)
    candidate_count = len(candidates)
    if candidate_count > MAX_DECK_BUILD_CANDIDATES:
      raise RuntimeError(
        f"Deck-build candidates exceeded MAX_DECK_BUILD_CANDIDATES: "
        f"{candidate_count} > {MAX_DECK_BUILD_CANDIDATES}"
      )
    context["candidate_card_def_ids"][:candidate_count] = np.asarray(candidates, dtype=np.int16)
    context["candidate_copy_counts"][:candidate_count] = np.asarray(copy_counts, dtype=np.uint8)
    context["candidate_count"] = int(candidate_count)
    return context

  def _deck_build_action_mask(self, player_index: int) -> dict[str, Any]:
    action_mask = _empty_action_mask()
    if not self._building or player_index != self._active_player_index:
      return action_mask
    candidates, _ = self._candidate_def_ids(player_index)
    candidate_count = len(candidates)
    if candidate_count <= 0:
      raise RuntimeError(f"Active deck builder {player_index} has no valid candidates")
    if candidate_count > MAX_DECK_BUILD_CANDIDATES:
      raise RuntimeError(
        f"Deck-build candidates exceeded MAX_DECK_BUILD_CANDIDATES: "
        f"{candidate_count} > {MAX_DECK_BUILD_CANDIDATES}"
      )
    action_mask["primary_action_mask"][int(ActionType.DECK_PICK_CARD)] = True
    action_mask["legal_action_count"] = int(candidate_count)
    legal = action_mask["legal_actions"]
    legal["legal_primary"][:candidate_count] = int(ActionType.DECK_PICK_CARD)
    legal["legal_sub1"][:candidate_count] = np.arange(candidate_count, dtype=np.int16)
    return action_mask

  def _building_observations(self) -> dict[int, dict[str, Any]]:
    observations: dict[int, dict[str, Any]] = {}
    for player_index, agent in enumerate(self.possible_agents):
      obs = empty_training_observation()
      is_active = player_index == self._active_player_index
      obs["deck_context"] = self._deck_context_for_player(player_index, include_candidates=is_active)
      obs["action_mask"] = self._deck_build_action_mask(player_index)
      observations[agent] = obs
    return observations

  def _battle_observations(self, observations: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out = {}
    for player_index, agent in enumerate(self.possible_agents):
      obs = _copy_with_sanitized_privileged_decks(observations[agent])
      action_mask = obs.get("action_mask", {})
      legal_actions = action_mask.get("legal_actions", {})
      legal_count = int(action_mask.get("legal_action_count", 0))
      legal_primary = legal_actions.get("legal_primary")
      if legal_primary is not None:
        primary = np.asarray(legal_primary).reshape(-1)
        if legal_count > 0 and np.any(primary[:legal_count] == int(ActionType.DECK_PICK_CARD)):
          raise RuntimeError("Battle action mask unexpectedly contains DECK_PICK_CARD")
      obs["deck_context"] = self._deck_context_for_player(player_index, include_candidates=False)
      out[agent] = obs
    return out

  def reset(self, seed=None, options=None):
    del options
    if seed is None:
      seed = int(self._rng.integers(0, 2**31 - 1))
    self._episode_seed = int(seed)
    self._rng = np.random.default_rng(self._episode_seed)
    self._states = self._initial_states()
    self._building = True
    self._active_player_index = 0
    self.agents = self.possible_agents[:]
    self.rewards = {agent: 0.0 for agent in self.possible_agents}
    self.terminations = {agent: False for agent in self.possible_agents}
    self.truncations = {agent: False for agent in self.possible_agents}
    self.infos = {agent: {} for agent in self.possible_agents}
    if all(state.is_complete for state in self._states):
      return self._start_battle(), self.infos
    if self._states[self._active_player_index].is_complete:
      battle_observations = self._advance_active_builder()
      if battle_observations is not None:
        return battle_observations, self.infos
    return self._building_observations(), self.infos

  def _deck_spec_for_player(self, player_index: int) -> NativeDeck:
    state = self._states[player_index]
    if not state.is_complete:
      raise RuntimeError(f"Cannot build deck spec for incomplete player {player_index}")
    counts = state.main_copy_counts()
    for card_def_id, quantity in counts.items():
      if quantity > MAX_MAIN_COPIES:
        raise RuntimeError(f"Player {player_index} has {quantity} copies of {card_def_id}; max {MAX_MAIN_COPIES}")
    cards: list[tuple[str, int]] = []
    leader = self._catalog.records_by_def_id[state.leader_card_def_id]
    gate = self._catalog.records_by_def_id[state.gate_card_def_id]
    cards.append((leader.card_code, 1))
    cards.append((gate.card_code, 1))
    for card_def_id in sorted(counts):
      record = self._catalog.records_by_def_id[card_def_id]
      cards.append((record.card_code, int(counts[card_def_id])))
    cards.append((self._catalog.ikz_record.card_code, IKZ_CARD_COUNT))
    total = sum(quantity for _, quantity in cards)
    expected = MAX_DECK_SIZE + 1 + 1 + IKZ_CARD_COUNT
    if total != expected:
      raise RuntimeError(f"Generated deck has {total} cards; expected {expected}")
    return tuple(cards)

  def _deckbuild_metrics_for_player(self, player_index: int) -> dict[str, float]:
    state = self._states[player_index]
    opponent_state = self._states[1 - player_index]
    if not state.is_complete:
      raise RuntimeError(f"Cannot emit deck-build metrics for incomplete player {player_index}")
    records = self._catalog.records_by_def_id
    gate = records[state.gate_card_def_id]
    opponent_gate = records[opponent_state.gate_card_def_id]
    leader = records[state.leader_card_def_id]
    main_ids = [
      int(card_id)
      for card_id in state.main_card_def_ids[: state.main_count]
      if int(card_id) >= 0
    ]
    counts = Counter(main_ids)
    type_counts = Counter(records[card_id].card_type for card_id in main_ids)
    element_counts = Counter(records[card_id].element for card_id in main_ids)
    copy_histogram = Counter(counts.values())
    unique_count = len(counts)
    main_total = max(len(main_ids), 1)
    probabilities = [float(quantity) / main_total for quantity in counts.values()]
    entropy = -sum(probability * float(np.log(probability)) for probability in probabilities if probability > 0.0)
    max_entropy = float(np.log(unique_count)) if unique_count > 1 else 0.0

    metrics: dict[str, float] = {
      "deckbuild/completed": 1.0,
      "deckbuild/picks": float(1 + MAX_DECK_SIZE),
      "deckbuild/main_count": float(len(main_ids)),
      "deckbuild/main_unique": float(unique_count),
      "deckbuild/main_unique_share": float(unique_count / main_total),
      "deckbuild/main_avg_copies_per_unique": float(len(main_ids) / max(unique_count, 1)),
      "deckbuild/main_max_copy_count": float(max(counts.values()) if counts else 0),
      "deckbuild/main_copy_entropy": float(entropy),
      "deckbuild/main_copy_entropy_norm": float(entropy / max_entropy) if max_entropy > 0.0 else 0.0,
      "deckbuild/main_singleton_count": float(copy_histogram.get(1, 0)),
      "deckbuild/main_pair_count": float(copy_histogram.get(2, 0)),
      "deckbuild/main_triplet_count": float(copy_histogram.get(3, 0)),
      "deckbuild/main_quad_count": float(copy_histogram.get(4, 0)),
      "deckbuild/main_singleton_slot_share": float(copy_histogram.get(1, 0) / main_total),
      "deckbuild/main_pair_slot_share": float((2 * copy_histogram.get(2, 0)) / main_total),
      "deckbuild/main_triplet_slot_share": float((3 * copy_histogram.get(3, 0)) / main_total),
      "deckbuild/main_quad_slot_share": float((4 * copy_histogram.get(4, 0)) / main_total),
      "deckbuild/main_normal_share": float(element_counts.get(NORMAL_ELEMENT, 0) / main_total),
      "deckbuild/main_gate_element_share": float(element_counts.get(gate.element, 0) / main_total),
      "deckbuild/gate_match": float(gate.element == opponent_gate.element),
    }

    for element in self._metric_gate_elements:
      metrics[f"deckbuild/gate/{element}"] = float(gate.element == element)
      metrics[f"deckbuild/opponent_gate/{element}"] = float(opponent_gate.element == element)
      metrics[f"deckbuild/leader/{element}"] = float(leader.element == element)

    for card_type in self._metric_main_types:
      count = float(type_counts.get(card_type, 0))
      metrics[f"deckbuild/main_type_count/{card_type}"] = count
      metrics[f"deckbuild/main_type_share/{card_type}"] = count / main_total

    for element in self._metric_main_elements:
      count = float(element_counts.get(element, 0))
      metrics[f"deckbuild/main_element_count/{element}"] = count
      metrics[f"deckbuild/main_element_share/{element}"] = count / main_total

    costs = [records[card_id].ikz_cost for card_id in main_ids]
    cost_total = max(len(costs), 1)
    avg_cost = float(sum(costs)) / cost_total
    bucket_counts = Counter(
      "0_1" if cost <= 1 else "2_3" if cost <= 3 else "4_5" if cost <= 5 else "6_plus"
      for cost in costs
    )
    metrics["deckbuild/main_avg_cost"] = avg_cost
    for bucket in ("0_1", "2_3", "4_5", "6_plus"):
      metrics[f"deckbuild/main_cost_share/{bucket}"] = float(bucket_counts.get(bucket, 0)) / cost_total

    metrics[f"deckbuild/leader_card/{leader.card_code}"] = 1.0

    # Sparse per-gate-card metrics: emitted only for the assigned gate, so the
    # trainer's per-key mean is the conditional mean given that gate.
    gate_prefix = f"deckbuild_gatecard/{gate.card_code}"
    metrics[f"{gate_prefix}/game"] = 1.0
    metrics[f"{gate_prefix}/avg_cost"] = avg_cost
    metrics[f"{gate_prefix}/main_unique"] = float(unique_count)
    metrics[f"{gate_prefix}/copy_entropy_norm"] = metrics["deckbuild/main_copy_entropy_norm"]
    metrics[f"{gate_prefix}/gate_element_share"] = metrics["deckbuild/main_gate_element_share"]
    for card_type in self._metric_main_types:
      metrics[f"{gate_prefix}/type_share/{card_type}"] = (
        float(type_counts.get(card_type, 0)) / main_total
      )
    for bucket in ("0_1", "2_3", "4_5", "6_plus"):
      metrics[f"{gate_prefix}/cost_share/{bucket}"] = float(bucket_counts.get(bucket, 0)) / cost_total
    metrics[f"{gate_prefix}/leader/{leader.card_code}"] = 1.0

    return metrics

  def _deckbuild_infos(self, *, include_step_metrics: bool) -> dict[int, dict[str, float]]:
    infos: dict[int, dict[str, float]] = {}
    for player_index, agent in enumerate(self.possible_agents):
      metrics = self._deckbuild_metrics_for_player(player_index)
      if include_step_metrics:
        metrics.update({f"azk_step_{key}": value for key, value in metrics.items()})
      infos[agent] = metrics
    return infos

  def _deckbuild_result_metrics_for_player(self, player_index: int, info: dict[str, Any]) -> dict[str, float]:
    state = self._states[player_index]
    opponent_state = self._states[1 - player_index]
    records = self._catalog.records_by_def_id
    gate = records[state.gate_card_def_id]
    opponent_gate = records[opponent_state.gate_card_def_id]
    gate_pair = f"{gate.element}_vs_{opponent_gate.element}"
    unordered_gate_pair = "_vs_".join(sorted((gate.element, opponent_gate.element)))
    win = float(info.get("win", 0.0))
    gate_match = gate.element == opponent_gate.element
    metrics = {
      "deckbuild_result/game": 1.0,
      "deckbuild_result/win": win,
      "deckbuild_result/gate_match": float(gate_match),
      "deckbuild_result/gate_match_win_joint": win if gate_match else 0.0,
      "deckbuild_result/gate_mismatch": float(not gate_match),
      "deckbuild_result/gate_mismatch_win_joint": win if not gate_match else 0.0,
      f"deckbuild_result/gate/{gate.element}/game": 1.0,
      f"deckbuild_result/gate/{gate.element}/win": win,
      f"deckbuild_result/gate_pair/{gate_pair}/game": 1.0,
      f"deckbuild_result/gate_pair/{gate_pair}/win": win,
      f"deckbuild_result/gate_pair_unordered/{unordered_gate_pair}/game": 1.0,
      f"deckbuild_result/gate_pair_unordered/{unordered_gate_pair}/win": win,
    }
    if gate_match:
      metrics["deckbuild_result/gate_match/game"] = 1.0
      metrics["deckbuild_result/gate_match/win"] = win
    else:
      metrics["deckbuild_result/gate_mismatch/game"] = 1.0
      metrics["deckbuild_result/gate_mismatch/win"] = win
    for element in self._metric_gate_elements:
      is_element = gate.element == element
      metrics[f"deckbuild_result/gate_freq/{element}"] = float(is_element)
      metrics[f"deckbuild_result/gate_win_joint/{element}"] = win if is_element else 0.0
    for candidate_pair in self._metric_gate_pairs:
      is_pair = gate_pair == candidate_pair
      metrics[f"deckbuild_result/gate_pair_freq/{candidate_pair}"] = float(is_pair)
      metrics[f"deckbuild_result/gate_pair_win_joint/{candidate_pair}"] = win if is_pair else 0.0
    for candidate_pair in self._metric_gate_pairs_unordered:
      is_pair = unordered_gate_pair == candidate_pair
      metrics[f"deckbuild_result/gate_pair_unordered_freq/{candidate_pair}"] = float(is_pair)
      metrics[f"deckbuild_result/gate_pair_unordered_win_joint/{candidate_pair}"] = (
        win if is_pair else 0.0
      )

    # Sparse per-gate-card result + playstyle metrics. Behavioral rates come from
    # the native env's terminal log entries already present in `info`.
    gate_prefix = f"deckbuild_result/gatecard/{gate.card_code}"
    metrics[f"{gate_prefix}/game"] = 1.0
    metrics[f"{gate_prefix}/win"] = win
    for metric_name, info_key in _BEHAVIOR_INFO_KEYS:
      value = info.get(info_key)
      if isinstance(value, (int, float)) and not isinstance(value, bool):
        metrics[f"{gate_prefix}/{metric_name}"] = float(value)
    ability_total = 0.0
    for info_key in (
      "azk_activate_garden_or_leader_ability_selected_rate",
      "azk_activate_alley_ability_selected_rate",
    ):
      value = info.get(info_key)
      if isinstance(value, (int, float)) and not isinstance(value, bool):
        ability_total += float(value)
    metrics[f"{gate_prefix}/ability_rate"] = ability_total
    return metrics

  def _deck_summary_for_player(self, player_index: int) -> dict[str, Any]:
    state = self._states[player_index]
    records = self._catalog.records_by_def_id
    counts = state.main_copy_counts()
    main_codes = {
      records[card_def_id].card_code: int(quantity)
      for card_def_id, quantity in sorted(counts.items())
    }
    main_ids = [card_id for card_id in state.main_card_def_ids[: state.main_count] if card_id >= 0]
    costs = [records[card_id].ikz_cost for card_id in main_ids]
    return {
      "gate": records[state.gate_card_def_id].card_code if state.gate_card_def_id >= 0 else None,
      "leader": records[state.leader_card_def_id].card_code if state.leader_card_def_id >= 0 else None,
      "main": main_codes,
      "avg_cost": round(float(sum(costs)) / max(len(costs), 1), 3),
    }

  def _maybe_write_deck_snapshot(self, infos: dict[int, dict[str, Any]]) -> None:
    if self._snapshot_dir is None:
      return
    if self._completed_episode_count % self._snapshot_every != 0:
      return
    if self._snapshot_path is None:
      self._snapshot_dir.mkdir(parents=True, exist_ok=True)
      self._snapshot_path = self._snapshot_dir / f"decks_pid{os.getpid()}.jsonl"
    players = []
    for player_index, agent in enumerate(self.possible_agents):
      info = infos.get(agent, {})
      summary = self._deck_summary_for_player(player_index)
      summary["win"] = float(info.get("win", 0.0) or 0.0)
      for metric_name, info_key in _BEHAVIOR_INFO_KEYS:
        value = info.get(info_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
          summary[metric_name] = round(float(value), 4)
      players.append(summary)
    record = {
      "ts": round(time.time(), 1),
      "episode": self._completed_episode_count,
      "seed": self._episode_seed,
      "players": players,
    }
    try:
      with self._snapshot_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    except OSError:
      pass

  def _start_battle(self) -> dict[int, dict[str, Any]]:
    player_decks = tuple(self._deck_spec_for_player(index) for index in range(self._agent_count))
    reset_with_decks = getattr(self.env, "reset_with_decks", None)
    if not callable(reset_with_decks):
      raise RuntimeError("Underlying Azuki env does not expose reset_with_decks")
    observations, infos = reset_with_decks(seed=self._episode_seed, player_decks=player_decks)
    self._building = False
    self._active_player_index = int(getattr(self.env, "_active_player_index", 0))
    self.agents = list(self.env.agents)
    deckbuild_infos = self._deckbuild_infos(include_step_metrics=True)
    self.infos = {
      agent: {
        **dict(infos.get(agent, {})),
        **deckbuild_infos[agent],
      }
      for agent in self.possible_agents
    }
    return self._battle_observations(observations)

  def _advance_active_builder(self) -> dict[int, dict[str, Any]] | None:
    for offset in range(1, self._agent_count + 1):
      next_index = (self._active_player_index + offset) % self._agent_count
      if not self._states[next_index].is_complete:
        self._active_player_index = next_index
        return None
    return self._start_battle()

  def _apply_pick(self, player_index: int, candidate_index: int) -> None:
    state = self._states[player_index]
    candidates, _ = self._candidate_def_ids(player_index)
    if candidate_index < 0 or candidate_index >= len(candidates):
      raise ValueError(
        f"Deck pick candidate_index {candidate_index} out of range for player {player_index} "
        f"(candidate_count={len(candidates)})"
      )
    selected_card_def_id = int(candidates[candidate_index])
    if state.leader_card_def_id < 0:
      state.leader_card_def_id = selected_card_def_id
      return
    if state.main_count >= MAX_DECK_SIZE:
      raise RuntimeError(f"Player {player_index} main deck is already full")
    current_copies = state.main_copy_counts().get(selected_card_def_id, 0)
    if current_copies >= MAX_MAIN_COPIES:
      raise ValueError(
        f"Player {player_index} selected card_def_id {selected_card_def_id} with {current_copies} copies"
      )
    state.main_card_def_ids[state.main_count] = selected_card_def_id
    state.main_count += 1

  def step(self, actions):
    if not self.agents:
      raise RuntimeError("step() called on finished environment")
    if not isinstance(actions, dict):
      raise TypeError("DeckBuildingParallelEnv expects a dict[action] keyed by agent")

    if not self._building:
      observations, rewards, terminations, truncations, infos = self.env.step(actions)
      self._active_player_index = int(getattr(self.env, "_active_player_index", 0))
      self.agents = list(self.env.agents)
      self.rewards = dict(rewards)
      self.terminations = dict(terminations)
      self.truncations = dict(truncations)
      enriched_infos: dict[int, dict[str, Any]] = {}
      episode_done = False
      for player_index, agent in enumerate(self.possible_agents):
        info = dict(infos.get(agent, {}))
        if bool(terminations.get(agent, False)) or bool(truncations.get(agent, False)):
          episode_done = True
          info.update(self._deckbuild_result_metrics_for_player(player_index, info))
        enriched_infos[agent] = info
      self.infos = enriched_infos
      if episode_done:
        self._maybe_write_deck_snapshot(enriched_infos)
        self._completed_episode_count += 1
      return self._battle_observations(observations), rewards, terminations, truncations, enriched_infos

    active_agent = self.possible_agents[self._active_player_index]
    action = actions.get(active_agent)
    if action is None:
      raise ValueError(f"Missing deck-build action for active agent {active_agent}")
    encoded = np.asarray(action, dtype=np.int32)
    if encoded.shape != (ACTION_COMPONENT_COUNT,):
      raise ValueError(f"Deck-build action must have shape {(ACTION_COMPONENT_COUNT,)}, got {encoded.shape}")
    if int(encoded[0]) != int(ActionType.DECK_PICK_CARD):
      raise ValueError(f"Deck-build action must use DECK_PICK_CARD, got {int(encoded[0])}")
    if int(encoded[2]) != 0 or int(encoded[3]) != 0:
      raise ValueError(f"Deck-build action sub2/sub3 must be zero, got {encoded.tolist()}")
    self._apply_pick(self._active_player_index, int(encoded[1]))

    battle_observations = self._advance_active_builder()
    self.rewards = {agent: 0.0 for agent in self.possible_agents}
    self.terminations = {agent: False for agent in self.possible_agents}
    self.truncations = {agent: False for agent in self.possible_agents}
    if battle_observations is not None:
      return battle_observations, dict(self.rewards), dict(self.terminations), dict(self.truncations), dict(self.infos)
    self.infos = {agent: {} for agent in self.possible_agents}
    return self._building_observations(), dict(self.rewards), dict(self.terminations), dict(self.truncations), dict(self.infos)

  def random_legal_action(self, rng: np.random.Generator) -> np.ndarray:
    if self._building:
      candidates, _ = self._candidate_def_ids(self._active_player_index)
      if not candidates:
        return np.asarray([0, 0, 0, 0], dtype=np.int32)
      return np.asarray(
        [int(ActionType.DECK_PICK_CARD), int(rng.integers(0, len(candidates))), 0, 0],
        dtype=np.int32,
      )
    env_random = getattr(self.env, "random_legal_action", None)
    if callable(env_random):
      return env_random(rng)
    active_index = int(getattr(self.env, "_active_player_index", 0))
    raw_obs = self.env._raw_observation(active_index)
    mask = raw_obs.action_mask
    legal_count = int(mask.legal_action_count)
    if legal_count <= 0:
      return np.asarray([0, 0, 0, 0], dtype=np.int32)
    choice = int(rng.integers(0, legal_count))
    return np.asarray(
      [
        int(mask.legal_primary[choice]),
        int(mask.legal_sub1[choice]),
        int(mask.legal_sub2[choice]),
        int(mask.legal_sub3[choice]),
      ],
      dtype=np.int32,
    )

  def render(self):
    if self._building:
      lines = ["Deck building"]
      for player_index, state in enumerate(self._states):
        mode = "leader" if state.leader_card_def_id < 0 else "main"
        active = " *" if player_index == self._active_player_index else ""
        lines.append(
          f"player_{player_index}{active}: mode={mode} gate={state.gate_card_def_id} "
          f"leader={state.leader_card_def_id} main={state.main_count}/{MAX_DECK_SIZE}"
        )
      return "\n".join(lines) + "\n"
    return self.env.render()

  def close(self):
    return self.env.close()
