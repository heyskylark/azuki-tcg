#!/usr/bin/env python3
"""Opportunity-normalized behavior descriptors for the paired specialist traces.

Per candidate game every metric is a (numerator, denominator) pair. Aggregates are ratio-of-sums.
Paired deltas (arm - u8223) use a block bootstrap over block_id (the two seat-swapped games that
share a world seed); identical blocks exist in every arm. Descriptor logic adapts
strategy_causal_followup_v1/report_followup.py (Rushfire/Hydromancy/Stonehaven funnels) and the
frozen strategy_descriptor.py (corrected Surge/Stormchain v3 helpers, game-level sequences).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RT = Path("/home/skylark/git/azuki-tcg-specialist-runtime")
sys.path[:0] = [str(RT / "python/src"), str(HERE)]
import strategy_descriptor as SD  # noqa: E402
SELECTION_ACTIONS = {"SELECT_EFFECT_TARGET", "SELECT_COST_TARGET", "CONFIRM_ABILITY", "SELECT_FROM_SELECTION",
                     "SELECT_TO_ALLEY", "SELECT_TO_GARDEN", "SELECT_TO_EQUIP", "BOTTOM_DECK_CARD", "BOTTOM_DECK_ALL",
                     "TOP_DECK_CARD"}


def annotate_turns(game):
  """Turn ids. The frozen analyze_selfplay_games.annotate_turns starts a new turn whenever the other
  seat acts in MAIN, which also fires when the non-active seat answers an ability selection during
  the owner's MAIN (e.g. choosing a target for an opponent's effect). Selections never start a turn."""
  turn, owner = 0, None
  for s in game["steps"]:
    if s["ph"] == "MULLIGAN":
      s["turn"] = 0
      continue
    if s["ph"] == "MAIN" and s["p"] != owner and s["d"].get("t") not in SELECTION_ACTIONS and not s.get("ability_src"):
      owner = s["p"]
      turn += 1
    s["turn"] = turn
    s["turn_owner"] = owner
  game["n_turns"] = turn
  return game

SD.CARD_METADATA_PATH = RT / "python/config/policy_card_metadata_v1.json"
META = SD._metadata()
GS = 5
NOOP, ATTACK, ATTACH, SPELL, DEFEND, PORTAL, ACTIVATE = 0, 6, 7, 8, 9, 10, 11
GATE = {"rushfire": "AZK01-122", "ragefire": "STT04-002", "surge": "STT01-002", "stormchain": "AZK01-120",
        "hydromancy": "STT02-002", "stonehaven": "STT03-002"}
LEADER = {"STT01-001": "Raizan", "AZK01-119": "Piko", "STT04-001": "Zero", "AZK01-121": "Kagoro",
          "STT02-001": "Shao", "AZK01-125": "Benzai", "STT03-001": "Bobu", "AZK01-123": "Goro"}
CARD_NAME = {"STT03-002": "Stonehaven", "STT02-002": "Hydromancy", "AZK01-122": "Rushfire", "STT04-002": "Ragefire",
             "STT01-002": "Surge", "AZK01-120": "Stormchain", **LEADER}
QUICKSAND = "STT03-016"
WATER_RESPONSE = {"STT02-016", "AZK01-029", "STT02-015", "AZK01-030"}  # Water Orb, Aquatic Veil, Commune, Lotus
WATER_BOUNCE = {"AZK01-022", "AZK01-087", "AZK01-088", "AZK01-089", "AZK01-093", "AZK01-028", "STT02-009",
                "AZK01-032", "STT02-017", "AZK01-090", "AZK01-024"}
FIRE_BURN = {"STT04-015", "STT04-016", "STT04-017", "AZK01-065", "AZK01-066"}
FIRE_REDIRECT = {"AZK01-062", "STT04-009"}  # Pekiro, Cinderwake Ritualist
HEAL_SPELLS = {"AZK01-002", "STT03-015", "STT03-017"}
EARTH_RESPONSE = {"AZK01-128", "AZK01-055", "AZK01-101", "AZK01-102"}
ELEMENT_OF_DECK_GATE = {"AZK01-122": "FIRE", "STT04-002": "FIRE", "STT01-002": "LIGHTNING", "AZK01-120": "LIGHTNING",
                        "STT02-002": "WATER", "AZK01-126": "WATER", "STT03-002": "EARTH", "AZK01-124": "EARTH"}


def entry(text):
  """code@slot:atk/hp[T][D][+weapons] -> dict."""
  code, rest = text.split("@", 1)
  slot, stats = rest.split(":", 1)
  weapons = stats.split("+", 1)[1].split(",") if "+" in stats else []
  core = stats.split("+", 1)[0]
  flags = core.split("/", 1)[1]
  hp = int("".join(ch for ch in flags if ch.isdigit() or ch == "-"))
  return {"code": code, "slot": int(slot), "atk": int(core.split("/", 1)[0]), "hp": hp,
          "tapped": "T" in flags, "defender": "D" in flags, "weapons": weapons}


def board(step, field):
  out = {}
  for text in step.get(field) or []:
    e = entry(text)
    out[e["slot"]] = e
  return out


def legal(step):
  return [tuple(int(v) for v in row) for row in step.get("legal", [])]


def act(step):
  return step["d"].get("t", "")


def post(step):
  return step.get("post") or {}


class M:
  """Per-game metric accumulator: name -> [num, den]."""

  def __init__(self):
    self.v = defaultdict(lambda: [0, 0])
    self.examples = defaultdict(list)

  def add(self, name, num, den=1):
    self.v[name][0] += int(num)
    self.v[name][1] += int(den)

  def ex(self, name, step_index):
    if len(self.examples[name]) < 1:
      self.examples[name].append(step_index)


def next_combat_end(steps, start):
  """First step after `start` (any actor) with no active combat; None if game ends first."""
  for j in range(start + 1, len(steps)):
    if not steps[j].get("combat"):
      return j
  return None


def own_view(step, seat):
  """Boards from candidate perspective at this step's pre-state."""
  if int(step["p"]) == seat:
    return board(step, "my_garden"), board(step, "opp_garden")
  return board(step, "opp_garden"), board(step, "my_garden")


def same_turn_attack_by_slot(steps, idx, seat, slot, turn, code=None):
  for j in range(idx + 1, len(steps)):
    s = steps[j]
    if s.get("turn") != turn:
      return None
    if int(s["p"]) != seat:
      continue
    if act(s) == "ATTACK" and int(s["a"][1]) == slot and (code is None or s["d"].get("attacker") == code):
      return j
  return None


def role_metrics(game, seat):
  m = M()
  steps = game["steps"]
  deck = game["decks"][seat]
  gate, leader = deck["gate"], deck["leader"]
  weapon_codes = {c for c in deck["main"] if META.get(c, {}).get("card_type") == "WEAPON"}
  mine = [i for i, s in enumerate(steps) if int(s["p"]) == seat]
  turns_leader_legal, turns_leader_used = set(), set()
  turns_portal_legal, turns_portal_used = set(), set()
  fav_turns, fav_taken = set(), set()
  burn_legal, burn_used = set(), set()
  bounce_legal, bounce_used = set(), set()
  qs_legal, qs_used = set(), set()
  heal_legal, heal_used = set(), set()
  attach_legal, attach_used = set(), set()
  spell_legal, spell_used = set(), set()
  own_turns = set()
  windows = []  # candidate RESPONSE windows: list of step indices
  last = None
  for i in mine:
    s = steps[i]
    if s["ph"] == "RESPONSE":
      if last is not None and last == i - 1 and windows and windows[-1][-1] == last:
        windows[-1].append(i)
      else:
        windows.append([i])
    last = i
  for k, i in enumerate(mine):
    s = steps[i]
    turn = s.get("turn")
    own = s.get("turn_owner") == seat
    if own:
      own_turns.add(turn)
    rows = legal(s)
    t = act(s)
    a = s["a"]
    hand = s.get("hand", [])
    # ---- leader -------------------------------------------------------
    leader_legal = any(r[0] == ACTIVATE and r[1] == GS for r in rows)
    is_leader = t == "ACTIVATE_GARDEN_OR_LEADER_ABILITY" and int(a[1]) == GS
    if leader_legal:
      turns_leader_legal.add(turn)
    if is_leader:
      turns_leader_used.add(turn)
      leader_conversion(m, steps, i, seat, leader, turn)
    # ---- gate ---------------------------------------------------------
    if own and any(r[0] == PORTAL for r in rows):
      turns_portal_legal.add(turn)
    if t == "GATE_PORTAL":
      turns_portal_used.add(turn)
      gate_conversion(m, steps, mine, k, seat, gate, turn, weapon_codes)
    # ---- attacks / trades ----------------------------------------------
    if s["ph"] == "MAIN" and own:
      my_g, opp_g = board(s, "my_garden"), board(s, "opp_garden")
      fav = set()
      face_by_attacker, board_by_attacker = set(), set()
      for r in rows:
        if r[0] != ATTACK:
          continue
        if r[2] == GS:
          face_by_attacker.add(r[1])
        elif r[2] < GS:
          board_by_attacker.add(r[1])
          att, tgt = my_g.get(r[1]), opp_g.get(r[2])
          if att and tgt and att["atk"] >= tgt["hp"] > 0 and tgt["atk"] < att["hp"]:
            fav.add((r[1], r[2]))
      if fav:
        fav_turns.add(turn)
        if t == "ATTACK" and (int(a[1]), int(a[2])) in fav:
          fav_taken.add(turn)
      if t == "ATTACK":
        att = int(a[1])
        if att in face_by_attacker and att in board_by_attacker:
          m.add("strategy/face_target_share_when_both_legal", int(a[2]) == GS)
        m.add("strategy/attacks_by_equipped_attacker",
              (att == GS and bool(s.get("my_leader_weapons"))) or (att < GS and bool(my_g.get(att, {}).get("weapons"))))
      # legal play classes (turn-level)
      play_cards = {hand[r[1]] for r in rows if r[0] == SPELL and 0 <= r[1] < len(hand)}
      ent_cards = {hand[r[1]] for r in rows if r[0] in (1, 2) and 0 <= r[1] < len(hand)}
      chosen = hand[int(a[1])] if t in ("PLAY_SPELL_FROM_HAND", "PLAY_ENTITY_TO_GARDEN", "PLAY_ENTITY_TO_ALLEY",
                                        "ATTACH_WEAPON_FROM_HAND") and 0 <= int(a[1]) < len(hand) else None
      if play_cards:
        spell_legal.add(turn)
        if t == "PLAY_SPELL_FROM_HAND":
          spell_used.add(turn)
      if play_cards & FIRE_BURN:
        burn_legal.add(turn)
        if chosen in FIRE_BURN and t == "PLAY_SPELL_FROM_HAND":
          burn_used.add(turn)
      if (play_cards | ent_cards) & WATER_BOUNCE:
        bounce_legal.add(turn)
        if chosen in WATER_BOUNCE:
          bounce_used.add(turn)
      if QUICKSAND in play_cards:
        qs_legal.add(turn)
        if chosen == QUICKSAND:
          qs_used.add(turn)
      my_hp = s["hp"][seat]
      if play_cards & HEAL_SPELLS and my_hp < 20:
        heal_legal.add(turn)
        if chosen in HEAL_SPELLS:
          heal_used.add(turn)
      if any(r[0] == ATTACH for r in rows):
        attach_legal.add(turn)
        if t == "ATTACH_WEAPON_FROM_HAND":
          attach_used.add(turn)
    # ---- card effects ---------------------------------------------------
    if t == "ATTACH_WEAPON_FROM_HAND":
      m.add("lightning/weapon_attach_to_entity_share", s["d"].get("target") != "MY_LEADER")
    if t == "PLAY_SPELL_FROM_HAND":
      card = s["d"].get("card")
      end = resolution_end(steps, i, seat)
      if card in FIRE_BURN:
        m.add("fire/burn_cast_then_opp_damage", opp_damaged(steps[i], steps[end], seat))
      if card == QUICKSAND:
        before = own_view(steps[i], seat)[1]
        after = view_after(steps[end], seat)[1]
        m.add("earth/quicksand_cast_multi_removal", len(set(before) - set(after)) >= 2)
      if card in WATER_BOUNCE:
        m.add("water/bounce_play_removes_opp_entity", opp_removed(steps[i], steps[end], seat))
    if t in ("PLAY_ENTITY_TO_GARDEN", "PLAY_ENTITY_TO_ALLEY") and s["d"].get("card") in WATER_BOUNCE:
      end = resolution_end(steps, i, seat)
      m.add("water/bounce_play_removes_opp_entity", opp_removed(steps[i], steps[end], seat))
    if t == "DECLARE_DEFENDER":
      block_outcome(m, steps, i, seat)
    if t == "SELECT_EFFECT_TARGET" and s["d"].get("src") == GATE["stonehaven"]:
      pass  # handled in gate_conversion
  # ---- response windows -------------------------------------------------
  for w in windows:
    first = steps[w[0]]
    rows = legal(first)
    hand = first.get("hand", [])
    response_spells = {hand[r[1]] for r in rows if r[0] == SPELL and 0 <= r[1] < len(hand)}
    acted = [steps[j] for j in w]
    if response_spells:
      m.add("response/spell_played_when_legal", any(act(x) == "PLAY_SPELL_FROM_HAND" for x in acted))
    if any(r[0] == DEFEND for r in rows):
      m.add("response/defender_declared_when_legal", any(act(x) == "DECLARE_DEFENDER" for x in acted))
    if any(r[0] not in (NOOP, DEFEND) for r in rows):
      m.add("response/any_nonblock_response_when_legal", any(act(x) not in ("NOOP", "DECLARE_DEFENDER") for x in acted))
  # ---- IKZ held at end of own turn -> later spent in opponent turn -------
  by_turn = defaultdict(list)
  for i in mine:
    by_turn[steps[i].get("turn")].append(i)
  for turn in sorted(own_turns):
    last_i = by_turn[turn][-1]
    p = post(steps[last_i])
    held = (p.get("ikz") or steps[last_i]["ikz"])[0]
    m.add("ikz/held_at_end_of_own_turn", held > 0)
    if held > 0:
      spent = any(steps[j]["ikz"][0] > (post(steps[j]).get("ikz") or steps[j]["ikz"])[0]
                  for j in by_turn.get(turn + 1, []))
      m.add("ikz/held_then_spent_in_opp_turn", spent)
  for name, legal_set, used_set in (
      ("leader/use_per_legal_turn", turns_leader_legal, turns_leader_used),
      ("gate/portal_per_legal_turn", turns_portal_legal, turns_portal_used),
      ("strategy/favorable_trade_taken_per_available_turn", fav_turns, fav_taken),
      ("fire/burn_cast_per_legal_turn", burn_legal, burn_used),
      ("water/bounce_play_per_legal_turn", bounce_legal, bounce_used),
      ("earth/quicksand_cast_per_legal_turn", qs_legal, qs_used),
      ("heal/heal_spell_per_legal_turn_below_max_hp", heal_legal, heal_used),
      ("lightning/weapon_attach_per_legal_turn", attach_legal, attach_used),
      ("strategy/spell_cast_per_legal_main_turn", spell_legal, spell_used)):
    if legal_set:
      m.add(name, len(used_set & legal_set), len(legal_set))
  # ---- game-level corrected sequences (frozen descriptor v3) ------------
  for row in SD._sequence_rows(game, seat, META):
    if row["eligible"] and row["element"] == ELEMENT_OF_DECK_GATE.get(gate):
      m.add(f"sequence/{row['id']}/completed_per_eligible_game", row["completed"])
      m.add(f"sequence/{row['id']}/converted_per_eligible_game", row["converted"])
  m.add("outcome/win", game["paired_eval"]["candidate_score"] == 1.0)
  m.add("outcome/own_turns", len(own_turns))
  m.add("outcome/battle_decisions", len(mine))
  return m


def view_after(step, seat):
  p = post(step)
  if int(step["p"]) == seat:
    return board(p, "my_garden"), board(p, "opp_garden")
  return board(p, "opp_garden"), board(p, "my_garden")


def resolution_end(steps, i, seat):
  """Last consecutive candidate step continuing the same ability chain (selections/confirms)."""
  j = i
  while j + 1 < len(steps) and int(steps[j + 1]["p"]) == seat and act(steps[j + 1]) in (
      "SELECT_EFFECT_TARGET", "SELECT_COST_TARGET", "CONFIRM_ABILITY", "SELECT_FROM_SELECTION",
      "SELECT_TO_ALLEY", "SELECT_TO_GARDEN", "SELECT_TO_EQUIP", "BOTTOM_DECK_CARD", "BOTTOM_DECK_ALL", "TOP_DECK_CARD"):
    j += 1
  return j


def opp_hp(step, seat, after=False):
  if after:
    p = post(step)
    return p.get("opp_hp") if int(step["p"]) == seat else p.get("my_hp")
  return step["hp"][1 - seat]


def opp_damaged(start, end, seat):
  before = own_view(start, seat)[1]
  after = view_after(end, seat)[1]
  hp_drop = (opp_hp(end, seat, after=True) or 99) < opp_hp(start, seat)
  board_hit = any(slot not in after or after[slot]["hp"] < e["hp"] for slot, e in before.items())
  return hp_drop or board_hit


def opp_removed(start, end, seat):
  before = own_view(start, seat)[1]
  after = view_after(end, seat)[1]
  return any(slot not in after or after[slot]["code"] != e["code"] for slot, e in before.items())


def leader_conversion(m, steps, i, seat, leader, turn):
  name = LEADER.get(leader, leader)
  sel = i + 1 if i + 1 < len(steps) and int(steps[i + 1]["p"]) == seat and act(steps[i + 1]) == "SELECT_EFFECT_TARGET" else None
  s = steps[i]
  if name in ("Raizan", "Piko", "Goro", "Zero"):
    if sel is None:
      m.add(f"leader/{name}/use_with_target", 0)
      return
    m.add(f"leader/{name}/use_with_target", 1)
    raw = steps[sel]["a"]
    slot = int(raw[1])
    my_g = board(steps[sel], "my_garden")
    target = my_g.get(slot)
    if name == "Raizan" and target is not None:
      fresh = slot not in board(first_step_of_turn(steps, i, seat), "my_garden") or \
        board(first_step_of_turn(steps, i, seat), "my_garden")[slot]["code"] != target["code"]
      m.add("leader/Raizan/target_entered_this_turn", fresh)
    if target is None:
      m.add(f"leader/{name}/target_then_attacks_same_turn", 0)
      return
    j = same_turn_attack_by_slot(steps, sel, seat, slot, turn, target["code"])
    m.add(f"leader/{name}/target_then_attacks_same_turn", j is not None)
    if j is not None:
      m.ex(f"leader/{name}/target_then_attacks_same_turn", i)
    if name == "Zero":
      if target["code"] in FIRE_REDIRECT:
        end = resolution_end(steps, sel, seat)
        m.add("fire/zero_ping_redirect_target_then_opp_damage", opp_damaged(steps[i], steps[end], seat))
      m.add("fire/zero_ping_target_is_redirect_card", target["code"] in FIRE_REDIRECT)
      # Zero ping -> Ragefire buff on the same entity this turn
      for k in range(sel + 1, len(steps)):
        x = steps[k]
        if x.get("turn") != turn:
          break
        if int(x["p"]) == seat and act(x) == "SELECT_EFFECT_TARGET" and x["d"].get("src") == GATE["ragefire"]:
          m.add("fire/zero_ping_then_ragefire_same_entity", int(x["a"][1]) == slot)
          break
  elif name == "Kagoro":
    j = same_turn_attack_by_slot(steps, i, seat, GS, turn)
    m.add("leader/Kagoro/use_then_leader_attacks_same_turn", j is not None)
  elif name == "Shao":
    combat = s.get("combat") or {}
    attacker = combat.get("attacker") if not combat.get("attacker_is_self") else None
    if sel is None:
      m.add("leader/Shao/target_is_current_attacker", 0)
      return
    tgt = steps[sel]["d"].get("target")
    m.add("leader/Shao/target_is_current_attacker", attacker is not None and tgt == attacker)
    if attacker is not None and tgt == attacker:
      m.ex("leader/Shao/target_is_current_attacker", i)
  elif name == "Benzai":
    played = any(act(x) in ("PLAY_SPELL_FROM_HAND", "PLAY_ENTITY_TO_GARDEN", "PLAY_ENTITY_TO_ALLEY", "ATTACH_WEAPON_FROM_HAND")
                 for x in steps[i + 1:] if x.get("turn") == turn and int(x["p"]) == seat)
    m.add("leader/Benzai/use_then_card_played_same_turn", played)
  elif name == "Bobu":
    # Earth entity of ours destroyed/sacrificed before our next turn, then leader HP rises at that step
    healed = False
    for j in range(i + 1, len(steps)):
      x = steps[j]
      if x.get("turn", 0) > turn + 1:
        break
      mine_before, _ = own_view(x, seat)
      mine_after, _ = view_after(x, seat)
      lost = [e for sl, e in mine_before.items() if sl not in mine_after or mine_after[sl]["code"] != e["code"]]
      if any(META.get(e["code"], {}).get("element") == "EARTH" for e in lost):
        p = post(x)
        hp_after = p.get("my_hp") if int(x["p"]) == seat else p.get("opp_hp")
        healed = hp_after is not None and hp_after > x["hp"][seat]
        m.add("leader/Bobu/earth_loss_before_next_turn", 1)
        break
    else:
      m.add("leader/Bobu/earth_loss_before_next_turn", 0)
    m.add("leader/Bobu/use_then_heal_observed", healed)


def first_step_of_turn(steps, i, seat):
  turn = steps[i].get("turn")
  j = i
  while j - 1 >= 0 and steps[j - 1].get("turn") == turn:
    j -= 1
  while int(steps[j]["p"]) != seat:
    j += 1
  return steps[j]


def gate_conversion(m, steps, mine, k, seat, gate, turn, weapon_codes):
  i = mine[k]
  s = steps[i]
  name = CARD_NAME.get(gate, gate)
  nxt = mine[k + 1] if k + 1 < len(mine) else None
  if gate == GATE["rushfire"]:
    payload = None
    for j in mine[k + 1:k + 4]:
      x = steps[j]
      if act(x) == "SELECT_TO_GARDEN" and x["d"].get("src") == gate:
        payload = j
        break
    m.add("gate/Rushfire/portal_then_payload", payload is not None)
    if payload is not None:
      slot = int(steps[payload]["a"][2])
      code = steps[payload]["d"].get("card")
      att = same_turn_attack_by_slot(steps, payload, seat, slot, turn, code)
      m.add("gate/Rushfire/payload_then_attack_same_turn", att is not None)
      if att is not None:
        m.ex("gate/Rushfire/payload_then_attack_same_turn", i)
  elif gate == GATE["ragefire"]:
    buff = None
    for j in mine[k + 1:k + 4]:
      x = steps[j]
      if act(x) == "SELECT_EFFECT_TARGET" and x["d"].get("src") == gate:
        buff = j
        break
    m.add("gate/Ragefire/portal_then_buff", buff is not None)
    if buff is not None:
      slot = int(steps[buff]["a"][1])
      target = board(steps[buff], "my_garden").get(slot)
      att = same_turn_attack_by_slot(steps, buff, seat, slot, turn, target["code"] if target else None)
      m.add("gate/Ragefire/buffed_then_attacks_same_turn", att is not None)
      if att is not None:
        m.ex("gate/Ragefire/buffed_then_attacks_same_turn", i)
  elif gate == GATE["hydromancy"]:
    end = resolution_end(steps, i, seat)
    before, after = s["ikz"][0], (post(steps[end]).get("ikz") or steps[end]["ikz"])[0]
    readied = after > before
    m.add("gate/Hydromancy/portal_readied_ikz", readied)
    if readied:
      spent = any(steps[j]["ikz"][0] > (post(steps[j]).get("ikz") or steps[j]["ikz"])[0]
                  for j in mine if j > end and steps[j].get("turn") == turn)
      m.add("gate/Hydromancy/readied_then_spent_same_turn", spent)
      if spent:
        m.ex("gate/Hydromancy/readied_then_spent_same_turn", i)
  elif gate == GATE["stonehaven"]:
    end = resolution_end(steps, i, seat)
    before = {sl: e for sl, e in board(s, "my_garden").items()}
    after = view_after(steps[end], seat)[0]
    granted = [sl for sl, e in after.items() if e["defender"] and (
      (sl in before and not before[sl]["defender"])
      or (sl not in before and "defender" not in META.get(e["code"], {}).get("keywords", [])))]
    m.add("gate/Stonehaven/portal_then_grant", bool(granted))
    if granted:
      blocked = False
      for j in range(end + 1, len(steps)):
        x = steps[j]
        if x.get("turn", 0) > turn + 1:
          break
        if int(x["p"]) == seat and act(x) == "DECLARE_DEFENDER" and int(x["a"][1]) in granted:
          blocked = True
          m.ex("gate/Stonehaven/grant_then_block", i)
          break
      m.add("gate/Stonehaven/grant_then_block", blocked)
  elif gate in (GATE["surge"], GATE["stormchain"]):
    helper = SD._surge_portal_weapons if gate == GATE["surge"] else SD._stormchain_portal_weapons
    eligible = helper(s, META, weapon_codes, selected_portal=True)
    m.add(f"gate/{name}/portal_with_eligible_weapon", bool(eligible))
    if eligible:
      events = [steps[j] for j in mine]
      eq = SD._immediate_gate_equip(events, k, gate, eligible)
      m.add(f"gate/{name}/eligible_portal_then_equip", eq is not None)
      if eq is not None:
        dest = SD._equip_destination(events[eq])
        weapon = str(events[eq]["d"].get("card", ""))
        slot = int(events[eq]["a"][2])
        att = SD._first_after(events, eq, lambda item: SD._selected_type(item, "ATTACK")
                              and item.get("d", {}).get("attacker") == dest and int(item["a"][1]) == slot
                              and (dest, slot) in SD._equipped_weapon_targets(item, {weapon}).get(weapon, set()))
        m.add(f"gate/{name}/equip_then_destination_attacks", att is not None)
        if att is not None:
          m.ex(f"gate/{name}/equip_then_destination_attacks", i)


def block_outcome(m, steps, i, seat):
  s = steps[i]
  combat = s.get("combat") or {}
  my_g, opp_g = board(s, "my_garden"), board(s, "opp_garden")
  blocker = my_g.get(int(s["a"][1]))
  attacker_code = combat.get("attacker")
  end = next_combat_end(steps, i)
  if blocker is None or end is None:
    return
  after_mine, after_opp = own_view(steps[end], seat)
  survived = int(s["a"][1]) in after_mine and after_mine[int(s["a"][1])]["code"] == blocker["code"]
  att_slots = [sl for sl, e in opp_g.items() if e["code"] == attacker_code]
  killed = attacker_code != "LEADER" and bool(att_slots) and all(
    sl not in after_opp or after_opp[sl]["code"] != attacker_code or after_opp[sl]["hp"] <= 0 for sl in att_slots)
  m.add("block/blocker_survives", survived)
  m.add("block/attacker_destroyed", killed)
  m.add("block/favorable_survive_or_kill", survived or killed)
  if survived or killed:
    m.ex("block/favorable_survive_or_kill", i)


# --------------------------------------------------------------------------------------------
def deck_profile(deck):
  main = deck["main"]
  counts = Counter(main)
  types = Counter(META[c]["card_type"] for c in main)
  normal = sum(META[c].get("element") == "NORMAL" for c in main)
  costs = [int(META[c].get("ikz_cost", 0) or 0) for c in main]
  curve = Counter(min(c, 7) for c in costs)
  return {"spell_share": types["SPELL"] / 50, "weapon_share": types["WEAPON"] / 50, "entity_share": types["ENTITY"] / 50,
          "normal_share": normal / 50, "unique_cards": len(counts), "mean_cost": sum(costs) / 50,
          "curve": {str(k): curve[k] for k in range(8)}, "counts": dict(counts)}


def wjacc(a, b):
  keys = set(a) | set(b)
  return sum(min(a.get(k, 0), b.get(k, 0)) for k in keys) / max(1, sum(max(a.get(k, 0), b.get(k, 0)) for k in keys))


def load_curated():
  pool = json.loads((RT / "train-specialist/decks/curated_deck_pool.json").read_text())
  out = []
  for idx, d in enumerate(pool["decks"]):
    counts = {c["card_id"]: c["quantity"] for c in d["cards"]
              if META.get(c["card_id"], {}).get("card_type") not in ("GATE", "LEADER")}
    out.append({"index": idx, "gate": d["gate_card_id"], "leader": d["leader_card_id"], "counts": counts})
  return out


# --------------------------------------------------------------------------------------------
def boot_delta(blocks_x, blocks_b, reps=2000, seed=7):
  """blocks_*: dict block_id -> (num, den). Paired over shared blocks."""
  shared = sorted(set(blocks_x) & set(blocks_b))
  if not shared:
    return None
  nx = [blocks_x[b][0] for b in shared]; dx = [blocks_x[b][1] for b in shared]
  nb = [blocks_b[b][0] for b in shared]; db = [blocks_b[b][1] for b in shared]
  if sum(dx) == 0 or sum(db) == 0:
    return {"x": None if sum(dx) == 0 else sum(nx) / sum(dx), "b": None if sum(db) == 0 else sum(nb) / sum(db),
            "nx": sum(dx), "nb": sum(db), "blocks": len(shared)}
  rx, rb = sum(nx) / sum(dx), sum(nb) / sum(db)
  rng = random.Random(seed)
  n = len(shared)
  deltas = []
  for _ in range(reps):
    idx = [rng.randrange(n) for _ in range(n)]
    sdx = sum(dx[i] for i in idx); sdb = sum(db[i] for i in idx)
    if sdx == 0 or sdb == 0:
      continue
    deltas.append(sum(nx[i] for i in idx) / sdx - sum(nb[i] for i in idx) / sdb)
  deltas.sort()
  lo = deltas[int(0.025 * len(deltas))] if deltas else None
  hi = deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))] if deltas else None
  return {"x": rx, "b": rb, "delta": rx - rb, "ci95": [lo, hi], "nx": sum(dx), "nb": sum(db),
          "num_x": sum(nx), "num_b": sum(nb), "blocks": n}


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--traces", default=str(ROOT / "traces/*.jsonl"))
  ap.add_argument("--out", type=Path, default=ROOT / "descriptors")
  ap.add_argument("--reps", type=int, default=2000)
  args = ap.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  # per (element, arm, mode): per game rows
  rows = defaultdict(list)
  examples = defaultdict(lambda: defaultdict(list))
  decks = defaultdict(list)
  invalid = 0
  curated = load_curated()
  for path in sorted(glob.glob(args.traces)):
    with open(path) as stream:
      for line in stream:
        game = json.loads(line)
        pe = game["paired_eval"]
        if pe["validation_errors"]:
          invalid += 1
          continue
        annotate_turns(game)
        seat = pe["candidate_seat"]
        metrics = role_metrics(game, seat)
        context = f"{CARD_NAME.get(pe['candidate_gate'], pe['candidate_gate'])}/{LEADER.get(pe['candidate_leader'], pe['candidate_leader'])}"
        key = (pe["element"], pe["arm"], pe["mode"])
        opp_el = ELEMENT_OF_DECK_GATE[pe["opponent_gate"]]
        v = dict(metrics.v)
        v[f"outcome/win_vs_{opp_el}"] = [int(pe["candidate_score"] == 1.0), 1]
        rows[key].append({"block": pe["block_id"], "suite": pe["suite"], "context": context,
                          "metrics": v, "task_id": pe["task_id"]})
        for name, idxs in metrics.examples.items():
          if len(examples[key][name]) < 3:
            examples[key][name].append({"task_id": pe["task_id"], "file": Path(path).name, "game": game["game"],
                                        "step": idxs[0]})
        if pe["suite"] == "free_draft":
          prof = deck_profile(game["decks"][seat])
          decks[key].append({"context": context, "gate": pe["candidate_gate"], "leader": pe["candidate_leader"],
                             "profile": prof, "block": pe["block_id"]})
          ref = [c for c in curated if c["gate"] == pe["candidate_gate"] and c["leader"] == pe["candidate_leader"]] or \
            [c for c in curated if c["gate"] == pe["candidate_gate"]]
          for share in ("spell_share", "weapon_share", "normal_share", "mean_cost"):
            v[f"draft/{share}"] = [prof[share] * 50, 50]
          v["draft/unique_cards"] = [prof["unique_cards"], 1]
          if ref:
            v["draft/max_wjaccard_to_curated"] = [max(wjacc(prof["counts"], c["counts"]) for c in ref), 1]
  print(f"games={sum(len(v) for v in rows.values())} invalid={invalid}", flush=True)

  def per_block(game_rows, metric):
    out = defaultdict(lambda: [0, 0])
    for r in game_rows:
      if metric in r["metrics"]:
        n, d = r["metrics"][metric]
        out[r["block"]][0] += n
        out[r["block"]][1] += d
      else:
        out[r["block"]]  # register block with zero denominator
    return {b: tuple(v) for b, v in out.items()}


  # descriptor per policy/mode
  for (element, arm, mode), game_rows in rows.items():
    desc = {"element": element, "arm": arm, "mode": mode, "games": len(game_rows), "slices": {}}
    for suite in ("fixed", "free_draft", "all"):
      sel = [r for r in game_rows if suite == "all" or r["suite"] == suite]
      for context in sorted({r["context"] for r in sel}) + ["ALL"]:
        sub = [r for r in sel if context == "ALL" or r["context"] == context]
        tot = defaultdict(lambda: [0, 0])
        for r in sub:
          for name, (n, d) in r["metrics"].items():
            tot[name][0] += n
            tot[name][1] += d
        desc["slices"][f"{suite}|{context}"] = {
          "games": len(sub),
          "metrics": {name: {"num": n, "den": d, "rate": (n / d if d else None)} for name, (n, d) in sorted(tot.items())}}
    if decks[(element, arm, mode)]:
      desc["draft"] = draft_summary(decks[(element, arm, mode)], curated)
    desc["examples"] = examples[(element, arm, mode)]
    (args.out / f"{element.lower()}__{arm}__{mode}.json").write_text(json.dumps(desc, indent=1, sort_keys=True))
  # paired comparisons
  comparisons = {}
  for (element, arm, mode), game_rows in rows.items():
    if arm == "u8223":
      continue
    base = rows.get((element, "u8223", mode))
    if not base:
      continue
    comp = {}
    for suite in ("fixed", "free_draft", "all"):
      sx = [r for r in game_rows if suite == "all" or r["suite"] == suite]
      sb = [r for r in base if suite == "all" or r["suite"] == suite]
      for context in sorted({r["context"] for r in sx}) + ["ALL"]:
        cx = [r for r in sx if context == "ALL" or r["context"] == context]
        cb = [r for r in sb if context == "ALL" or r["context"] == context]
        names = sorted({n for r in cx + cb for n in r["metrics"]})
        comp[f"{suite}|{context}"] = {n: boot_delta(per_block(cx, n), per_block(cb, n), args.reps) for n in names}
    comparisons[f"{element}|{arm}|{mode}"] = comp
  (args.out / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=1, sort_keys=True))
  print("wrote", args.out)


def draft_summary(entries, curated):
  out = {}
  by_ctx = defaultdict(list)
  for e in entries:
    by_ctx[e["context"]].append(e)
  for ctx, es in sorted(by_ctx.items()):
    gate, leader = es[0]["gate"], es[0]["leader"]
    same_ctx = [c for c in curated if c["gate"] == gate and c["leader"] == leader]
    same_gate = [c for c in curated if c["gate"] == gate]
    ref = same_ctx or same_gate
    sims = [max(wjacc(e["profile"]["counts"], c["counts"]) for c in ref) for e in es] if ref else []
    pair = [wjacc(a["profile"]["counts"], b["profile"]["counts"]) for i, a in enumerate(es) for b in es[i + 1:]]
    keys = ("spell_share", "weapon_share", "entity_share", "normal_share", "unique_cards", "mean_cost")
    card_freq = Counter()
    for e in es:
      card_freq.update(e["profile"]["counts"])
    out[ctx] = {
      "decks": len(es), **{k: sum(e["profile"][k] for e in es) / len(es) for k in keys},
      "curve_mean": {str(k): sum(e["profile"]["curve"][str(k)] for e in es) / len(es) for k in range(8)},
      "curated_reference": "same gate+leader" if same_ctx else ("same gate" if same_gate else "none"),
      "curated_reference_indices": [c["index"] for c in ref],
      "max_wjaccard_to_curated_mean": sum(sims) / len(sims) if sims else None,
      "within_context_wjaccard_mean": sum(pair) / len(pair) if pair else None,
      "top_cards_per_deck": {c: round(n / len(es), 2) for c, n in card_freq.most_common(15)},
      "per_deck": [{"block": e["block"], "max_wjacc_curated": (max(wjacc(e["profile"]["counts"], c["counts"]) for c in ref) if ref else None),
                    **{k: e["profile"][k] for k in keys}} for e in es],
      "_counts": [e["profile"]["counts"] for e in es],
    }
  # sibling differentiation: between-context similarity vs within-context similarity
  ctxs = sorted(out)
  sib = {}
  for i, a in enumerate(ctxs):
    for b in ctxs[i + 1:]:
      between = [wjacc(x, y) for x in out[a]["_counts"] for y in out[b]["_counts"]]
      sib[f"{a} vs {b}"] = {"between_wjaccard_mean": sum(between) / len(between)}
  for v in out.values():
    v.pop("_counts")
  return {"contexts": out, "context_pairs": sib}


if __name__ == "__main__":
  main()
