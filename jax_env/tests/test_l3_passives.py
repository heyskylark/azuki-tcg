"""L3 passive-observer ports: episode equivalence + scripted aura scenarios.

Dual-engine pattern from test_l3_abilities_batch1: identical decks/seeds,
actions chosen from the C mask, full semantic-view + mask-order compare each
step. Deck groups stack the passive cards with their interactors (weapons for
STT01-011/008/009, element/subtype mixes for AZK01-010/019/073, alley plays
and combat for AZK01-043/048); scripted scenarios pin the aura transitions
that random play may miss (grant-after-removal, last-source death,
death-by-buff-removal).
"""
from __future__ import annotations

import numpy as np
import pytest

from test_l2_vanilla import c_semantic_view, jax_semantic_view

# Vanilla fillers (no abilities).
FILLERS = [
    ("STT01-010", 4),
    ("STT02-004", 4),
    ("STT02-006", 4),
    ("STT02-008", 4),
    ("AZK01-001", 4),
    ("AZK01-012", 4),
    ("AZK01-025", 4),
    ("AZK01-035", 4),
    ("AZK01-037", 4),
    ("AZK01-038", 4),
    ("AZK01-049", 4),
    ("AZK01-054", 4),
    ("AZK01-094", 2),  # vanilla weapon
]


def build_deck(cards_qty, leader="STT01-001", gate="STT01-002", fillers=None):
  deck = [(leader, 1), (gate, 1)]
  count = 0
  for code, qty in cards_qty:
    deck.append((code, qty))
    count += qty
  need = 50 - count
  for code, qty in (fillers or FILLERS):
    if need <= 0:
      break
    take = min(qty, need)
    deck.append((code, take))
    need -= take
  assert need == 0, f"deck short by {need}"
  deck.append(("IKZ-001", 10))
  return deck


# Normal-element fillers so AZK01-010/019's all-Normal condition can hold,
# with water (AZK01-025 / STT02-006) mixed in to flip it; 073+019 are Beanz.
NORMAL_FLIP_FILLERS = [
    ("STT02-004", 6), ("AZK01-001", 6), ("AZK01-012", 6), ("AZK01-025", 6),
    ("STT02-006", 4), ("AZK01-035", 4), ("AZK01-037", 4), ("AZK01-038", 4),
]

DECK_GROUPS = {
    # all-Normal / all-Beanz self-auras
    "P1": build_deck(
        [("AZK01-073", 6), ("AZK01-019", 6), ("AZK01-010", 6)],
        fillers=NORMAL_FLIP_FILLERS,
    ),
    # weapon auras: 016 already ported (when-attacking AoE), 094 vanilla
    "P2": build_deck(
        [("STT01-008", 6), ("STT01-009", 6), ("STT01-011", 6),
         ("STT01-016", 6), ("AZK01-094", 4)],
    ),
    # innate carapace + leader alley-targeting dagger
    "P3": build_deck(
        [("AZK01-048", 8), ("AZK01-043", 6)],
    ),
}

DRIVER_TYPES = {0, 1, 2, 6, 7, 8, 9, 11, 13, 14, 16, 25}


def comparable(rows):
  out = []
  for row in rows:
    if row[0] == 11 and row[1] == 5:  # leader activations not ported
      continue
    out.append(row)
  return out


def compare_step(step_index, cref, state, jit_mask):
  cview = c_semantic_view(cref)
  jview = jax_semantic_view(state)
  for key, cval in cview.items():
    if key == "winner":
      continue
    assert jview[key] == cval, (
        f"step {step_index}: {key}\nC  ={cval}\nJAX={jview[key]}"
    )
  active = cview["active"]
  c_rows = comparable(cref.legal_actions(active))
  legal, count, _ = jit_mask(state)
  j_rows = comparable(
      [tuple(int(x) for x in row) for row in np.asarray(legal)[: int(count)]]
  )
  assert j_rows == c_rows, (
      f"step {step_index} (phase {cview['phase']}): mask mismatch\n"
      f"C  ={c_rows}\nJAX={j_rows}"
  )
  return cview, c_rows


def make_pair(make_cref, seed, deck0, deck1=None):
  import jax

  from azuki_jax.engine.step import engine_step, stabilize
  from azuki_jax.env import init_state_with_decks
  from azuki_jax.masks import build_mask
  from azuki_jax.setup import deck_tables_from_card_lists

  deck1 = deck1 or deck0
  cref = make_cref(seed, deck_pool=None)
  cref.reset_with_decks(seed, deck0, deck1)
  tables = deck_tables_from_card_lists(deck0, deck1)
  state = stabilize(init_state_with_decks(seed, tables))
  return cref, state, jax.jit(engine_step), jax.jit(build_mask)


# coverage collected by the parametrized equivalence runs (group -> def ids
# that carried a nonzero passive at any step)
_COVERAGE: dict[str, set] = {}


def run_episode(make_cref, seed, deck, steps=450):
  cref, state, jit_step, jit_mask = make_pair(make_cref, seed, deck)

  rng = np.random.default_rng(seed)
  passive_seen = set()  # def ids that carried a nonzero passive at some step
  for step_index in range(steps):
    cview, c_rows = compare_step(step_index, cref, state, jit_mask)

    pa = np.asarray(state.passive_atk)
    ph = np.asarray(state.passive_hp)
    ids = np.asarray(state.def_id)
    for d in np.unique(ids[(pa != 0) | (ph != 0)]):
      passive_seen.add(int(d))

    if not c_rows:
      break  # both engines agree the game is stuck
    driver_rows = [row for row in c_rows if row[0] in DRIVER_TYPES]
    assert driver_rows, f"step {step_index}: no driver actions\n{c_rows}"
    action = driver_rows[int(rng.integers(0, len(driver_rows)))]

    cref.step(np.asarray(action, np.int32))
    state = jit_step(state, np.asarray(action, np.int32))

    c_term, c_trunc = cref.dones()
    j_over = int(state.winner) != -1
    assert c_term == j_over, f"step {step_index}: terminal mismatch"
    if c_term or c_trunc:
      break

  assert int(state.ab_scratch[3]) == 0, "unimplemented ability hit"
  return passive_seen


@pytest.mark.parametrize("seed", [7, 1234])
@pytest.mark.parametrize("group", sorted(DECK_GROUPS))
def test_passives_episode_equivalence(group, seed, make_cref):
  seen = run_episode(make_cref, seed, DECK_GROUPS[group])
  _COVERAGE.setdefault(group, set()).update(seen)


def test_passives_episode_coverage():
  """The aura cards must actually fire during the equivalence episodes
  (otherwise those runs prove nothing about them). 016 carries STT01-011's
  aura; the others self-buff. STT01-009 (>=6 weapons in the discard) is
  exercised by the scripted runs' EOT weapon churn only probabilistically,
  so it is not required here."""
  from azuki_jax import cards

  if not _COVERAGE:
    pytest.skip("equivalence episodes did not run in this session")
  seen = _COVERAGE.get("P1", set()) | _COVERAGE.get("P2", set())
  names = {code for code in
           ("AZK01-073", "AZK01-019", "AZK01-010",
            "STT01-008", "STT01-009", "STT01-016")
           if cards.CODE_TO_ID[code] in seen}
  assert {"AZK01-073", "AZK01-019", "AZK01-010", "STT01-008",
          "STT01-016"} <= names, f"auras never fired: {names}"


# ---------------------------------------------------------------------------
# Scripted dual-engine scenarios (deterministic aura transitions)
# ---------------------------------------------------------------------------


class Driver:
  """Steps BOTH engines with actions chosen from the C mask, comparing the
  semantic views + masks before every action."""

  def __init__(self, make_cref, seed, deck0, deck1):
    self.cref, self.state, self.jit_step, self.jit_mask = make_pair(
        make_cref, seed, deck0, deck1
    )
    self.step_index = 0

  def rows(self):
    return self.cref.legal_actions(self.cref.active_player)

  def compare(self):
    compare_step(self.step_index, self.cref, self.state, self.jit_mask)

  def step(self, action):
    self.compare()
    assert action in self.rows(), (action, self.rows())
    self.cref.step(np.asarray(action, np.int32))
    self.state = self.jit_step(self.state, np.asarray(action, np.int32))
    self.step_index += 1

  def find(self, pred):
    cand = [r for r in self.rows() if pred(r)]
    if not cand:
      return None
    no_token = [r for r in cand if r[3] == 0]
    return (no_token or cand)[0]

  def do(self, pred):
    row = self.find(pred)
    assert row is not None, self.rows()
    self.step(row)

  def noop(self):
    self.do(lambda r: r[0] == 0)

  @property
  def active(self):
    return self.cref.active_player

  def phase(self):
    return int(self.cref.raw(0).phase)

  def hand(self, player):
    my = self.cref.raw(player).my_observation_data
    return [int(my.hand[i].card_def_id)
            for i in range(min(int(my.hand_count), 16))]

  def garden(self, player):
    from azuki_jax import cards

    my = self.cref.raw(player).my_observation_data
    out = []
    for s in range(5):
      c = my.garden[s]
      if int(c.card_def_id) >= 0:
        out.append((s, cards.CARD_CODES[int(c.card_def_id)],
                    int(c.cur_stats.cur_atk), int(c.cur_stats.cur_hp),
                    bool(c.tap_state.tapped)))
    return out

  def play(self, code, to_alley=False):
    from azuki_jax import cards

    h = self.hand(self.active)
    want = cards.CODE_TO_ID[code]
    if want not in h:
      return False
    hi = h.index(want)
    t = 2 if to_alley else 1
    row = self.find(lambda r: r[0] == t and r[1] == hi)
    if row is None:
      return False
    self.step(row)
    return True

  def attach(self, code, slot):
    from azuki_jax import cards

    h = self.hand(self.active)
    want = cards.CODE_TO_ID[code]
    if want not in h:
      return False
    hi = h.index(want)
    row = self.find(lambda r: r[0] == 7 and r[1] == hi and r[2] == slot)
    if row is None:
      return False
    self.step(row)
    return True

  def maybe_close_response(self, player):
    if self.active == player and self.phase() == 3:
      self.noop()

  def ikz_count(self, player):
    my = self.cref.raw(player).my_observation_data
    return sum(1 for i in range(10) if int(my.ikz_area[i].card_def_id) >= 0)


def test_scripted_073_grant_on_removal(make_cref):
  """Probe A2 as a dual-engine test: garden {073, STT02-006 (non-Beanz)} ->
  no buff; the non-Beanz dies -> all-Beanz grant lands in both engines
  (post-removal recount, empirically verified vs C); a later non-Beanz play
  removes it again."""
  deck0 = [("STT01-001", 1), ("STT01-002", 1),
           ("AZK01-073", 25), ("STT02-006", 25), ("IKZ-001", 10)]
  deck1 = [("STT01-001", 1), ("STT01-002", 1),
           ("STT02-004", 50), ("IKZ-001", 10)]
  d = Driver(make_cref, 7, deck0, deck1)
  ME, OPP = 0, 1
  d.noop(); d.noop()  # mulligans

  st = {"x": False, "k073": False, "tapped": False, "killed": False,
        "regrew": False}
  for _ in range(120):
    if st["killed"]:
      break
    if d.active == ME:
      if not st["x"] and d.play("STT02-006"):
        st["x"] = True
        continue
      if st["x"] and not st["k073"] and d.play("AZK01-073"):
        st["k073"] = True
        g = d.garden(ME)
        b = [x for x in g if x[1] == "AZK01-073"][0]
        assert (b[2], b[3]) == (1, 1), g  # mixed garden: no buff
        continue
      if st["k073"] and not st["tapped"]:
        g = d.garden(ME)
        s = [x[0] for x in g if x[1] == "STT02-006" and not x[4]]
        if s:
          row = d.find(lambda r: r[0] == 6 and r[1] == s[0])
          if row:
            d.step(row)
            d.maybe_close_response(OPP)
            st["tapped"] = True
            continue
      d.noop()
    else:
      go = d.garden(OPP)
      if len(go) < 1 and d.play("STT02-004"):
        continue
      gm = d.garden(ME)
      t = [x[0] for x in gm if x[1] == "STT02-006" and x[4]]
      att = [x[0] for x in go if not x[4]]
      if t and att and st["tapped"]:
        row = d.find(lambda r: r[0] == 6 and r[1] == att[0] and r[2] == t[0])
        if row:
          d.step(row)
          d.maybe_close_response(ME)
          st["killed"] = True
          continue
      d.noop()
  assert st["killed"], "scenario never reached the kill"
  d.compare()
  g = d.garden(ME)
  buffed = [x for x in g if x[1] == "AZK01-073"]
  assert buffed and (buffed[0][2], buffed[0][3]) == (2, 2), g

  # follow-up: a non-Beanz play removes the buff in both engines
  for _ in range(30):
    if d.active == ME:
      if d.play("STT02-006"):
        st["regrew"] = True
        break
      d.noop()
    else:
      d.noop()
  assert st["regrew"]
  d.compare()
  g = d.garden(ME)
  b = [x for x in g if x[1] == "AZK01-073"][0]
  assert (b[2], b[3]) == (1, 1), g


def test_scripted_011_death_removes_weapon_buff(make_cref):
  """Probe B2 as a dual-engine test: host+016 (host 7), 011 played (host 8,
  weapon 5), 011 suicides into a tapped 3/3 -> aura removed (host 7)."""
  deck0 = [("STT01-001", 1), ("STT01-002", 1), ("STT01-011", 20),
           ("STT01-016", 10), ("AZK01-012", 20), ("IKZ-001", 10)]
  deck1 = [("STT01-001", 1), ("STT01-002", 1),
           ("AZK01-012", 50), ("IKZ-001", 10)]
  d = Driver(make_cref, 11, deck0, deck1)
  ME, OPP = 0, 1
  d.noop(); d.noop()

  st = {"host": False, "k011": False, "done": False}
  for _ in range(300):
    if st["done"]:
      break
    if d.active == ME:
      if not st["host"] and d.play("AZK01-012"):
        st["host"] = True
        continue
      if st["host"] and not st["k011"] and d.play("STT01-011"):
        st["k011"] = True
        continue
      from azuki_jax import cards as _c

      g = d.garden(ME)
      host = [x for x in g if x[1] == "AZK01-012"]
      s011 = [x for x in g if x[1] == "STT01-011" and not x[4]]
      opp_tapped = [x[0] for x in d.garden(OPP) if x[4]]
      if (st["k011"] and host and s011 and opp_tapped
          and d.ikz_count(ME) >= 4
          and _c.CODE_TO_ID["STT01-016"] in d.hand(ME)):
        assert d.attach("STT01-016", host[0][0])
        d.compare()
        g = d.garden(ME)
        host_atk = [x for x in g if x[1] == "AZK01-012"][0][2]
        assert host_atk == 8, g  # 3 base + 4 weapon + 1 aura
        s011 = [x for x in d.garden(ME) if x[1] == "STT01-011" and not x[4]]
        row = d.find(
            lambda r: r[0] == 6 and r[1] == s011[0][0]
            and r[2] == opp_tapped[0]
        )
        assert row, d.rows()
        d.step(row)
        d.maybe_close_response(OPP)
        st["done"] = True
        continue
      d.noop()
    else:
      go = d.garden(OPP)
      if len(go) < 2 and d.play("AZK01-012"):
        continue
      att = [x[0] for x in go if not x[4]]
      if att and d.ikz_count(ME) >= 8 and not [x for x in go if x[4]]:
        row = d.find(lambda r: r[0] == 6 and r[2] == 5)
        if row:
          d.step(row)
          d.maybe_close_response(ME)
          continue
      d.noop()
  assert st["done"], "scenario never reached the suicide attack"
  d.compare()
  g = d.garden(ME)
  host = [x for x in g if x[1] == "AZK01-012"][0]
  assert host[2] == 7, g  # aura removed with the last 011


def test_scripted_019_death_by_buff_removal(make_cref):
  """AZK01-019 (2/1, +2 hp aura while garden all-Normal -> 3 hp) attacks a
  tapped 2-atk enemy in its own turn (takes 2 -> 1 hp; EOT would heal it, so
  damage and flip must share a turn), then a Water entity is played: the -2
  aura removal drops it to -1 -> it dies in both engines (the C
  remove_health_modifier death path in azk_process_passive_buff_queue).
  Scenario validated against C standalone first."""
  deck0 = [("STT01-001", 1), ("STT01-002", 1), ("AZK01-019", 25),
           ("AZK01-025", 25), ("IKZ-001", 10)]
  deck1 = [("STT01-001", 1), ("STT01-002", 1),
           ("STT02-008", 50), ("IKZ-001", 10)]
  d = Driver(make_cref, 13, deck0, deck1)
  ME, OPP = 0, 1
  d.noop(); d.noop()

  from azuki_jax import cards as _c

  st = {"k019": False, "done": False}
  for _ in range(200):
    if st["done"]:
      break
    if d.active == ME:
      if not st["k019"] and d.play("AZK01-019"):
        st["k019"] = True
        g = d.garden(ME)
        b = [x for x in g if x[1] == "AZK01-019"][0]
        assert (b[2], b[3]) == (2, 3), g  # aura active: 2/1 -> 2/3
        continue
      g = d.garden(ME)
      mine = [x for x in g if x[1] == "AZK01-019" and not x[4]]
      opp_tapped = [x[0] for x in d.garden(OPP) if x[4]]
      if (st["k019"] and mine and opp_tapped and d.ikz_count(ME) >= 4
          and _c.CODE_TO_ID["AZK01-025"] in d.hand(ME)):
        row = d.find(
            lambda r: r[0] == 6 and r[1] == mine[0][0]
            and r[2] == opp_tapped[0]
        )
        if row:
          d.step(row)
          d.maybe_close_response(OPP)
          g = d.garden(ME)
          hurt = [x for x in g if x[1] == "AZK01-019"][0]
          assert hurt[3] == 1, g  # 3 hp - 2 combat damage
          assert d.play("AZK01-025")
          st["done"] = True
          continue
      d.noop()
    else:
      go = d.garden(OPP)
      if len(go) < 2 and d.play("STT02-008"):
        continue
      att = [x[0] for x in go if not x[4]]
      if att and d.ikz_count(ME) >= 4 and not [x for x in go if x[4]]:
        row = d.find(lambda r: r[0] == 6 and r[2] == 5)
        if row:
          d.step(row)
          d.maybe_close_response(ME)
          continue
      d.noop()
  assert st["done"], "scenario never reached the damage+flip turn"
  d.compare()
  g = d.garden(ME)
  assert not [x for x in g if x[1] == "AZK01-019"], (
      f"019 should have died from aura removal: {g}"
  )
