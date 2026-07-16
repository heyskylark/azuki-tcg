#!/usr/bin/env python3
"""Aggregate strategy/playstyle statistics from logged self-play games.

Input: JSONL from play_selfplay_games.py (one game per line).
Output: a stats JSON (for the write-up) + printed summary.

Turn derivation: MULLIGAN steps are turn 0. A new turn starts at a MAIN-phase
step whose actor differs from the current MAIN owner (control changes inside
a turn only via RESPONSE windows, which keep the turn index).

Usage:
  python3 analyze_selfplay_games.py GAMES.jsonl... --json OUT.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_games(paths):
    games = []
    for p in paths:
        for line in open(p):
            line = line.strip()
            if line:
                games.append(json.loads(line))
    return games


def annotate_turns(game):
    turn = 0
    owner = None
    for s in game["steps"]:
        if s["ph"] == "MULLIGAN":
            s["turn"] = 0
            continue
        if s["ph"] == "MAIN" and s["p"] != owner:
            owner = s["p"]
            turn += 1
        s["turn"] = turn
        s["turn_owner"] = owner
    game["n_turns"] = turn
    return game


def code_only(s):
    return s.split("@")[0].split(":")[0]


ELEMENT_OF_GATE = {
    "STT01-002": "LIGHTNING", "AZK01-120": "LIGHTNING",
    "STT02-002": "WATER", "AZK01-126": "WATER",
    "AZK01-122": "FIRE", "STT04-002": "FIRE",
    "AZK01-124": "EARTH", "STT03-002": "EARTH",
}

PLAY_ACTIONS = {
    "PLAY_ENTITY_TO_GARDEN": "play_garden",
    "PLAY_ENTITY_TO_ALLEY": "play_alley",
    "PLAY_SPELL_FROM_HAND": "play_spell",
    "ATTACH_WEAPON_FROM_HAND": "play_weapon",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    games = [annotate_turns(g) for g in load_games(args.inputs)]
    print(f"loaded {len(games)} games")

    # ---------------- global outcomes ----------------
    seat_wins = Counter()
    gate_games = Counter()
    gate_wins = Counter()
    elem_games = Counter()
    elem_wins = Counter()
    matchup = defaultdict(lambda: [0, 0])  # (elemA, elemB) -> [gamesA, winsA]
    lengths, turns = [], []
    for g in games:
        w = g["outcome"]["winner"]
        seat_wins[w] += 1
        lengths.append(g["battle_steps"])
        turns.append(g["n_turns"])
        for p in range(2):
            gate = g["decks"][p]["gate"]
            el = ELEMENT_OF_GATE[gate]
            gate_games[gate] += 1
            elem_games[el] += 1
            if w == p:
                gate_wins[gate] += 1
                elem_wins[el] += 1
        e0, e1 = (ELEMENT_OF_GATE[g["decks"][p]["gate"]] for p in range(2))
        matchup[(e0, e1)][0] += 1
        if w == 0:
            matchup[(e0, e1)][1] += 1

    # ---------------- per-card stats ----------------
    # play events, portals, attacks, defends, abilities; turn of play; wins
    card = defaultdict(lambda: defaultdict(float))
    card_play_turns = defaultdict(list)
    card_played_game_wins = defaultdict(lambda: [0, 0])   # [games_played, wins]
    card_indeck_game_wins = defaultdict(lambda: [0, 0])
    for g in games:
        w = g["outcome"]["winner"]
        played_by = [set(), set()]
        for s in g["steps"]:
            d = s["d"]
            p = s["p"]
            t = d["t"]
            if t in PLAY_ACTIONS and "card" in d and not d["card"].startswith(("hand[", "sel[")):
                c = d["card"]
                card[c][PLAY_ACTIONS[t]] += 1
                card_play_turns[c].append(s.get("turn", 0))
                played_by[p].add(c)
            elif t == "GATE_PORTAL" and "card" in d and not d["card"].startswith("a"):
                card[d["card"]]["portaled"] += 1
            elif t == "ATTACK":
                a = d.get("attacker", "?")
                tgt = d.get("target", "?")
                if not a.endswith("?"):
                    card[a]["attacks"] += 1
                    if tgt == "OPP_LEADER":
                        card[a]["attacks_face"] += 1
                    elif tgt.startswith("ALLEY:"):
                        card[a]["attacks_alley"] += 1
                    else:
                        card[a]["attacks_board"] += 1
                if not tgt.endswith("?") and tgt not in ("OPP_LEADER",):
                    card[tgt.replace("ALLEY:", "")]["attacked_as_target"] += 1
            elif t == "DECLARE_DEFENDER" and "card" in d and not d["card"].endswith("?"):
                card[d["card"]]["defends"] += 1
            elif t in ("ACTIVATE_GARDEN_OR_LEADER_ABILITY", "ACTIVATE_ALLEY_ABILITY"):
                c = d.get("card", "?")
                if c == "MY_LEADER":
                    c = g["decks"][p]["leader"]
                if not c.endswith("?"):
                    card[c]["ability_uses"] += 1
        for p in range(2):
            deck = set(g["decks"][p]["main"])
            for c in deck:
                card_indeck_game_wins[c][0] += 1
                if w == p:
                    card_indeck_game_wins[c][1] += 1
            for c in played_by[p]:
                card_played_game_wins[c][0] += 1
                if w == p:
                    card_played_game_wins[c][1] += 1

    card_out = {}
    for c, stats in card.items():
        pg, pw = card_played_game_wins.get(c, [0, 0])
        dg, dw = card_indeck_game_wins.get(c, [0, 0])
        turns_played = card_play_turns.get(c, [])
        card_out[c] = {
            **{k: int(v) for k, v in stats.items()},
            "games_played": pg,
            "win_rate_when_played": round(pw / pg, 3) if pg else None,
            "games_in_deck": dg,
            "win_rate_in_deck": round(dw / dg, 3) if dg else None,
            "mean_play_turn": round(sum(turns_played) / len(turns_played), 2)
            if turns_played
            else None,
        }

    # ---------------- within-turn sequences ----------------
    def tok(s):
        d = s["d"]
        t = d["t"]
        if t in PLAY_ACTIONS:
            return f"{PLAY_ACTIONS[t]}:{d.get('card', '?')}"
        if t == "GATE_PORTAL":
            return f"portal:{d.get('card', '?')}"
        if t == "ATTACK":
            tgt = d.get("target", "?")
            tgt = "FACE" if tgt == "OPP_LEADER" else ("ALLEY" if tgt.startswith("ALLEY:") else "BOARD")
            return f"attack:{d.get('attacker', '?')}>{tgt}"
        if t == "ACTIVATE_GARDEN_OR_LEADER_ABILITY":
            return f"ability:{d.get('card', '?')}"
        if t == "ACTIVATE_ALLEY_ABILITY":
            return f"alley_ability:{d.get('card', '?')}"
        if t == "DECLARE_DEFENDER":
            return f"defend:{d.get('card', '?')}"
        return None

    bigrams = defaultdict(Counter)  # element -> Counter
    trigrams = defaultdict(Counter)
    combo = Counter()
    combo_examples = defaultdict(list)
    leader_ability = Counter()
    leader_ability_games = Counter()

    for gi, g in enumerate(games):
        elems = [ELEMENT_OF_GATE[g["decks"][p]["gate"]] for p in range(2)]
        # group by (turn, actor) for owner actions (skip response-window steps
        # of the non-owner; they are separate reactive sequences)
        seqs = defaultdict(list)
        for s in g["steps"]:
            if s["ph"] == "MULLIGAN":
                continue
            token = tok(s)
            if token:
                seqs[(s["turn"], s["p"])].append((s, token))
        used_leader_ability = [False, False]
        for (turn, p), items in seqs.items():
            el = elems[p]
            toks = [t for _, t in items]
            for i in range(len(toks) - 1):
                bigrams[el][(toks[i], toks[i + 1])] += 1
            for i in range(len(toks) - 2):
                trigrams[el][(toks[i], toks[i + 1], toks[i + 2])] += 1
            # combos
            played_alley = {}
            portaled = {}
            for s, token in items:
                d = s["d"]
                t = d["t"]
                if t == "PLAY_ENTITY_TO_ALLEY":
                    played_alley[d.get("card")] = s["i"]
                if t == "GATE_PORTAL":
                    c = d.get("card")
                    portaled[c] = s["i"]
                    if c in played_alley:
                        combo["alley_play_then_portal_same_turn"] += 1
                        if len(combo_examples["alley_play_then_portal_same_turn"]) < 5:
                            combo_examples["alley_play_then_portal_same_turn"].append(
                                (gi, turn, c)
                            )
                if t == "ATTACK":
                    a = d.get("attacker")
                    if a in portaled:
                        combo["portal_then_attack_same_turn"] += 1
                        if len(combo_examples["portal_then_attack_same_turn"]) < 5:
                            combo_examples["portal_then_attack_same_turn"].append(
                                (gi, turn, a, d.get("target"))
                            )
                if t == "ACTIVATE_GARDEN_OR_LEADER_ABILITY" and d.get("card") == "MY_LEADER":
                    leader_ability[g["decks"][p]["leader"]] += 1
                    used_leader_ability[p] = True
        for p in range(2):
            if used_leader_ability[p]:
                leader_ability_games[g["decks"][p]["leader"]] += 1

    # response-window plays (reactive card use)
    response_plays = Counter()
    for g in games:
        for s in g["steps"]:
            if s["ph"] == "RESPONSE" and s["d"]["t"] in PLAY_ACTIONS and "card" in s["d"]:
                response_plays[s["d"]["card"]] += 1

    # ---------------- long-term arcs ----------------
    # winner-vs-loser HP by turn; ikz spend discipline; attack target mix by game third
    hp_by_turn = defaultdict(lambda: [[], []])  # turn -> [winner_hps, loser_hps]
    target_mix = {"early": Counter(), "mid": Counter(), "late": Counter()}
    portals_per_game = []
    first_attack_turn = []
    for g in games:
        w = g["outcome"]["winner"]
        if w < 0:
            continue
        nt = max(g["n_turns"], 1)
        fat = None
        portals = 0
        for s in g["steps"]:
            if s["ph"] == "MULLIGAN":
                continue
            hp = s["hp"]
            hp_by_turn[s["turn"]][0].append(hp[w])
            hp_by_turn[s["turn"]][1].append(hp[1 - w])
            d = s["d"]
            if d["t"] == "GATE_PORTAL":
                portals += 1
            if d["t"] == "ATTACK":
                if fat is None:
                    fat = s["turn"]
                tgt = d.get("target", "?")
                bucket = "early" if s["turn"] <= nt / 3 else ("mid" if s["turn"] <= 2 * nt / 3 else "late")
                kind = "face" if tgt == "OPP_LEADER" else ("alley" if tgt.startswith("ALLEY:") else "board")
                target_mix[bucket][kind] += 1
        portals_per_game.append(portals)
        if fat is not None:
            first_attack_turn.append(fat)

    hp_curve = {
        t: {
            "winner_mean_hp": round(sum(v[0]) / len(v[0]), 2),
            "loser_mean_hp": round(sum(v[1]) / len(v[1]), 2),
            "n": len(v[0]),
        }
        for t, v in sorted(hp_by_turn.items())
        if len(v[0]) >= 20
    }

    # ---------------- assemble ----------------
    out = {
        "n_games": len(games),
        "seat_wins": {str(k): v for k, v in seat_wins.items()},
        "mean_battle_steps": round(sum(lengths) / len(lengths), 1),
        "mean_turns": round(sum(turns) / len(turns), 1),
        "gate_winrates": {
            k: {"games": gate_games[k], "win_rate": round(gate_wins[k] / gate_games[k], 3)}
            for k in sorted(gate_games)
        },
        "element_winrates": {
            k: {"games": elem_games[k], "win_rate": round(elem_wins[k] / elem_games[k], 3)}
            for k in sorted(elem_games)
        },
        "element_matchups_p0_perspective": {
            f"{a}v{b}": {"games": n, "p0_win_rate": round(wn / n, 3)}
            for (a, b), (n, wn) in sorted(matchup.items())
            if n >= 3
        },
        "cards": card_out,
        "response_window_plays": dict(response_plays.most_common(40)),
        "leader_ability_uses": dict(leader_ability),
        "leader_ability_games": dict(leader_ability_games),
        "combos": dict(combo),
        "combo_examples": {k: v for k, v in combo_examples.items()},
        "top_bigrams_by_element": {
            el: [{"seq": " -> ".join(k), "n": n} for k, n in c.most_common(25)]
            for el, c in bigrams.items()
        },
        "top_trigrams_by_element": {
            el: [{"seq": " -> ".join(k), "n": n} for k, n in c.most_common(15)]
            for el, c in trigrams.items()
        },
        "hp_curve_by_turn": hp_curve,
        "attack_target_mix_by_game_third": {k: dict(v) for k, v in target_mix.items()},
        "mean_portals_per_game_both_seats": round(sum(portals_per_game) / len(portals_per_game), 2),
        "mean_first_attack_turn": round(sum(first_attack_turn) / len(first_attack_turn), 2),
    }

    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")

    print(json.dumps({k: v for k, v in out.items() if k not in ("cards", "top_bigrams_by_element", "top_trigrams_by_element", "hp_curve_by_turn")}, indent=2)[:4000])


if __name__ == "__main__":
    main()
