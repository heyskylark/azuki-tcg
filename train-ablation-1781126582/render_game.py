#!/usr/bin/env python3
"""Render a logged self-play game as a readable turn-by-turn narrative.

Usage: python3 render_game.py GAMES.jsonl --game N [--names cards.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_selfplay_games import ELEMENT_OF_GATE, annotate_turns, load_games

NAMES_PATH = Path(__file__).parent.parent / "scripts" / "azuki-card-defs.jsonl"


def load_names():
    names = {}
    for line in open(NAMES_PATH):
        c = json.loads(line)
        names[c["card_id"]] = c["name"]
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--max-steps", type=int, default=500)
    args = ap.parse_args()
    names = load_names()

    def nm(code):
        if code in ("MY_LEADER", "OPP_LEADER", "FACE"):
            return code
        base = code.replace("ALLEY:", "")
        return f"{names.get(base, base)}" if base in names else code

    games = load_games(args.inputs)
    g = annotate_turns(games[args.game])
    d0, d1 = g["decks"]
    print(f"game {g['game']} seed {g['seed']}")
    for p, d in enumerate(g["decks"]):
        print(f"  P{p}: {ELEMENT_OF_GATE[d['gate']]} {d['gate_name']} gate={d['gate']} leader={nm(d['leader'])}")
    o = g["outcome"]
    print(f"  outcome: winner=P{o['winner']} rewards={o['terminal_rewards']} turns={g['n_turns']} steps={g['battle_steps']}\n")

    cur_turn = None
    for s in g["steps"][: args.max_steps]:
        t = s.get("turn", 0)
        if t != cur_turn:
            cur_turn = t
            owner = s.get("turn_owner", "-")
            print(f"--- turn {t} (P{owner}) hp={s['hp']} ---")
        d = s["d"]
        typ = d["t"]
        p = s["p"]
        extra = ""
        if typ == "NOOP":
            line = "pass"
        elif typ == "MULLIGAN_SHUFFLE":
            line = "mulligan"
        elif typ in ("PLAY_ENTITY_TO_GARDEN", "PLAY_ENTITY_TO_ALLEY"):
            zone = "garden" if typ.endswith("GARDEN") else "alley"
            line = f"play {nm(d.get('card', '?'))} -> {zone}"
        elif typ == "PLAY_SPELL_FROM_HAND":
            line = f"cast {nm(d.get('card', '?'))}"
        elif typ == "ATTACH_WEAPON_FROM_HAND":
            line = f"equip {nm(d.get('card', '?'))} -> {nm(d.get('target', '?'))}"
        elif typ == "GATE_PORTAL":
            line = f"PORTAL {nm(d.get('card', '?'))} alley->garden"
        elif typ == "ATTACK":
            line = f"attack {nm(d.get('attacker', '?'))} -> {nm(d.get('target', '?'))}"
        elif typ == "DECLARE_DEFENDER":
            line = f"defend with {nm(d.get('card', '?'))}"
        elif typ == "ACTIVATE_GARDEN_OR_LEADER_ABILITY":
            line = f"activate {nm(d.get('card', '?'))} ability"
        elif typ == "ACTIVATE_ALLEY_ABILITY":
            line = f"activate alley ability {nm(d.get('card', '?'))}"
        elif typ in ("SELECT_FROM_SELECTION", "SELECT_TO_GARDEN", "SELECT_TO_ALLEY", "SELECT_TO_EQUIP",
                     "BOTTOM_DECK_CARD", "TOP_DECK_CARD"):
            line = f"{typ.lower()} {nm(d.get('card', '?'))}"
        elif typ in ("SELECT_COST_TARGET", "SELECT_EFFECT_TARGET", "CONFIRM_ABILITY", "BOTTOM_DECK_ALL"):
            line = typ.lower()
        else:
            line = typ
        src = d.get("src")
        if src and typ not in ("PLAY_SPELL_FROM_HAND",):
            extra += f"  [resolving {nm(src)}]"
        if s["ph"] == "RESPONSE":
            extra += "  (response)"
        if "r" in s and abs(s["r"][0]) >= 1:
            extra += f"  => terminal {s['r']}"
        print(f"  P{p}: {line}{extra}")
    print()


if __name__ == "__main__":
    main()
