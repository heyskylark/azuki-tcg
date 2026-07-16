#!/usr/bin/env python3
"""Deeper pattern queries over logged self-play games (companion to
analyze_selfplay_games.py). Focus: gate-followup resolution, charge-enabled
same-turn attacks, response-window behavior, mana discipline, lethal shape.

Usage: python3 query_selfplay_patterns.py GAMES.jsonl... --json OUT.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from analyze_selfplay_games import ELEMENT_OF_GATE, annotate_turns, load_games

GATE_CODES = set(ELEMENT_OF_GATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    games = [annotate_turns(g) for g in load_games(args.inputs)]
    print(f"loaded {len(games)} games")

    out = {}

    # ---- 1. gate portal followups: ability steps with src == own gate ------
    gate_followup = defaultdict(Counter)      # gate -> Counter of follow-up detail
    portal_counts = Counter()                 # gate -> portals taken
    portal_turns = defaultdict(list)
    for g in games:
        gates = [g["decks"][p]["gate"] for p in range(2)]
        steps = g["steps"]
        for idx, s in enumerate(steps):
            if s["d"]["t"] != "GATE_PORTAL":
                continue
            p = s["p"]
            gate = gates[p]
            portal_counts[gate] += 1
            portal_turns[gate].append(s.get("turn", 0))
            # scan the ability-resolution window right after the portal
            j = idx + 1
            saw = []
            while j < len(steps) and j < idx + 12:
                nxt = steps[j]
                if nxt["p"] != p:
                    break
                src = nxt.get("ability_src")
                if src is None or src != gate:
                    break
                d = nxt["d"]
                if d["t"] in ("SELECT_FROM_SELECTION", "SELECT_TO_EQUIP", "SELECT_TO_GARDEN",
                              "SELECT_TO_ALLEY") and "card" in d:
                    saw.append(f"{d['t']}:{d['card']}")
                elif d["t"] in ("SELECT_COST_TARGET", "SELECT_EFFECT_TARGET"):
                    saw.append(d["t"])
                elif d["t"] == "CONFIRM_ABILITY":
                    saw.append("CONFIRM")
                elif d["t"] == "NOOP":
                    saw.append("DECLINE")
                j += 1
            if saw:
                gate_followup[gate][" | ".join(saw[:3])] += 1
            else:
                gate_followup[gate]["(no ability window)"] += 1
    out["portals_per_gate"] = {
        k: {"portals": portal_counts[k],
            "mean_turn": round(sum(portal_turns[k]) / len(portal_turns[k]), 2)}
        for k in sorted(portal_counts)
    }
    out["gate_followups"] = {k: dict(c.most_common(12)) for k, c in gate_followup.items()}

    # ---- 2. same-turn played->attacked (charge realized) --------------------
    charge_attacks = Counter()   # card -> times attacked same turn it entered garden
    charge_enablers = Counter()  # spells cast that turn when it happened
    portal_attacks = Counter()   # card -> attacked same turn it was portaled
    for g in games:
        per_turn = defaultdict(list)
        for s in g["steps"]:
            if s["ph"] == "MULLIGAN":
                continue
            per_turn[(s["turn"], s["p"])].append(s)
        for (_, p), items in per_turn.items():
            entered = {}
            spells = []
            for s in items:
                d = s["d"]
                if d["t"] == "PLAY_ENTITY_TO_GARDEN" and "card" in d:
                    entered[d["card"]] = "played"
                elif d["t"] == "GATE_PORTAL" and "card" in d:
                    entered[d["card"]] = "portaled"
                elif d["t"] == "PLAY_SPELL_FROM_HAND" and "card" in d:
                    spells.append(d["card"])
                elif d["t"] == "ATTACK":
                    a = d.get("attacker")
                    if a in entered:
                        if entered[a] == "played":
                            charge_attacks[a] += 1
                        else:
                            portal_attacks[a] += 1
                        for sp in spells:
                            charge_enablers[sp] += 1
    out["same_turn_entry_attacks_played"] = dict(charge_attacks.most_common(20))
    out["same_turn_entry_attacks_portaled"] = dict(portal_attacks.most_common(20))
    out["spells_cast_in_those_turns"] = dict(charge_enablers.most_common(15))

    # ---- 3. response-window behavior ----------------------------------------
    resp = Counter()
    resp_actions = Counter()
    for g in games:
        for s in g["steps"]:
            if s["ph"] != "RESPONSE":
                continue
            d = s["d"]
            resp_actions[d["t"]] += 1
            if "card" in d and not d["card"].endswith("?"):
                resp[f"{d['t']}:{d['card']}"] += 1
    out["response_action_mix"] = dict(resp_actions.most_common())
    out["response_cards"] = dict(resp.most_common(30))

    # ---- 4. mana discipline: floating IKZ at end of own MAIN ----------------
    floating = defaultdict(list)  # turn_bucket -> leftover untapped ikz at pass
    for g in games:
        steps = g["steps"]
        for idx, s in enumerate(steps):
            if s["ph"] != "MAIN" or s["d"]["t"] != "NOOP":
                continue
            nxt = steps[idx + 1] if idx + 1 < len(steps) else None
            # a NOOP that hands over the turn (next MAIN step is the opponent's)
            if nxt is None or (nxt["ph"] == "MAIN" and nxt["p"] != s["p"]) or nxt["ph"] == "MULLIGAN":
                t = s.get("turn", 0)
                bucket = "t1-4" if t <= 4 else ("t5-10" if t <= 10 else "t11+")
                floating[bucket].append(s["ikz"][0])
    out["floating_ikz_at_turn_pass"] = {
        k: {"mean": round(sum(v) / len(v), 2),
            "zero_rate": round(sum(1 for x in v if x == 0) / len(v), 3),
            "n": len(v)}
        for k, v in sorted(floating.items())
    }

    # ---- 5. lethal shape: how games end --------------------------------------
    end_kinds = Counter()
    winner_last_face_turns = []
    for g in games:
        w = g["outcome"]["winner"]
        if w < 0:
            end_kinds["draw/trunc"] += 1
            continue
        last = None
        for s in reversed(g["steps"]):
            if s["p"] == w and s["d"]["t"] != "NOOP":
                last = s
                break
        if last is None:
            end_kinds["opponent_side_end"] += 1
            continue
        d = last["d"]
        if d["t"] == "ATTACK" and d.get("target") == "OPP_LEADER":
            end_kinds["face_attack"] += 1
        elif d["t"] == "ATTACK":
            end_kinds["board_attack"] += 1
        elif d["t"] == "PLAY_SPELL_FROM_HAND":
            end_kinds[f"spell:{d.get('card')}"] += 1
        else:
            end_kinds[d["t"]] += 1
    out["winning_final_actions"] = dict(end_kinds.most_common(15))

    # ---- 6. defender / interception rate -------------------------------------
    n_attacks = 0
    n_defended = 0
    for g in games:
        steps = g["steps"]
        for idx, s in enumerate(steps):
            if s["d"]["t"] != "ATTACK":
                continue
            n_attacks += 1
            j = idx + 1
            while j < len(steps) and steps[j]["ph"] == "RESPONSE":
                if steps[j]["d"]["t"] == "DECLARE_DEFENDER":
                    n_defended += 1
                    break
                j += 1
    out["attacks_total"] = n_attacks
    out["attacks_intercepted_rate"] = round(n_defended / max(n_attacks, 1), 3)

    # ---- 7. mulligan rate -----------------------------------------------------
    mull = Counter()
    for g in games:
        for s in g["steps"]:
            if s["ph"] == "MULLIGAN":
                mull[s["d"]["t"]] += 1
    out["mulligan"] = dict(mull)

    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")
    print(json.dumps(out, indent=2)[:5000])


if __name__ == "__main__":
    main()
