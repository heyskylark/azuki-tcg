#!/usr/bin/env python3
"""Render paired_comparisons.json into markdown tables (tables.md)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ELEMENT_PREFIX = {"FIRE": ("fire/", "gate/Rushfire", "gate/Ragefire", "leader/Zero", "leader/Kagoro"),
                  "LIGHTNING": ("lightning/", "gate/Surge", "gate/Stormchain", "leader/Raizan", "leader/Piko"),
                  "WATER": ("water/", "gate/Hydromancy", "leader/Shao", "leader/Benzai"),
                  "EARTH": ("earth/", "gate/Stonehaven", "leader/Bobu", "leader/Goro", "block/")}
COMMON = ("leader/use_per_legal_turn", "gate/portal_per_legal_turn", "strategy/face_target_share_when_both_legal",
          "strategy/favorable_trade_taken_per_available_turn", "strategy/attacks_by_equipped_attacker",
          "strategy/spell_cast_per_legal_main_turn", "ikz/held_at_end_of_own_turn", "ikz/held_then_spent_in_opp_turn",
          "response/spell_played_when_legal", "response/any_nonblock_response_when_legal",
          "response/defender_declared_when_legal", "block/favorable_survive_or_kill", "heal/heal_spell_per_legal_turn_below_max_hp",
          "outcome/win", "outcome/own_turns")


def fmt(r, pct=True):
  if r is None:
    return "–"
  return f"{100 * r:.1f}" if pct else f"{r:.2f}"


def row(name, c, min_den):
  if c is None or (c.get("nx", 0) < min_den and c.get("nb", 0) < min_den):
    return None
  pct = not name.endswith(("own_turns", "unique_cards", "mean_cost", "battle_decisions"))
  scale = 100 if pct else 1
  if "delta" not in c:
    return f"| {name} | {fmt(c['b'], pct)} ({c['nb']}) | {fmt(c['x'], pct)} ({c['nx']}) | – | |"
  lo, hi = c["ci95"]
  sig = "**" if lo is not None and (lo > 0 or hi < 0) else ""
  return (f"| {name} | {fmt(c['b'], pct)} ({c['nb']}) | {fmt(c['x'], pct)} ({c['nx']}) | "
          f"{sig}{scale * c['delta']:+.1f}{sig} | [{scale * lo:+.1f}, {scale * hi:+.1f}] |")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--min-den", type=int, default=8)
  args = ap.parse_args()
  comps = json.loads((ROOT / "descriptors/paired_comparisons.json").read_text())
  out = ["# Paired behavior tables (auto-generated)",
         "Rates in %, (n) = opportunities (denominator) for that arm; delta = arm − u8223 in pp "
         "(own_turns/unique_cards/mean_cost in raw units); 95% block-bootstrap CI over seed blocks; ** = CI excludes 0.", ""]
  for element in ("FIRE", "LIGHTNING", "WATER", "EARTH"):
    out.append(f"## {element}")
    for key in sorted(k for k in comps if k.startswith(element + "|")):
      _, arm, mode = key.split("|")
      for slice_name in sorted(comps[key], key=lambda s: (s.split("|")[0] != "all", s)):
        metrics = comps[key][slice_name]
        names = [n for n in metrics if n in COMMON or any(n.startswith(p) for p in ELEMENT_PREFIX[element])
                 or n.startswith(("sequence/", "draft/", "outcome/win_vs"))]
        if slice_name.split("|")[1] != "ALL":
          names = [n for n in names if not n.startswith("outcome/win_vs")]
        lines = [r for n in names if (r := row(n, metrics[n], args.min_den))]
        if not lines:
          continue
        out += [f"### {arm} vs u8223 — {mode} — {slice_name}", "",
                f"| metric | u8223 | {arm} | Δ | 95% CI |", "|---|---|---|---|---|", *lines, ""]
  (ROOT / "tables.md").write_text("\n".join(out) + "\n")
  print("wrote", ROOT / "tables.md")


if __name__ == "__main__":
  main()
