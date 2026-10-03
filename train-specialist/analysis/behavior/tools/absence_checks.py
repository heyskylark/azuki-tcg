#!/usr/bin/env python3
"""Direct trace checks backing absence/zero claims (Raizan/Piko legality, entity weapon attach, Stormchain, drafts)."""
import glob, json
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
out = defaultdict(Counter)
for path in sorted(glob.glob(str(ROOT / "traces/lightning__*.jsonl")) + sorted(glob.glob(str(ROOT / "traces/fire__*.jsonl")))):
  arm, mode = Path(path).name.split("__")[1:3]
  for line in open(path):
    g = json.loads(line); pe = g["paired_eval"]; s = pe["candidate_seat"]
    leader = g["decks"][s]["leader"]; gate = g["decks"][s]["gate"]
    k = f"{pe['element']}|{arm}|{mode}|{gate}|{leader}"
    c = out[k]; c["games"] += 1
    for st in g["steps"]:
      if st["p"] != s: continue
      legal = st["legal"]
      c["decisions"] += 1
      if any(r[0] == 11 and r[1] == 5 for r in legal): c["leader_legal_decisions"] += 1
      if st["d"]["t"] == "ACTIVATE_GARDEN_OR_LEADER_ABILITY" and st["a"][1] == 5: c["leader_used"] += 1
      att = [r for r in legal if r[0] == 7]
      if att:
        c["attach_legal_decisions"] += 1
        if any(r[2] < 5 for r in att): c["attach_to_entity_legal_decisions"] += 1
      if st["d"]["t"] == "ATTACH_WEAPON_FROM_HAND":
        c["attach_chosen"] += 1; c["attach_chosen_entity"] += int(st["a"][2] < 5)
      if any("+" in e for e in st["my_garden"]): c["decisions_with_weapon_on_garden_entity"] += 1
      if any(r[0] == 10 for r in legal): c["portal_legal_decisions"] += 1
print(json.dumps({k: dict(v) for k, v in sorted(out.items())}, indent=1))
