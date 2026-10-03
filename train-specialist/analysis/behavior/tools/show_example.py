#!/usr/bin/env python3
"""Print a compact trace excerpt: show_example.py FILE TASK_ID STEP [--n 6]."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("file")
  ap.add_argument("task_id")
  ap.add_argument("step", type=int)
  ap.add_argument("--n", type=int, default=6)
  ap.add_argument("--before", type=int, default=0)
  args = ap.parse_args()
  path = Path(args.file)
  if not path.exists():
    path = ROOT / "traces" / args.file
  for line in path.open():
    game = json.loads(line)
    if game["paired_eval"]["task_id"] != args.task_id:
      continue
    seat = game["paired_eval"]["candidate_seat"]
    print(f"{args.task_id} arm={game['paired_eval']['arm']} mode={game['paired_eval']['mode']} cand_seat={seat} "
          f"deck={game['decks'][seat]['gate']}/{game['decks'][seat]['leader']} score={game['paired_eval']['candidate_score']}")
    for i in range(max(0, args.step - args.before), min(len(game["steps"]), args.step + args.n)):
      s = game["steps"][i]
      who = "CAND" if s["p"] == seat else "opp "
      d = {k: v for k, v in s["d"].items() if k != "raw"}
      print(f"  [{i}] {who} {s['ph']:<8} hp={s['hp']} ikz={s['ikz']} {d} | my_g={s['my_garden']} opp_g={s['opp_garden']}"
            + (f" combat={s['combat']}" if s.get("combat") else ""))
    return
  raise SystemExit("task not found")


if __name__ == "__main__":
  main()
