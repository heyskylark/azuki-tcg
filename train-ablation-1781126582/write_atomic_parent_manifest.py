#!/usr/bin/env python3
"""Write a hash-locked model/trainer/league parent manifest for the next stage."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _entry(path: Path) -> dict[str, str]:
  if not path.is_file():
    raise FileNotFoundError(path)
  return {"path": str(path), "sha256": _sha256(path)}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, required=True)
  parser.add_argument("--trainer", type=Path, required=True)
  parser.add_argument("--league", type=Path, required=True)
  parser.add_argument("--promotion", type=Path, required=True)
  parser.add_argument("--source-decision", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  metadata_path = Path(f"{args.model}.meta.json")
  metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
  epoch = metadata.get("update")
  completed = metadata.get("env_completed_episodes")
  if not isinstance(epoch, int) or epoch < 0:
    raise ValueError("checkpoint metadata has no valid update")
  if not isinstance(completed, int) or completed < 0:
    raise ValueError("checkpoint metadata has no valid env_completed_episodes")

  payload = {
    "schema_version": 1,
    "epoch": epoch,
    "completed_episodes": completed,
    "model": _entry(args.model),
    "trainer": _entry(args.trainer),
    "metadata": _entry(metadata_path),
    "league": _entry(args.league),
    "promotion": _entry(args.promotion),
    "source_decision": _entry(args.source_decision),
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  print(
    f"[parent-manifest] epoch={epoch} completed_episodes={completed} "
    f"output={args.output}"
  )


if __name__ == "__main__":
  main()
