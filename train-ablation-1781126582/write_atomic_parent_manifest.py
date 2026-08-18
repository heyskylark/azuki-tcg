#!/usr/bin/env python3
"""Export a portable, hash-closed production training parent."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TARGETS = (
  "python/src/azk_puffer/trainer.py",
  "python/src/league_training.py",
  "python/src/draft_prefix_outcome.py",
  "python/src/policy/v2/tcg_policy.py",
  "python/src/policy/v2/tcg_sampler.py",
  "python/src/v2/tcg.py",
  "python/src/v2/tcg_parallel.py",
  "python/src/v2/observation.py",
  "python/src/deck_building.py",
  "python/src/tcg.h",
  "python/src/production_runtime.py",
  "python/src/train.py",
  "python/src/training_deck_pool.py",
  "python/src/training_utils.py",
  ".codex/docs/azuki_tcg_decks_final.json",
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _portable_path(path: Path) -> str:
  return str(path.resolve().relative_to(REPO_ROOT))


def _entry(path: Path) -> dict[str, str | int]:
  if not path.is_file():
    raise FileNotFoundError(path)
  return {
    "path": _portable_path(path),
    "sha256": _sha256(path),
    "bytes": path.stat().st_size,
  }


def _copy(source: Path, destination: Path) -> Path:
  if not source.is_file():
    raise FileNotFoundError(source)
  destination.parent.mkdir(parents=True, exist_ok=True)
  temporary = destination.with_suffix(destination.suffix + ".tmp")
  shutil.copy2(source, temporary)
  temporary.replace(destination)
  return destination




def _required_league_checkpoints(payload: dict) -> tuple[set[str], set[Path]]:
  policies = payload.get("policies")
  if not isinstance(policies, dict):
    raise ValueError("league state has no policies object")
  required_ids = {
    str(policy_id)
    for policy_id, raw in policies.items()
    if isinstance(raw, dict) and bool(raw.get("active", False))
  }
  for key in ("champion_policy_id", "current_candidate_policy_id"):
    value = payload.get(key)
    if isinstance(value, str) and value:
      required_ids.add(value)

  paths: set[Path] = set()
  for policy_id in required_ids:
    raw = policies.get(policy_id)
    if not isinstance(raw, dict):
      raise ValueError(f"required league policy is missing: {policy_id}")
    checkpoint = raw.get("checkpoint_path")
    if not isinstance(checkpoint, str):
      raise ValueError(f"required league policy has no checkpoint: {policy_id}")
    path = Path(checkpoint).expanduser()
    if not path.is_absolute():
      path = REPO_ROOT / path
    if not path.is_file():
      raise FileNotFoundError(path)
    paths.add(path.resolve())
  return required_ids, paths


def _required_promotion_checkpoints(
  promotion: dict,
  league: dict,
) -> tuple[set[str], set[Path]]:
  required_ids: set[str] = set()
  anchor_id = promotion.get("production_anchor_policy_id")
  if isinstance(anchor_id, str) and anchor_id:
    required_ids.add(anchor_id)
  active_version = promotion.get("active_panel_version")
  panels = promotion.get("panels")
  if not isinstance(panels, list):
    raise ValueError("promotion state has no panels array")
  for panel in panels:
    if not isinstance(panel, dict) or panel.get("version") != active_version:
      continue
    members = panel.get("members")
    if isinstance(members, list):
      required_ids.update(
        str(member["policy_id"])
        for member in members
        if isinstance(member, dict) and isinstance(member.get("policy_id"), str)
      )
  archive = promotion.get("quality_archive")
  if isinstance(archive, list):
    required_ids.update(
      str(entry["policy_id"])
      for entry in archive
      if isinstance(entry, dict)
      and bool(entry.get("active", False))
      and isinstance(entry.get("policy_id"), str)
    )

  paths_by_id: dict[str, Path] = {}
  league_policies = league.get("policies")
  if isinstance(league_policies, dict):
    for policy_id, raw in league_policies.items():
      if isinstance(raw, dict) and isinstance(raw.get("checkpoint_path"), str):
        paths_by_id[str(policy_id)] = Path(raw["checkpoint_path"])
  for panel in panels:
    if not isinstance(panel, dict):
      continue
    members = panel.get("members")
    if not isinstance(members, list):
      continue
    for member in members:
      if isinstance(member, dict) and isinstance(member.get("policy_id"), str):
        checkpoint = member.get("checkpoint_path")
        if isinstance(checkpoint, str):
          paths_by_id[str(member["policy_id"])] = Path(checkpoint)

  paths: set[Path] = set()
  for policy_id in required_ids:
    raw_path = paths_by_id.get(policy_id)
    if raw_path is None:
      raise ValueError(f"required promotion policy is missing: {policy_id}")
    path = raw_path.expanduser()
    if not path.is_absolute():
      path = REPO_ROOT / path
    if not path.is_file():
      raise FileNotFoundError(path)
    paths.add(path.resolve())
  return required_ids, paths


def _rewrite_paths(value: object, mapping: dict[Path, Path]) -> object:
  if isinstance(value, dict):
    return {key: _rewrite_paths(item, mapping) for key, item in value.items()}
  if isinstance(value, list):
    return [_rewrite_paths(item, mapping) for item in value]
  if isinstance(value, str):
    path = Path(value).expanduser()
    if not path.is_absolute():
      path = REPO_ROOT / path
    try:
      resolved = path.resolve()
    except OSError:
      return value
    destination = mapping.get(resolved)
    if destination is not None:
      return _portable_path(destination)
  return value


def _write_json(path: Path, payload: object) -> Path:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
  temporary.replace(path)
  return path


def _copy_checkpoint_dependency(
  source: Path,
  destination: Path,
) -> tuple[Path, Path | None]:
  copied = _copy(source, destination)
  metadata_source = source.with_suffix(source.suffix + ".meta.json")
  if not metadata_source.is_file():
    return copied, None
  metadata_destination = destination.with_suffix(destination.suffix + ".meta.json")
  return copied, _copy(metadata_source, metadata_destination)


def _verify_entry(raw: object) -> None:
  if not isinstance(raw, dict):
    raise TypeError("manifest entry must be an object")
  path_raw = raw.get("path")
  expected = raw.get("sha256")
  if not isinstance(path_raw, str) or not isinstance(expected, str):
    raise ValueError("manifest entry requires path and sha256")
  path = REPO_ROOT / path_raw
  actual = _sha256(path)
  if actual != expected:
    raise ValueError(f"hash mismatch for {path}: expected={expected} actual={actual}")


def _git_revision() -> str:
  result = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=REPO_ROOT,
    check=True,
    capture_output=True,
    text=True,
  )
  return result.stdout.strip()


def export_bundle(args: argparse.Namespace) -> Path:
  output = args.output.resolve()
  try:
    bundle_root = output.parent.relative_to(REPO_ROOT)
  except ValueError as exc:
    raise ValueError("output must be inside the repository") from exc
  bundle_root = REPO_ROOT / bundle_root

  metadata_source = Path(f"{args.model}.meta.json")
  metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
  epoch = metadata.get("update")
  completed = metadata.get("env_completed_episodes")
  if not isinstance(epoch, int) or epoch < 0:
    raise ValueError("checkpoint metadata has no valid update")
  if not isinstance(completed, int) or completed < 0:
    raise ValueError("checkpoint metadata has no valid env_completed_episodes")

  league_payload = json.loads(args.league.read_text(encoding="utf-8"))
  promotion_payload = json.loads(args.promotion.read_text(encoding="utf-8"))
  required_ids, league_dependencies = _required_league_checkpoints(league_payload)
  required_promotion_ids, promotion_dependencies = _required_promotion_checkpoints(
    promotion_payload,
    league_payload,
  )
  dependencies = league_dependencies | promotion_dependencies | {
    args.production_anchor.resolve(),
  }

  resume_model = bundle_root / "resume" / args.model.name
  resume_trainer = bundle_root / "resume" / args.trainer.name
  resume_metadata = resume_model.with_suffix(resume_model.suffix + ".meta.json")
  anchor_destination = bundle_root / "anchors" / "production_anchor.pt"
  mapping: dict[Path, Path] = {
    args.model.resolve(): resume_model,
    args.production_anchor.resolve(): anchor_destination,
  }

  _copy(args.model, resume_model)
  _copy(args.trainer, resume_trainer)
  _copy(metadata_source, resume_metadata)
  _copy_checkpoint_dependency(args.production_anchor, anchor_destination)

  dependency_entries = []
  for source in sorted(dependencies, key=str):
    if source in mapping:
      destination = mapping[source]
    else:
      digest = _sha256(source)
      destination = bundle_root / "opponents" / f"{digest[:16]}-{source.name}"
      mapping[source] = destination
      _copy_checkpoint_dependency(source, destination)
    dependency_entries.append(
      {
        "source_name": source.name,
        **_entry(destination),
      }
    )

  rewritten_league = _rewrite_paths(league_payload, mapping)
  rewritten_promotion = _rewrite_paths(promotion_payload, mapping)
  league_destination = _write_json(
    bundle_root / "state" / "league_state.json",
    rewritten_league,
  )
  promotion_destination = _write_json(
    bundle_root / "state" / "league_state_promotion.json",
    rewritten_promotion,
  )

  config_destination = _copy(args.config, bundle_root / "config" / args.config.name)
  catalog_destination = _copy(
    args.card_catalog,
    bundle_root / "config" / "card_catalog.json",
  )
  panel_destination = _copy(
    args.bootstrap_panel,
    bundle_root / "config" / "panel-v1.json",
  )
  binding_destination = _copy(
    args.binding,
    bundle_root / "runtime" / args.binding.name,
  )
  decision_destination = _copy(
    args.source_decision,
    bundle_root / "qualification" / args.source_decision.name,
  )

  source_entries = [_entry(REPO_ROOT / relative) for relative in SOURCE_TARGETS]
  payload = {
    "schema_version": 2,
    "epoch": epoch,
    "completed_episodes": completed,
    "git_revision": _git_revision(),
    "model": _entry(resume_model),
    "trainer": _entry(resume_trainer),
    "metadata": _entry(resume_metadata),
    "league": _entry(league_destination),
    "promotion": _entry(promotion_destination),
    "source_decision": _entry(decision_destination),
    "config": _entry(config_destination),
    "card_catalog": _entry(catalog_destination),
    "native_binding": _entry(binding_destination),
    "promotion_bootstrap_panel": _entry(panel_destination),
    "production_anchor": _entry(anchor_destination),
    "required_promotion_policy_ids": sorted(required_promotion_ids),
    "required_league_policy_ids": sorted(required_ids),
    "checkpoint_dependencies": dependency_entries,
    "source_files": source_entries,
  }
  _write_json(output, payload)

  for key in (
    "model",
    "trainer",
    "metadata",
    "league",
    "promotion",
    "source_decision",
    "config",
    "card_catalog",
    "native_binding",
    "promotion_bootstrap_panel",
    "production_anchor",
  ):
    _verify_entry(payload[key])
  for entry in dependency_entries:
    _verify_entry(entry)
  for entry in source_entries:
    _verify_entry(entry)

  print(
    f"[parent-bundle] epoch={epoch} completed_episodes={completed} "
    f"league_dependencies={len(league_dependencies)} "
    f"promotion_dependencies={len(promotion_dependencies)} output={output}"
  )
  return output


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, required=True)
  parser.add_argument("--trainer", type=Path, required=True)
  parser.add_argument("--league", type=Path, required=True)
  parser.add_argument("--promotion", type=Path, required=True)
  parser.add_argument("--source-decision", type=Path, required=True)
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--card-catalog", type=Path, required=True)
  parser.add_argument("--binding", type=Path, required=True)
  parser.add_argument("--bootstrap-panel", type=Path, required=True)
  parser.add_argument("--production-anchor", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  export_bundle(args)


if __name__ == "__main__":
  main()
