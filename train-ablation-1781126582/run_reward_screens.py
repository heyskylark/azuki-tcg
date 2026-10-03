from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRATION = ROOT / "train-ablation-1781126582/results/reward_screens/registration.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run registered reward-screen arms serially.")
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--arm", action="append", default=[], help="Run only the named arm; repeatable.")
    args = parser.parse_args()

    registration_path = args.registration.resolve()
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("schema_id") != "azuki.ablation_registration":
        raise ValueError("invalid reward-screen registration schema")
    status = str(registration.get("status", "registered"))
    if status not in {"registered", "ready", "planned"}:
        raise ValueError(f"registration is not launchable: {status}")
    if registration.get("production_qualified") is False:
        raise ValueError("registration explicitly fails production qualification")
    selected = set(args.arm)
    arms = [arm for arm in registration["arms"] if not selected or arm["name"] in selected]
    if selected and selected != {arm["name"] for arm in arms}:
        missing = sorted(selected - {arm["name"] for arm in arms})
        raise ValueError(f"unknown reward-screen arms: {missing}")

    source_paths = [
        ROOT / "python/src/tcg.h",
        ROOT / "python/src/binding.c",
        ROOT / "python/src/azk_native.py",
        ROOT / "python/src/azk_puffer/vector.py",
        ROOT / "python/src/training_utils.py",
        ROOT / "python/src/league_training.py",
        ROOT / "python/src/league_manager.py",
        ROOT / "python/src/league_state.py",
        ROOT / "python/src/league_archive.py",
        ROOT / "python/src/train.py",
    ]
    source_hashes = {str(path.relative_to(ROOT)): _sha256(path) for path in source_paths}
    env = os.environ.copy()
    python_paths = [str(ROOT / "build/python/src"), str(ROOT / "python/src")]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/azuki-pyc")

    for arm in arms:
        config = (ROOT / arm["config"]).resolve()
        actual_hash = _sha256(config)
        if actual_hash != arm["config_sha256"]:
            raise RuntimeError(
                f"config hash changed for {arm['name']}: {actual_hash} != {arm['config_sha256']}"
            )
        result_root = (ROOT / arm["result_root"]).resolve()
        status_path = result_root / "run_status.json"
        if status_path.exists():
            existing = json.loads(status_path.read_text(encoding="utf-8"))
            if existing.get("state") == "completed":
                print(f"[reward-screen] skipping completed arm {arm['name']}", flush=True)
                continue
            raise RuntimeError(f"refusing to overwrite incomplete arm status: {status_path}")

        started = time.time()
        status: dict[str, object] = {
            "schema_id": "azuki.ablation_run_status",
            "schema_version": 1,
            "family": registration["family"],
            "arm": arm["id"],
            "name": arm["name"],
            "state": "running",
            "started_at": started,
            "config": str(config.relative_to(ROOT)),
            "config_sha256": actual_hash,
            "source_sha256": source_hashes,
            "sampled_rows": registration["sampled_rows_per_arm"],
            "seed": registration["seed"],
        }
        _write_status(status_path, status)
        command = [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "python/src/train.py"),
            "--config",
            str(config),
        ]
        print(f"[reward-screen] starting {arm['id']} ({arm['name']})", flush=True)
        completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
        status["finished_at"] = time.time()
        status["wall_time_seconds"] = status["finished_at"] - started
        status["returncode"] = completed.returncode
        status["state"] = "completed" if completed.returncode == 0 else "failed"
        experiment_dirs = sorted(
            [path for path in ROOT.glob(f"experiments/azuki_local_{arm['tag']}_*") if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
        )
        if experiment_dirs:
            status["latest_experiment_dir"] = str(experiment_dirs[-1].relative_to(ROOT))
        _write_status(status_path, status)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        print(f"[reward-screen] completed {arm['id']} ({arm['name']})", flush=True)


if __name__ == "__main__":
    main()
