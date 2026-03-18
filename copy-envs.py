#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

SKIP_DIRS = {
    ".git",
    ".next",
    ".turbo",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "venv",
}


def print_usage() -> None:
    script_name = Path(sys.argv[0]).name
    print(
        f"Usage: ./{script_name} <path-to-source-root> [path-to-destination-root]",
        file=sys.stderr,
    )


def collect_env_files(source_root: Path) -> list[Path]:
    env_files: list[Path] = []
    for root, dirnames, filenames in os.walk(source_root):
        dirnames[:] = [dirname for dirname in dirnames if dirname not in SKIP_DIRS]
        if ".env.local" not in filenames:
            continue
        env_files.append(Path(root) / ".env.local")
    return env_files


def main() -> int:
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print_usage()
        return 1

    script_root = Path(__file__).resolve().parent
    source_root = Path(sys.argv[1]).expanduser().resolve()
    destination_root = (
        Path(sys.argv[2]).expanduser().resolve()
        if len(sys.argv) == 3
        else script_root
    )

    if not source_root.is_dir():
        print(f"Source root does not exist or is not a directory: {source_root}", file=sys.stderr)
        return 1

    if not destination_root.exists():
        print(
            f"Destination root does not exist or is not a directory: {destination_root}",
            file=sys.stderr,
        )
        return 1

    if not destination_root.is_dir():
        print(f"Destination root is not a directory: {destination_root}", file=sys.stderr)
        return 1

    env_files = collect_env_files(source_root)
    if not env_files:
        print(f"No .env.local files found under {source_root}", file=sys.stderr)
        return 1

    copied_count = 0
    skipped_count = 0

    for source_file in env_files:
        relative_path = source_file.relative_to(source_root)
        destination_file = destination_root / relative_path

        if source_file == destination_file:
            print(f"Skipped {relative_path} (source and destination are the same file)")
            skipped_count += 1
            continue

        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination_file)
        print(f"Copied {relative_path}")
        copied_count += 1

    print(
        f"Done. Copied {copied_count} .env.local file(s)"
        + (f", skipped {skipped_count}." if skipped_count else ".")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
