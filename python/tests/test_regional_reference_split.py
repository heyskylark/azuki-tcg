from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = REPO_ROOT / ".codex" / "docs" / "azuki_garden_arena_2026-08-15_decks.json"
TRAINING_CORPUS = (
    REPO_ROOT / ".codex" / "docs" / "azuki_garden_arena_2026-08-15_training_decks.json"
)
SPLIT_MANIFEST = (
    REPO_ROOT / ".codex" / "docs" / "azuki_garden_arena_2026-08-15_reference_split.json"
)


def test_regional_reference_split_is_reproducible_and_signature_disjoint(
    tmp_path: Path,
) -> None:
    generated_corpus = tmp_path / "training.json"
    generated_manifest = tmp_path / "split.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_regional_reference_split.py"),
            "--corpus",
            str(CORPUS),
            "--training-corpus",
            str(generated_corpus),
            "--manifest",
            str(generated_manifest),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    assert generated_corpus.read_bytes() == TRAINING_CORPUS.read_bytes()
    generated = json.loads(generated_manifest.read_text(encoding="utf-8"))
    committed = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    generated["training"]["corpus_path"] = committed["training"]["corpus_path"]
    assert generated == committed

    training_hashes = set(committed["training"]["content_sha256"])
    holdout_hashes = set(committed["holdout"]["content_sha256"])
    assert training_hashes.isdisjoint(holdout_hashes)
    assert committed["training"]["rows"] == 222
    assert committed["holdout"]["rows"] == 15
    assert committed["training"]["unique_signatures"] == 198
    assert committed["holdout"]["unique_signatures"] == 9
