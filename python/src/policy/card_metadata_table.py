from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch


ARTIFACT_VERSION = "policy_card_metadata_v1"


@dataclass(frozen=True)
class PolicyCardMetadataTable:
    artifact_version: str
    table_len: int
    real_card_count: int
    embedding_dim: int
    keyword_vocab: tuple[str, ...]
    subtype_vocab: tuple[str, ...]
    card_present_mask: torch.Tensor
    card_type_ids: torch.Tensor
    element_ids: torch.Tensor
    ikz_cost: torch.Tensor
    attack: torch.Tensor
    health: torch.Tensor
    gate_points: torch.Tensor
    has_ability: torch.Tensor
    ability_timing_ids: torch.Tensor
    ability_is_optional: torch.Tensor
    keyword_multi_hot: torch.Tensor
    name_embeddings: torch.Tensor
    effect_embeddings: torch.Tensor
    subtype_pooled_embeddings: torch.Tensor
    subtype_embeddings: torch.Tensor

    @property
    def vocab_size(self) -> int:
        return self.table_len

    @property
    def keyword_vocab_size(self) -> int:
        return len(self.keyword_vocab)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_manifest_path() -> Path:
    return _repo_root() / "python" / "config" / "policy_card_metadata_v1.json"


def _default_array_path() -> Path:
    return _repo_root() / "python" / "config" / "policy_card_metadata_v1.npz"


def _require_array(bundle: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    if name not in bundle:
        raise KeyError(f"Missing '{name}' in metadata artifact array bundle")
    return bundle[name]


def _to_tensor(
    array: np.ndarray,
    *,
    device: torch.device | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.as_tensor(array, device=device, dtype=dtype)


def load_policy_card_metadata_table(
    *,
    manifest_path: Path | None = None,
    array_path: Path | None = None,
    device: torch.device | None = None,
) -> PolicyCardMetadataTable:
    manifest_path = manifest_path or _default_manifest_path()
    array_path = array_path or _default_array_path()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_version = str(manifest.get("artifact_version", ""))
    if artifact_version != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported metadata artifact version '{artifact_version}' in {manifest_path}"
        )

    with np.load(array_path, allow_pickle=False) as bundle:
        table_len = int(manifest["table_len"])
        real_card_count = int(manifest["real_card_count"])
        embedding_dim = int(manifest["embedding_dim"])
        keyword_vocab = tuple(str(value) for value in manifest.get("keyword_vocab", []))
        subtype_vocab = tuple(str(value) for value in manifest.get("subtype_vocab", []))

        return PolicyCardMetadataTable(
            artifact_version=artifact_version,
            table_len=table_len,
            real_card_count=real_card_count,
            embedding_dim=embedding_dim,
            keyword_vocab=keyword_vocab,
            subtype_vocab=subtype_vocab,
            card_present_mask=_to_tensor(
                _require_array(bundle, "card_present_mask"),
                device=device,
                dtype=torch.float32,
            ),
            card_type_ids=_to_tensor(
                _require_array(bundle, "card_type_ids"),
                device=device,
                dtype=torch.long,
            ),
            element_ids=_to_tensor(
                _require_array(bundle, "element_ids"),
                device=device,
                dtype=torch.long,
            ),
            ikz_cost=_to_tensor(
                _require_array(bundle, "ikz_cost"),
                device=device,
                dtype=torch.float32,
            ),
            attack=_to_tensor(
                _require_array(bundle, "attack"),
                device=device,
                dtype=torch.float32,
            ),
            health=_to_tensor(
                _require_array(bundle, "health"),
                device=device,
                dtype=torch.float32,
            ),
            gate_points=_to_tensor(
                _require_array(bundle, "gate_points"),
                device=device,
                dtype=torch.float32,
            ),
            has_ability=_to_tensor(
                _require_array(bundle, "has_ability"),
                device=device,
                dtype=torch.float32,
            ),
            ability_timing_ids=_to_tensor(
                _require_array(bundle, "ability_timing_ids"),
                device=device,
                dtype=torch.long,
            ),
            ability_is_optional=_to_tensor(
                _require_array(bundle, "ability_is_optional"),
                device=device,
                dtype=torch.float32,
            ),
            keyword_multi_hot=_to_tensor(
                _require_array(bundle, "keyword_multi_hot"),
                device=device,
                dtype=torch.float32,
            ),
            name_embeddings=_to_tensor(
                _require_array(bundle, "name_embeddings"),
                device=device,
                dtype=torch.float32,
            ),
            effect_embeddings=_to_tensor(
                _require_array(bundle, "effect_embeddings"),
                device=device,
                dtype=torch.float32,
            ),
            subtype_pooled_embeddings=_to_tensor(
                _require_array(bundle, "subtype_pooled_embeddings"),
                device=device,
                dtype=torch.float32,
            ),
            subtype_embeddings=_to_tensor(
                _require_array(bundle, "subtype_embeddings"),
                device=device,
                dtype=torch.float32,
            ),
        )
