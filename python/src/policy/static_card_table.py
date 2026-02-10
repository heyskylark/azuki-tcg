from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

import torch


_CARD_TYPE_TO_ID = {
    "LEADER": 0,
    "GATE": 1,
    "ENTITY": 2,
    "WEAPON": 3,
    "SPELL": 4,
    "IKZ": 5,
    "EXTRA_IKZ": 6,
}

_TIMING_TAG_TO_ID = {
    "NONE": 0,
    "AOnPlay": 1,
    "AStartOfTurn": 2,
    "AEndOfTurn": 3,
    "AWhenEquipping": 4,
    "AWhenEquipped": 5,
    "AMain": 6,
    "AWhenAttacking": 7,
    "AWhenAttacked": 8,
    "AResponse": 9,
    "AWhenReturnedToHand": 10,
    "AOnGatePortal": 11,
}


@dataclass(frozen=True)
class PolicyStaticCardTable:
    card_count: int
    card_type: torch.Tensor
    base_ikz_cost: torch.Tensor
    base_attack: torch.Tensor
    base_health: torch.Tensor
    base_gate_points: torch.Tensor
    innate_has_charge: torch.Tensor
    innate_has_defender: torch.Tensor
    innate_has_infiltrate: torch.Tensor
    has_ability: torch.Tensor
    ability_timing: torch.Tensor
    ability_is_optional: torch.Tensor

    @property
    def vocab_size(self) -> int:
        # +1 slot 0 for null/empty card, real card IDs are shifted by +1.
        return self.card_count + 1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _enum_suffix_to_card_code(enum_suffix: str) -> str:
    if enum_suffix.startswith("IKZ_"):
        return enum_suffix.replace("_", "-")
    if re.fullmatch(r"STT\d{2}_\d{3}", enum_suffix):
        return enum_suffix.replace("_", "-", 1)
    raise ValueError(f"Unsupported CardDef enum suffix '{enum_suffix}'")


def _load_card_def_id_map(header_path: Path) -> dict[str, int]:
    text = header_path.read_text(encoding="utf-8")
    pattern = re.compile(r"CARD_DEF_([A-Z0-9_]+)\s*=\s*(\d+)")
    card_def_id_map: dict[str, int] = {}
    for enum_suffix, value in pattern.findall(text):
        if enum_suffix == "COUNT":
            continue
        card_code = _enum_suffix_to_card_code(enum_suffix)
        card_def_id_map[card_code] = int(value)
    if not card_def_id_map:
        raise ValueError(f"No CardDefId entries found in {header_path}")
    return card_def_id_map


def _load_card_defs_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    cards: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            card_code = payload["card_id"]
            cards[card_code] = payload
    if not cards:
        raise ValueError(f"No card records loaded from {path}")
    return cards


def _load_ability_metadata(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cards = payload.get("cards", {})
    if not isinstance(cards, dict):
        raise ValueError(f"Expected object at 'cards' in {path}")
    return cards


def _to_float_tensor(values: list[float], device: torch.device | None) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=device)


def _to_long_tensor(values: list[int], device: torch.device | None) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long, device=device)


def load_policy_static_card_table(
    *, device: torch.device | None = None
) -> PolicyStaticCardTable:
    root = _repo_root()
    card_defs_path = root / "scripts" / "azuki-card-defs.jsonl"
    card_defs_header = root / "include" / "generated" / "card_defs.h"
    ability_metadata_path = root / "python" / "config" / "ability_static_metadata.json"

    card_defs = _load_card_defs_jsonl(card_defs_path)
    card_def_id_map = _load_card_def_id_map(card_defs_header)
    ability_metadata = _load_ability_metadata(ability_metadata_path)

    card_count = max(card_def_id_map.values()) + 1

    # index 0 = null/empty card, real ids shifted by +1
    table_len = card_count + 1
    card_type = [0] * table_len
    base_ikz_cost = [0.0] * table_len
    base_attack = [0.0] * table_len
    base_health = [0.0] * table_len
    base_gate_points = [0.0] * table_len
    innate_has_charge = [0.0] * table_len
    innate_has_defender = [0.0] * table_len
    innate_has_infiltrate = [0.0] * table_len
    has_ability = [0.0] * table_len
    ability_timing = [0] * table_len
    ability_is_optional = [0.0] * table_len

    for card_code, card_def_id in card_def_id_map.items():
        card_record = card_defs.get(card_code)
        if card_record is None:
            raise ValueError(
                f"Card '{card_code}' found in generated header but missing in {card_defs_path}"
            )

        index = card_def_id + 1
        card_type_name = card_record["type"]
        if card_type_name not in _CARD_TYPE_TO_ID:
            raise ValueError(f"Unknown card type '{card_type_name}' for {card_code}")
        card_type[index] = _CARD_TYPE_TO_ID[card_type_name]
        base_ikz_cost[index] = float(card_record.get("ikz_cost", 0))
        base_attack[index] = float(card_record.get("attack", 0))
        base_health[index] = float(card_record.get("health", 0))
        base_gate_points[index] = float(card_record.get("gate_points", 0))

        keywords = set(card_record.get("keywords", []))
        innate_has_charge[index] = 1.0 if "charge" in keywords else 0.0
        innate_has_defender[index] = 1.0 if "defender" in keywords else 0.0
        innate_has_infiltrate[index] = 1.0 if "infiltrate" in keywords else 0.0

        ability_record = ability_metadata.get(card_code)
        if ability_record is not None:
            has_ability[index] = 1.0 if ability_record.get("has_ability", False) else 0.0
            ability_is_optional[index] = (
                1.0 if ability_record.get("is_optional", False) else 0.0
            )
            timing_tag = str(ability_record.get("timing_tag", "NONE"))
            ability_timing[index] = _TIMING_TAG_TO_ID.get(timing_tag, 0)

    return PolicyStaticCardTable(
        card_count=card_count,
        card_type=_to_long_tensor(card_type, device),
        base_ikz_cost=_to_float_tensor(base_ikz_cost, device),
        base_attack=_to_float_tensor(base_attack, device),
        base_health=_to_float_tensor(base_health, device),
        base_gate_points=_to_float_tensor(base_gate_points, device),
        innate_has_charge=_to_float_tensor(innate_has_charge, device),
        innate_has_defender=_to_float_tensor(innate_has_defender, device),
        innate_has_infiltrate=_to_float_tensor(innate_has_infiltrate, device),
        has_ability=_to_float_tensor(has_ability, device),
        ability_timing=_to_long_tensor(ability_timing, device),
        ability_is_optional=_to_float_tensor(ability_is_optional, device),
    )
