#!/usr/bin/env python3
"""
Generate Flecs-ready card definition C code from a JSONL source file.

Each line in the input file must be a JSON object describing a single card.
The generated C file contains a `CardDef` table, an enum covering every card,
and a lookup table that ties card IDs back to their generated definition
entries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


RARITY_ORDER = ["L", "G", "C", "UC", "R", "SR", "IKZ"]
ELEMENT_ORDER = ["NORMAL", "LIGHTNING", "WATER", "EARTH", "FIRE"]
TYPE_ORDER = ["LEADER", "GATE", "ENTITY", "WEAPON", "SPELL", "IKZ", "EXTRA_IKZ"]

RARITY_ENUM = {value: f"CARD_RARITY_{value}" for value in RARITY_ORDER}
ELEMENT_ENUM = {value: f"CARD_ELEMENT_{value}" for value in ELEMENT_ORDER}
TYPE_ENUM = {value: f"CARD_TYPE_{value}" for value in TYPE_ORDER}
BASE_REQUIRED_FIELDS = {"card_id", "name", "rarity", "element", "type"}

TYPE_RULES = {
    "LEADER": {
        "allowed": BASE_REQUIRED_FIELDS | {"attack", "health"},
        "required": {"health"},
    },
    "GATE": {
        "allowed": BASE_REQUIRED_FIELDS,
        "required": set(),
    },
    "ENTITY": {
        "allowed": BASE_REQUIRED_FIELDS | {"attack", "health", "ikz_cost", "gate_points"},
        "required": {"attack", "health", "ikz_cost"},
    },
    "WEAPON": {
        "allowed": BASE_REQUIRED_FIELDS | {"attack", "ikz_cost"},
        "required": {"attack", "ikz_cost"},
    },
    "SPELL": {
        "allowed": BASE_REQUIRED_FIELDS | {"ikz_cost"},
        "required": {"ikz_cost"},
    },
    "IKZ": {
        "allowed": BASE_REQUIRED_FIELDS,
        "required": set(),
    },
    "EXTRA_IKZ": {
        "allowed": BASE_REQUIRED_FIELDS,
        "required": set(),
    },
}

CARD_PREFAB_COMPONENT_STRUCTS = [
    ("BaseStats", "int8_t attack, health;"),
    ("CurStats", "int8_t cur_atk, cur_hp;"),
    ("TapState", "uint8_t tapped, cooldown;"),
    ("Element", "uint8_t element;"),
    ("GatePoints", "uint8_t gate_points;"),
    ("IKZCost", "int8_t ikz_cost;"),
]

CARD_PREFAB_COMPONENT_DECLARATIONS = [
    "CardId",
    "Name",
    "Type",
    "BaseStats",
    "CurStats",
    "TapState",
    "Element",
    "GatePoints",
    "IKZCost",
]

CARD_PREFAB_TAG_DECLARATIONS = [
    "TLeader",
    "TGate",
    "TEntity",
    "TWeapon",
    "TSpell",
    "TIKZ",
    "TExtraIKZCard",
]

CARD_PREFAB_ON_INSTANTIATE_COMPONENTS = [
    "CardId",
    "Name",
    "Element",
    "Type",
    "TapState",
    "BaseStats",
    "GatePoints",
    "IKZCost",
]


class CardValidationError(Exception):
    """Wrap validation problems with line numbers for easier debugging."""

    def __init__(self, lineno: int, message: str) -> None:
        super().__init__(f"Line {lineno}: {message}")


@dataclass(frozen=True)
class CardRecord:
    card_id: str
    name: str
    rarity: str
    rarity_enum: str
    element: str
    element_enum: str
    card_type: str
    type_enum: str
    enum_name: str
    attack: Optional[int]
    health: Optional[int]
    gate_points: Optional[int]
    ikz_cost: Optional[int]

    @property
    def has_base_stats(self) -> bool:
        return self.attack is not None or self.health is not None

    @property
    def base_attack(self) -> int:
        return self.attack if self.attack is not None else 0

    @property
    def base_health(self) -> int:
        return self.health if self.health is not None else 0

    @property
    def has_gate_points(self) -> bool:
        return self.gate_points is not None

    @property
    def gate_points_value(self) -> int:
        return self.gate_points if self.gate_points is not None else 0

    @property
    def has_ikz_cost(self) -> bool:
        return self.ikz_cost is not None

    @property
    def ikz_cost_value(self) -> int:
        return self.ikz_cost if self.ikz_cost is not None else 0

    @property
    def prefab_tag(self) -> str:
        mapping = {
            "IKZ": "TIKZ",
            "EXTRA_IKZ": "TExtraIKZCard",
            "ENTITY": "TEntity",
            "GATE": "TGate",
            "LEADER": "TLeader",
            "WEAPON": "TWeapon",
            "SPELL": "TSpell",
        }
        try:
            return mapping[self.card_type]
        except KeyError as exc:  # pragma: no cover - validated earlier
            raise ValueError(f"Unsupported card type '{self.card_type}' for prefab tag") from exc

    @property
    def prefab_entity_name(self) -> str:
        return f"CardPrefab::{self.card_id}"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Flecs card definition C code from a JSONL file."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the JSONL file containing card definitions.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("src/generated/card_defs.c"),
        help="Destination path for the generated C source file.",
    )
    parser.add_argument(
        "--header",
        type=Path,
        default=Path("include/generated/card_defs.h"),
        help="Destination path for the generated C header file.",
    )
    return parser.parse_args(argv)


def load_cards(path: Path) -> List[CardRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    records: List[CardRecord] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CardValidationError(lineno, f"Invalid JSON: {exc.msg}") from exc

            record = validate_card_payload(payload, lineno)

            if record.card_id in seen_ids:
                raise CardValidationError(lineno, f"Duplicate card_id '{record.card_id}'")
            seen_ids.add(record.card_id)
            records.append(record)

    if not records:
        raise ValueError(f"No card definitions found in {path}")

    return records


def validate_card_payload(payload: object, lineno: int) -> CardRecord:
    if not isinstance(payload, dict):
        raise CardValidationError(lineno, "Card entry must be a JSON object.")

    missing = [field for field in BASE_REQUIRED_FIELDS if field not in payload]
    if missing:
        raise CardValidationError(
            lineno, f"Missing required field(s): {', '.join(sorted(missing))}"
        )

    card_type = expect_string(payload["type"], "type", lineno).upper()
    if card_type not in TYPE_RULES:
        allowed = ", ".join(TYPE_RULES)
        raise CardValidationError(
            lineno, f"Unknown type '{card_type}'. Expected one of: {allowed}."
        )

    rules = TYPE_RULES[card_type]
    extras = sorted(set(payload.keys()) - rules["allowed"])
    if extras:
        raise CardValidationError(
            lineno,
            f"Unexpected field(s) for type {card_type}: {', '.join(extras)}",
        )

    missing_type_specific = [field for field in rules["required"] if payload.get(field) is None]
    if missing_type_specific:
        raise CardValidationError(
            lineno,
            f"Type {card_type} missing required field(s): {', '.join(sorted(missing_type_specific))}",
        )

    card_id = expect_non_empty_string(payload["card_id"], "card_id", lineno)
    name = expect_non_empty_string(payload["name"], "name", lineno)

    rarity_raw = expect_string(payload["rarity"], "rarity", lineno).upper()
    if rarity_raw not in RARITY_ENUM:
        allowed = ", ".join(RARITY_ORDER)
        raise CardValidationError(
            lineno, f"Unknown rarity '{rarity_raw}'. Expected one of: {allowed}."
        )

    element_raw = expect_string(payload["element"], "element", lineno).upper()
    if element_raw not in ELEMENT_ENUM:
        allowed = ", ".join(ELEMENT_ORDER)
        raise CardValidationError(
            lineno, f"Unknown element '{element_raw}'. Expected one of: {allowed}."
        )

    attack = parse_optional_int(payload, "attack", lineno, -128, 127)
    health = parse_optional_int(payload, "health", lineno, -128, 127)
    gate_points = parse_optional_int(payload, "gate_points", lineno, 0, 255)
    ikz_cost = parse_optional_int(payload, "ikz_cost", lineno, 0, 127)

    if card_type != "ENTITY" and gate_points is not None:
        raise CardValidationError(
            lineno, f"gate_points is only permitted for ENTITY cards."
        )

    if card_type == "LEADER" and attack is None and health is None:
        # Leader is allowed to omit attack, but health is required as per rules.
        raise CardValidationError(
            lineno, "Leader cards must include a health value."
        )

    if card_type == "ENTITY" and gate_points is None:
        # Not strictly required by spec, but helpful for future-proofing.
        pass

    return CardRecord(
        card_id=card_id,
        name=name,
        rarity=rarity_raw,
        rarity_enum=RARITY_ENUM[rarity_raw],
        element=element_raw,
        element_enum=ELEMENT_ENUM[element_raw],
        card_type=card_type,
        type_enum=TYPE_ENUM[card_type],
        enum_name=make_enum_name(card_id),
        attack=attack,
        health=health,
        gate_points=gate_points,
        ikz_cost=ikz_cost,
    )


def expect_string(value: object, field: str, lineno: int) -> str:
    if not isinstance(value, str):
        raise CardValidationError(lineno, f"Field '{field}' must be a string.")
    return value


def expect_non_empty_string(value: object, field: str, lineno: int) -> str:
    text = expect_string(value, field, lineno).strip()
    if not text:
        raise CardValidationError(lineno, f"Field '{field}' cannot be empty.")
    return text


def parse_optional_int(
    payload: dict,
    field: str,
    lineno: int,
    minimum: Optional[int],
    maximum: Optional[int],
) -> Optional[int]:
    if field not in payload or payload[field] is None:
        return None
    value = payload[field]
    return coerce_int(value, field, lineno, minimum, maximum)


def coerce_int(
    value: object,
    field: str,
    lineno: int,
    minimum: Optional[int],
    maximum: Optional[int],
) -> int:
    if isinstance(value, bool):
        raise CardValidationError(lineno, f"Field '{field}' must be an integer.")

    if isinstance(value, int):
        result = value
    elif isinstance(value, float) and value.is_integer():
        result = int(value)
    else:
        raise CardValidationError(lineno, f"Field '{field}' must be an integer.")

    if minimum is not None and result < minimum:
        raise CardValidationError(
            lineno, f"Field '{field}' must be >= {minimum} (got {result})."
        )
    if maximum is not None and result > maximum:
        raise CardValidationError(
            lineno, f"Field '{field}' must be <= {maximum} (got {result})."
        )
    return result


def make_enum_name(card_id: str) -> str:
    cleaned = []
    for ch in card_id:
        if ch.isalnum():
            cleaned.append(ch.upper())
        else:
            cleaned.append("_")
    return "CARD_DEF_" + "".join(cleaned)


def render_enum(name: str, order: Sequence[str], mapping: dict[str, str]) -> List[str]:
    lines = [f"typedef enum {{"]
    for idx, value in enumerate(order):
        enum_name = mapping[value]
        suffix = "," if idx < len(order) - 1 else ""
        lines.append(f"    {enum_name} = {idx}{suffix}")
    lines.append(f"}} {name};")
    return lines


def render_card_id_enum(records: Sequence[CardRecord]) -> List[str]:
    lines = ["typedef enum {"]
    for idx, record in enumerate(records):
        lines.append(f"    {record.enum_name} = {idx},")
    lines.append(f"    CARD_DEF_COUNT = {len(records)}")
    lines.append("} CardDefId;")
    return lines


def render_card_def_entry(record: CardRecord) -> Iterable[str]:
    indent_outer = "    "
    indent_inner = "        "
    lines = [f"{indent_outer}{{"]
    lines.append(f"{indent_inner}.card_id = {c_string_literal(record.card_id)},")
    lines.append(f"{indent_inner}.name = {c_string_literal(record.name)},")
    lines.append(f"{indent_inner}.rarity = {record.rarity_enum},")
    lines.append(f"{indent_inner}.element = {record.element_enum},")
    lines.append(f"{indent_inner}.type = {record.type_enum},")
    lines.append(f"{indent_inner}.has_base_stats = {c_bool(record.has_base_stats)},")
    lines.append(
        f"{indent_inner}.base_stats = {{ .attack = {record.base_attack}, .health = {record.base_health} }},"
    )
    lines.append(f"{indent_inner}.has_gate_points = {c_bool(record.has_gate_points)},")
    lines.append(
        f"{indent_inner}.gate_points = {{ .gate_points = {record.gate_points_value} }},"
    )
    lines.append(f"{indent_inner}.has_ikz_cost = {c_bool(record.has_ikz_cost)},")
    lines.append(
        f"{indent_inner}.ikz_cost = {{ .ikz_cost = {record.ikz_cost_value} }},"
    )
    lines.append(f"{indent_outer}}},")
    return lines


def render_card_component_structs() -> List[str]:
    lines: List[str] = []
    for name, fields in CARD_PREFAB_COMPONENT_STRUCTS:
        lines.append("typedef struct {")
        lines.append(f"    {fields}")
        lines.append(f"}} {name};")
        lines.append("")
    if lines:
        lines.pop()  # drop trailing blank
    return lines


def render_lookup_table(records: Sequence[CardRecord]) -> List[str]:
    lines: List[str] = []
    lines.append("static const CardDefLookupEntry kGeneratedCardLookup[CARD_DEF_COUNT] = {")
    for record in records:
        lines.append(
            f"    {{ .card_id = {c_string_literal(record.card_id)}, .def = &kGeneratedCardDefs[{record.enum_name}] }},"
        )
    lines.append("};")
    return lines


def render_prefab_component_sets(record: CardRecord, indent: str) -> List[str]:
    lines: List[str] = []
    lines.append(
        f"{indent}ecs_set(world, prefab, CardId, {{ .id = {record.enum_name}, .code = {c_string_literal(record.card_id)} }});"
    )
    lines.append(
        f"{indent}ecs_set(world, prefab, Name, {{ .value = {c_string_literal(record.name)} }});"
    )
    lines.append(
        f"{indent}ecs_set(world, prefab, Element, {{ .element = {record.element_enum} }});"
    )
    lines.append(
        f"{indent}ecs_set(world, prefab, Type, {{ .value = {record.type_enum} }});"
    )
    lines.append(
        f"{indent}ecs_set(world, prefab, TapState, {{ .tapped = 0, .cooldown = 0 }});"
    )

    if record.card_type in {"ENTITY", "LEADER", "WEAPON"}:
        lines.append(
            f"{indent}ecs_set(world, prefab, BaseStats, {{ .attack = {record.base_attack}, .health = {record.base_health} }});"
        )

    if record.card_type in {"ENTITY", "LEADER", "WEAPON"}:
        lines.append(
            f"{indent}ecs_set(world, prefab, CurStats, {{ .cur_atk = {record.base_attack}, .cur_hp = {record.base_health} }});"
        )

    if record.card_type == "ENTITY":
        lines.append(
            f"{indent}ecs_set(world, prefab, GatePoints, {{ .gate_points = {record.gate_points_value} }});"
        )

    if record.card_type in {"ENTITY", "WEAPON", "SPELL"}:
        lines.append(
            f"{indent}ecs_set(world, prefab, IKZCost, {{ .ikz_cost = {record.ikz_cost_value} }});"
        )

    return lines


def render_prefab_block(record: CardRecord) -> List[str]:
    add_ids = ", ".join(["EcsPrefab", record.prefab_tag])
    lines: List[str] = []
    lines.append("    {")
    lines.append("        ecs_entity_desc_t desc = {")
    lines.append(f"            .name = {c_string_literal(record.prefab_entity_name)},")
    lines.append(f"            .add = (ecs_id_t[]){{ {add_ids}, 0 }}")
    lines.append("        };")
    lines.append("        ecs_entity_t prefab = ecs_entity_init(world, &desc);")
    lines.append(
        f"        ecs_assert(prefab != 0, ECS_INVALID_PARAMETER, \"failed to create prefab for card {record.card_id}\");"
    )
    lines.append(f"        kGeneratedPrefabs[{record.enum_name}] = prefab;")
    lines.extend(render_prefab_component_sets(record, "        "))
    lines.append("    }")
    return lines


def render_card_struct_definition() -> List[str]:
    lines = ["typedef struct {"]
    lines.append("    const char *card_id;")
    lines.append("    const char *name;")
    lines.append("    CardRarity rarity;")
    lines.append("    CardElement element;")
    lines.append("    CardType type;")
    lines.append("    bool has_base_stats;")
    lines.append("    BaseStats base_stats;")
    lines.append("    bool has_gate_points;")
    lines.append("    GatePoints gate_points;")
    lines.append("    bool has_ikz_cost;")
    lines.append("    IKZCost ikz_cost;")
    lines.append("} CardDef;")
    return lines


def render_prefab_component_structs() -> List[str]:
    lines = ["typedef struct {"]
    lines.append("    CardDefId id;")
    lines.append("    const char *code;")
    lines.append("} CardId;")
    lines.append("")
    lines.append("typedef struct {")
    lines.append("    const char *value;")
    lines.append("} Name;")
    lines.append("")
    lines.append("typedef struct {")
    lines.append("    CardType value;")
    lines.append("} Type;")
    return lines


def render_lookup_struct_definition() -> List[str]:
    lines = ["typedef struct {"]
    lines.append("    const char *card_id;")
    lines.append("    const CardDef *def;")
    lines.append("} CardDefLookupEntry;")
    return lines


def render_component_and_tag_declarations() -> List[str]:
    lines: List[str] = []
    for component in CARD_PREFAB_COMPONENT_DECLARATIONS:
        lines.append(f"extern ECS_COMPONENT_DECLARE({component});")
    lines.append("")
    for tag in CARD_PREFAB_TAG_DECLARATIONS:
        lines.append(f"extern ECS_TAG_DECLARE({tag});")
    return lines


def render_header(records: Sequence[CardRecord], include_guard: str) -> str:
    lines: List[str] = []
    lines.append("// This file is auto-generated by scripts/generate_card_defs.py. Do not edit manually.")
    lines.append(f"#ifndef {include_guard}")
    lines.append(f"#define {include_guard}")
    lines.append("")
    lines.append("#include <stdbool.h>")
    lines.append("#include <stddef.h>")
    lines.append("#include <stdint.h>")
    lines.append("#include <flecs.h>")
    lines.append("")

    lines.extend(render_enum("CardRarity", RARITY_ORDER, RARITY_ENUM))
    lines.append("")
    lines.extend(render_enum("CardElement", ELEMENT_ORDER, ELEMENT_ENUM))
    lines.append("")
    lines.extend(render_enum("CardType", TYPE_ORDER, TYPE_ENUM))
    lines.append("")
    lines.extend(render_card_component_structs())
    if CARD_PREFAB_COMPONENT_STRUCTS:
        lines.append("")
    lines.extend(render_card_struct_definition())
    lines.append("")
    lines.extend(render_card_id_enum(records))
    lines.append("")
    lines.extend(render_prefab_component_structs())
    lines.append("")
    lines.extend(render_lookup_struct_definition())
    lines.append("")
    lines.extend(render_component_and_tag_declarations())
    lines.append("")
    lines.append("#ifdef __cplusplus")
    lines.append("extern \"C\" {")
    lines.append("#endif")
    lines.append("")
    lines.append("void azk_register_card_def_resources(ecs_world_t *world);")
    lines.append("size_t azk_card_def_count(void);")
    lines.append("const CardDef *azk_card_def_from_id(CardDefId id);")
    lines.append("const CardDefLookupEntry *azk_card_def_lookup_table(size_t *out_count);")
    lines.append("ecs_entity_t azk_prefab_from_id(CardDefId id);")
    lines.append("")
    lines.append("#ifdef __cplusplus")
    lines.append("}")
    lines.append("#endif")
    lines.append("")
    lines.append(f"#endif /* {include_guard} */")
    lines.append("")
    return "\n".join(lines)


def render_c_file(records: Sequence[CardRecord], header_include: str) -> str:
    lines: List[str] = []
    lines.append("// This file is auto-generated by scripts/generate_card_defs.py. Do not edit manually.")
    lines.append(f"#include \"{header_include}\"")
    lines.append("")
    for component in CARD_PREFAB_COMPONENT_DECLARATIONS:
        lines.append(f"ECS_COMPONENT_DECLARE({component});")
    lines.append("")
    for tag in CARD_PREFAB_TAG_DECLARATIONS:
        lines.append(f"ECS_TAG_DECLARE({tag});")
    lines.append("ECS_TAG_DECLARE(CardDefComponentsRegisteredTag);")
    lines.append("ECS_TAG_DECLARE(CardDefPrefabsRegisteredTag);")
    lines.append("")
    lines.append("static const CardDef kGeneratedCardDefs[CARD_DEF_COUNT] = {")
    for record in records:
        lines.extend(render_card_def_entry(record))
    lines.append("};")
    lines.append("")
    lines.append("static ecs_entity_t kGeneratedPrefabs[CARD_DEF_COUNT];")
    lines.append("")
    lines.append("size_t azk_card_def_count(void) {")
    lines.append("    return CARD_DEF_COUNT;")
    lines.append("}")
    lines.append("")
    lines.append("const CardDef *azk_card_def_from_id(CardDefId id) {")
    lines.append("    if ((size_t)id >= CARD_DEF_COUNT) {")
    lines.append("        return NULL;")
    lines.append("    }")
    lines.append("    return &kGeneratedCardDefs[id];")
    lines.append("}")
    lines.append("")
    lines.extend(render_lookup_table(records))
    lines.append("")
    lines.append("const CardDefLookupEntry *azk_card_def_lookup_table(size_t *out_count) {")
    lines.append("    if (out_count) {")
    lines.append("        *out_count = CARD_DEF_COUNT;")
    lines.append("    }")
    lines.append("    return kGeneratedCardLookup;")
    lines.append("}")
    lines.append("")
    lines.append("void azk_register_card_def_resources(ecs_world_t *world) {")
    lines.append("    ECS_TAG_DEFINE(world, CardDefComponentsRegisteredTag);")
    lines.append("    ECS_TAG_DEFINE(world, CardDefPrefabsRegisteredTag);")
    lines.append("")
    lines.append("    if (!ecs_has_id(world, ecs_id(CardDefComponentsRegisteredTag), ecs_id(CardDefComponentsRegisteredTag))) {")
    lines.append("        ecs_singleton_add(world, CardDefComponentsRegisteredTag);")
    for component in CARD_PREFAB_COMPONENT_DECLARATIONS:
        lines.append(f"        ECS_COMPONENT_DEFINE(world, {component});")
    if CARD_PREFAB_COMPONENT_DECLARATIONS:
        lines.append("")
    for tag in CARD_PREFAB_TAG_DECLARATIONS:
        lines.append(f"        ECS_TAG_DEFINE(world, {tag});")
    for component in CARD_PREFAB_ON_INSTANTIATE_COMPONENTS:
        lines.append(
            f"        ecs_add_pair(world, ecs_id({component}), EcsOnInstantiate, EcsInherit);"
        )
    if CARD_PREFAB_ON_INSTANTIATE_COMPONENTS:
        lines.append("")
    lines.append("    }")
    lines.append("")
    lines.append("    if (ecs_has_id(world, ecs_id(CardDefPrefabsRegisteredTag), ecs_id(CardDefPrefabsRegisteredTag))) {")
    lines.append("        return;")
    lines.append("    }")
    lines.append("    ecs_singleton_add(world, CardDefPrefabsRegisteredTag);")
    lines.append("")
    for record in records:
        lines.extend(render_prefab_block(record))
    lines.append("}")
    lines.append("")
    lines.append("ecs_entity_t azk_prefab_from_id(CardDefId id) {")
    lines.append("    if ((size_t)id >= CARD_DEF_COUNT) {")
    lines.append("        return 0;")
    lines.append("    }")
    lines.append("    return kGeneratedPrefabs[id];")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def c_string_literal(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def c_bool(value: bool) -> str:
    return "true" if value else "false"


def ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def make_include_guard(path: Path) -> str:
    text = path.as_posix()
    guard_chars = []
    for ch in text:
        if ch.isalnum():
            guard_chars.append(ch.upper())
        else:
            guard_chars.append("_")
    guard = "".join(guard_chars) or "CARD_DEFS_H"
    if guard[0].isdigit():
        guard = "_" + guard
    return guard


def compute_header_include(c_path: Path, header_path: Path) -> str:
    rel = os.path.relpath(header_path, start=c_path.parent)
    return Path(rel).as_posix()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        records = load_cards(args.input)
    except Exception as exc:  # noqa: BLE001 - surface useful error to CLI
        print(exc, file=sys.stderr)
        return 1

    header_guard = make_include_guard(args.header)
    header_content = render_header(records, header_guard)
    ensure_parent_directory(args.header)
    args.header.write_text(header_content, encoding="utf-8")

    header_include = compute_header_include(args.output, args.header)
    rendered = render_c_file(records, header_include)
    ensure_parent_directory(args.output)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        f"Wrote {len(records)} card definition(s) to {args.output} "
        f"and declarations to {args.header}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
