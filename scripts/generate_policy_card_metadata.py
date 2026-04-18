#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any
from urllib import error, request

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_BASENAME = "policy_card_metadata_v1"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "python" / "config" / f"{DEFAULT_ARTIFACT_BASENAME}.json"
DEFAULT_ARRAY_PATH = REPO_ROOT / "python" / "config" / f"{DEFAULT_ARTIFACT_BASENAME}.npz"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LOCAL_DATABASE_URL = "postgres://azuki:azuki@localhost:5432/azuki"
ARTIFACT_VERSION = "policy_card_metadata_v1"
EMBEDDING_BATCH_SIZE = 64
OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"

CARD_TYPE_TO_ID = {
    "LEADER": 0,
    "GATE": 1,
    "ENTITY": 2,
    "WEAPON": 3,
    "SPELL": 4,
    "IKZ": 5,
    "EXTRA_IKZ": 6,
}

ELEMENT_TO_ID = {
    "NORMAL": 0,
    "FIRE": 1,
    "EARTH": 2,
    "LIGHTNING": 3,
    "WATER": 4,
}

TIMING_TAG_TO_ID = {
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
class CardMetadataRecord:
    card_code: str
    name: str
    effect_text: str
    keywords: tuple[str, ...]
    subtypes: tuple[str, ...]
    card_type: str
    element: str
    ikz_cost: int
    attack: int
    health: int
    gate_points: int
    has_ability: bool
    ability_timing_id: int
    ability_is_optional: bool


def _enum_suffix_to_card_code(enum_suffix: str) -> str:
    if re.fullmatch(r"[A-Z0-9]+_\d{3}", enum_suffix):
        return enum_suffix.replace("_", "-", 1)
    raise ValueError(f"Unsupported CardDef enum suffix '{enum_suffix}'")


def _load_card_def_id_map(header_path: Path) -> dict[str, int]:
    text = header_path.read_text(encoding="utf-8")
    pattern = re.compile(r"CARD_DEF_([A-Z0-9_]+)\s*=\s*(\d+)")
    card_def_id_map: dict[str, int] = {}
    for enum_suffix, value in pattern.findall(text):
        if enum_suffix == "COUNT":
            continue
        card_def_id_map[_enum_suffix_to_card_code(enum_suffix)] = int(value)
    if not card_def_id_map:
        raise ValueError(f"No CardDefId entries found in {header_path}")
    return card_def_id_map


def _load_ability_metadata(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cards = payload.get("cards", {})
    if not isinstance(cards, dict):
        raise ValueError(f"Expected object at 'cards' in {path}")
    return cards


def _load_generated_card_defs(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            records[str(payload["card_id"]).upper()] = payload
    if not records:
        raise ValueError(f"No generated card defs loaded from {path}")
    return records


def _default_database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_LOCAL_DATABASE_URL)


def _run_psql_json_query(database_url: str, sql: str) -> list[dict[str, Any]]:
    cmd = ["psql", database_url, "-X", "-v", "ON_ERROR_STOP=1", "-Atqc", sql]
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("psql is required to generate the metadata artifact") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        raise RuntimeError(
            "Failed to query card metadata from Postgres. "
            "Start the local DB with `bun run dev:infra` or pass --database-url."
            + (f" psql stderr: {stderr}" if stderr else "")
        ) from exc

    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def _load_card_records_from_postgres(database_url: str) -> dict[str, dict[str, Any]]:
    sql = """
    WITH ranked_cards AS (
      SELECT
        UPPER(card_code) AS card_code,
        name,
        COALESCE(effect_text, '') AS effect_text,
        COALESCE(keywords, ARRAY[]::text[]) AS keywords,
        COALESCE(subtypes, ARRAY[]::text[]) AS subtypes,
        card_type::text AS card_type,
        element::text AS element,
        COALESCE(ikz_cost, 0) AS ikz_cost,
        COALESCE(attack, 0) AS attack,
        COALESCE(health, 0) AS health,
        COALESCE(gate_points, 0) AS gate_points,
        ROW_NUMBER() OVER (
          PARTITION BY UPPER(card_code)
          ORDER BY (special_rarity IS NOT NULL), rarity::text
        ) AS row_rank
      FROM cards
    )
    SELECT json_build_object(
      'card_code', card_code,
      'name', name,
      'effect_text', effect_text,
      'keywords', keywords,
      'subtypes', subtypes,
      'card_type', card_type,
      'element', element,
      'ikz_cost', ikz_cost,
      'attack', attack,
      'health', health,
      'gate_points', gate_points
    )::text
    FROM ranked_cards
    WHERE row_rank = 1
    ORDER BY card_code;
    """
    rows = _run_psql_json_query(database_url, sql)
    if not rows:
        raise RuntimeError("No card metadata rows were returned from Postgres")
    return {str(row["card_code"]).upper(): row for row in rows}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_text_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _build_card_metadata_records(
    card_def_id_map: dict[str, int],
    postgres_records: dict[str, dict[str, Any]],
    ability_metadata: dict[str, dict[str, Any]],
    generated_card_defs: dict[str, dict[str, Any]],
) -> dict[str, CardMetadataRecord]:
    records: dict[str, CardMetadataRecord] = {}
    missing_codes: list[str] = []
    for card_code in sorted(card_def_id_map):
        postgres_record = postgres_records.get(card_code)
        if postgres_record is None:
            missing_codes.append(card_code)
            continue

        card_type = str(postgres_record["card_type"]).upper()
        element = str(postgres_record["element"]).upper()
        if card_type not in CARD_TYPE_TO_ID:
            raise ValueError(f"Unknown card_type '{card_type}' for {card_code}")
        if element not in ELEMENT_TO_ID:
            raise ValueError(f"Unknown element '{element}' for {card_code}")

        generated_card_def = generated_card_defs.get(card_code)
        if generated_card_def is None:
            raise RuntimeError(
                f"Generated card defs are missing card code '{card_code}'"
            )

        ability_record = ability_metadata.get(card_code, {})
        timing_tag = str(ability_record.get("timing_tag", "NONE"))
        if timing_tag not in TIMING_TAG_TO_ID:
            raise ValueError(f"Unknown ability timing tag '{timing_tag}' for {card_code}")

        records[card_code] = CardMetadataRecord(
            card_code=card_code,
            name=_normalize_text(postgres_record.get("name")),
            effect_text=_normalize_text(postgres_record.get("effect_text")),
            keywords=_normalize_text_list(generated_card_def.get("keywords")),
            subtypes=_normalize_text_list(postgres_record.get("subtypes")),
            card_type=card_type,
            element=element,
            ikz_cost=int(postgres_record.get("ikz_cost") or 0),
            attack=int(postgres_record.get("attack") or 0),
            health=int(postgres_record.get("health") or 0),
            gate_points=int(postgres_record.get("gate_points") or 0),
            has_ability=bool(ability_record.get("has_ability", False)),
            ability_timing_id=TIMING_TAG_TO_ID[timing_tag],
            ability_is_optional=bool(ability_record.get("is_optional", False)),
        )

    if missing_codes:
        preview = ", ".join(missing_codes[:8])
        raise RuntimeError(
            "Postgres card metadata is missing card codes present in include/generated/card_defs.h: "
            f"{preview}{' ...' if len(missing_codes) > 8 else ''}"
        )

    return records


def _normalize_embedding(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    return (matrix / norms).astype(np.float32, copy=False)


def _embed_unique_texts(
    texts: list[str],
    *,
    model: str,
    api_key: str,
) -> dict[str, np.ndarray]:
    normalized_texts = sorted({text for text in texts if text})
    if not normalized_texts:
        return {}

    embeddings: dict[str, np.ndarray] = {}
    for start in range(0, len(normalized_texts), EMBEDDING_BATCH_SIZE):
        batch = normalized_texts[start : start + EMBEDDING_BATCH_SIZE]
        payload = json.dumps(
            {
                "input": batch,
                "model": model,
                "encoding_format": "float",
            }
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response_payload: dict[str, Any] | None = None
        for attempt in range(5):
            req = request.Request(
                OPENAI_EMBEDDINGS_URL,
                data=payload,
                headers=headers,
                method="POST",
            )
            try:
                with request.urlopen(req, timeout=120) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                break
            except error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504) and attempt < 4:
                    time.sleep(2**attempt)
                    continue
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"Embedding request failed with HTTP {exc.code}: {body}"
                ) from exc
            except error.URLError as exc:
                if attempt < 4:
                    time.sleep(2**attempt)
                    continue
                raise RuntimeError(f"Embedding request failed: {exc}") from exc

        if response_payload is None:
            raise RuntimeError("Embedding request failed without a response payload")

        items = sorted(response_payload.get("data", []), key=lambda item: item["index"])
        if len(items) != len(batch):
            raise RuntimeError(
                f"Embedding batch size mismatch: expected {len(batch)}, got {len(items)}"
            )
        batch_matrix = np.asarray(
            [item["embedding"] for item in items],
            dtype=np.float32,
        )
        batch_matrix = _normalize_embedding(batch_matrix)
        for text, vector in zip(batch, batch_matrix, strict=True):
            embeddings[text] = vector

    return embeddings


def _build_artifact_arrays(
    card_def_id_map: dict[str, int],
    records: dict[str, CardMetadataRecord],
    *,
    embedding_model: str,
    api_key: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    table_len = max(card_def_id_map.values()) + 2
    keyword_vocab = sorted({keyword for record in records.values() for keyword in record.keywords})
    subtype_vocab = sorted({subtype for record in records.values() for subtype in record.subtypes})

    text_corpus: list[str] = []
    for record in records.values():
        if record.name:
            text_corpus.append(record.name)
        if record.effect_text:
            text_corpus.append(record.effect_text)
        text_corpus.extend(record.subtypes)
    embeddings_by_text = _embed_unique_texts(
        text_corpus,
        model=embedding_model,
        api_key=api_key,
    )

    if not embeddings_by_text:
        raise RuntimeError("No text embeddings were generated")

    embedding_dim = next(iter(embeddings_by_text.values())).shape[0]
    zero_vector = np.zeros(embedding_dim, dtype=np.float32)

    card_present_mask = np.zeros(table_len, dtype=np.float32)
    card_type_ids = np.zeros(table_len, dtype=np.int16)
    element_ids = np.zeros(table_len, dtype=np.int16)
    ikz_cost = np.zeros(table_len, dtype=np.float32)
    attack = np.zeros(table_len, dtype=np.float32)
    health = np.zeros(table_len, dtype=np.float32)
    gate_points = np.zeros(table_len, dtype=np.float32)
    has_ability = np.zeros(table_len, dtype=np.float32)
    ability_timing_ids = np.zeros(table_len, dtype=np.int16)
    ability_is_optional = np.zeros(table_len, dtype=np.float32)
    keyword_multi_hot = np.zeros((table_len, len(keyword_vocab)), dtype=np.float32)
    name_embeddings = np.zeros((table_len, embedding_dim), dtype=np.float32)
    effect_embeddings = np.zeros((table_len, embedding_dim), dtype=np.float32)
    subtype_pooled_embeddings = np.zeros((table_len, embedding_dim), dtype=np.float32)

    subtype_embedding_matrix = np.zeros((len(subtype_vocab), embedding_dim), dtype=np.float32)
    for subtype_index, subtype in enumerate(subtype_vocab):
        subtype_embedding_matrix[subtype_index] = embeddings_by_text.get(subtype, zero_vector)

    subtype_to_index = {subtype: index for index, subtype in enumerate(subtype_vocab)}
    keyword_to_index = {keyword: index for index, keyword in enumerate(keyword_vocab)}

    manifest_records: list[dict[str, Any]] = []
    for card_code, record in sorted(records.items(), key=lambda item: card_def_id_map[item[0]]):
        table_index = card_def_id_map[card_code] + 1
        card_present_mask[table_index] = 1.0
        card_type_ids[table_index] = CARD_TYPE_TO_ID[record.card_type]
        element_ids[table_index] = ELEMENT_TO_ID[record.element]
        ikz_cost[table_index] = float(record.ikz_cost)
        attack[table_index] = float(record.attack)
        health[table_index] = float(record.health)
        gate_points[table_index] = float(record.gate_points)
        has_ability[table_index] = 1.0 if record.has_ability else 0.0
        ability_timing_ids[table_index] = np.int16(record.ability_timing_id)
        ability_is_optional[table_index] = 1.0 if record.ability_is_optional else 0.0
        name_embeddings[table_index] = embeddings_by_text.get(record.name, zero_vector)
        effect_embeddings[table_index] = embeddings_by_text.get(record.effect_text, zero_vector)

        if record.subtypes:
            pooled = np.stack(
                [subtype_embedding_matrix[subtype_to_index[subtype]] for subtype in record.subtypes],
                axis=0,
            ).mean(axis=0)
            pooled_norm = float(np.linalg.norm(pooled))
            if pooled_norm > 0.0:
                pooled = pooled / pooled_norm
            subtype_pooled_embeddings[table_index] = pooled.astype(np.float32)

        for keyword in record.keywords:
            keyword_multi_hot[table_index, keyword_to_index[keyword]] = 1.0

        manifest_records.append(
            {
                "card_code": record.card_code,
                "card_def_id": card_def_id_map[card_code],
                "table_index": table_index,
                "name": record.name,
                "effect_text": record.effect_text,
                "keywords": list(record.keywords),
                "subtypes": list(record.subtypes),
                "card_type": record.card_type,
                "element": record.element,
                "ikz_cost": record.ikz_cost,
                "attack": record.attack,
                "health": record.health,
                "gate_points": record.gate_points,
                "has_ability": record.has_ability,
                "ability_timing_id": record.ability_timing_id,
                "ability_is_optional": record.ability_is_optional,
            }
        )

    arrays = {
        "card_present_mask": card_present_mask,
        "card_type_ids": card_type_ids,
        "element_ids": element_ids,
        "ikz_cost": ikz_cost,
        "attack": attack,
        "health": health,
        "gate_points": gate_points,
        "has_ability": has_ability,
        "ability_timing_ids": ability_timing_ids,
        "ability_is_optional": ability_is_optional,
        "keyword_multi_hot": keyword_multi_hot,
        "name_embeddings": name_embeddings,
        "effect_embeddings": effect_embeddings,
        "subtype_pooled_embeddings": subtype_pooled_embeddings,
        "subtype_embeddings": subtype_embedding_matrix,
    }
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "table_len": table_len,
        "real_card_count": len(records),
        "real_card_index_offset": 1,
        "keyword_vocab": keyword_vocab,
        "subtype_vocab": subtype_vocab,
        "card_type_to_id": CARD_TYPE_TO_ID,
        "element_to_id": ELEMENT_TO_ID,
        "ability_timing_to_id": TIMING_TAG_TO_ID,
        "records": manifest_records,
    }
    return arrays, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the versioned policy card metadata artifact for the v2 model."
    )
    parser.add_argument(
        "--database-url",
        default=_default_database_url(),
        help="Postgres connection string. Defaults to DATABASE_URL or the local docker-compose DSN.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="JSON manifest output path.",
    )
    parser.add_argument(
        "--array-path",
        type=Path,
        default=DEFAULT_ARRAY_PATH,
        help="NPZ tensor bundle output path.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="OpenAI embedding model to use for name/effect/subtype text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to generate the metadata artifact")

    card_def_id_map = _load_card_def_id_map(REPO_ROOT / "include" / "generated" / "card_defs.h")
    ability_metadata = _load_ability_metadata(
        REPO_ROOT / "python" / "config" / "ability_static_metadata.json"
    )
    generated_card_defs = _load_generated_card_defs(
        REPO_ROOT / "scripts" / "azuki-card-defs.jsonl"
    )
    postgres_records = _load_card_records_from_postgres(args.database_url)
    records = _build_card_metadata_records(
        card_def_id_map,
        postgres_records,
        ability_metadata,
        generated_card_defs,
    )
    arrays, manifest = _build_artifact_arrays(
        card_def_id_map,
        records,
        embedding_model=args.embedding_model,
        api_key=api_key,
    )

    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.array_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(args.array_path, **arrays)

    print(
        json.dumps(
            {
                "artifact_version": ARTIFACT_VERSION,
                "manifest_path": str(args.manifest_path),
                "array_path": str(args.array_path),
                "real_card_count": manifest["real_card_count"],
                "embedding_dim": manifest["embedding_dim"],
                "keyword_vocab_size": len(manifest["keyword_vocab"]),
                "subtype_vocab_size": len(manifest["subtype_vocab"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
