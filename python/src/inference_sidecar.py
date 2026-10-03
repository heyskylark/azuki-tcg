from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import threading
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - startup failure path
    np = None
    _NUMPY_IMPORT_ERROR = str(exc)
else:
    _NUMPY_IMPORT_ERROR = None

try:
    import torch
except Exception as exc:  # pragma: no cover - startup failure path
    torch = None
    _TORCH_IMPORT_ERROR = str(exc)
else:
    _TORCH_IMPORT_ERROR = None

try:
    from observation import (
        DECKBUILD_OBSERVATION_CTYPE as _DECKBUILD_OBSERVATION_CTYPE,
        DECKBUILD_OBSERVATION_STRUCT_SIZE as _DECKBUILD_OBSERVATION_STRUCT_SIZE,
        OBSERVATION_CTYPE as _OBSERVATION_CTYPE,
        OBSERVATION_STRUCT_SIZE as _OBSERVATION_STRUCT_SIZE,
        observation_to_dict as _observation_to_dict,
    )
except Exception as exc:  # pragma: no cover - startup failure path
    _DECKBUILD_OBSERVATION_CTYPE = None
    _DECKBUILD_OBSERVATION_STRUCT_SIZE = None
    _OBSERVATION_CTYPE = None
    _OBSERVATION_STRUCT_SIZE = None
    _observation_to_dict = None
    _OBSERVATION_IMPORT_ERROR = str(exc)
else:
    _OBSERVATION_IMPORT_ERROR = None

# Keep a fallback for environments where observation.py fails to import.
# The current packed TrainingObservationData size is 6308 bytes.
OBSERVATION_BYTE_SIZE = (
    int(_OBSERVATION_STRUCT_SIZE) if _OBSERVATION_STRUCT_SIZE is not None else 6308
)
DECKBUILD_OBSERVATION_BYTE_SIZE = (
    int(_DECKBUILD_OBSERVATION_STRUCT_SIZE)
    if _DECKBUILD_OBSERVATION_STRUCT_SIZE is not None
    else None
)
ACTION_COMPONENT_COUNT = 4
SESSION_TTL_SECONDS = 60 * 15
DEFAULT_MAX_CONCURRENT_INFERENCES = 2
DEFAULT_MAX_QUEUE_SIZE = 8
DEFAULT_QUEUE_WAIT_TIMEOUT_MS = 15000


class InferenceError(Exception):
    pass


class InferenceBusyError(InferenceError):
    pass

def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _canonical_deck_hash(
    gate_card_code: str,
    leader_card_code: str,
    ordered_main_card_codes: list[str],
) -> str:
    return _canonical_sha256(
        {
            "gateCardCode": gate_card_code,
            "leaderCardCode": leader_card_code,
            "orderedMainCardCodes": ordered_main_card_codes,
        }
    )


def _validate_deck_generate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "modelKey",
        "sessionKey",
        "draftSeed",
        "aiSlot",
        "gateCardCode",
        "leaderCardCode",
        "premadeDeckSlug",
    }
    unknown = set(payload) - expected
    missing = expected - set(payload)
    if missing:
        raise InferenceError(
            "Missing deck generation fields: " + ", ".join(sorted(missing))
        )
    if unknown:
        raise InferenceError(
            "Unknown deck generation fields: " + ", ".join(sorted(unknown))
        )
    for field in ("modelKey", "sessionKey", "gateCardCode", "leaderCardCode"):
        value = payload[field]
        if not isinstance(value, str) or not value.strip():
            raise InferenceError(f"{field} must be a non-empty string")
    draft_seed = payload["draftSeed"]
    if (
        not isinstance(draft_seed, int)
        or isinstance(draft_seed, bool)
        or not 0 <= draft_seed <= 0xFFFFFFFF
    ):
        raise InferenceError("draftSeed must be an unsigned 32-bit integer")
    ai_slot = payload["aiSlot"]
    if not isinstance(ai_slot, int) or isinstance(ai_slot, bool) or ai_slot not in (0, 1):
        raise InferenceError("aiSlot must be 0 or 1")
    premade_deck_slug = payload["premadeDeckSlug"]
    if premade_deck_slug is not None and (
        not isinstance(premade_deck_slug, str) or not premade_deck_slug.strip()
    ):
        raise InferenceError("premadeDeckSlug must be null or a non-empty string")
    return {
        "modelKey": payload["modelKey"].strip(),
        "sessionKey": payload["sessionKey"].strip(),
        "draftSeed": draft_seed,
        "aiSlot": ai_slot,
        "gateCardCode": payload["gateCardCode"].strip(),
        "leaderCardCode": payload["leaderCardCode"].strip(),
        "premadeDeckSlug": (
            premade_deck_slug.strip() if premade_deck_slug is not None else None
        ),
    }


def _deck_context_to_dict(context: Any) -> dict[str, Any]:
    return {
        "mode": int(context.mode),
        "gate_card_def_id": int(context.gate_card_def_id),
        "leader_card_def_id": int(context.leader_card_def_id),
        "main_card_def_ids": np.fromiter(
            (int(value) for value in context.main_card_def_ids),
            dtype=np.int16,
        ),
        "main_count": int(context.main_count),
        "candidate_card_def_ids": np.fromiter(
            (int(value) for value in context.candidate_card_def_ids),
            dtype=np.int16,
        ),
        "candidate_copy_counts": np.fromiter(
            (int(value) for value in context.candidate_copy_counts),
            dtype=np.uint8,
        ),
        "candidate_count": int(context.candidate_count),
    }


def _decode_observation_bytes(observation_bytes: bytes) -> dict[str, Any]:
    if _OBSERVATION_CTYPE is None or _observation_to_dict is None:
        raise InferenceError(
            "Packed observation decoder is unavailable: "
            f"{_OBSERVATION_IMPORT_ERROR or 'observation module did not load'}"
        )
    if len(observation_bytes) == OBSERVATION_BYTE_SIZE:
        observation = _OBSERVATION_CTYPE.from_buffer_copy(observation_bytes)
        return _observation_to_dict(observation)
    if (
        DECKBUILD_OBSERVATION_BYTE_SIZE is not None
        and len(observation_bytes) == DECKBUILD_OBSERVATION_BYTE_SIZE
        and _DECKBUILD_OBSERVATION_CTYPE is not None
    ):
        observation = _DECKBUILD_OBSERVATION_CTYPE.from_buffer_copy(observation_bytes)
        decoded = _observation_to_dict(observation)
        decoded["deck_context"] = _deck_context_to_dict(observation.deck_context)
        return decoded
    expected = str(OBSERVATION_BYTE_SIZE)
    if DECKBUILD_OBSERVATION_BYTE_SIZE is not None:
        expected += f" or {DECKBUILD_OBSERVATION_BYTE_SIZE}"
    raise InferenceError(
        f"Invalid observation size. Expected {expected}, got {len(observation_bytes)}"
    )


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        env_key = key.strip()
        if not env_key:
            continue

        env_value = value.strip()
        if (
            (env_value.startswith('"') and env_value.endswith('"'))
            or (env_value.startswith("'") and env_value.endswith("'"))
        ) and len(env_value) >= 2:
            env_value = env_value[1:-1]
        elif " #" in env_value:
            env_value = env_value.split(" #", 1)[0].rstrip()

        values[env_key] = env_value

    return values


def _load_local_env_files() -> None:
    script_repo_root = Path(__file__).resolve().parents[2]
    cwd = Path.cwd().resolve()

    candidate_roots = [cwd]
    if script_repo_root not in candidate_roots:
        candidate_roots.append(script_repo_root)

    merged_values: dict[str, str] = {}
    for root in candidate_roots:
        env_path = root / ".env"
        env_local_path = root / ".env.local"
        if env_path.exists():
            merged_values.update(_parse_env_file(env_path))
        if env_local_path.exists():
            merged_values.update(_parse_env_file(env_local_path))

    for key, value in merged_values.items():
        os.environ.setdefault(key, value)


def resolve_device(requested_device: str) -> str:
    if requested_device in {"cpu", "cuda", "mps"}:
        if requested_device == "cuda" and torch is not None and torch.cuda.is_available():
            return "cuda"
        if (
            requested_device == "mps"
            and torch is not None
            and getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        ):
            return "mps"
        if requested_device == "cpu":
            return "cpu"
        raise InferenceError(f"Requested device '{requested_device}' is not available")

    if torch is None:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_built() and mps_backend.is_available():
        return "mps"

    return "cpu"


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise InferenceError(f"Invalid S3 URI: {uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket or not key:
        raise InferenceError(f"Invalid S3 URI: {uri}")
    return bucket, key


def _normalize_s3_prefix(prefix: str) -> str:
    parsed = urlparse(prefix)
    if parsed.scheme != "s3":
        raise InferenceError(
            "AZK_INFER_S3_MODEL_PREFIX must be an s3:// URI prefix"
        )

    bucket = parsed.netloc
    key_prefix = parsed.path.lstrip("/")
    if not bucket:
        raise InferenceError(
            "AZK_INFER_S3_MODEL_PREFIX must include an S3 bucket name"
        )

    if key_prefix and not key_prefix.endswith("/"):
        key_prefix = f"{key_prefix}/"

    return f"s3://{bucket}/{key_prefix}"


def _resolve_aws_region() -> str | None:
    return (
        os.getenv("AZK_AWS_REGION")
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
    )


@dataclass
class SessionState:
    lstm_h: Any
    lstm_c: Any
    last_used_at: float
    deck_context: dict[str, Any] | None = None


@dataclass
class ModelRuntime:
    model: Any
    model_path: Path
    loaded_at: float


class InferenceEngine:
    def __init__(
        self,
        *,
        config_path: Path,
        requested_device: str,
        model_cache_dir: Path,
        session_ttl_seconds: int,
        max_concurrent_inferences: int,
        max_queue_size: int,
        queue_wait_timeout_ms: int,
    ) -> None:
        # Draft serving is always greedy; battle serving defaults to sampling.
        self._battle_action_mode = os.getenv("AZK_INFER_BATTLE_ACTION_MODE", "sample")
        if self._battle_action_mode not in ("sample", "argmax"):
            raise ValueError("AZK_INFER_BATTLE_ACTION_MODE must be sample or argmax")
        self.config_path = config_path
        self.requested_device = requested_device
        self.session_ttl_seconds = session_ttl_seconds
        self.model_cache_dir = model_cache_dir
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        if max_concurrent_inferences <= 0:
            raise ValueError("max_concurrent_inferences must be greater than 0")
        if max_queue_size < 0:
            raise ValueError("max_queue_size must be greater than or equal to 0")
        if queue_wait_timeout_ms <= 0:
            raise ValueError("queue_wait_timeout_ms must be greater than 0")
        self.max_concurrent_inferences = max_concurrent_inferences
        self.max_queue_size = max_queue_size
        self.queue_wait_timeout_seconds = queue_wait_timeout_ms / 1000.0

        self._lock = threading.Lock()
        self._models: dict[str, ModelRuntime] = {}
        self._model_load_locks: dict[str, threading.Lock] = {}
        self._sessions: dict[str, SessionState] = {}
        self._runtime_error: str | None = None
        self._device = "cpu"
        self._s3_client: Any | None = None
        self._inference_slots = threading.BoundedSemaphore(self.max_concurrent_inferences)
        self._request_slots = threading.BoundedSemaphore(
            self.max_concurrent_inferences + self.max_queue_size
        )
        self._inflight_inferences = 0
        self._tcg_argmax_logits = None
        self._queued_requests = 0

        self._puffer_sample_logits = None
        self._build_policy = None
        self._build_vecenv = None
        self._install_tcg_sampler = None
        self._load_training_config = None
        self._load_model_weights = None
        self._vecenv = None
        self._trainer_args = None

        self._initialize_runtime()
    @property
    def device(self) -> str:
        return self._device

    @property
    def runtime_error(self) -> str | None:
        return self._runtime_error

    def health_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": "ok" if self._runtime_error is None else "degraded",
                "device": self._device,
                "battleActionMode": self._battle_action_mode,
                "observationByteSize": OBSERVATION_BYTE_SIZE,
                "deckBuildObservationByteSize": DECKBUILD_OBSERVATION_BYTE_SIZE,
                "observationImportError": _OBSERVATION_IMPORT_ERROR,
                "runtimeError": self._runtime_error,
                "loadedModelCount": len(self._models),
                "activeSessionCount": len(self._sessions),
                "awsRegion": _resolve_aws_region(),
                "s3EndpointUrl": os.getenv("AZK_AWS_S3_ENDPOINT_URL"),
                "s3ModelPrefix": os.getenv("AZK_INFER_S3_MODEL_PREFIX"),
                "maxConcurrentInferences": self.max_concurrent_inferences,
                "maxQueueSize": self.max_queue_size,
                "queueWaitTimeoutSeconds": self.queue_wait_timeout_seconds,
                "inflightInferences": self._inflight_inferences,
                "queuedRequests": self._queued_requests,
            }

    def _model_action(
        self,
        *,
        model: Any,
        session_key: str,
        observation_bytes: bytes,
        deterministic: bool,
    ) -> list[int]:
        with self._lock:
            session = self._sessions.get(session_key)
        if _OBSERVATION_CTYPE is None or _observation_to_dict is None:
            obs_array = (
                np.frombuffer(observation_bytes, dtype=np.uint8)
                .copy()
                .reshape(1, len(observation_bytes))
            )
            obs_input: Any = torch.from_numpy(obs_array).to(device=self._device)
        else:
            obs_input = _decode_observation_bytes(observation_bytes)
            if (
                "deck_context" not in obs_input
                and session is not None
                and session.deck_context is not None
            ):
                obs_input["deck_context"] = session.deck_context
        state: dict[str, Any] = {
            "mask": torch.ones(1, dtype=torch.bool, device=self._device),
        }
        if session is not None:
            state["lstm_h"] = session.lstm_h
            state["lstm_c"] = session.lstm_c
        else:
            hidden_size = int(model.hidden_size)
            state["lstm_h"] = torch.zeros(1, hidden_size, device=self._device)
            state["lstm_c"] = torch.zeros(1, hidden_size, device=self._device)
        with torch.no_grad():
            logits, _ = model.forward_eval(obs_input, state)
            if deterministic:
                sampled_actions = self._tcg_argmax_logits(logits)
            else:
                sampled_actions, _, _ = self._puffer_sample_logits(logits)
        action_values = (
            sampled_actions.detach().cpu().numpy().astype(np.int32, copy=True).reshape(-1)
        )
        if action_values.shape[0] != ACTION_COMPONENT_COUNT:
            raise InferenceError(
                "Inference produced invalid action size: "
                f"{action_values.shape[0]} (expected {ACTION_COMPONENT_COUNT})"
            )
        with self._lock:
            self._sessions[session_key] = SessionState(
                lstm_h=state["lstm_h"],
                lstm_c=state["lstm_c"],
                last_used_at=time.time(),
                deck_context=session.deck_context if session is not None else None,
            )
        return [int(value) for value in action_values.tolist()]

    def infer(
        self,
        *,
        model_key: str,
        session_key: str,
        observation_b64: str,
        reset_session: bool,
        require_session: bool = False,
    ) -> list[int]:
        if self._runtime_error is not None:
            raise InferenceError(self._runtime_error)

        if not model_key:
            raise InferenceError("modelKey is required")
        if not session_key:
            raise InferenceError("sessionKey is required")

        try:
            observation_bytes = base64.b64decode(observation_b64, validate=True)
        except Exception as exc:
            raise InferenceError(f"Invalid observationBase64 payload: {exc}") from exc

        valid_sizes = {OBSERVATION_BYTE_SIZE}
        if DECKBUILD_OBSERVATION_BYTE_SIZE is not None:
            valid_sizes.add(DECKBUILD_OBSERVATION_BYTE_SIZE)
        if len(observation_bytes) not in valid_sizes:
            expected = " or ".join(str(size) for size in sorted(valid_sizes))
            raise InferenceError(
                f"Invalid observation size. Expected {expected}, got {len(observation_bytes)}"
            )

        if not self._try_enter_request_queue():
            raise InferenceBusyError("Inference queue is full, try again shortly")

        inference_slot_acquired = False
        try:
            if not self._enter_inference_slot():
                self._drop_queued_request()
                raise InferenceBusyError(
                    "Inference queue wait timed out, try again shortly"
                )

            inference_slot_acquired = True
            self._promote_queued_request_to_inflight()
            self._evict_stale_sessions()

            if reset_session:
                self.end_session(session_key)
            if require_session:
                with self._lock:
                    session_exists = session_key in self._sessions
                if not session_exists:
                    raise InferenceError("Required inference session is not active")
            model = self._get_or_load_model(model_key)

            return self._model_action(
                model=model,
                session_key=session_key,
                observation_bytes=observation_bytes,
                deterministic=self._battle_action_mode == "argmax",
            )
        finally:
            if inference_slot_acquired:
                self._leave_inference_slot()
            self._leave_request_slot()

    def end_session(self, session_key: str) -> None:
        with self._lock:
            self._sessions.pop(session_key, None)

    def session_active(self, session_key: str) -> bool:
        self._evict_stale_sessions()
        with self._lock:
            return session_key in self._sessions

    def generate_deck(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = _validate_deck_generate_payload(payload)
        if self._runtime_error is not None:
            raise InferenceError(self._runtime_error)
        env_config = self._trainer_args.get("env", {})
        if not isinstance(env_config, dict) or not bool(
            env_config.get("deck_building_enabled", False)
        ):
            raise InferenceError(
                "Configured policy is not a deck-building policy "
                "(env.deck_building_enabled=true is required)"
            )
        if not self._try_enter_request_queue():
            raise InferenceBusyError("Inference queue is full, try again shortly")

        inference_slot_acquired = False
        generation_env = None
        session_key = request["sessionKey"]
        try:
            if not self._enter_inference_slot():
                self._drop_queued_request()
                raise InferenceBusyError(
                    "Inference queue wait timed out, try again shortly"
                )
            inference_slot_acquired = True
            self._promote_queued_request_to_inflight()
            self._evict_stale_sessions()
            self.end_session(session_key)

            model_key = request["modelKey"]
            model = self._get_or_load_model(model_key)
            with self._lock:
                runtime = self._models.get(model_key)
            if runtime is None:
                raise InferenceError("Loaded model runtime is unavailable")

            from azk_native import AzukiNativeEnv  # noqa: WPS433
            from deck_building import (  # noqa: WPS433
                GATE_CARD_TYPE,
                LEADER_CARD_TYPE,
                MAIN_CARD_TYPES,
                MAX_MAIN_COPIES,
                build_deck_build_catalog,
            )
            from observation import (  # noqa: WPS433
                DECK_CONTEXT_MODE_BATTLE,
                DECK_CONTEXT_MODE_PICK_LEADER,
                DECK_CONTEXT_MODE_PICK_MAIN,
            )
            from training_deck_pool import (  # noqa: WPS433
                load_training_deck_labels,
                load_training_deck_pool,
            )

            deck_pool_path = env_config.get("deck_pool_path")
            if deck_pool_path is not None and not isinstance(deck_pool_path, str):
                raise InferenceError("env.deck_pool_path must be a string")
            deck_pool = load_training_deck_pool(deck_pool_path)
            if not deck_pool:
                raise InferenceError("Configured deck pool is empty")
            catalog = build_deck_build_catalog(deck_pool)
            gate_record = catalog.records_by_code.get(request["gateCardCode"])
            leader_record = catalog.records_by_code.get(request["leaderCardCode"])
            if (
                gate_record is None
                or gate_record.card_type != GATE_CARD_TYPE
                or gate_record.card_def_id not in set(catalog.gate_def_id_population)
            ):
                raise InferenceError(
                    f"gateCardCode is not a production draft gate: {request['gateCardCode']}"
                )
            if (
                leader_record is None
                or leader_record.card_type != LEADER_CARD_TYPE
                or leader_record.element != gate_record.element
                or leader_record.card_def_id
                not in catalog.leader_def_ids_by_element.get(gate_record.element, ())
            ):
                raise InferenceError(
                    "leaderCardCode is not compatible with gateCardCode"
                )

            ai_slot = request["aiSlot"]
            reference_slot = 1 - ai_slot
            uniform_assignment = bool(
                env_config.get("draft_uniform_assignment", False)
            )
            generation_env = AzukiNativeEnv(
                num_envs=1,
                deck_pool=deck_pool,
                seed=request["draftSeed"],
                deck_building=True,
                draft_uniform_assignment=uniform_assignment,
                evaluation_mode=True,
            )
            forced_leader = leader_record.card_def_id if uniform_assignment else -1
            premade_deck_slug = request["premadeDeckSlug"]
            if premade_deck_slug is not None:
                # Constructed opponent: open the same native evaluation game the
                # specialist trained on (a supplied pool deck that starts directly
                # in battle) and serve with that game's battle deck context.
                deck_labels = load_training_deck_labels(deck_pool_path)
                if premade_deck_slug not in deck_labels:
                    raise InferenceError(
                        "premadeDeckSlug is not in the configured deck pool: "
                        f"{premade_deck_slug}"
                    )
                deck_index = deck_labels.index(premade_deck_slug)
                premade_gate_codes: list[str] = []
                premade_leader_codes: list[str] = []
                premade_main_codes: list[str] = []
                for card_code, quantity in deck_pool[deck_index]:
                    record = catalog.records_by_code.get(card_code)
                    if record is None:
                        raise InferenceError(
                            f"Premade deck {premade_deck_slug} references unknown card {card_code}"
                        )
                    if record.card_type == GATE_CARD_TYPE:
                        premade_gate_codes.extend([card_code] * quantity)
                    elif record.card_type == LEADER_CARD_TYPE:
                        premade_leader_codes.extend([card_code] * quantity)
                    elif record.card_type in MAIN_CARD_TYPES:
                        premade_main_codes.extend([card_code] * quantity)
                if (
                    premade_gate_codes != [request["gateCardCode"]]
                    or premade_leader_codes != [request["leaderCardCode"]]
                    or len(premade_main_codes) != 50
                ):
                    raise InferenceError(
                        "premadeDeckSlug does not match gateCardCode/leaderCardCode"
                    )
                generation_env.reset_evaluation_games(
                    [
                        {
                            "env_index": 0,
                            "seed": request["draftSeed"],
                            "gate0": gate_record.card_def_id,
                            "gate1": gate_record.card_def_id,
                            "leader0": forced_leader,
                            "leader1": forced_leader,
                            "reference_seat": ai_slot,
                            "reference_deck_index": deck_index,
                            "other_deck_index": deck_index,
                        }
                    ]
                )
                decoded = _decode_observation_bytes(
                    generation_env.observations[ai_slot].tobytes()
                )
                premade_context = decoded.get("deck_context")
                if (
                    not isinstance(premade_context, dict)
                    or premade_context["mode"] != DECK_CONTEXT_MODE_BATTLE
                    or premade_context["gate_card_def_id"] != gate_record.card_def_id
                    or premade_context["leader_card_def_id"] != leader_record.card_def_id
                    or premade_context["main_count"] != 50
                    or [
                        catalog.records_by_def_id[int(card_id)].card_code
                        if int(card_id) in catalog.records_by_def_id
                        else None
                        for card_id in premade_context["main_card_def_ids"][:50]
                    ]
                    != premade_main_codes
                ):
                    raise InferenceError(
                        "Native premade battle context does not match the premade deck"
                    )
                self._store_battle_deck_context(session_key, premade_context, model=model)
                return self._deck_generation_result(
                    request, catalog, runtime, premade_main_codes, []
                )
            generation_env.reset_evaluation_games(
                [
                    {
                        "env_index": 0,
                        "seed": request["draftSeed"],
                        "gate0": gate_record.card_def_id,
                        "gate1": gate_record.card_def_id,
                        "leader0": forced_leader,
                        "leader1": forced_leader,
                        "reference_seat": reference_slot,
                        "reference_deck_index": 0,
                    }
                ]
            )

            picks: list[dict[str, Any]] = []
            leader_pick_seen = uniform_assignment
            while len(picks) < 50 or not leader_pick_seen:
                active_players = generation_env.active_players()
                if active_players.shape != (1,) or int(active_players[0]) != ai_slot:
                    raise InferenceError(
                        "Native evaluation draft did not keep the model in the active seat"
                    )
                observation_bytes = generation_env.observations[ai_slot].tobytes()
                decoded = _decode_observation_bytes(observation_bytes)
                context = decoded.get("deck_context")
                if not isinstance(context, dict):
                    raise InferenceError("Deck-build observation is missing deck_context")
                mode = context["mode"]
                candidate_count = context["candidate_count"]
                candidate_ids = list(
                    context["candidate_card_def_ids"][:candidate_count]
                )
                if (
                    mode not in (DECK_CONTEXT_MODE_PICK_LEADER, DECK_CONTEXT_MODE_PICK_MAIN)
                    or not isinstance(candidate_count, int)
                    or candidate_count <= 0
                    or len(candidate_ids) != candidate_count
                ):
                    raise InferenceError("Native draft returned invalid candidates")
                candidate_codes: list[str] = []
                for card_id in candidate_ids:
                    record = catalog.records_by_def_id.get(card_id)
                    if record is None:
                        raise InferenceError(
                            f"Native draft returned unknown card def id {card_id}"
                        )
                    candidate_codes.append(record.card_code)

                action = self._model_action(
                    model=model,
                    session_key=session_key,
                    observation_bytes=observation_bytes,
                    deterministic=True,
                )
                if (
                    action[0] != 3
                    or action[2] != 0
                    or action[3] != 0
                    or action[1] < 0
                    or action[1] >= candidate_count
                ):
                    raise InferenceError(
                        f"Model produced invalid deck-building action {action}"
                    )
                selected_index = action[1]
                selected_code = candidate_codes[selected_index]
                if mode == DECK_CONTEXT_MODE_PICK_LEADER:
                    if leader_pick_seen:
                        raise InferenceError("Native draft requested more than one leader pick")
                    leader_pick_seen = True
                    if selected_code != request["leaderCardCode"]:
                        raise InferenceError(
                            "Configured non-uniform policy selected a leader that "
                            "does not match leaderCardCode"
                        )
                else:
                    picks.append(
                        {
                            "ordinal": len(picks) + 1,
                            "candidateCardCodes": candidate_codes,
                            "selectedIndex": selected_index,
                            "selectedCardCode": selected_code,
                        }
                    )
                generation_env.actions[ai_slot] = np.asarray(action, dtype=np.int32)
                generation_env.step()

            snapshot = generation_env.draft_snapshot(ai_slot)
            if (
                snapshot["gate"] != gate_record.card_def_id
                or snapshot["leader"] != leader_record.card_def_id
            ):
                raise InferenceError("Native draft snapshot does not match forced context")
            ordered_main_codes: list[str] = []
            for card_id in snapshot["main"]:
                record = catalog.records_by_def_id.get(card_id)
                if record is None:
                    raise InferenceError(
                        f"Native snapshot returned unknown card def id {card_id}"
                    )
                ordered_main_codes.append(record.card_code)
            card_counts = Counter(ordered_main_codes)
            if (
                len(ordered_main_codes) != 50
                or any(count > MAX_MAIN_COPIES for count in card_counts.values())
                or [pick["selectedCardCode"] for pick in picks] != ordered_main_codes
            ):
                raise InferenceError("Native draft snapshot failed deck validation")
            battle_deck_context = {
                **context,
                "mode": DECK_CONTEXT_MODE_BATTLE,
                "gate_card_def_id": snapshot["gate"],
                "leader_card_def_id": snapshot["leader"],
                "main_card_def_ids": np.asarray(snapshot["main"], dtype=np.int16),
                "main_count": 50,
                "candidate_card_def_ids": np.full(
                    len(context["candidate_card_def_ids"]), -1, dtype=np.int16
                ),
                "candidate_copy_counts": np.zeros(
                    len(context["candidate_copy_counts"]), dtype=np.uint8
                ),
                "candidate_count": 0,
            }
            self._store_battle_deck_context(session_key, battle_deck_context, model=None)
            return self._deck_generation_result(
                request, catalog, runtime, ordered_main_codes, picks
            )
        except Exception:
            self.end_session(session_key)
            raise
        finally:
            if generation_env is not None:
                generation_env.close()
            if inference_slot_acquired:
                self._leave_inference_slot()
            self._leave_request_slot()

    def _store_battle_deck_context(
        self, session_key: str, deck_context: dict[str, Any], *, model: Any
    ) -> None:
        """Attach the battle deck context to the recurrent session.

        Drafts already own a session (model=None requires it); premade decks
        start a fresh zero-state session like any first battle decision.
        """
        with self._lock:
            session = self._sessions.get(session_key)
            if session is None:
                if model is None:
                    raise InferenceError(
                        "Draft recurrent session disappeared before completion"
                    )
                hidden_size = int(model.hidden_size)
                session = SessionState(
                    lstm_h=torch.zeros(1, hidden_size, device=self._device),
                    lstm_c=torch.zeros(1, hidden_size, device=self._device),
                    last_used_at=time.time(),
                )
                self._sessions[session_key] = session
            session.deck_context = deck_context
            session.last_used_at = time.time()

    def _deck_generation_result(
        self,
        request: dict[str, Any],
        catalog: Any,
        runtime: ModelRuntime,
        ordered_main_codes: list[str],
        picks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        catalog_payload = [
            {
                "cardCode": record.card_code,
                "cardDefId": record.card_def_id,
                "cardType": record.card_type,
                "element": record.element,
                "ikzCost": record.ikz_cost,
            }
            for record in sorted(
                catalog.records_by_def_id.values(),
                key=lambda item: item.card_def_id,
            )
        ]
        return {
            "gateCardCode": request["gateCardCode"],
            "leaderCardCode": request["leaderCardCode"],
            "premadeDeckSlug": request["premadeDeckSlug"],
            "orderedMainCardCodes": ordered_main_codes,
            "cardCounts": dict(sorted(Counter(ordered_main_codes).items())),
            "picks": picks,
            "deckHash": _canonical_deck_hash(
                request["gateCardCode"],
                request["leaderCardCode"],
                ordered_main_codes,
            ),
            "catalogHash": _canonical_sha256(catalog_payload),
            "checkpointSha256": self._sha256_file(runtime.model_path),
        }

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


    def _try_enter_request_queue(self) -> bool:
        acquired = self._request_slots.acquire(blocking=False)
        if not acquired:
            return False

        with self._lock:
            self._queued_requests += 1
        return True

    def _leave_request_slot(self) -> None:
        self._request_slots.release()

    def _enter_inference_slot(self) -> bool:
        return self._inference_slots.acquire(timeout=self.queue_wait_timeout_seconds)

    def _promote_queued_request_to_inflight(self) -> None:
        with self._lock:
            if self._queued_requests > 0:
                self._queued_requests -= 1
            self._inflight_inferences += 1

    def _drop_queued_request(self) -> None:
        with self._lock:
            if self._queued_requests > 0:
                self._queued_requests -= 1

    def _leave_inference_slot(self) -> None:
        with self._lock:
            if self._inflight_inferences > 0:
                self._inflight_inferences -= 1
        self._inference_slots.release()

    def _initialize_runtime(self) -> None:
        if torch is None:
            self._runtime_error = (
                "torch import failed. Install runtime dependencies before running inference: "
                f"{_TORCH_IMPORT_ERROR}"
            )
            self._device = "cpu"
            return
        if np is None:
            self._runtime_error = (
                "numpy import failed. Install runtime dependencies before running inference: "
                f"{_NUMPY_IMPORT_ERROR}"
            )
            self._device = "cpu"
            return

        try:
            self._device = resolve_device(self.requested_device)

            from training_utils import (  # noqa: WPS433
                build_policy,
                build_vecenv,
                install_tcg_sampler,
                load_training_config,
            )
            from train import _load_model_weights  # noqa: WPS433
            import azk_puffer.pytorch as azk_pytorch  # noqa: WPS433
            from policy.v2.tcg_sampler import tcg_argmax_logits  # noqa: WPS433

            self._build_policy = build_policy
            self._build_vecenv = build_vecenv
            self._install_tcg_sampler = install_tcg_sampler
            self._load_training_config = load_training_config
            self._load_model_weights = _load_model_weights

            self._install_tcg_sampler()
            self._puffer_sample_logits = azk_pytorch.sample_logits
            self._tcg_argmax_logits = tcg_argmax_logits

            trainer_args = self._load_training_config(self.config_path, [])
            trainer_args["train"]["device"] = self._device
            vecenv = self._build_vecenv(trainer_args)

            self._trainer_args = trainer_args
            self._vecenv = vecenv
        except Exception as exc:  # pragma: no cover - startup failure path
            self._runtime_error = f"Failed to initialize inference runtime: {exc}"

    def _download_model_from_s3(self, model_s3_uri: str) -> Path:
        bucket, key = _parse_s3_uri(model_s3_uri)
        local_name = f"{bucket}_{key.replace('/', '_')}"
        local_path = self.model_cache_dir / local_name
        if local_path.exists():
            return local_path

        s3 = self._get_or_create_s3_client()
        temp_path = Path(f"{local_path}.{threading.get_ident()}.tmp")
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if temp_path.exists():
                temp_path.unlink()
            s3.download_file(bucket, key, str(temp_path))
            temp_path.replace(local_path)
        except Exception as exc:
            if temp_path.exists():
                temp_path.unlink()
            raise InferenceError(
                f"Failed to download model from s3://{bucket}/{key}: {exc}"
            ) from exc

        return local_path

    def _get_or_create_s3_client(self) -> Any:
        with self._lock:
            if self._s3_client is not None:
                return self._s3_client

        s3_client = self._build_s3_client()

        with self._lock:
            if self._s3_client is None:
                self._s3_client = s3_client
            return self._s3_client

    def _build_s3_client(self) -> Any:
        try:
            import boto3  # noqa: WPS433
        except Exception as exc:  # pragma: no cover - optional dependency
            raise InferenceError(
                "boto3 is required for s3:// model keys but is not installed"
            ) from exc

        access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        session_token = os.getenv("AWS_SESSION_TOKEN")
        profile_name = os.getenv("AWS_PROFILE")
        endpoint_url = os.getenv("AZK_AWS_S3_ENDPOINT_URL")
        region_name = _resolve_aws_region()

        if (access_key_id and not secret_access_key) or (
            secret_access_key and not access_key_id
        ):
            raise InferenceError(
                "Both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set together"
            )

        session_kwargs: dict[str, Any] = {}
        if region_name:
            session_kwargs["region_name"] = region_name
        if profile_name and not access_key_id and not secret_access_key:
            session_kwargs["profile_name"] = profile_name

        session = boto3.session.Session(**session_kwargs)

        client_kwargs: dict[str, Any] = {}
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        if access_key_id and secret_access_key:
            client_kwargs["aws_access_key_id"] = access_key_id
            client_kwargs["aws_secret_access_key"] = secret_access_key
            if session_token:
                client_kwargs["aws_session_token"] = session_token

        return session.client("s3", **client_kwargs)

    def _build_model_s3_uri(self, model_key: str) -> str:
        model_key_trimmed = model_key.strip().lstrip("/")
        if not model_key_trimmed:
            raise InferenceError("modelKey must not be empty")

        raw_prefix = os.getenv("AZK_INFER_S3_MODEL_PREFIX")
        if not raw_prefix:
            raise InferenceError(
                "AZK_INFER_S3_MODEL_PREFIX is required to resolve model keys"
            )

        normalized_prefix = _normalize_s3_prefix(raw_prefix)
        return f"{normalized_prefix}{model_key_trimmed}"

    def _get_model_load_lock(self, model_key: str) -> threading.Lock:
        with self._lock:
            existing_lock = self._model_load_locks.get(model_key)
            if existing_lock is not None:
                return existing_lock

            created_lock = threading.Lock()
            self._model_load_locks[model_key] = created_lock
            return created_lock

    def _resolve_model_path(self, model_key: str) -> Path:
        local_root_raw = os.getenv("AZK_INFER_LOCAL_MODEL_ROOT")
        if local_root_raw:
            relative = Path(model_key.strip())
            if not model_key.strip() or relative.is_absolute() or ".." in relative.parts:
                raise InferenceError("modelKey is not a safe local model path")
            local_root = Path(local_root_raw).expanduser().resolve()
            candidate = (local_root / relative).resolve()
            if not candidate.is_relative_to(local_root) or not candidate.is_file():
                raise InferenceError(
                    f"Local modelKey does not resolve to a file under {local_root}"
                )
            return candidate
        model_s3_uri = self._build_model_s3_uri(model_key)
        return self._download_model_from_s3(model_s3_uri)

    def _get_or_load_model(self, model_key: str):
        with self._lock:
            cached = self._models.get(model_key)
            if cached is not None:
                return cached.model

        model_load_lock = self._get_model_load_lock(model_key)
        with model_load_lock:
            with self._lock:
                cached_after_lock = self._models.get(model_key)
                if cached_after_lock is not None:
                    return cached_after_lock.model

            model_path = self._resolve_model_path(model_key)
            build_trainer_args = self._trainer_args
            checkpoint_load_device = self._device

            # MPS does not support float64 tensors. Build/load on CPU first,
            # then cast/move the model to MPS float32.
            if self._device == "mps":
                build_trainer_args = dict(self._trainer_args)
                train_cfg = dict(build_trainer_args.get("train", {}))
                train_cfg["device"] = "cpu"
                build_trainer_args["train"] = train_cfg
                checkpoint_load_device = "cpu"

            policy = self._build_policy(self._vecenv, build_trainer_args)
            self._load_model_weights(
                policy,
                model_path,
                device=checkpoint_load_device,
                strict=False,
            )

            if self._device == "mps":
                policy = policy.to(device=self._device, dtype=torch.float32)

            policy.eval()

            with self._lock:
                self._models[model_key] = ModelRuntime(
                    model=policy,
                    model_path=model_path,
                    loaded_at=time.time(),
                )
                return policy

    def _evict_stale_sessions(self) -> None:
        cutoff = time.time() - float(self.session_ttl_seconds)
        with self._lock:
            stale_keys = [
                key for key, state in self._sessions.items() if state.last_used_at < cutoff
            ]
            for key in stale_keys:
                self._sessions.pop(key, None)


class InferenceRequestHandler(BaseHTTPRequestHandler):
    engine: InferenceEngine

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        content_length_raw = self.headers.get("Content-Length", "0")
        try:
            content_length = int(content_length_raw)
        except ValueError as exc:
            raise InferenceError("Invalid Content-Length header") from exc

        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise InferenceError(f"Invalid JSON payload: {exc}") from exc

        if not isinstance(payload, dict):
            raise InferenceError("JSON payload must be an object")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._send_json(404, {"error": "Not found"})
            return
        self._send_json(200, self.engine.health_payload())

    def _post_authorized(self) -> bool:
        shared_secret = os.getenv("AZK_INFER_SHARED_SECRET")
        if not shared_secret:
            return True
        authorization = self.headers.get("Authorization", "")
        if not isinstance(authorization, str):
            return False
        return hmac.compare_digest(
            authorization.encode("utf-8"),
            f"Bearer {shared_secret}".encode("utf-8"),
        )

    def do_POST(self) -> None:  # noqa: N802
        if not self._post_authorized():
            self._send_json(401, {"error": "Unauthorized"})
            return
        if self.path == "/infer":
            self._handle_infer()
            return
        if self.path == "/deck/generate":
            self._handle_generate_deck()
            return
        if self.path == "/session/status":
            self._handle_session_status()
            return
        if self.path == "/session/end":
            self._handle_end_session()
            return
        self._send_json(404, {"error": "Not found"})

    def _handle_infer(self) -> None:
        try:
            payload = self._read_json_body()
            model_key = payload.get("modelKey")
            session_key = payload.get("sessionKey")
            observation_b64 = payload.get("observationBase64")
            reset_session = bool(payload.get("resetSession", False))
            require_session = payload.get("requireSession", False)
            if not isinstance(model_key, str):
                raise InferenceError("modelKey must be a string")
            if not isinstance(session_key, str):
                raise InferenceError("sessionKey must be a string")
            if not isinstance(observation_b64, str):
                raise InferenceError("observationBase64 must be a string")
            if not isinstance(require_session, bool):
                raise InferenceError("requireSession must be a boolean")
            action = self.engine.infer(
                model_key=model_key,
                session_key=session_key,
                observation_b64=observation_b64,
                reset_session=reset_session,
                require_session=require_session,
            )
            self._send_json(
                200,
                {
                    "action": action,
                    "device": self.engine.device,
                },
            )
        except InferenceBusyError as exc:
            self._send_json(503, {"error": str(exc)})
        except InferenceError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - unexpected failure path
            traceback.print_exc()
            self._send_json(500, {"error": f"Unexpected inference failure: {exc}"})

    def _handle_generate_deck(self) -> None:
        try:
            payload = self._read_json_body()
            result = self.engine.generate_deck(payload)
            self._send_json(200, result)
        except InferenceBusyError as exc:
            self._send_json(503, {"error": str(exc)})
        except InferenceError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - unexpected failure path
            traceback.print_exc()
            self._send_json(500, {"error": f"Unexpected deck generation failure: {exc}"})

    def _handle_session_status(self) -> None:
        try:
            payload = self._read_json_body()
            if set(payload) != {"sessionKey"}:
                raise InferenceError(
                    "Session status payload must contain only sessionKey"
                )
            session_key = payload["sessionKey"]
            if not isinstance(session_key, str) or not session_key:
                raise InferenceError("sessionKey must be a non-empty string")
            self._send_json(200, {"active": self.engine.session_active(session_key)})
        except InferenceError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - unexpected failure path
            traceback.print_exc()
            self._send_json(500, {"error": f"Unexpected session failure: {exc}"})

    def _handle_end_session(self) -> None:
        try:
            payload = self._read_json_body()
            session_key = payload.get("sessionKey")
            if not isinstance(session_key, str):
                raise InferenceError("sessionKey must be a string")
            self.engine.end_session(session_key)
            self._send_json(200, {"ok": True})
        except InferenceError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - unexpected failure path
            traceback.print_exc()
            self._send_json(500, {"error": f"Unexpected session failure: {exc}"})

    def log_message(self, _format: str, *_args: Any) -> None:  # noqa: A003
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Azuki model inference sidecar")
    parser.add_argument("--host", type=str, default=os.getenv("AZK_INFER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("AZK_INFER_PORT", "8002")))
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default=os.getenv("AZK_INFER_DEVICE", "auto"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.getenv("AZK_INFER_CONFIG", "python/config/azuki.ini")),
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=Path(os.getenv("AZK_INFER_MODEL_CACHE_DIR", "/tmp/azk-model-cache")),
    )
    parser.add_argument(
        "--session-ttl-seconds",
        type=int,
        default=int(os.getenv("AZK_INFER_SESSION_TTL_SECONDS", str(SESSION_TTL_SECONDS))),
    )
    parser.add_argument(
        "--max-concurrent-inferences",
        type=int,
        default=int(
            os.getenv(
                "AZK_INFER_MAX_CONCURRENT_INFERENCES",
                str(DEFAULT_MAX_CONCURRENT_INFERENCES),
            )
        ),
    )
    parser.add_argument(
        "--max-queue-size",
        type=int,
        default=int(os.getenv("AZK_INFER_MAX_QUEUE_SIZE", str(DEFAULT_MAX_QUEUE_SIZE))),
    )
    parser.add_argument(
        "--queue-wait-timeout-ms",
        type=int,
        default=int(
            os.getenv(
                "AZK_INFER_QUEUE_WAIT_TIMEOUT_MS",
                str(DEFAULT_QUEUE_WAIT_TIMEOUT_MS),
            )
        ),
    )
    return parser.parse_args()


def main() -> None:
    _load_local_env_files()
    args = parse_args()
    engine = InferenceEngine(
        config_path=args.config,
        requested_device=args.device,
        model_cache_dir=args.model_cache_dir,
        session_ttl_seconds=args.session_ttl_seconds,
        max_concurrent_inferences=args.max_concurrent_inferences,
        max_queue_size=args.max_queue_size,
        queue_wait_timeout_ms=args.queue_wait_timeout_ms,
    )

    class Handler(InferenceRequestHandler):
        pass

    Handler.engine = engine
    server = ThreadingHTTPServer((args.host, int(args.port)), Handler)
    print(
        json.dumps(
            {
                "event": "inference_sidecar_started",
                "host": args.host,
                "port": int(args.port),
                "requestedDevice": args.device,
                "resolvedDevice": engine.device,
                "runtimeError": engine.runtime_error,
                "maxConcurrentInferences": engine.max_concurrent_inferences,
                "maxQueueSize": engine.max_queue_size,
                "queueWaitTimeoutSeconds": engine.queue_wait_timeout_seconds,
            }
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
