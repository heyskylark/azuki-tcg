from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import time
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
        OBSERVATION_CTYPE as _OBSERVATION_CTYPE,
        OBSERVATION_STRUCT_SIZE as _OBSERVATION_STRUCT_SIZE,
        observation_to_dict as _observation_to_dict,
    )
except Exception as exc:  # pragma: no cover - startup failure path
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
ACTION_COMPONENT_COUNT = 4
SESSION_TTL_SECONDS = 60 * 15
DEFAULT_MAX_CONCURRENT_INFERENCES = 2
DEFAULT_MAX_QUEUE_SIZE = 8
DEFAULT_QUEUE_WAIT_TIMEOUT_MS = 15000


class InferenceError(Exception):
    pass


class InferenceBusyError(InferenceError):
    pass


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
                "observationByteSize": OBSERVATION_BYTE_SIZE,
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

    def infer(
        self,
        *,
        model_key: str,
        session_key: str,
        observation_b64: str,
        reset_session: bool,
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

        if len(observation_bytes) != OBSERVATION_BYTE_SIZE:
            raise InferenceError(
                f"Invalid observation size. Expected {OBSERVATION_BYTE_SIZE}, got {len(observation_bytes)}"
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

            model = self._get_or_load_model(model_key)
            if reset_session:
                self.end_session(session_key)

            with self._lock:
                session = self._sessions.get(session_key)

            if _OBSERVATION_CTYPE is not None and _observation_to_dict is not None:
                observation_struct = _OBSERVATION_CTYPE.from_buffer_copy(observation_bytes)
                obs_input: Any = _observation_to_dict(observation_struct)
            else:
                # np.frombuffer returns a read-only view; copy to avoid non-writable tensor warnings.
                obs_array = (
                    np.frombuffer(observation_bytes, dtype=np.uint8)
                    .copy()
                    .reshape(1, OBSERVATION_BYTE_SIZE)
                )
                obs_input = torch.from_numpy(obs_array).to(device=self._device)

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
                )

            return [int(v) for v in action_values.tolist()]
        finally:
            if inference_slot_acquired:
                self._leave_inference_slot()
            self._leave_request_slot()

    def end_session(self, session_key: str) -> None:
        with self._lock:
            self._sessions.pop(session_key, None)

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
            import pufferlib.pytorch  # noqa: WPS433

            self._build_policy = build_policy
            self._build_vecenv = build_vecenv
            self._install_tcg_sampler = install_tcg_sampler
            self._load_training_config = load_training_config
            self._load_model_weights = _load_model_weights

            self._install_tcg_sampler()
            self._puffer_sample_logits = pufferlib.pytorch.sample_logits

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

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/infer":
            self._handle_infer()
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

            if not isinstance(model_key, str):
                raise InferenceError("modelKey must be a string")
            if not isinstance(session_key, str):
                raise InferenceError("sessionKey must be a string")
            if not isinstance(observation_b64, str):
                raise InferenceError("observationBase64 must be a string")

            action = self.engine.infer(
                model_key=model_key,
                session_key=session_key,
                observation_b64=observation_b64,
                reset_session=reset_session,
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
            self._send_json(500, {"error": f"Unexpected inference failure: {exc}"})

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
