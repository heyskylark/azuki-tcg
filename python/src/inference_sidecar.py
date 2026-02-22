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


class InferenceError(Exception):
    pass


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
    ) -> None:
        self.config_path = config_path
        self.requested_device = requested_device
        self.session_ttl_seconds = session_ttl_seconds
        self.model_cache_dir = model_cache_dir
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._models: dict[str, ModelRuntime] = {}
        self._sessions: dict[str, SessionState] = {}
        self._runtime_error: str | None = None
        self._device = "cpu"

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

        action_values = sampled_actions.detach().cpu().numpy().astype(np.int32, copy=True).reshape(-1)
        if action_values.shape[0] != ACTION_COMPONENT_COUNT:
            raise InferenceError(
                f"Inference produced invalid action size: {action_values.shape[0]} (expected {ACTION_COMPONENT_COUNT})"
            )

        with self._lock:
            self._sessions[session_key] = SessionState(
                lstm_h=state["lstm_h"],
                lstm_c=state["lstm_c"],
                last_used_at=time.time(),
            )

        return [int(v) for v in action_values.tolist()]

    def end_session(self, session_key: str) -> None:
        with self._lock:
            self._sessions.pop(session_key, None)

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

    def _download_model_from_s3(self, model_key: str) -> Path:
        bucket, key = _parse_s3_uri(model_key)
        local_name = f"{bucket}_{key.replace('/', '_')}"
        local_path = self.model_cache_dir / local_name
        if local_path.exists():
            return local_path

        try:
            import boto3  # noqa: WPS433
        except Exception as exc:  # pragma: no cover - optional dependency
            raise InferenceError(
                "boto3 is required for s3:// model keys but is not installed"
            ) from exc

        s3 = boto3.client("s3")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(local_path))
        return local_path

    def _resolve_model_path(self, model_key: str) -> Path:
        if model_key.startswith("s3://"):
            return self._download_model_from_s3(model_key)

        path = Path(model_key)
        if not path.exists():
            raise InferenceError(f"Model path does not exist: {model_key}")
        return path

    def _get_or_load_model(self, model_key: str):
        with self._lock:
            cached = self._models.get(model_key)
            if cached is not None:
                return cached.model

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = InferenceEngine(
        config_path=args.config,
        requested_device=args.device,
        model_cache_dir=args.model_cache_dir,
        session_ttl_seconds=args.session_ttl_seconds,
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
            }
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
