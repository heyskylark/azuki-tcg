from __future__ import annotations

import hashlib
import json
import base64
import threading
from typing import Any

import pytest

import inference_sidecar as sidecar


def _valid_payload() -> dict[str, Any]:
    return {
        "modelKey": "models/frozen.pt",
        "sessionKey": "evaluation:match-1",
        "draftSeed": 42,
        "aiSlot": 1,
        "gateCardCode": "GATE-001",
        "leaderCardCode": "LEADER-001",
    }


def test_deck_generate_payload_is_strict_and_normalized() -> None:
    payload = _valid_payload()
    payload["modelKey"] = " models/frozen.pt "
    assert sidecar._validate_deck_generate_payload(payload)["modelKey"] == "models/frozen.pt"

    for field in tuple(_valid_payload()):
        invalid = _valid_payload()
        invalid.pop(field)
        with pytest.raises(sidecar.InferenceError, match="Missing deck generation fields"):
            sidecar._validate_deck_generate_payload(invalid)

    invalid = _valid_payload()
    invalid["extra"] = True
    with pytest.raises(sidecar.InferenceError, match="Unknown deck generation fields"):
        sidecar._validate_deck_generate_payload(invalid)

    for seed in (-1, 2**32, True, "42"):
        invalid = _valid_payload()
        invalid["draftSeed"] = seed
        with pytest.raises(sidecar.InferenceError, match="draftSeed"):
            sidecar._validate_deck_generate_payload(invalid)

    for slot in (-1, 2, True, "1"):
        invalid = _valid_payload()
        invalid["aiSlot"] = slot
        with pytest.raises(sidecar.InferenceError, match="aiSlot"):
            sidecar._validate_deck_generate_payload(invalid)


def test_canonical_deck_hash_preserves_pick_order() -> None:
    ordered = ["CARD-B", "CARD-A", "CARD-C"]
    payload = {
        "gateCardCode": "GATE",
        "leaderCardCode": "LEADER",
        "orderedMainCardCodes": ordered,
    }
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    assert sidecar._canonical_deck_hash("GATE", "LEADER", ordered) == expected
    assert sidecar._canonical_deck_hash("GATE", "LEADER", list(reversed(ordered))) != expected


def test_deck_build_observation_decoder_includes_context() -> None:
    if sidecar._DECKBUILD_OBSERVATION_CTYPE is None:
        pytest.skip("observation ctypes unavailable")
    observation = sidecar._DECKBUILD_OBSERVATION_CTYPE()
    observation.deck_context.mode = 2
    observation.deck_context.gate_card_def_id = 11
    observation.deck_context.leader_card_def_id = 12
    observation.deck_context.main_count = 1
    observation.deck_context.main_card_def_ids[0] = 13
    observation.deck_context.candidate_count = 2
    observation.deck_context.candidate_card_def_ids[0] = 13
    observation.deck_context.candidate_card_def_ids[1] = 14
    decoded = sidecar._decode_observation_bytes(bytes(observation))
    context = decoded["deck_context"]
    assert context["mode"] == 2
    assert context["main_card_def_ids"].dtype.name == "int16"
    assert context["main_card_def_ids"][0] == 13
    assert context["candidate_card_def_ids"].dtype.name == "int16"
    assert context["candidate_card_def_ids"][:2].tolist() == [13, 14]
    assert context["candidate_copy_counts"].dtype.name == "uint8"


def test_native_draft_snapshot_validates_binding_result(monkeypatch) -> None:
    from azk_native import AzukiNativeEnv
    import azk_native

    env = object.__new__(AzukiNativeEnv)
    env._deck_building = True
    env._num_envs = 1
    env._handle = object()
    valid = {"gate": 1, "leader": 2, "main_count": 50, "main": list(range(3, 53))}
    monkeypatch.setattr(
        azk_native.binding,
        "vec_draft_snapshot",
        lambda *_args: valid,
        raising=False,
    )
    assert env.draft_snapshot(1) == valid
    assert env.draft_snapshot(1)["main"] is not valid["main"]

    monkeypatch.setattr(
        azk_native.binding,
        "vec_draft_snapshot",
        lambda *_args: {**valid, "main_count": 49},
        raising=False,
    )
    with pytest.raises(RuntimeError, match="invalid deck data"):
        env.draft_snapshot(1)


def test_deck_generate_handler_maps_expected_errors() -> None:
    handler = object.__new__(sidecar.InferenceRequestHandler)
    handler._read_json_body = _valid_payload
    responses: list[tuple[int, dict[str, Any]]] = []
    handler._send_json = lambda status, payload: responses.append((status, payload))

    class Engine:
        def generate_deck(self, _payload: dict[str, Any]) -> dict[str, Any]:
            raise sidecar.InferenceError("invalid context")

    handler.engine = Engine()
    handler._handle_generate_deck()
    assert responses == [(400, {"error": "invalid context"})]

    responses.clear()

    class BusyEngine:
        def generate_deck(self, _payload: dict[str, Any]) -> dict[str, Any]:
            raise sidecar.InferenceBusyError("queue full")

    handler.engine = BusyEngine()
    handler._handle_generate_deck()
    assert responses == [(503, {"error": "queue full"})]


def test_post_auth_is_optional_and_uses_bearer_secret(monkeypatch) -> None:
    handler = object.__new__(sidecar.InferenceRequestHandler)
    handler.path = "/infer"
    handler.headers = {}
    responses: list[tuple[int, dict[str, Any]]] = []
    calls: list[str] = []
    handler._send_json = lambda status, payload: responses.append((status, payload))
    handler._handle_infer = lambda: calls.append("infer")

    monkeypatch.delenv("AZK_INFER_SHARED_SECRET", raising=False)
    handler.do_POST()
    assert calls == ["infer"]

    calls.clear()
    monkeypatch.setenv("AZK_INFER_SHARED_SECRET", "private-value")
    handler.do_POST()
    assert calls == []
    assert responses[-1] == (401, {"error": "Unauthorized"})

    handler.headers = {"Authorization": "Bearer private-value"}
    handler.do_POST()
    assert calls == ["infer"]


def test_session_status_payload_is_strict() -> None:
    handler = object.__new__(sidecar.InferenceRequestHandler)
    responses: list[tuple[int, dict[str, Any]]] = []
    handler._send_json = lambda status, payload: responses.append((status, payload))

    class Engine:
        def session_active(self, session_key: str) -> bool:
            return session_key == "active-session"

    handler.engine = Engine()
    handler._read_json_body = lambda: {"sessionKey": "active-session"}
    handler._handle_session_status()
    assert responses == [(200, {"active": True})]

    responses.clear()
    handler._read_json_body = lambda: {"sessionKey": "active-session", "extra": True}
    handler._handle_session_status()
    assert responses[0][0] == 400


def test_require_session_rejects_missing_state_before_model_load() -> None:
    engine = object.__new__(sidecar.InferenceEngine)
    engine._runtime_error = None
    engine._sessions = {}
    engine._lock = threading.Lock()
    engine._try_enter_request_queue = lambda: True
    engine._enter_inference_slot = lambda: True
    engine._promote_queued_request_to_inflight = lambda: None
    engine._evict_stale_sessions = lambda: None
    engine._leave_inference_slot = lambda: None
    engine._leave_request_slot = lambda: None
    engine._get_or_load_model = lambda _key: pytest.fail(
        "model must not load for a missing required session"
    )
    observation = base64.b64encode(bytes(sidecar.OBSERVATION_BYTE_SIZE)).decode()
    with pytest.raises(sidecar.InferenceError, match="not active"):
        engine.infer(
            model_key="model",
            session_key="missing",
            observation_b64=observation,
            reset_session=False,
            require_session=True,
        )


def test_infer_handler_requires_boolean_require_session() -> None:
    handler = object.__new__(sidecar.InferenceRequestHandler)
    responses: list[tuple[int, dict[str, Any]]] = []
    handler._send_json = lambda status, payload: responses.append((status, payload))
    handler._read_json_body = lambda: {
        "modelKey": "model",
        "sessionKey": "session",
        "observationBase64": "AA==",
        "requireSession": "true",
    }
    handler.engine = object()
    handler._handle_infer()
    assert responses == [(400, {"error": "requireSession must be a boolean"})]
