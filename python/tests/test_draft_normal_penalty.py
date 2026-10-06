import hashlib
import json

import pytest

import draft_normal_penalty as penalty


def test_leader_allowance_and_weight_apply_to_card_copies():
    settings = {121: (0.26, 1.0), 170: (0.64, 0.5)}
    assert penalty.leader_normal_cost(13, 121, settings) == 0
    assert penalty.leader_normal_cost(14, 121, settings) > 0
    assert penalty.leader_normal_cost(32, 170, settings) == 0
    assert penalty.leader_normal_cost(33, 170, settings) > 0
    assert penalty.leader_normal_cost(50, 121, settings) == 1
    assert penalty.leader_normal_cost(50, 170, settings) == 0.5
    with pytest.raises(ValueError):
        penalty.leader_normal_cost(51, 121, settings)


def test_configuration_rejects_missing_leaders_and_catalog_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(penalty, "ROOT", tmp_path)
    metadata_path = tmp_path / "python/config/policy_card_metadata_v1.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(json.dumps({"records": [
        {"card_code": "STT03-001", "card_def_id": 121, "card_type": "LEADER", "element": "EARTH"},
        {"card_code": "AZK01-123", "card_def_id": 170, "card_type": "LEADER", "element": "EARTH"},
        {"card_code": "AZK01-127", "card_def_id": 158, "card_type": "SPELL", "element": "NORMAL"},
    ]}))
    payload = {"schema_id": "azuki.leader_normal_penalty", "schema_version": 1, "main_deck_size": 50,
               "leaders": {"STT03-001": {"max_normal_fraction": 0.26, "weight": 1}},
               "provenance": {"metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest()}}
    config_path = tmp_path / "penalty.json"
    config_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="every supported leader"):
        penalty.load_leader_normal_penalty(config_path)
    payload["leaders"]["AZK01-123"] = {"max_normal_fraction": 0.64, "weight": 1}
    config_path.write_text(json.dumps(payload))
    _, normal_ids = penalty.load_leader_normal_penalty(config_path)
    assert 158 in normal_ids
    metadata_path.write_text(metadata_path.read_text() + "\n")
    with pytest.raises(ValueError, match="catalog differs"):
        penalty.load_leader_normal_penalty(config_path)
