import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def league_monitor(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "train-ablation-1781126582"
    monkeypatch.syspath_prepend(str(scripts))
    monitor = importlib.import_module("monitor_strategy_discovery")

    def save(relative, value):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")
        return path

    plan_path = save("experiment.json", {
        "schema_id": "azuki.bptt_league_experiment", "schema_version": 1,
        "comparison": {"arms": {"control": {}, "candidate": {}},
                       "additional_learner_steps_per_arm": 15000000},
        "evaluation": {"initial_common_canary_checkpoint": 1,
                       "midpoint_two_arm_games": 2, "final_two_arm_games": 2},
    })
    save("orchestration_manifest.json", {
        "sha256": {str(plan_path): hashlib.sha256(plan_path.read_bytes()).hexdigest()}
    })
    save("common_parent.json", {"learner_global_step": 100, "update": 10})
    save("campaign_status.json", {"state": "training_control_final", "completed": []})
    for arm in ("control", "candidate"):
        save(f"runs/{arm}/stages/midpoint/result.json", {
            "learner_global_step": 7500110, "update": 100,
        })
        save(f"evaluation/midpoint/fixed/sample/{arm}/shard00.jsonl", {"paired_eval": {"complete": True}})
    save("evaluation/canary/fixed/sample/canary/shard00.jsonl", {"paired_eval": {"complete": True}})
    return monitor, tmp_path, save


def test_league_progress_ignores_partial_metric_until_record_completes(league_monitor):
    monitor, campaign, save = league_monitor
    path = save("runs/control/logs/control_final.jsonl", {"_step": 9000100, "epoch": 120})
    unfinished = json.dumps({"_step": 10000100, "epoch": 130})
    with path.open("a") as handle:
        handle.write(unfinished)
    state = {}
    first = monitor.snapshot(campaign, None, state)
    assert first["arms"][0]["rows"] == 9000000
    with path.open("a") as handle:
        handle.write("\n")
    second = monitor.snapshot(campaign, None, state)
    assert second["arms"][0]["rows"] == 10000000
    assert monitor.snapshot(campaign, None, state)["arms"][0]["rows"] == 10000000


def test_league_completion_requires_terminal_result_and_full_trace_coverage(league_monitor):
    monitor, campaign, save = league_monitor
    state = {}
    current = monitor.snapshot(campaign, None, state)
    assert current["state"] == "running"
    notices = monitor.events(current, state, 100, 1800, True)
    assert "campaign_complete" not in notices
    state["sent_events"] = list(notices)
    state["pending_events"] = {}

    save("campaign_status.json", {"state": "evaluating_midpoint", "completed": []})
    current = monitor.snapshot(campaign, None, state)
    assert current["state"] == "running"
    assert "process_exited" in monitor.events(current, state, 101, 1800, False)

    save("campaign_status.json", {"state": "failed", "completed": []})
    current = monitor.snapshot(campaign, None, state)
    notices = monitor.events(current, state, 102, 1800, False)
    assert "campaign:failed" in notices
    assert "campaign_complete" not in notices

    save("campaign_status.json", {"state": "complete", "completed": ["evaluation/final"]})
    save("result.json", {"status": "complete", "evaluation_games": 5})
    with pytest.raises(ValueError):
        monitor.snapshot(campaign, None, {})
    for arm in ("control", "candidate"):
        save(f"evaluation/final/fixed/sample/{arm}/shard00.jsonl", {"paired_eval": {"complete": True}})
    current = monitor.snapshot(campaign, None, {})
    assert "campaign_complete" in monitor.events(current, {}, 103, 1800, False)
