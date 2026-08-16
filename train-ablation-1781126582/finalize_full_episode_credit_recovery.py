#!/usr/bin/env python3
"""Finalize an interrupted retained-row ladder from two exact metric ranges."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def _load_rows(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    duplicates: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            epoch = row.get("epoch")
            if not isinstance(epoch, (int, float)) or isinstance(epoch, bool):
                continue
            epoch_int = int(epoch)
            if epoch_int in rows:
                duplicates.add(epoch_int)
            rows[epoch_int] = row
    if duplicates:
        raise RuntimeError(
            f"{path} contains duplicate epochs: {sorted(duplicates)[:20]}"
        )
    return rows


def _select_exact_range(
    rows: dict[int, dict],
    *,
    lower: int,
    upper: int,
    label: str,
) -> dict[int, dict]:
    selected = {
        epoch: row for epoch, row in rows.items() if lower <= epoch <= upper
    }
    expected = set(range(lower, upper + 1))
    actual = set(selected)
    if actual != expected:
        raise RuntimeError(
            f"{label} epoch mismatch: missing={sorted(expected - actual)[:20]} "
            f"extra={sorted(actual - expected)[:20]}"
        )
    return selected


def _values(rows: list[dict], key: str) -> list[float]:
    return [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float))
        and not isinstance(row.get(key), bool)
    ]


def _training_summary(
    rows_by_epoch: dict[int, dict],
    *,
    parent_epoch: int,
    target_epoch: int,
    resume_epoch: int,
) -> dict:
    rows = [rows_by_epoch[epoch] for epoch in sorted(rows_by_epoch)]
    intervals = []
    for left, right in zip(rows, rows[1:]):
        elapsed = float(right["uptime"]) - float(left["uptime"])
        steps = float(right["agent_steps"]) - float(left["agent_steps"])
        if elapsed > 0.0 and steps > 0.0:
            intervals.append(steps / elapsed)
    steady = intervals[20:]
    if not steady:
        raise RuntimeError("candidate has no sustained SPS window")

    trained = [
        row
        for row in rows
        if float(row.get("losses/draft_episode_credit_examples", 0.0)) > 0.0
    ]
    draft_picks = _values(rows, "environment/deckbuild/picks")
    prefix = "environment/draft_prefix_outcome/"
    prefix_quartiles = {
        str(index): sum(_values(rows, prefix + key + "_deltas"))
        for index, key in enumerate(
            ("q1_00_12", "q2_13_25", "q3_26_37", "q4_38_50"), start=1
        )
    }
    return {
        "arm": "full_episode_credit",
        "credit_mode": "retained_rows",
        "epochs": len(rows),
        "parent_epoch": parent_epoch,
        "target_epoch": target_epoch,
        "resume_boundary_epoch": resume_epoch,
        "source_rows": resume_epoch - parent_epoch,
        "resumed_rows": target_epoch - resume_epoch,
        "median_sps": statistics.median(steady),
        "tail100_median_sps": statistics.median(steady[-100:]),
        "p10_sps": sorted(steady)[max(0, int(0.1 * (len(steady) - 1)))],
        "timeout_rate_max": max(
            _values(rows, "environment/timeout_truncation_rate"), default=0.0
        ),
        "draft_picks_mean": (
            statistics.mean(draft_picks) if draft_picks else None
        ),
        "captured_records": sum(
            _values(rows, "environment/draft_episode_credit/captured")
        ),
        "labeled_records": sum(
            _values(rows, "environment/draft_episode_credit/labeled")
        ),
        "completed_episodes": sum(
            _values(rows, "environment/draft_episode_credit/completed_episodes")
        ),
        "draw_episodes": sum(
            _values(rows, "environment/draft_episode_credit/draw_episodes")
        ),
        "decisive_episodes": sum(
            _values(rows, "environment/draft_episode_credit/decisive_episodes")
        ),
        "win_episodes": sum(
            _values(rows, "environment/draft_episode_credit/win_episodes")
        ),
        "loss_episodes": sum(
            _values(rows, "environment/draft_episode_credit/loss_episodes")
        ),
        "truncated_records": sum(
            _values(rows, "environment/draft_episode_credit/truncated_records")
        ),
        "incomplete_episodes": sum(
            _values(rows, "environment/draft_episode_credit/incomplete_episodes")
        ),
        "trained_examples": sum(
            _values(rows, "losses/draft_episode_credit_examples")
        ),
        "training_updates": len(trained),
        "gradient_norm_max": max(
            _values(rows, "losses/draft_episode_credit_gradient_norm"),
            default=0.0,
        ),
        "standard_actor_rows": sum(
            _values(rows, "losses/draft_episode_credit_standard_actor_rows")
        ),
        "standard_masked_rows": sum(
            _values(rows, "losses/draft_episode_credit_standard_masked_rows")
        ),
        "aux_wall_seconds": sum(
            _values(rows, "losses/draft_episode_credit_train_seconds")
        ),
        "aux_gpu_seconds": sum(
            _values(rows, "losses/draft_episode_credit_gpu_seconds")
        ),
        "importance_mean_median": (
            statistics.median(
                float(row["losses/draft_episode_credit_importance_mean"])
                for row in trained
            )
            if trained
            else None
        ),
        "clipfrac_max": max(
            (
                float(row["losses/draft_episode_credit_clipfrac"])
                for row in trained
            ),
            default=None,
        ),
        "quartile_examples": {
            str(index): sum(
                _values(rows, f"losses/draft_episode_credit_q{index}_examples")
            )
            for index in range(1, 5)
        },
        "prefix_outcome": {
            "telemetry_rows": len(_values(rows, prefix + "delta_count")),
            "delta_count": sum(_values(rows, prefix + "delta_count")),
            "residual_count": sum(_values(rows, prefix + "residual_count")),
            "completed": sum(_values(rows, prefix + "completed")),
            "truncated": sum(_values(rows, prefix + "truncated")),
            "unsynchronized": sum(_values(rows, prefix + "unsynchronized")),
            "telescope_abs_max": max(
                _values(rows, prefix + "telescope_abs_max"), default=0.0
            ),
            "delta_abs_max": max(
                _values(rows, prefix + "delta_abs_max"), default=0.0
            ),
            "prediction_std_mean": (
                statistics.mean(_values(rows, prefix + "prediction_std"))
                if _values(rows, prefix + "prediction_std")
                else 0.0
            ),
            "inference_seconds": sum(
                _values(rows, prefix + "inference_seconds")
            ),
            "quartile_deltas": prefix_quartiles,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--resume-jsonl", type=Path, required=True)
    parser.add_argument("--parent-epoch", type=int, required=True)
    parser.add_argument("--resume-epoch", type=int, required=True)
    parser.add_argument("--target-epoch", type=int, required=True)
    parser.add_argument("--relative-floor", type=float, default=0.95)
    parser.add_argument("--absolute-floor", type=float, default=1235.0)
    parser.add_argument("--waiver-reason", required=True)
    args = parser.parse_args()

    source_rows_all = _load_rows(args.source_jsonl)
    resume_rows_all = _load_rows(args.resume_jsonl)
    source_rows = _select_exact_range(
        source_rows_all,
        lower=args.parent_epoch + 1,
        upper=args.resume_epoch,
        label="source",
    )
    resume_rows = _select_exact_range(
        resume_rows_all,
        lower=args.resume_epoch + 1,
        upper=args.target_epoch,
        label="resume",
    )
    merged = {**source_rows, **resume_rows}
    if len(merged) != args.target_epoch - args.parent_epoch:
        raise RuntimeError("merged row count does not match the requested ladder")

    arm_root = args.root / "full_episode_credit"
    arm_root.mkdir(parents=True, exist_ok=True)
    merged_path = arm_root / "train.jsonl"
    with merged_path.open("w", encoding="utf-8") as handle:
        for epoch in sorted(merged):
            handle.write(
                json.dumps(merged[epoch], separators=(",", ":"), allow_nan=False)
                + "\n"
            )

    summary = _training_summary(
        merged,
        parent_epoch=args.parent_epoch,
        target_epoch=args.target_epoch,
        resume_epoch=args.resume_epoch,
    )
    (arm_root / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    control = json.loads(
        (args.root / "control/training_summary.json").read_text(encoding="utf-8")
    )
    ratio = float(summary["median_sps"]) / float(control["median_sps"])
    relative_pass = ratio >= args.relative_floor
    absolute_pass = float(summary["median_sps"]) >= args.absolute_floor
    finite_signal = all(
        value is not None and math.isfinite(float(value))
        for value in (
            summary["importance_mean_median"],
            summary["clipfrac_max"],
        )
    )
    integrity = (
        summary["epochs"] == args.target_epoch - args.parent_epoch
        and summary["captured_records"] > 0
        and summary["labeled_records"] > 0
        and summary["completed_episodes"] > 0
        and summary["decisive_episodes"] > 0
        and summary["win_episodes"] > 0
        and summary["loss_episodes"] > 0
        and summary["draw_episodes"] < summary["completed_episodes"]
        and summary["trained_examples"] > 0
        and summary["training_updates"] > 0
        and summary["gradient_norm_max"] > 0
        and summary["standard_actor_rows"] > 0
        and summary["standard_masked_rows"] > 0
        and all(value > 0 for value in summary["quartile_examples"].values())
        and summary["truncated_records"] == 0
        and summary["incomplete_episodes"] == 0
        and summary["timeout_rate_max"] == 0
        and summary["draft_picks_mean"] is not None
        and abs(float(summary["draft_picks_mean"]) - 50.0) < 1e-6
        and finite_signal
    )
    gate = {
        "integrity_pass": integrity,
        "credit_mode": "retained_rows",
        "candidate_arm": "full_episode_credit",
        "performance_pass": relative_pass and absolute_pass,
        "continuation_pass": integrity,
        "throughput_diagnostic_only": True,
        "candidate_over_control_sps": ratio,
        "relative_floor": args.relative_floor,
        "hard_floor": args.absolute_floor,
        "relative_sps_pass": relative_pass,
        "absolute_sps_pass": absolute_pass,
        "relative_sps_waived": False,
        "absolute_sps_waived": False,
        "relative_sps_waiver_reason": "",
        "absolute_sps_waiver_reason": "",
    }
    (args.root / "training_gate.json").write_text(
        json.dumps(gate, indent=2) + "\n", encoding="utf-8"
    )

    discarded = sorted(
        epoch
        for epoch in source_rows_all
        if args.resume_epoch < epoch <= args.target_epoch
    )
    recovery = {
        "schema_version": 1,
        "parent_epoch": args.parent_epoch,
        "resume_epoch": args.resume_epoch,
        "target_epoch": args.target_epoch,
        "source_rows_used": len(source_rows),
        "resume_rows_used": len(resume_rows),
        "source_rows_discarded_after_resume_boundary": discarded,
        "merged_rows": len(merged),
        "integrity_pass": integrity,
        "throughput_diagnostic_only": True,
    }
    (arm_root / "recovery_summary.json").write_text(
        json.dumps(recovery, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"training": summary, "gate": gate}, sort_keys=True))
    if not integrity:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
