#!/usr/bin/env bash
# External throughput guard for the monitored archive-admission continuation.
# It reads JSONL once per minute and never shares the trainer process or GPU.
set -euo pipefail

ROOT=/home/skylark/git/azuki-tcg
SESSION=${SESSION:-azuki_promotion_archive45}
LOG_GLOB=${LOG_GLOB:-'experiments/runlogs/promotionv2_archive45_final_resume*.jsonl'}
STATUS_LOG=${STATUS_LOG:-/tmp/promotionv2_archive45_sps_guard.log}
THRESHOLD=${THRESHOLD:-1300}
WINDOW=${WINDOW:-20}
REQUIRED_STRIKES=${REQUIRED_STRIKES:-2}
POLL_SECONDS=${POLL_SECONDS:-60}

cd "$ROOT"
strikes=0
strike_end_epoch=0
last_checked_epoch=0

while true; do
  pane_dead=$(tmux list-panes -t "=$SESSION" -F '#{pane_dead}' 2>/dev/null | head -n 1 || true)
  if [[ "$pane_dead" != "0" ]]; then
    echo "$(date --iso-8601=seconds) trainer session ended" >> "$STATUS_LOG"
    exit 0
  fi

  log_path=""
  latest_mtime=-1
  while IFS= read -r candidate; do
    candidate_mtime=$(stat -c %Y "$candidate")
    if (( candidate_mtime > latest_mtime )); then
      latest_mtime=$candidate_mtime
      log_path=$candidate
    fi
  done < <(compgen -G "$LOG_GLOB" || true)
  if [[ -z "$log_path" ]]; then
    sleep "$POLL_SECONDS"
    continue
  fi

  stats=$(.venv/bin/python - "$log_path" "$WINDOW" <<'PY'
import json
import statistics
import sys

path = sys.argv[1]
window = int(sys.argv[2])
rows = []
with open(path, encoding="utf-8") as handle:
    for line in handle:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row.get("epoch"), (int, float)) and isinstance(row.get("SPS"), (int, float)):
            rows.append(row)

tail = rows[-window:]
if not tail:
    print("0 0 0 0 0")
else:
    values = [float(row["SPS"]) for row in tail]
    pool = float(tail[-1].get("environment/league/pool_size_active", 0.0))
    print(
        int(tail[-1]["epoch"]),
        len(tail),
        f"{statistics.mean(values):.6f}",
        f"{statistics.median(values):.6f}",
        f"{pool:.0f}",
    )
PY
  )
  read -r epoch count mean median pool <<< "$stats"
  if (( epoch <= last_checked_epoch )); then
    sleep "$POLL_SECONDS"
    continue
  fi
  last_checked_epoch=$epoch
  echo "$(date --iso-8601=seconds) epoch=$epoch n=$count mean=$mean median=$median pool=$pool" >> "$STATUS_LOG"

  if (( count >= WINDOW && epoch >= 1002 )) &&
     awk -v value="$median" -v limit="$THRESHOLD" 'BEGIN { exit !(value < limit) }'; then
    if (( strikes == 0 )); then
      strikes=1
      strike_end_epoch=$epoch
    elif (( epoch - strike_end_epoch >= WINDOW )); then
      ((strikes += 1))
      strike_end_epoch=$epoch
    else
      echo "$(date --iso-8601=seconds) pending: low window overlaps strike ending at epoch=$strike_end_epoch" >> "$STATUS_LOG"
    fi
  else
    strikes=0
    strike_end_epoch=0
  fi

  if (( strikes >= REQUIRED_STRIKES )); then
    echo "$(date --iso-8601=seconds) HALT median SPS remained below $THRESHOLD" >> "$STATUS_LOG"
    tmux send-keys -t "=$SESSION" C-c
    exit 2
  fi
  sleep "$POLL_SECONDS"
done
