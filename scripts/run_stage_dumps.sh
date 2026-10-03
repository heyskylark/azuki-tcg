#!/bin/bash
# Detached stage-dump runner: 3 checkpoint deck dumps, sequential, survives shell exit.
cd /home/skylark/git/azuki-tcg
setsid nohup bash -c '
run_dump() {
  PYTHONPATH=build/python/src:python/src .venv/bin/python python/src/dump_checkpoint_decks.py \
    --config python/config/azuki_deckbuild_3090.ini --checkpoint "$1" --episodes 60 \
    --out-dir "experiments/stage_dumps/$2" --device cuda --max-steps 700 > "/tmp/dump_$2.log" 2>&1
  echo "$2 exit=$?" >> /tmp/stage_dumps_status.log
}
rm -f /tmp/stage_dumps_status.log
run_dump "experiments/azuki_local_base-deckbuild-02_178113787430/model_azuki_local_000250.pt" stage250
run_dump "experiments/azuki_local_base-deckbuild-02_178113787430/model_azuki_local_000750.pt" stage750
run_dump "experiments/azuki_local_base-deckbuild-02_178117983197/model_azuki_local_001500.pt" stage1500
echo ALL_DONE >> /tmp/stage_dumps_status.log
' > /dev/null 2>&1 < /dev/null &
echo "stage dumps detached pid=$!"
