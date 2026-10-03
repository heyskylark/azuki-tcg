# Native training

Run commands below from the repository root. Native training requires the engine
and Python binding built for the active Python environment:

```sh
cmake -S . -B build -DBUILD_PYTHON_BINDINGS=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build --target azuki azuki_puffer_env -j 8
```

### Card catalog corrections

Sundering Strike (`AZK01-127`) is **NORMAL**. Its canonical engine input is
`scripts/azuki-card-defs.jsonl`; regenerate both engine and JAX tables after
changing that catalog:

```sh
.venv/bin/python scripts/generate_card_defs.py scripts/azuki-card-defs.jsonl
.venv/bin/python jax_env/tools/gen_card_tables.py
```

Database migration `0022_fix-azk01-127-element.sql` repairs existing card rows.
The policy JSON and NPZ element feature agree with the corrected catalog.
Normal cards are draftable in every elemental context, subject to the four-copy
limit; the optional leader Normal-composition penalty counts their copies.

Rebuild active curriculum pools after catalog corrections. Keep historical
evaluation reports, checkpoints, and their fingerprints unchanged. Existing
runs are not retrospectively corrected; do not bypass source/pool compatibility
checks to treat them as same-catalog continuations.

## Recurrent PPO updates

With `train.use_rnn=true`, both trainers unroll each PPO trajectory for
`train.bptt_horizon` recorded steps. Each segment starts from its detached rollout
hidden/cell state; gradients flow through subsequent recurrent steps rather than
restarting from a detached state at every observation. The horizon counts stored
rows, not just active-player decisions.

Terminal and truncation flags reset the recurrent state **after** the flagged
observation, matching rollout timing. Draft-to-battle transitions do not reset it.
Masked actor rows can still provide recurrent context for later losses; frozen
opponent rows remain excluded from learner losses. When
`train.recompute_old_logprobs=true`, the frozen PPO reference uses the same
sequence and reset path as the learner.

League rollout histories use global game/seat IDs, not worker-batch row indices.
Learner, current-policy opponent, and frozen-opponent histories therefore remain
isolated when different workers reuse the same batch rows. Terminal resets use
those same global IDs.

The separate delayed terminal/draft-credit updates remain per-decision replays.
They are not full-episode BPTT and are not controlled by `train.bptt_horizon`.

## Additional learner-step budgets

Use `--stop-after-learner-steps N` to stop after a positive number of additional
logged learner steps, measured from the fresh or resumed `trainer.global_step`:

```sh
PYTHONPATH=build/python/src:python/src .venv/bin/python python/src/train.py \
  --config python/config/azuki_prebuilt_curriculum.ini \
  --stop-after-learner-steps 1000000
```

The stop occurs after a complete update and retains the normal final checkpoint.
The final log reports the actual increment and whether the target was reached.
League learner steps count trainable-seat rows, not all sampled rows or just
active battle decisions; one update can overshoot the target.

This flag does not change `train.total_timesteps`, restart the learning-rate
schedule, or extend the configured training horizon. Set that horizon separately;
an earlier configured end can prevent the additional-step target being reached.

## Both-seat prebuilt curriculum

`python/config/azuki_prebuilt_curriculum.ini` enables the native league curriculum.
At each new game, the environment samples whether **both** seats receive complete
prebuilt decks or both seats draft normally. Supplied games begin with normal
opening/mulligan decisions, produce no setup reward, and contribute no draft picks
or draft-credit examples. Ordinary games make all 50 main-deck picks per seat;
forced-prefix settings must remain empty.

Each supplied seat independently samples a gate/leader context uniformly, then a
distinct deck uniformly within that context. Contexts with many tournament lists
therefore do not crowd out sparse contexts. Explicit evaluation schedules bypass
the curriculum.

The preset retains gamma 1, terminal rewards of +/-5, and the uncapped zero-sum
0.05 bonus for completed nonempty gate/leader effects. Other battle shaping is off.
Deck exposure does not add another reward.

### Hand-size telemetry

League curriculum logs report actual native hand counts under
`environment/hand_size/{learner,opponent}/*`. They count active, nonterminal battle
decisions with a non-NOOP legal action, excluding draft and passive rows. Learner
counts follow the trainable-seat mask, including both trainable current/current
seats; opponent counts cover nontrainable seats.

These metrics are emitted per update. For multi-update log windows, sum
`decisions`, `over_capacity_decisions`, and `card_count_sum`, and take the maximum
of `max`. Oversized hands exceed the visible hand-card array's capacity. The
counters read the native `hand_count` scalar rather than the
possibly clipped card array; the model's 30 visible hand slots remain unchanged.

### Deck-pool inputs

Build a production pool after receiving the human lists:

```sh
PYTHONPATH=build/python/src:python/src .venv/bin/python \
  scripts/build_prebuilt_deck_pool.py \
  --additional-pool human-decks.json \
  --output python/config/prebuilt_training_decks.json
```

Then set `env.deck_pool_path` to that output. `--additional-pool` is repeatable.
Inputs use the existing pool JSON format: `schema_version: 1`, a `decks` array,
and per-deck `leader_card_id`, `gate_card_id`, and `cards` entries containing
`card_id` and positive integer `quantity`. Each list includes 50 mains, one leader,
one gate, and ten `IKZ-001` cards.

The builder:

- Requires at least **two distinct legal training lists in each of the 16
  same-element gate/leader contexts**.
- Preserves the original evaluation panel at indices 0 through 17.
- Excludes every evaluation-panel deck signature from training, including
  duplicates reintroduced by supplemental inputs.
- Deduplicates complete card/count signatures and records source provenance.
- Audits and excludes off-element tournament-source training candidates; rejects
  invalid additional training inputs rather than silently repairing them.
- Reports missing coverage instead of generating substitute decks.

Historical evaluation lists are preserved for comparison even when their main
cards disagree with current element metadata. They never enter supplied training.
The output records these reference indices and excluded tournament candidates.

The checked-in preset currently points to the explicitly labeled validation pool
under `train-ablation-1781126582/results/prebuilt_curriculum_v1/`. Its supplement
uses existing documented strategic variants, not the pending optimized human
lists. Replace this input before starting a long run.

### Schedule and resume

| Setting | Preset |
| --- | --- |
| `env.prebuilt_curriculum` | `true` |
| `env.prebuilt_probability` | `0.8` |
| `train.prebuilt_final_probability` | `0.2` |
| `train.prebuilt_anneal_start_battle_decisions` | `5000000` |
| `train.prebuilt_anneal_end_battle_decisions` | `40000000` |

Hold 80% until the first threshold, anneal linearly to 20% at the second, then
hold 20%. The clock counts active, trainable battle decisions: not draft picks,
passive NOOP-only rows, frozen-opponent rows, or terminal bookkeeping. Both
trainable seats count in current-policy self-play. These thresholds are separate
from the sampled-row budget in `train.total_timesteps`.

Probability changes affect new games only; they never replace a draft or battle
already in progress. Serial and multiprocessing vectors share the same control.

Checkpoints and metadata store `prebuilt_battle_decisions`. Resume restores it
before the first worker reset, including model-only resume. Pool content and
schedule parameters are included in compatibility checks. Use
`--resume-checkpoint` and `--resume-load-optimizer` for optimizer continuation.
This preserves curriculum progress, not trajectory-exact worker/environment
state: live games still restart under the existing resume contract.

### Metrics and verification

- `environment/prebuilt/configured_probability`: next-game lottery probability.
- `environment/prebuilt/learner_battle_decisions`: cumulative schedule clock.
- `environment/curriculum/{prebuilt,drafted}_game_fraction`: completed-game mode
  fractions, distinct from the configured probability.
- `environment/curriculum/battle_length` and mode-specific battle lengths:
  completed-game lengths on a common denominator, with zero contribution from
  the opposite mode.
- Existing `draft_episode_credit` metrics contain actual learner draft decisions
  only; supplied games do not trigger the empty-draft-label watchdog.

Focused regression coverage:

```sh
PYTHONPATH=build/python/src:python/src .venv/bin/python -m pytest -q \
  python/tests/test_prebuilt_curriculum.py \
  python/tests/test_native_prebuilt.py \
  python/tests/test_prebuilt_deck_pool.py \
  python/tests/test_train_resume.py \
  python/tests/test_native_vector_actions.py \
  python/tests/test_native_reference_eval.py
```

The implementation's bounded training, optimizer-resume, and supplied-only
verification artifacts are under
`train-ablation-1781126582/results/prebuilt_curriculum_v1/`.