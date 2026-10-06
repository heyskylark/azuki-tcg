# Blind human evaluation: Earth specialist u019500

One model is evaluated: `earth_specialist_u019500.pt` (sha256
`5e5c06c1d8b5114455a3ce9fbc6aca7057dc59adceac0aceb4cb575cee3bf81e`).

## What each match does

The roster `train-specialist/human-eval/earth_specialist_u019500.roster.json` stores an
evaluation plan on the model row (`ai_models.human_evaluation_plan`). When a session is created,
each match gets one of two deck sources. The split is exact: half of the games are PREMADE, and
the order is shuffled from the stored seed.

- **PREMADE**: one of the curated Earth tournament decks, picked uniformly at random:
  `pasadena_cat` (cat's Bobu/Stonehaven, Garden Arena Top 8), `case1_bobu_3rd` (CoreTCG Case 3rd,
  Bobu/Stonehaven) or `pasadena_dkb_goro_36th` (dkb's Goro/Stonehaven). The inference sidecar finds
  the deck by slug in its configured pool and serves it with the same battle deck context it was
  trained with. There is no draft.
- **DRAFT**: the model drafts 50 cards itself. The gate is picked uniformly from the plan's draft
  gates, then the leader uniformly from the leaders valid for that gate (Bobu `STT03-001` or Goro
  `AZK01-123`). **Only Stonehaven `STT03-002` is enabled.** The curated pool now also contains Gate
  of Devotion `AZK01-124` decks (added 2026-10-05 for training), so the sidecar would accept it, but
  u019500 never trained on Devotion and the roster plan keeps draft gates to Stonehaven. Drafts are **sampled** from the model's pick
  distribution: every pick uses an RNG seeded from (`draft_seed`, gate, leader, pick), so a match's
  deck is reproducible from its stored `draft_seed` while different matches get different decks.
  Set `AZK_INFER_DRAFT_ACTION_MODE=argmax` to restore greedy drafting (one fixed deck per gate/leader).

Every match row stores `deck_source`, `premade_deck_slug`, `gate_card_code`, `leader_card_code`,
`draft_seed`, `battle_seed`, `ai_slot` and `starting_player`. All of them can be recomputed from
`human_evaluation_sessions.schedule_seed` plus the stored model and match ids (see
`buildHumanEvaluationModelAssignments`). None of this is shown during the blind phase. After
reveal, the session table shows the source, deck, gate and leader of each match, and each match
review shows "Premade deck" or "Drafted deck".

## Laptop setup

1. **Get the code**
   ```bash
   git fetch origin && git checkout skylark/specialist-earth && git pull
   bun install
   ```
2. **Copy and verify the model** (from the desktop):
   ```bash
   rsync -av <desktop>:~/azuki-models/earth_specialist_u019500/ ~/azuki-models/earth_specialist_u019500/
   (cd ~/azuki-models/earth_specialist_u019500 && sha256sum -c SHA256SUMS)   # macOS: shasum -a 256 -c SHA256SUMS
   ```
3. **Build the engine and Python binding** (the sidecar needs `build/python/src/binding.so`, with
   the regenerated card defs):
   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
   ```
   The Python venv at `.venv` needs torch and numpy, the same as for earlier evals.
4. **Start Postgres, run migrations, start the web app** (runs migrations up to
   `0025_human-eval-deck-source`, including the new-card seed `0024`):
   ```bash
   bun run dev:infra          # docker: db + migrate + web on :3000
   ```
5. **Register the model** (this disables any previously enabled eval models):
   ```bash
   DATABASE_URL=postgres://azuki:azuki@localhost:5432/azuki \
     bun core human-eval:models -- train-specialist/human-eval/earth_specialist_u019500.roster.json
   ```
6. **Start the inference sidecar** on the host (port 8002):
   ```bash
   AZK_INFER_LOCAL_MODEL_ROOT=~/azuki-models/earth_specialist_u019500 \
   AZK_INFER_CONFIG=python/config/azuki_human_eval_earth_specialist.ini \
     bun run dev:ai:eval
   # optional: AZK_INFER_BATTLE_ACTION_MODE=argmax (default: sample)
   # optional: AZK_INFER_DRAFT_ACTION_MODE=argmax (default: sample, seeded by draft_seed)
   curl -s localhost:8002/health        # "status": "ok"
   ```
7. **Start the websocket server.** It rebuilds the native addon with the new cards:
   ```bash
   docker compose up --build --no-deps ws      # Docker builds engine + addon from source
   ```
   Or run it natively: `bun ws build:native && bun ws build`, then start it with Node 18–22
   (uWebSockets.js does not load under Bun or Node ≥ 23):
   `cd apps/websocket && INFERENCE_URL=http://localhost:8002 INFERENCE_TIMEOUT_MS=60000 node dist/server.js`.
   The ws server needs the same `JWT_SECRET` as the web app.
8. **Run a session**: open http://localhost:3000/evaluations, choose your deck, pick 8 or 16
   games, then "Start blind session" → "Start match N". Rate each match as soon as it ends. Reveal
   once everything is played and annotated.
9. **Export results**:
   ```bash
   docker compose exec -T db psql -U azuki -d azuki --csv \
     -f - < train-specialist/human-eval/export_human_eval_matches.sql > earth_eval_results.csv
   ```
   You get one row per match: deck source, premade deck, gate, leader, seeds, outcome, ratings,
   notes and the number of recorded actions. Every decision is also stored in
   `human_evaluation_actions` (observation + legal mask).

## Opponent-chooses cards

Gou the Iron Judge, Hōren of Two Paths and Gin and Tonika (Fatedealer) give the choice to the
opponent of the player who played them. When the AI plays one, the human sees a prompt with the
card's name and both printed modes. The first mode sends CONFIRM and the second sends NOOP. Raiko's
Wrath, Shin uses the same prompt for its own controller. Gurugumi Imitator and the follow-up
choices (sacrifice / discard / Shock target) use the normal selection highlighting. Drafted Earth
decks contain Gou.
