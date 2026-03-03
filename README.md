# PufferTank for RL

```bash
PROJECT=~/git/azuki-tcg
PUFFER=~/git/rl/SkyPufferLib

docker run -d --name puffertank-dev \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  --network host \
  --ipc host \
  --security-opt label=disable \
  --security-opt seccomp=unconfined \
  --restart unless-stopped \
  -v "$PROJECT":/workspace \
  -v "$PUFFER":/ext/SkyPufferLib \
  -v "$HOME/.cache/pip":/root/.cache/pip \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HOME/.cache/npm":/root/.npm \
  -w /workspace \
  pufferai/puffertank:3.0 \
  bash -lc "/workspace/scripts/wait_for_cuda.sh && exec sleep infinity"
```

```bash
docker exec -it puffertank-dev bash
```

`scripts/wait_for_cuda.sh` gates startup on `cuInit(0) == 0`. With `--restart unless-stopped`, reboot-time CUDA races auto-retry instead of leaving a broken container running.
This launch path intentionally uses `--runtime nvidia` + `NVIDIA_VISIBLE_DEVICES` instead of `--gpus all` to avoid stale CDI mount entries after host driver updates.

## Building C Env

```bash
# In <root-directory> 
cmake -S . -B build && cmake --build build --target azuki_puffer_env
```

## Running Training

```bash
# In python/src/
PYTHONPATH=build/python/src:python/src:$PYTHONPATH uv run --active python/src/train.py --config python/config/azuki.ini --train.device cuda --train.total-timesteps 1_000_000
```

```bash
WANDB_API_KEY=$WANDB_KEY WANDB_ENTITY=heyskylark-self-affiliated \
  PYTHONPATH=build/python/src:python/src:$PYTHONPATH uv run --active python/src/train.py \
    --config python/config/azuki.ini --wandb --wandb-project azuki-tcg --wandb-group azuki \
    --tag tcg-mvp --train.device cuda --train.total-timesteps 1_000_000
```

## Websocket + AI Sidecar (Dev)

Run these from the repo root:

```bash
# core app services (no AI sidecar)
bun run dev:infra

# AI sidecar in Docker
bun run dev:ai

# AI sidecar on host (local Python process)
bun run dev:ai:local
```

Check sidecar health:

```bash
curl http://localhost:8002/health
```

Expected: `status` is `"ok"` and `runtimeError` is `null`.

### Local sidecar prerequisites

If you use `bun run dev:ai:local`, build Python bindings and install Python deps first (repo root):

```bash
cmake -S . -B build -DBUILD_PYTHON_BINDINGS=ON
cmake --build build -j
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy pufferlib pettingzoo gymnasium boto3
```

`dev:ai:local` now auto-loads env files in this order:
- `.env`
- `.env.local` (overrides `.env`)

Shell-exported env vars still take precedence over both files.

### INFERENCE_URL notes

- If websocket runs in Docker and sidecar runs locally, set websocket `INFERENCE_URL` to:
  - `http://host.docker.internal:8002`
- If both websocket and sidecar run locally, use:
  - `http://localhost:8002`

### AI model registry + model key format

- Room creation now loads AI options from the `ai_models` table.
- The room-create dropdown shows models with `status = ENABLED`.
- Each `ai_models.model_key` should be an S3 object key, e.g. `model_009646.pt`.
- The inference sidecar resolves model keys as:
  - `${AZK_INFER_S3_MODEL_PREFIX}${model_key}`
- Example:
  - `AZK_INFER_S3_MODEL_PREFIX=s3://azuki-tcg-models/`
  - `model_key=model_009646.pt`
  - resolved path: `s3://azuki-tcg-models/model_009646.pt`

### S3 credentials for sidecar

The inference sidecar uses boto3 and supports standard AWS credential resolution.

- Static credentials (optional):
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_SESSION_TOKEN` (optional)
- Profile-based credentials (optional):
  - `AWS_PROFILE`
- Region selection (optional):
  - `AZK_AWS_REGION` (preferred)
  - or `AWS_REGION`
  - or `AWS_DEFAULT_REGION`
- Custom S3-compatible endpoint (optional):
  - `AZK_AWS_S3_ENDPOINT_URL`
- Required model prefix:
  - `AZK_INFER_S3_MODEL_PREFIX` (must be `s3://...`)
- Inference concurrency controls (optional):
  - `AZK_INFER_MAX_CONCURRENT_INFERENCES` (default `2`)
  - `AZK_INFER_MAX_QUEUE_SIZE` (default `8`)
  - `AZK_INFER_QUEUE_WAIT_TIMEOUT_MS` (default `15000`)

## Rendering & Playback

- The Python env now supports `render(mode="ansi")` and exposes a playback helper at `python/src/playback.py` that loads a checkpoint, rolls out single-env self-play, and emits text frames you can pipe to `ttyrec`/`asciinema` or convert with `ffmpeg`. Example:  
  `PYTHONPATH=build/python/src:python/src:$PYTHONPATH python python/src/playback.py --checkpoint <path/to/model.pt> --output renders/epoch_010000.ansi --episodes 1 --max-steps 200`
- Training can optionally trigger playback automatically: `--render-playback-interval N` runs a short render every N epochs, `--render-playback-final` runs once at the end, `--render-playback-dir` saves frames instead of spamming stdout, and `--render-playback-device` lets you offload playback to CPU.
- To avoid bloating storage with random early games, start rendering once the policy stabilizes (e.g., after 70–80% of planned epochs) and keep intervals coarse (every 25–50 epochs) with `--render-playback-steps` around 200 so each capture stays small.

# Dependencies

- **Linux**: Install development headers via your package manager (Ubuntu/Debian `sudo apt install libncurses-dev`, Fedora `sudo dnf install ncurses-devel`, Arch-based `sudo pacman -S ncurses` or `yay -S ncurses`).
- **macOS**: `brew install ncurses`.
- **Windows**: Use MSYS2 (`pacman -S mingw-w64-x86_64-ncurses`) or another curses-compatible port such as PDCurses when targeting MSVC.

# How to Build 

```bash
cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Debug && cmake --build ./build -j
```

## Clean

```bash
cmake --build ./build --target clean
```

## Release Build

```bash
cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release
```

## Card Definition Generation

Use `scripts/generate_card_defs.py` to convert a JSONL list of card definitions into Flecs-friendly C code and a companion header.

```bash
python3 scripts/generate_card_defs.py path/to/cards.jsonl \
  -o src/generated/card_defs.c \
  --header include/generated/card_defs.h
```

Each line of the JSONL file must describe a single card object containing the card's base stats, IKZ cost, element, type, and other metadata. The script validates the shape of each record before emitting both the source (lookup tables) and header (enums, structs, and accessors). Override `-o` or `--header` as needed to place the generated files elsewhere.

# Resources

- [Dota 2 with Large Scale Deep Reinforcement Learning](https://arxiv.org/pdf/1912.06680)

- [Cardsformer: Grounding Language to Learna Generalizable Policy in Hearthstone](https://www.researchgate.net/publication/374299909_Cardsformer_Grounding_Language_to_Learn_a_Generalizable_Policy_in_Hearthstone)

- [Learning With Generalised Card Representations for “Magic: The Gathering”](https://arxiv.org/html/2407.05879v1?utm_source=chatgpt.com)
