# PufferTank for RL

```bash
PROJECT=~/git/azuki-tcg
PUFFER=~/git/rl/SkyPufferLib

docker run -d --name puffertank-dev \
  --gpus all \
  --network host \
  --ipc host \
  --restart unless-stopped \
  -v "$PROJECT":/workspace \
  -v "$PUFFER":/ext/SkyPufferLib \
  -v "$HOME/.cache/pip":/root/.cache/pip \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HOME/.cache/npm":/root/.npm \
  -w /workspace \
  pufferai/puffertank:3.0 bash -lc "sleep infinity"
```

```bash
docker exec -it puffertank-dev bash
```

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
