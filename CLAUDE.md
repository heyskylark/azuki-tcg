# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Azuki TCG is a reinforcement learning environment for a custom trading card game, built with:
- **C Core Engine**: Flecs ECS-based game logic with deterministic simulation
- **Python Bindings**: PufferLib/PettingZoo integration for RL training
- **Multi-head Policy**: LSTM-based neural network with 4 action heads

The project trains competitive agents via self-play using PPO on a multi-agent card game environment.

## Development Environment

### Docker Setup (Recommended)

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

docker exec -it puffertank-dev bash
```

### System Dependencies

- **Linux**: `sudo apt install libncurses-dev` (Ubuntu/Debian), `sudo dnf install ncurses-devel` (Fedora), `sudo pacman -S ncurses` (Arch)
- **macOS**: `brew install ncurses`
- **Windows**: Use MSYS2 (`pacman -S mingw-w64-x86_64-ncurses`)

## Build Commands

### Building the C Environment

```bash
# Debug build (from repository root)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j

# Release build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# Build only the Python binding target
cmake -S . -B build && cmake --build build --target azuki_puffer_env
```

### Cleaning Build Artifacts

```bash
cmake --build build --target clean
```

### Running Tests

```bash
# Build and run C unit tests
cmake --build build
ctest --test-dir build

# Or run specific test directly
./build/world_tests
```

## Running RL Training

```bash
# From repository root
PYTHONPATH=build/python/src:python/src:$PYTHONPATH uv run --active python/src/train.py \
  --config python/config/azuki.ini \
  --train.device cuda \
  --train.total-timesteps 1_000_000
```

**Important**: The `PYTHONPATH` must include `build/python/src` to import the compiled `binding` module.

### Training Configuration

Training parameters are configured in `python/config/azuki.ini`. Key parameters:
- `vec.num_envs`: Number of parallel environments
- `train.device`: `cuda` or `cpu`
- `train.total_timesteps`: Total training steps
- `train.batch_size`: PPO batch size
- `train.bptt_horizon`: LSTM truncation length

Command-line overrides use dot notation: `--train.device cuda --vec.num_envs 12`

## Architecture Overview

### C Core Engine (`src/`, `include/`)

The game engine uses **Flecs ECS** (Entity Component System) architecture:

- **`world.c/h`**: ECS world initialization and core game state singleton
- **`azuki_engine.c`**: High-level API wrapping ECS operations
- **`components.h`**: ECS component definitions (cards, players, phases)
- **`systems/`**: ECS systems for game phases (draw, main, combat, end turn)
- **`queries/`**: Reusable ECS queries for common entity lookups
- **`validation/`**: Action validation and legal move enumeration
- **`utils/`**: Phase transitions, card utilities, RNG helpers
- **`generated/`**: Auto-generated card definitions from JSONL

**Key Concepts**:
- Game state is stored in ECS components, primarily in a `GameState` singleton
- Systems process entities in phases (pregame, main, combat, response, end turn)
- The engine is **deterministic** with explicit RNG state for reproducible training
- All game logic runs in C for performance; Python only handles RL orchestration

### Python Bindings (`python/src/`)

- **`binding.c`**: C extension module exporting PufferLib-compatible interface
- **`tcg.h`**: C header defining observation/action layouts
- **`tcg.py`**: PettingZoo AEC wrapper around the C binding
- **`observation.py`**: Observation space encoding utilities
- **`action.py`**: Action space and multi-head action utilities
- **`train.py`**: PufferLib training entrypoint
- **`policy/tcg_policy.py`**: Multi-head LSTM policy network

### Multi-Head Action Space

Actions are represented as 4 discrete heads:
1. **Head 0** (13 actions): Action type (NOOP, PLAY_ENTITY, ATTACK, END_TURN, etc.)
2. **Head 1** (16 values): Hand index / slot index / ability index
3. **Head 2** (8 values): Target kind/slot combinations
4. **Head 3** (8 values): Auxiliary parameters (gate slot, replacement flag)

Legal action masks are provided per-head via `env.infos[agent]["action_mask"]`.

### Turn-Based AEC Flow

- Two players alternate as `agent_selection`
- During **response windows** (e.g., defender declaring blockers), control switches to the non-active player
- The `tcg.py` wrapper handles agent switching and mask propagation

## Card Generation Pipeline

Card definitions are stored as structured data and code-generated:

```bash
# Generate card definitions from JSONL
python3 scripts/generate_card_defs.py path/to/cards.jsonl \
  -o src/generated/card_defs.c \
  --header include/generated/card_defs.h
```

Card schema is documented in `.codex/docs/cards.schema.md`. Generated files are included in the CMake build automatically.

## Testing Strategy

### C Unit Tests (`tests/`)

Located in `tests/test_world.c` and run via CTest:
```bash
cmake --build build
ctest --test-dir build --verbose
```

### Python Integration Tests

```bash
# From python/src/
pytest python/src/test.py -v
```

### Key Invariants to Test

- **Determinism**: Same seed produces identical game outcomes
- **Action Masking**: No illegal actions should pass validation
- **Zone Consistency**: Cards never occupy invalid zones or duplicate slots
- **Resource Tracking**: IKZ (mana) taps/untaps correctly
- **Combat Resolution**: Damage, death, and weapon attachment logic

## Important Development Notes

### Observation Space

Observations are flat `float32` vectors (`AZK_OBS_LEN` elements) containing:
1. Phase one-hot encoding
2. Leader stats (attack, health, keywords) for both players
3. Gate features (tapped, cooldown)
4. Garden/Alley slots (5 each per player): card stats, tapped, cooldown, keywords
5. Hand encoding (top-K cards with type/cost/element)
6. IKZ (mana) pool state
7. Combat stack summary

See `python/src/observation.py` for encoding details and `.codex/docs/azuki-observations.md` for specification.

### Phase System

Game phases follow a state machine:
```
PREGAME_MULLIGAN_P0 → PREGAME_MULLIGAN_P1 → START_OF_TURN → MAIN →
COMBAT_DECLARED → RESPONSE_WINDOW → COMBAT_RESOLVE → END_TURN
```

- **Main Phase**: Active player plays cards, activates abilities, attacks
- **Response Window**: Defender can play response spells or declare defenders
- **Combat Resolve**: Simultaneous damage, death triggers

Systems in `src/systems/` correspond to these phases.

### ECS Component-System Model

When modifying game logic:
1. **Define components** in `include/components/components.h` (structs with game state)
2. **Create systems** in `src/systems/` that query and modify components
3. **Register systems** in `world.c` initialization with correct phase ordering
4. **Update queries** in `src/queries/` if new entity lookups are needed

Flecs uses a single `ecs_world_t*` (aliased as `AzkEngine*`) for all operations.

### PufferLib Integration

The C binding implements the PufferLib `env_` interface:
- `env_init`: Initialize environment with numpy buffer pointers
- `env_step`: Execute one agent action, update observations/rewards/terminals
- `env_reset`: Reset game state to initial conditions
- `env_get`: Return metadata (obs/action sizes, agent count)

The binding does **not** manage Python memory—PufferLib allocates buffers and passes pointers.

## Documentation

Comprehensive design documents are in `.codex/docs/`:
- `azuki-env-tech-spec.md`: Full C engine specification
- `azuki-training-spec.md`: RL training pipeline details
- `azuki-observations.md`: Observation space layout
- `azuki-product-spec.md`: Game rules and mechanics
- `cards.schema.md`: Card definition format

Refer to these for architectural decisions and implementation details.

## Common Gotchas

- **Build path**: Python training requires `build/python/src` in `PYTHONPATH` to import `binding`
- **Seed management**: Both C engine and Python RNG must be seeded for reproducibility
- **Action masking**: Policies must apply masks before argmax to prevent illegal moves
- **Response windows**: Agent selection switches to defender during combat; ensure policy handles role correctly
- **ECS assertions**: Debug builds use `ecs_assert`; enable them when diagnosing engine bugs

## Project Structure Summary

```
azuki-tcg/
├── src/              # C game engine source
├── include/          # C headers
├── tests/            # C unit tests
├── python/
│   ├── src/          # Python bindings, policy, training
│   └── config/       # Training config files
├── scripts/          # Card generation utilities
├── .codex/docs/      # Design documentation
└── build/            # CMake build output (gitignored)
```

When making changes, ensure both C engine tests and Python integration tests pass.
