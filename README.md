# How to Build 

```bash
cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Debug
cmake --build ./build -j
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

Use `scripts/generate_card_defs.py` to convert a JSONL list of card definitions into Flecs-friendly C code.

```bash
python3 scripts/generate_card_defs.py path/to/cards.jsonl -o src/generated/card_defs.c
```

Each line of the JSONL file must describe a single card object containing the card's base stats, IKZ cost, element, type, and other metadata. The script validates the shape of each record before emitting the generated C source.
