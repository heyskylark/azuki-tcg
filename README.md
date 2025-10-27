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
