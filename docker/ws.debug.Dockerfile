FROM oven/bun:1-debian

# Install build + debug dependencies
# Node.js and npm are needed for node-gyp to compile the native N-API addon
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3 \
    git \
    libncurses-dev \
    gdb \
    ccache \
    ninja-build \
    libasan8 \
    libubsan1 \
    nodejs \
    npm \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Sanitizer flags for both the engine and native addon
ENV SANITIZER_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g"
ENV CCACHE_DIR="/ccache"
ENV CC="ccache gcc"
ENV CXX="ccache g++"
ENV FETCHCONTENT_BASE_DIR="/fetchcontent"

# Runtime sanitizer settings (override in compose if needed)
ENV ASAN_OPTIONS="detect_leaks=0,abort_on_error=1,fast_unwind_on_malloc=0"
ENV UBSAN_OPTIONS="print_stacktrace=1"
RUN set -e; \
    LIBASAN_PATH="$(ls /usr/lib/*/libasan.so.* | head -n 1)"; \
    ln -sf "${LIBASAN_PATH}" /usr/lib/libasan.so
ENV LD_PRELOAD="/usr/lib/libasan.so"

# Copy C engine source files (flecs is fetched by CMake via FetchContent)
COPY CMakeLists.txt ./
COPY include/ ./include/
COPY src/ ./src/
COPY tests/ ./tests/

# Build C engine library (Debug + sanitizers)
RUN --mount=type=cache,target=/ccache \
  --mount=type=cache,target=/fetchcontent \
  cmake -G Ninja -S . -B build \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_PYTHON_BINDINGS=OFF \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DFETCHCONTENT_BASE_DIR="${FETCHCONTENT_BASE_DIR}" \
    -DCMAKE_C_FLAGS="${SANITIZER_FLAGS}" \
    -DCMAKE_EXE_LINKER_FLAGS="${SANITIZER_FLAGS}" \
    -DCMAKE_SHARED_LINKER_FLAGS="${SANITIZER_FLAGS}" \
  && cmake --build build -j$(nproc) --target azuki_lib \
  && rm -rf /app/build/_deps \
  && cp -R "${FETCHCONTENT_BASE_DIR}" /app/build/_deps

# Install dependencies
COPY package.json bun.lock ./
COPY packages/backend-core/package.json ./packages/backend-core/
COPY apps/websocket/package.json ./apps/websocket/
RUN bun install

# Copy source code
COPY tsconfig.base.json ./
COPY packages/backend-core ./packages/backend-core
COPY apps/websocket ./apps/websocket

# Build native module (Debug) + TypeScript
RUN --mount=type=cache,target=/ccache \
  --mount=type=cache,target=/root/.cache/node-gyp \
  cd apps/websocket && \
  CFLAGS="${SANITIZER_FLAGS}" CXXFLAGS="${SANITIZER_FLAGS}" LDFLAGS="${SANITIZER_FLAGS}" \
  bun run build:native --debug
RUN bun run --filter '@tcg/backend-core' build && bun run --filter '@azuki/websocket' build

EXPOSE 3001

CMD ["bun", "run", "--filter", "@azuki/websocket", "start"]
