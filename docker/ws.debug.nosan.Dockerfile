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
    nodejs \
    npm \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV CCACHE_DIR="/ccache"
ENV CC="ccache gcc"
ENV CXX="ccache g++"
ENV FETCHCONTENT_BASE_DIR="/fetchcontent"

# Copy C engine source files (flecs is fetched by CMake via FetchContent)
COPY CMakeLists.txt ./
COPY include/ ./include/
COPY src/ ./src/
COPY tests/ ./tests/

# Build C engine library (Debug symbols, no sanitizers)
RUN --mount=type=cache,target=/ccache \
  --mount=type=cache,target=/fetchcontent \
  cmake -G Ninja -S . -B build \
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_PYTHON_BINDINGS=OFF \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DFETCHCONTENT_BASE_DIR="${FETCHCONTENT_BASE_DIR}" \
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
  cd apps/websocket && bun run build:native --debug
RUN bun run --filter '@tcg/backend-core' build && bun run --filter '@azuki/websocket' build

EXPOSE 3001

CMD ["bun", "run", "--filter", "@azuki/websocket", "start"]
