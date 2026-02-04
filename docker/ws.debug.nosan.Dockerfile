FROM node:22-slim

# Install build + debug dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3 \
    git \
    libncurses-dev \
    gdb \
    ccache \
    ninja-build \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV CCACHE_DIR="/ccache"
ENV CC="ccache gcc"
ENV CXX="ccache g++"
ENV FETCHCONTENT_BASE_DIR="/fetchcontent"
ENV YARN_REGISTRY="https://registry.npmjs.org"
ENV YARN_NETWORK_TIMEOUT="600000"
ENV YARN_NETWORK_CONCURRENCY="1"

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

# Install Node.js dependencies
COPY package.json yarn.lock ./
COPY packages/backend-core/package.json ./packages/backend-core/
COPY apps/websocket/package.json ./apps/websocket/
RUN --mount=type=cache,target=/root/.cache/yarn \
  bash -lc 'set -e; \
    for i in 1 2 3; do \
      yarn config set registry "${YARN_REGISTRY}"; \
      yarn install --frozen-lockfile --network-timeout "${YARN_NETWORK_TIMEOUT}" --network-concurrency "${YARN_NETWORK_CONCURRENCY}" && exit 0; \
      echo "yarn install failed (attempt ${i}); clearing cache and retrying..."; \
      yarn cache clean; \
      sleep 2; \
    done; \
    exit 1'

# Copy source code
COPY tsconfig.base.json ./
COPY packages/backend-core ./packages/backend-core
COPY apps/websocket ./apps/websocket

# Build native module (Debug) + TypeScript
RUN --mount=type=cache,target=/ccache \
  --mount=type=cache,target=/root/.cache/node-gyp \
  cd apps/websocket && yarn build:native --debug
RUN yarn core build && yarn ws build

EXPOSE 3001

CMD ["yarn", "ws", "start"]
