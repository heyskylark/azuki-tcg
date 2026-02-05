FROM oven/bun:1-debian AS builder

# Install build dependencies for C engine and native module
# Node.js and npm are needed for node-gyp to compile the native N-API addon
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3 \
    git \
    libncurses-dev \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy C engine source files (flecs is fetched by CMake via FetchContent)
COPY CMakeLists.txt ./
COPY include/ ./include/
COPY src/ ./src/
COPY tests/ ./tests/

# Build C engine library (Release mode for production)
# Disable Python bindings since we don't need them for the WebSocket server
# This also fetches flecs via FetchContent
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=OFF \
    && cmake --build build -j$(nproc) --target azuki_lib

# Install dependencies
COPY package.json bun.lock ./
COPY packages/backend-core/package.json ./packages/backend-core/
COPY apps/websocket/package.json ./apps/websocket/
RUN bun install

# Copy source code
COPY tsconfig.base.json ./
COPY packages/backend-core ./packages/backend-core
COPY apps/websocket ./apps/websocket

# Build native module (requires C engine library and flecs from build/_deps/)
RUN cd apps/websocket && bun run build:native

# Build backend-core and websocket TypeScript
RUN bun run --filter '@tcg/backend-core' build && bun run --filter '@azuki/websocket' build

# Production image
FROM oven/bun:1-debian

# Install runtime dependencies (ncurses for the engine)
RUN apt-get update && apt-get install -y \
    libncurses6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built artifacts
COPY --from=builder /app/package.json ./
COPY --from=builder /app/packages/backend-core/package.json ./packages/backend-core/
COPY --from=builder /app/apps/websocket/package.json ./apps/websocket/

# Install production dependencies only
COPY --from=builder /app/bun.lock ./
RUN bun install --production

# Copy built code
COPY --from=builder /app/packages/backend-core/dist ./packages/backend-core/dist
COPY --from=builder /app/apps/websocket/dist ./apps/websocket/dist
COPY --from=builder /app/apps/websocket/native/build ./apps/websocket/native/build

# Copy the shared libraries needed at runtime (if dynamically linked)
# The native module links against these at runtime
COPY --from=builder /app/build ./build

EXPOSE 3001

CMD ["bun", "apps/websocket/dist/server.js"]
