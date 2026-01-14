FROM node:20-slim AS builder

# Install build dependencies for C engine and native module
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3 \
    git \
    libncurses-dev \
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

# Install Node.js dependencies
COPY package.json yarn.lock ./
COPY packages/backend-core/package.json ./packages/backend-core/
COPY apps/websocket/package.json ./apps/websocket/
RUN yarn install --frozen-lockfile

# Copy source code
COPY tsconfig.base.json ./
COPY packages/backend-core ./packages/backend-core
COPY apps/websocket ./apps/websocket

# Build native module (requires C engine library and flecs from build/_deps/)
RUN cd apps/websocket && yarn build:native

# Build backend-core and websocket TypeScript
RUN yarn core build && yarn ws build

# Production image
FROM node:20-slim

# Install runtime dependencies (ncurses for the engine)
RUN apt-get update && apt-get install -y \
    libncurses6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built artifacts
COPY --from=builder /app/package.json /app/yarn.lock ./
COPY --from=builder /app/packages/backend-core/package.json ./packages/backend-core/
COPY --from=builder /app/apps/websocket/package.json ./apps/websocket/

# Install production dependencies only
RUN yarn install --frozen-lockfile --production

# Copy built code
COPY --from=builder /app/packages/backend-core/dist ./packages/backend-core/dist
COPY --from=builder /app/apps/websocket/dist ./apps/websocket/dist
COPY --from=builder /app/apps/websocket/native/build ./apps/websocket/native/build

# Copy the shared libraries needed at runtime (if dynamically linked)
# The native module links against these at runtime
COPY --from=builder /app/build ./build

EXPOSE 3001

CMD ["yarn", "ws", "start"]
