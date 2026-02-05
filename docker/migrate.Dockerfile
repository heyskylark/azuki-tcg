FROM oven/bun:1-debian

WORKDIR /app

# Install dependencies
COPY package.json bun.lock ./
COPY packages/backend-core/package.json ./packages/backend-core/
RUN bun install

# Copy source code
COPY tsconfig.base.json ./
COPY packages/backend-core ./packages/backend-core

# Build backend-core
RUN bun run --filter '@tcg/backend-core' build

# Run migrations
CMD ["bun", "run", "--filter", "@tcg/backend-core", "db:migrate"]
