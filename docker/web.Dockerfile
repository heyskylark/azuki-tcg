FROM oven/bun:1-debian

WORKDIR /app

# Install dependencies
COPY package.json bun.lock ./
COPY packages/backend-core/package.json ./packages/backend-core/
COPY apps/web/package.json ./apps/web/
RUN bun install

# Copy source code
COPY tsconfig.base.json ./
COPY packages/backend-core ./packages/backend-core
COPY apps/web ./apps/web

EXPOSE 3000

# Run Next.js in development mode for hot reloading
CMD ["bun", "run", "--filter", "@azuki/web", "dev"]
