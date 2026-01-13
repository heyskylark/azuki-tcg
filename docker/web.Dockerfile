FROM node:20-alpine

WORKDIR /app

# Install dependencies
COPY package.json yarn.lock ./
COPY packages/backend-core/package.json ./packages/backend-core/
COPY apps/web/package.json ./apps/web/
RUN yarn install --frozen-lockfile

# Copy source code
COPY tsconfig.base.json ./
COPY packages/backend-core ./packages/backend-core
COPY apps/web ./apps/web

EXPOSE 3000

# Run Next.js in development mode for hot reloading
CMD ["yarn", "web", "dev"]
