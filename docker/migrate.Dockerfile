FROM node:22-alpine

WORKDIR /app

# Install dependencies
COPY package.json yarn.lock ./
COPY packages/backend-core/package.json ./packages/backend-core/
RUN yarn install --frozen-lockfile

# Copy source code
COPY tsconfig.base.json ./
COPY packages/backend-core ./packages/backend-core

# Build backend-core
RUN yarn core build

# Run migrations
CMD ["yarn", "core", "db:migrate"]
