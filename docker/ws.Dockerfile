FROM node:20-slim

WORKDIR /app

# Install dependencies
COPY package.json yarn.lock ./
COPY packages/backend-core/package.json ./packages/backend-core/
COPY apps/websocket/package.json ./apps/websocket/
RUN yarn install --frozen-lockfile

# Copy source code
COPY tsconfig.base.json ./
COPY packages/backend-core ./packages/backend-core
COPY apps/websocket ./apps/websocket

# Build backend-core first, then websocket
RUN yarn core build && yarn ws build

EXPOSE 3001

CMD ["yarn", "ws", "start"]
