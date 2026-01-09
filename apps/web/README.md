# @azuki/web

Next.js web application for Azuki TCG - API routes and game client.

## Requirements

- **Node.js 20 LTS**
- Yarn 1.x

```bash
# Check Node version
node --version  # Should be v20.x.x

# Switch to Node 20 using nvm
nvm use 20
```

## Installation

From the repository root:

```bash
yarn install
```

Or install this package specifically:

```bash
yarn workspace @azuki/web install
```

## Development

```bash
# From repository root
yarn web dev

# Or from this directory
yarn dev
```

The app starts on `http://localhost:3000`.

## Production

```bash
# Build
yarn web build

# Start
yarn web start
```

## Linting

```bash
yarn web lint
```

## Project Structure

```
src/
└── app/
    ├── layout.tsx    # Root layout
    ├── page.tsx      # Home page
    ├── globals.css   # Global styles (Tailwind)
    └── api/          # API routes (future)
```

## Tech Stack

- **Next.js 15** - React framework with App Router
- **React 19** - UI library
- **Tailwind CSS 4** - Utility-first CSS
- **TypeScript** - Type safety
- **@azuki/shared** - Shared types and constants

## Import Conventions

Use absolute path aliases (no relative imports):

```typescript
// Internal imports use @/
import { Button } from "@/components/ui/Button";
import { useAuth } from "@/hooks/useAuth";

// Shared package imports use @shared/
import { RoomStatus, CardType } from "@shared/types";
import { DECK_SIZE } from "@shared/constants";
```

## API Routes (Planned)

See `.claude/docs/web-service.md` for the full API specification:

- `/api/auth/*` - Authentication (register, login, refresh)
- `/api/users/*` - User management
- `/api/decks/*` - Deck CRUD operations
- `/api/rooms/*` - Game room management
- `/api/cards/*` - Card database queries
- `/api/matches/*` - Match history
