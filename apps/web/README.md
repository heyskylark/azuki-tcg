# @azuki/web

Next.js web application for Azuki TCG - API routes and game client.

## Requirements

- **[Bun](https://bun.sh)** - JavaScript/TypeScript runtime and package manager
- **CMake 3.16+** (for building the C engine, if needed)

## Installation

From the repository root:

```bash
bun install
```

## Development

```bash
# From repository root
bun web dev
```

The app starts on `http://localhost:3000`.

## Production

```bash
# Build
bun web build

# Start
bun web start
```

## Linting

```bash
bun web lint
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
