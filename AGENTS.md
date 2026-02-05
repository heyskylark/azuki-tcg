## AGENTS.md (read first)

When you start any task in this repo, open and follow this file.

### C engine safety + logs
- Do **not** wrap function calls in `ecs_assert(...)`. `ecs_assert` may compile out in release builds and can skip the call entirely.
- If you want an assertion, call the function first, store the result, then use `ecs_assert` only on the **data output** (return value or computed state).
  - Example pattern:
    - `bool ok = do_thing(...);`
    - `ecs_assert(ok, ECS_INVALID_OPERATION, "…");`
- Keep engine changes deterministic: use the engine RNG (`GameState.rng_state`) and avoid non-deterministic sources.
- Do not hand-edit generated card definitions in `src/generated/` or `include/generated/`; regenerate via `scripts/generate_card_defs.py`.

### TypeScript/Bun conventions (from CLAUDE.md)
- **Bun** is used as the JavaScript/TypeScript runtime and package manager.
- **No relative imports** in TS. Use aliases: `@/*` or `@tcg/backend-core/*`.
- When importing backend-core from apps, use `@tcg/backend-core/*` (not `@tcg/backend-core`).
- **No barrel exports** (index files that only re-export).
- Exception: `packages/backend-core/src/drizzle/schemas/index.ts` (required by Drizzle).
- Services are **functional**, not class-based.
- Validate all API request bodies with **Zod `.strict()`**.
- Avoid `as` assertions and `!` non-null assertions; use runtime checks or Zod.
- API routes: wrap handlers with `withErrorHandler`, and `withAuth` for protected routes.
- Custom errors must extend `ApiError` (from `packages/backend-core/src/errors/`).

### Build reminders
- If you change C engine code (`src/`, `include/`), rebuild the engine and the native module before running the web service.

### Database / migrations (Drizzle)
- Schema changes: update schema files then run `bun core db:generate`.
- Never edit generated migration files by hand.
- Use custom migrations only for seed/data transforms (`drizzle-kit generate --custom`).

### Python/RL training
- Training requires `build/python/src` in `PYTHONPATH` to import the compiled `binding` module.
