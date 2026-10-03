## AGENTS.md (read first)

When you start any task in this repo, open and follow this file.

### Git autonomy
- Agents have standing approval to create branches, commit completed work, and push as often as useful on any non-protected branch without asking for approval.
- Keep commits coherent and verified. Do not include unrelated user changes in a commit.
- Never push directly to a protected branch, force-push a shared branch, merge a pull request, or bypass branch protection without explicit user approval.

### Autonomous side quests
- Treat worthwhile out-of-scope findings—dead code, stale documentation, a reproducible bug, a confirmed TODO, or a development-workflow failure—as side-quest candidates immediately. Keep them out of the current diff unless the fix is trivial and already inside a file being edited.
- Dispatch a side quest without asking when the finding is verified or reproducible, the work is bounded and independent, and no unresolved product or scope decision is required. If the evidence is incomplete, dispatch an isolated read-only investigation first and authorize edits only after it confirms the issue.
- Before dispatching work, check open and closed pull requests plus active branches for the same path or symbol. Do not duplicate work already in flight; report the existing item instead.
- Use the active agent platform's isolated worktree or session mechanism. Give the executor a standalone prompt with file paths, evidence, acceptance criteria, required verification, and any dependency on unmerged work.
- The isolated executor owns the side quest end to end: reproduce or verify the issue, implement the fix, run focused verification, create and push its own branch, and open a dedicated pull request. It may do this without further approval, but must never merge the pull request or push directly to a protected branch.
- Dispatch confirmed side quests as soon as they are independent, then keep the primary session moving. Report each side quest's pull-request URL and verification separately; never merge or copy its patch into the primary task's branch.
- Repository-external tooling failures are not repository side quests. Report them through the relevant tool or platform feedback channel instead of opening an unrelated repository pull request.

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
