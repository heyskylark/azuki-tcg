---
name: submit
description: Create or update a pull request for the current branch with a complete summary, repo-specific risk assessment, and concrete validation plan. Use when the user asks to open a PR, update an existing PR, prepare a PR description, or run the repo's standard PR submission workflow for azuki-tcg.
---

# Submit

## Overview

Create or update a GitHub pull request for the current branch against `main` by default, or a user-specified base branch.
Build the PR body from the actual branch diff, existing repo instructions, and validation you really ran.

## Inputs

- Interpret the user request as:
  - `submit` -> create or update a PR against `main`
  - `submit <base-branch>` -> create or update a PR against the specified base branch
- Do not ask follow-up questions unless the intended base branch or validation scope is genuinely ambiguous and cannot be inferred from the branch diff or conversation.

## Required References

Read these before drafting the PR summary:

- `AGENTS.md`
- `CLAUDE.md`

Use `.codex/docs/*` only when the branch touches that documented area and the PR summary or test plan needs the extra context.

## Workflow

### 1. Inspect the branch state

Run these commands to understand the full scope:

- `git status`
- `git diff <base-branch>...HEAD`
- `git log <base-branch>..HEAD --oneline`
- `gh pr list --head $(git branch --show-current) --json number,title,url --state open`

Understand the full branch, not just the last commit. If an open PR already exists for the branch, update it instead of creating a new one.

### 2. Build PR context

Extract from the conversation and diff:

- **Trigger**: What initiated the work
- **Investigation**: What you discovered while exploring the codebase
- **Approach**: What implementation path was chosen and why
- **Key decisions**: Scope cuts, tradeoffs, or design choices that matter to reviewers

### 3. Assess risk

Rate each dimension:

| Dimension | LOW | MEDIUM | HIGH |
|-----------|-----|--------|------|
| Domain Sensitivity | Docs, isolated utilities, narrow tests | Shared gameplay helpers, training utilities, reusable validation | Engine rules, determinism, legal actions, observation/action schema, native bindings, generated card pipeline |
| Blast Radius | 1-2 isolated files | Several files in one subsystem | Cross-layer change touching C engine plus bindings/training/docs/tests |
| Runtime Exposure | Dev-only tooling or docs | Training-only or narrow feature path | Core gameplay loop, turn flow, combat, action masking, RNG, or shared interfaces |
| Reversibility | Clean revert, no generated outputs or interface changes | Revert possible with regenerated artifacts or minor follow-up | Interface/schema/generator changes that require coordinated rebuilds or regenerated outputs |
| Pattern Familiarity | Follows established codebase pattern exactly | Adapts existing pattern to new context | Novel approach with no prior pattern in codebase |
| Data Flow Impact | No interface or data-shape changes | Internal data transformation changes | Changes observation layout, action layout, binding contract, training inputs, or card-definition generation |

Apply these repo-specific rules strictly:

- Domain Sensitivity sets the floor for overall risk.
- Any single `HIGH` dimension means overall risk is `HIGH`.
- Three or more `MEDIUM` dimensions means overall risk is `HIGH`.
- Two `MEDIUM` dimensions means overall risk is `MEDIUM`.
- Otherwise overall risk is `LOW`.

High-sensitivity domains:

- Deterministic engine behavior in `src/` or `include/`
- Turn sequencing, combat resolution, effect timing, or action legality
- `GameState.rng_state` or randomness behavior
- Observation/action schemas, Python binding contract, or RL training inputs
- Generated card definitions or the generator pipeline
- Native module build flow used by downstream services

Translate overall risk into review guidance:

- `LOW` -> Standard review with targeted validation
- `MEDIUM` -> Human review plus explicit validation notes in the PR
- `HIGH` -> Careful human audit of determinism, interfaces, and validation evidence before merge

### 4. Apply repo-specific checks

- If C engine code under `src/` or `include/` changed, call out determinism and action-legality risk explicitly.
- If generated card definitions changed, confirm they came from `scripts/generate_card_defs.py` and do not describe them as hand-edited.
- If observation/action layout, bindings, or training code changed, call out compatibility risk across C, Python, and training configs.
- Never claim verification you did not run.

### 5. Push if needed

- If not already on a feature branch, create one.
- Push with upstream tracking before opening the PR:
  - `git push -u origin <branch-name>`

### 6. Draft the PR body

Use this structure:

```md
## Context
**Trigger**: [What initiated this work]

**Investigation**: [What was discovered]

**Approach**: [What plan was chosen and why]

## Risk Assessment: [LOW / MEDIUM / HIGH]

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Domain Sensitivity | [LOW/MED/HIGH] | [What domain does this touch?] |
| Blast Radius | [LOW/MED/HIGH] | [Files/services affected] |
| Runtime Exposure | [LOW/MED/HIGH] | [How central is the affected runtime path?] |
| Reversibility | [LOW/MED/HIGH] | [Can this be cleanly reverted?] |
| Pattern Familiarity | [LOW/MED/HIGH] | [Established pattern or novel approach?] |
| Data Flow Impact | [LOW/MED/HIGH] | [Does this change how data moves?] |

**Review guidance**: [One sentence summary]

## Changes
<Organized list of specific code changes, grouped by area>

## Test Plan

### Prerequisites
- [ ] Build prerequisites for the touched subsystem are installed
- [ ] Use the repo root as the working directory
- [ ] If Python bindings or training code are involved, include `build/python/src` in `PYTHONPATH`

### Steps to Verify
1. **[Build/command]** - Run the exact command used to validate the change
   - Expected: [what should succeed or what output matters]
2. **[Behavior check]** - Exercise the changed subsystem directly
   - Expected: [what behavior or invariant should hold]

### What to Look For
- [ ] Deterministic behavior remains intact where applicable
- [ ] No new build or test failures in the touched subsystem
- [ ] Generated artifacts, if any, match the source change and were not hand-edited

### Areas Affected
- [ ] [Relevant subsystem, file group, or command path]

## Rollback
<How to revert if something goes wrong>
```

Test plan rules:

- Write steps as if the reader has never seen the repo.
- Use exact commands, files, and subsystems.
- Include expected results for every step.
- Cover the happy path and at least one edge or error check when applicable.
- Prefer existing repo commands such as:
  - `cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug`
  - `cmake --build build -j`
  - `ctest --test-dir build`
  - `PYTHONPATH=build/python/src:python/src:$PYTHONPATH ...`
- If C engine code changed, include a build plus `ctest` unless you explicitly could not run them.
- If only docs changed, say runtime behavior did not change and validation was limited accordingly.
- If validation was not run, say so plainly in the PR summary and test plan.

### 7. Create or update the PR

Write the PR body to a temp file, then use:

- New PR:
  - `gh pr create --title "<title>" --body-file /tmp/pr-body.md`
- Existing PR:
  - `gh pr edit <pr-number> --title "<title>" --body-file /tmp/pr-body.md`

Title prefixes:

- `feat:` for new features
- `fix:` for bug fixes
- `refactor:` for restructuring without behavior change
- `chore:` for config, dependency, or tooling work
- `docs:` for documentation-only changes

Apply a matching risk label after the PR exists:

- `risk:low` -> color `0E8A16`, description `Low risk - eligible for auto-merge after CI`
- `risk:medium` -> color `FBCA04`, description `Medium risk - requires human review`
- `risk:high` -> color `D93F0B`, description `High risk - requires careful audit`

Commands:

```bash
gh label create "risk:<level>" --color "<color>" --description "<description>" --force 2>/dev/null || true
gh pr edit <pr-number> --add-label "risk:<level>"
```

## Output

Return the PR URL and a short summary including the overall risk level.
