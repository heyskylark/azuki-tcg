---
name: commit-main
description: Inspect the current working tree, stage the intended files, write a relevant git commit message from the actual diff, and create a local commit instead of opening a pull request. Use when the user asks to commit current work directly, wants a commit-ready message for the touched files, or wants a direct-commit workflow rather than the repo's PR submission flow.
---

# Commit Main

## Overview

Create a local git commit for the current task using a message derived from the actual staged diff.
Use this skill when the user wants a direct commit instead of the `submit` pull-request workflow.

## Inputs

- Interpret the user request as:
  - `commit-main` -> inspect current changes, stage the intended files, write a relevant commit message, and create a commit on the current branch
  - `commit-main <scope>` -> limit the commit to the files or subsystem implied by the request
- Do not switch branches, merge, or push unless the user explicitly asks.
- If already on `main`, commit on `main`.
- If on another branch, commit there and report it plainly rather than changing branch topology as part of this skill.

## Required References

Read these before staging or committing:

- `AGENTS.md`
- `CLAUDE.md`

Read `.codex/docs/*` only when the changed area needs extra repo-specific context to understand the diff or validation.

## Workflow

### 1. Inspect the worktree

Run these commands to understand the current state:

- `git branch --show-current`
- `git status --short`
- `git diff --stat`
- `git diff`
- `git diff --cached`

If there are no changes, do not create an empty commit unless the user explicitly asks for one.

### 2. Determine commit scope

Use the conversation and the diff to decide which files belong in the commit.

- Do not assume every modified file belongs in scope.
- Leave unrelated user changes untouched.
- Stage exact paths when possible: `git add <path>...`
- Use broad staging such as `git add -A` only when the entire worktree is clearly in scope.
- Avoid interactive git flows. If a single file mixes in-scope and out-of-scope edits and cannot be separated safely with non-interactive commands, stop and ask before committing.

Re-check the staged snapshot with:

- `git diff --cached --stat`
- `git diff --cached`

### 3. Validate appropriately

Run validation proportional to the touched files.

- Prefer the validation already performed during the task.
- Never claim validation you did not run.
- If C engine code under `src/` or `include/` changed, rebuild the engine and native module before service validation as required by `AGENTS.md`.
- If generated card definitions changed, confirm they were regenerated via `scripts/generate_card_defs.py`.
- If no validation was run, say so plainly in the final response.

### 4. Write the commit message

Base the message on the staged diff, not just filenames.

- Keep the subject concise and specific.
- Prefer a single-line subject when that is enough.
- Add a body only when the change spans multiple concerns, generated outputs, or notable risk.
- Use a conventional prefix when it genuinely improves clarity, for example:
  - `feat:` for new behavior
  - `fix:` for bug fixes
  - `refactor:` for structure changes without behavior change
  - `chore:` for tooling or maintenance
  - `docs:` for documentation-only updates
  - `test:` for test-only changes
- Match the repository's existing history when possible, but optimize for accuracy over style imitation.

Good subjects:

- `fix: rebuild legal action mask after combat cleanup`
- `refactor card ability target resolution`
- `docs: add training environment setup notes`

Weak subjects:

- `update files`
- `misc changes`
- `wip`

### 5. Create the commit

Commit only after the staged diff and message both match the intended scope.

- `git commit -m "<subject>"`
- `git commit -m "<subject>" -m "<body>"` when a body is warranted

Do not amend prior commits unless the user explicitly asks.
Do not push as part of this skill unless the user explicitly asks.

## Output

Return:

- the branch name
- the new commit hash
- the exact commit message
- the files included in the commit
- the validation that was run, or a plain statement that validation was not run
- any leftover modified files that were intentionally excluded
