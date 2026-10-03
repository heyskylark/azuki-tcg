# OMP Project Instructions

@../AGENTS.md

## Workspace boundaries

- Keep one deliverable in each top-level Git worktree.
- Never have separate top-level sessions edit the same checkout.
- Use OMP subagents for bounded slices within the current deliverable. Define shared interfaces before parallel implementation.

## Subagent usage

- Use a read-only scout for discovery when the issue, affected files, or call sites are not yet confirmed.
- Request an isolated editing subagent for an independent implementation slice or side quest. Read-only research does not need filesystem isolation.
- Do not isolate tightly coupled slices that must repeatedly edit the same symbols. Keep those changes in one agent or agree on the interface and ownership boundary first.
- Give every subagent a standalone prompt containing the exact target, evidence, constraints, acceptance criteria, required verification, and relevant dependency branches or pull requests.
- Batch genuinely independent work concurrently. Keep the primary session active instead of waiting when other primary-task work remains.
- Require editing subagents to leave the primary checkout and branch untouched and to return concrete verification evidence.

## Autonomous side-quest execution

- The root side-quest policy is standing approval to dispatch isolated OMP background agents or standalone OMP sessions without asking the user.
- Once a side quest is confirmed, instruct its isolated agent to complete the work through focused verification, coherent commits, a push to its dedicated non-protected branch, and a dedicated pull request.
- Side-quest agents may commit and push as often as useful on their non-protected branches without further approval. They must never push to a protected branch, merge their pull request, or copy their patch into the primary deliverable.
- Return each side quest's pull-request URL and verification evidence separately.
- If a side quest depends on unmerged work, provide the dependency branch and pull-request number and require a stacked pull request, or hold dispatch until the dependency is stable.

## Prompt scope

- Use broad multi-agent workflows only for research, reviews, migrations, or implementations with several substantial independent slices.
- Use lighter delegation when the decomposition and ownership boundaries are already clear.
- Use a single agent for narrow, tightly coupled, or mostly sequential work.
