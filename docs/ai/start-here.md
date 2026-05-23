# AI Memory Start Here

This directory is the persistent AI project memory for Codex sessions.

## Read order for a new Codex session

1. `../../AGENTS.md`
2. `context.md`
3. `task-board.md`
4. `handoff.md`
5. `chat-index.md`
6. `chat-roles.md`
7. `decision-log.md` if the task affects architecture, dataset format, training, evaluation, or documentation.
8. `sessions/` (read the most recent session note for your role)
9. `experiments/` (read notes relevant to the run you are touching)

## Core files

- `context.md`: consolidated current project state.
- `task-board.md`: backlog, in-progress work, completed work.
- `handoff.md`: immediate continuation notes for the next chat.
- `decision-log.md`: durable technical decisions.
- `chat-index.md`: logical Codex session registry.
- `chat-roles.md`: ownership model for chat roles.
- `tests-and-results.md`: latest validation and experiment outcomes.

## Operational notes

- Sandbox is currently unavailable on the server (Codex bubblewrap / `bwrap` failures). Prefer non-sandboxed commands and minimal, targeted edits for memory files until further notice.

## Directories

- `sessions/`: chronological notes from individual Codex sessions.
- `experiments/`: reproducible training/evaluation/test records.
- `runbooks/`: stable operational procedures.

## Current project

See `context.md`.
