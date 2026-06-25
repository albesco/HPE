# AI Memory Start Here

This directory is the persistent AI project memory for Codex sessions.

## Read order for a new Codex session

1. `../../AGENTS.md`
2. `context.md`
3. `task-board.md`
4. `chat-index.md`
5. `chat-roles.md`
6. `handoff.md` (short compatibility pointer only; not a source of truth)
7. `decision-log.md` if the task affects architecture, dataset format, training, evaluation, or documentation.
8. `tests-and-results.md` if the task affects dataset preparation, training, evaluation, metrics, or reproducibility.
9. `sessions/` only for historical detail relevant to the current task.
10. `experiments/` and `runbooks/` only when relevant to the run or workflow being touched.

## Core files

- `context.md`: consolidated current project state.
- `task-board.md`: backlog, in-progress work, completed work.
- `decision-log.md`: durable technical decisions.
- `tests-and-results.md`: latest validation and experiment outcomes.
- `chat-index.md`: logical Codex session registry.
- `chat-roles.md`: ownership model for chat roles.
- `handoff.md`: deprecated short pointer kept for compatibility with older instructions.

## Operational notes

- Current operational scripts live under `script/`.
- Legacy/archive scripts live under `script_old/`; do not use them for new work unless explicitly needed for historical compatibility.
- `docs/ai/sessions/` is historical. Paths and plans there may be stale; prefer the core files above for current state.
- Sandbox is currently unavailable on the server (Codex bubblewrap / `bwrap` failures). Prefer non-sandboxed commands and minimal, targeted edits for memory files until further notice.
- Memory hygiene: record only durable decisions, important project facts, and important results. Do not keep transient discarded attempts in consolidated memory; remove or supersede them once they are not part of the useful project state.

## Directories

- `sessions/`: chronological notes from individual Codex sessions.
- `experiments/`: reproducible training/evaluation/test records.
- `runbooks/`: stable operational procedures.

## Current project

See `context.md`.
