## AI memory protocol

This repository uses `docs/ai/` as the persistent project memory shared by Codex sessions.

## Sandbox / tooling note

This server does not support the sandbox mechanism used by Codex tools (bubblewrap / `bwrap`).
Until further notice:

- Avoid relying on sandboxed operations (they may fail with `bwrap` errors).
- Prefer running commands and making memory-only edits outside the sandbox.
- When editing `docs/ai/` or other memory files, keep changes minimal and scoped; a small targeted script is acceptable.

Before any non-trivial task, read:

1. `AGENTS.md`
2. `docs/ai/context.md`
3. `docs/ai/task-board.md`
4. `docs/ai/chat-index.md`
5. `docs/ai/chat-roles.md`
6. `docs/ai/handoff.md` if it exists

Also read `docs/ai/decision-log.md` when the task affects:
- architecture
- dataset format
- training configuration
- evaluation
- documentation
- reproducibility

For training, evaluation, dataset conversion, or experiment-related work, also read:
- `docs/ai/tests-and-results.md` if it exists
- relevant files under `docs/ai/experiments/`
- relevant files under `docs/ai/runbooks/`

Before ending a long session or when context is becoming limited:

1. Update `docs/ai/handoff.md`.
2. Update `docs/ai/context.md` only with consolidated current state.
3. Update `docs/ai/task-board.md` if task status changed.
4. Update `docs/ai/chat-index.md` if the session role or active files changed.
5. Add or update a session note under `docs/ai/sessions/` for any substantial work.
6. Update `docs/ai/tests-and-results.md` for any test, training, evaluation, or metrics result.
7. Add or update an experiment note under `docs/ai/experiments/` for any reproducible training/eval run (include run id, config, and outcomes).

Do not use chat history as the source of truth. Treat repository files and `docs/ai/` as the durable source of truth.
