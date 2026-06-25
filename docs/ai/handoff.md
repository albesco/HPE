# AI Handoff

Last updated: 2026-06-24 UTC
Status: deprecated as an operational handoff.

## Purpose

This file is kept only as a short compatibility pointer because `AGENTS.md` may still ask new sessions to read it. The project no longer uses long chat-to-chat handoff notes as the source of truth.

## Source Of Truth

- Current project state: `docs/ai/context.md`
- Active and backlog work: `docs/ai/task-board.md`
- Durable decisions: `docs/ai/decision-log.md`
- Validation, metrics, and run outcomes: `docs/ai/tests-and-results.md`
- Stable procedures: `docs/ai/runbooks/`
- Historical session notes: `docs/ai/sessions/`

## Memory Rules

- Use `script/` for current operational workflows.
- Treat `script_old/` as legacy/archive.
- Treat `docs/ai/sessions/` as historical; paths there may be stale.
- Do not add new operational state here; consolidate it in the source-of-truth files above.
