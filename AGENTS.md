# AGENTS.md

## Project
Training pipeline for HPE VitPose++ on a subset of SwimXYZ.

## Mandatory workflow for Codex
Before making changes:
1. Read this file.
2. Read docs/ai/context.md.
3. Read docs/ai/decision-log.md if the task affects architecture, dataset format, training, or documentation.

After making relevant changes:
1. Update docs/ai/context.md with the current state.
2. Update docs/ai/decision-log.md only for durable decisions.
3. Update docs/ai/task-board.md if task status changed.
4. Do not store secrets, absolute private paths, credentials, or large logs.

## Coding rules
- Use Python.
- Prefer small, testable modules.
- Keep dataset conversion, training, evaluation, and documentation separated.
- Use clear names for constants and variables.
- Add CLI entry points for scripts that are meant to be run on the server.
- Avoid hardcoded paths; use config files or command-line arguments.

## Repository discipline
- Do not overwrite existing dataset labels.
- Do not commit large datasets, videos, checkpoints, or generated logs.
- Keep reproducibility notes updated.

## Codex session discipline

Because Codex VS Code chat titles are auto-generated and may not be manually renamed, every session must identify itself through docs/ai/chat-index.md.

At the beginning of each session:
1. Read docs/ai/chat-index.md.
2. Ask which logical chat role this session belongs to if unclear.
3. Use that role consistently.

At the end of relevant work:
1. Update docs/ai/context.md.
2. Update docs/ai/chat-index.md.
3. Update docs/ai/decision-log.md only for durable decisions.

