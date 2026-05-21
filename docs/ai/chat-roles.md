# Codex Chat Roles

## Documentation / AI memory maintenance (CHAT-DOCS)
Owns:
- docs/ai/ memory structure
- start-here / handoff hygiene
- experiment note templates and indexing
- maintaining chat index and role registry

Must update:
- docs/ai/handoff.md when changes affect next steps
- docs/ai/chat-index.md when roles/files change
- docs/ai/task-board.md only when task status changes

## Dataset conversion chat
Owns:
- SwimXYZ parsing
- video/frame extraction
- annotation conversion
- dataset validation

Must update:
- docs/ai/context.md
- docs/ai/decision-log.md for schema decisions

## Training chat
Owns:
- VitPose++ config
- training launcher
- checkpoints
- logs
- evaluation

Must update:
- docs/ai/context.md
- docs/ai/decision-log.md for experiment decisions

## Workspace Q&A chat
Owns:
- repo navigation
- Linux/SSH issues
- dependency/debugging questions
- GitHub workflow

Must not make architectural decisions unless recorded.

## Documentation chat
Owns:
- README
- reproducibility guide
- experiment notes
- usage documentation
