# Chat Index (Codex Sessions)

This file tracks Codex chat sessions for this repository.

## Active / Recent

### CHAT-DATASET-2 (formerly: Data-Cleaning 2)
- Logical title: Dataset conversion (data cleaning & preparation)
- Role: dataset-conversion
- Status: active
- Predecessor: `Elenca file e cartelle`
- Purpose: inspect and prepare the current dataset/workspace state for dataset conversion work, using repository files and `docs/ai/` as source of truth
- Key files:
  - `AGENTS.md`
  - `docs/ai/context.md`
  - `docs/ai/task-board.md`
  - `docs/ai/decision-log.md`
  - `docs/ai/chat-index.md`
  - `docs/ai/chat-roles.md`
  - `docs/ai/handoff.md`
  - `docs/ai/tests-and-results.md`
  - `docs/ai/sessions/`
  - `docs/ai/experiments/`
- Started: 2026-05-20 (UTC)

### CHAT-DOCS
- Logical title: AI memory maintenance
- Role: documentation / AI memory maintenance
- Purpose: manutenzione di `docs/ai/` per handoff tra sessioni Codex con finestra di contesto limitata
- Key files:
  - `AGENTS.md`
  - `docs/ai/start-here.md`
  - `docs/ai/context.md`
  - `docs/ai/task-board.md`
  - `docs/ai/handoff.md`
  - `docs/ai/decision-log.md`
  - `docs/ai/tests-and-results.md`
  - `docs/ai/chat-index.md`
  - `docs/ai/chat-roles.md`
  - `docs/ai/sessions/`
  - `docs/ai/experiments/`
  - `docs/ai/runbooks/`
- Started: 2026-05-15 (UTC)

### CHAT-DATASET
- Logical title: SwimXYZ 2 VitPose++
- Role: dataset-conversion
- Purpose: conversione del subset SwimXYZ nel formato richiesto da VitPose++/MMPose
- Key files:
  - `AGENTS.md`
  - `docs/ai/context.md`
  - `docs/ai/decision-log.md`
  - `docs/ai/chat-index.md`
  - `scripts/convert_swimxyz_to_vitpose.py`
  - `scripts/validate_vitpose_dataset.py`
  - `docs/dataset-format.md`
- Started: 2026-05-11 (UTC)

### CHAT-TRAINING
- Logical title: VitPose++ Training
- Role: training
- Purpose: training/eval pipeline, configs, launch, checkpoints, metrics
- Status: closing; successor planned as `CHAT-TRAINING-2`
- Key files:
  - `AGENTS.md`
  - `docs/ai/context.md`
  - `docs/ai/decision-log.md`
  - `docs/ai/chat-index.md`
  - `docs/ai/task-board.md`
  - `configs/`
  - `script/`
  - `script/yolo_training/`
  - `script/visualize_gt_bboxes.py`
  - `src/`
  - `docs/ai/sessions/2026-05-12-CHAT-TRAINING.md`
  - `docs/ai/experiments/EXP-20260514-vitpose-aniso-resume-30ep.md`
- Started: 2026-05-11 (UTC)

### CHAT-TRAINING-2
- Logical title: VitPose++ Training 2
- Role: training
- Status: active
- Predecessor: `CHAT-TRAINING`
- Purpose: continue VitPose++/YOLO training evaluation after CHAT-TRAINING handoff, using workspace files as source of truth
- Key files:
  - `AGENTS.md`
  - `docs/ai/start-here.md`
  - `docs/ai/context.md`
  - `docs/ai/task-board.md`
  - `docs/ai/handoff.md`
  - `docs/ai/decision-log.md`
  - `docs/ai/tests-and-results.md`
  - `docs/ai/sessions/2026-05-15-CHAT-TRAINING-handoff-to-CHAT-TRAINING-2.md`
  - `docs/ai/sessions/2026-05-20-CHAT-TRAINING-2.md`
  - `docs/ai/experiments/EXP-20260514-vitpose-aniso-resume-30ep.md`
  - `runs/vitposepp_side_above_water_aniso_20x25_min15/training_status.txt`
  - `script/run_resume_side_above_water_to_25ep_tmux.sh`
  - `data/output/experiments/YoloVitPose_mAP/`
  - `docs/ai/experiments/EXP-20260520-YoloVitPose-consolidation.md`
  - `script/yolo_training/evaluate_yolo_vitpose_map.py`
- Started: 2026-05-15 (UTC)
- Handoff updated: 2026-05-19 (UTC)
