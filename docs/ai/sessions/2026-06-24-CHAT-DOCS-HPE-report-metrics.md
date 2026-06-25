# 2026-06-24 - CHAT-DOCS HPE report metrics

## Purpose

Consolidate the report workflow for direct and cross train metrics into the AI project memory.

## Sources read

- `AGENTS.md`
- `docs/ai/context.md`
- `docs/ai/task-board.md`
- `docs/ai/chat-index.md`
- `docs/ai/chat-roles.md`
- `docs/ai/handoff.md`
- `docs/ai/decision-log.md`
- `docs/ai/tests-and-results.md`
- `script/hpe_report/build_hpe_report_tables.py`
- `script/hpe_report/hpe_report_config.example.json`
- `script/hpe_report/7) Diagrammi sulle metriche dirette e cross.md`
- `script/hpe_report/20260617_Report_Fine-Tuning_Senza-Immagini.pptx`

## Consolidated memory updates

- Added `docs/ai/runbooks/hpe-report-direct-cross-metrics.md`.
- Updated `docs/ai/context.md` with the durable report workflow, inputs, outputs, defaults, and covered slides.
- Updated `docs/ai/tests-and-results.md` with key direct/cross AP values extracted from the PPTX.
- Updated `docs/ai/decision-log.md` to record the report script as the reproducible source for the HPE metric tables.
- Updated `docs/ai/handoff.md` and `docs/ai/chat-index.md`.

## Important facts

- Actual script path in the workspace: `script/hpe_report/build_hpe_report_tables.py`.
- Example config: `script/hpe_report/hpe_report_config.example.json`.
- Default output: `data/output/experiments/hpe_report/hpe_report_tables.xlsx`.
- The workflow is headless and does not require GPU; it reads existing COCO GT JSON, `kp_Test.json`, and `val_metrics_by_epoch.csv` artifacts.
- The project MD examples mention `script/report/...`, but the actual workspace path is `script/hpe_report/...`.

## Key PPTX values recorded

- Direct Test AP on `SAW_frames_EntireSwim`: YOLO26x-Pose `0.93190`, VitPose++ `0.97822`.
- Direct Test AP on `SAW_frames`: YOLO26x-Pose `0.87474`, VitPose++ `0.93288`.
- Cross Train `SAW_frames` -> Test `SAW_frames_EntireSwim` thresholded AP: YOLO26x-Pose `0.9774`, VitPose++ `0.9914`.
- Cross Train `SAW_frames` -> Test `SAW_frames_EntireSwim` without-threshold AP: YOLO26x-Pose `0.9854`, VitPose++ `0.9914`.

## Notes

- No application code was changed.
- The attached PPTX was used only to extract durable report facts; binary content was not copied into memory.
