# HPE Report Direct/Cross Metrics

Purpose: document the reproducible report-table workflow for direct and cross evaluation metrics used in the HPE fine-tuning report.

## Source files

- Script: `script/hpe_report/build_hpe_report_tables.py`
- Example config: `script/hpe_report/hpe_report_config.example.json`
- Project documentation: `script/hpe_report/7) Diagrammi sulle metriche dirette e cross.md`
- Report deck: `script/hpe_report/20260617_Report_Fine-Tuning_Senza-Immagini.pptx`

Note: the project documentation examples mention `script/report/...`, but the actual workspace path is `script/hpe_report/...`.

## What the script produces

The script builds a headless XLSX report table workbook from already-generated GT, prediction, and validation-metric files. Default output:

```text
data/output/experiments/hpe_report/hpe_report_tables.xlsx
```

Covered PPTX tables/sheets:

- Slide 8: Val/Test AP and AR for `SAW_frames` and `SAW_frames_EntireSwim`.
- Slide 9: global geometric accuracy (`mean`, `median`, `P90`, `PCK`).
- Slide 10: reliability (`frames with predictions`, missing visible KP, invisible GT KP).
- Slide 16: cross-test AP/AR with and without thresholds.
- Slide 17: per-KP mean/median/P90 in cross-test.
- Slide 18: direct vs cross per-KP deltas.
- Slide 19: per-KP difficulty by combined P90.
- Slide 20: VitPose++ vs YOLO26x-Pose per-KP advantage.

## Inputs

- GT COCO keypoint JSON for `SAW_frames` and `SAW_frames_EntireSwim`.
- YOLO26x-Pose prediction JSON files (`kp_Test.json`).
- YOLO26x-Detection -> VitPose++ prediction JSON files (`kp_Test.json`).
- Validation metric CSV files (`val_metrics_by_epoch.csv`).

Direct scenarios:

- Train/Test `SAW_frames`.
- Train/Test `SAW_frames_EntireSwim`.

Cross scenario:

- Train `SAW_frames` -> Test `SAW_frames_EntireSwim`.

## Operational defaults

- YOLO26x-Pose confidence threshold: `0.30`.
- VitPose++ confidence threshold: `0.20`.
- Val best-epoch reconstruction: `patience=3`, `min_delta=0.007`.
- Difficulty grouping: Easy `<= 6.0`, Medium `<= 9.0`, High `<= 12.0`, Challenging `> 12.0` on combined P90.
- Dependencies: `numpy`, `openpyxl`, `pycocotools`.
- GPU is not required; the script only reads existing JSON/CSV artifacts.

## Standard commands

Validate inputs/calculations without writing XLSX:

```bash
python script/hpe_report/build_hpe_report_tables.py \
  --config script/hpe_report/hpe_report_config.example.json \
  --validate-only
```

Generate the workbook:

```bash
python script/hpe_report/build_hpe_report_tables.py \
  --config script/hpe_report/hpe_report_config.example.json \
  --output data/output/experiments/hpe_report/hpe_report_tables.xlsx
```

Generate workbook plus intermediate CSV checks:

```bash
python script/hpe_report/build_hpe_report_tables.py \
  --config script/hpe_report/hpe_report_config.example.json \
  --output data/output/experiments/hpe_report/hpe_report_tables.xlsx \
  --export-intermediate-csv
```

## Interpretation notes

- The thresholded analyses use model-specific operational thresholds; YOLO and VitPose++ confidence values are not necessarily calibrated comparably.
- The 12-KP/no-head analysis is post-hoc on models trained with all 17 COCO keypoints. It is not equivalent to evaluating models trained without head keypoints.
- For report use, validate `scenario_summary.csv`, `per_kp_metrics.csv`, and `val_best_epoch_summary.csv` before copying numbers into slides.
