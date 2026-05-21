# EXP-20260514-yolo-padded10-epoch10

## Summary

This note documents the best-known checkpoint/metrics for the YOLO one-class `swimmer` detector trained on the Side_above_water dataset using **uniform GT bbox padding ratio `0.10`** in the prepared dataset (YOLO conversion used `bbox_padding_ratio=0.0` to avoid double-padding).

This experiment was later superseded by the anisotropic GT bbox padding convention (see `docs/ai/decision-log.md`).

## Run identity

- Area: YOLO detector (Side_above_water)
- Run/work dir: `runs/yolo_side_above_water/yolo26x_swimmer_gt_bbox_padded10_20ep/`
- Status: stopped early after epoch `11` was recorded in `results.csv` (best checkpoint selected at epoch `10`)

## Key results (epoch-10 / best)

Metrics recorded as the best-known validation outcome for this run:
- precision: `0.985`
- recall: `0.986`
- mAP50: `0.992`
- mAP50-95: `0.872`

## Notes

- This was an early YOLO run used to validate the detector workflow and bbox padding convention.
- Later runs switched to anisotropic padding (horizontal `0.20`, vertical `0.25`, min `15 px`) for better vertical context.

## Sources

- `docs/ai/context.md`
- `docs/ai/task-board.md`
- `docs/ai/tests-and-results.md`
- `docs/ai/decision-log.md`
