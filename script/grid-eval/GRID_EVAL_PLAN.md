# Grid-evaluation plan

## Scope

Questo documento pianifica tre script indipendenti e parametrici per grid-evaluation locale:

- `script/grid-eval/grid_eval_Yolo26x-Detection.py`
- `script/grid-eval/grid_eval_Yolo26x-Pose.py`
- `script/grid-eval/grid_eval_VitPose.py`

In questa fase non vengono creati gli script definitivi. La grid-evaluation serve a scegliere configurazioni promettenti su validation; il test set non deve essere usato per scegliere iperparametri.

## Fonti analizzate

Memoria e contesto:

- `AGENTS.md`
- `docs/ai/start-here.md`
- `docs/ai/context.md`
- `docs/ai/task-board.md`
- `docs/ai/decision-log.md`
- `docs/ai/tests-and-results.md`
- `docs/ai/chat-index.md`
- `docs/ai/chat-roles.md`
- `docs/ai/handoff.md`
- `docs/ai/experiments/EXP-20260523-yolo26x-detector-grid2x2-v2.md`
- `docs/ai/experiments/EXP-20260527-yolo26x-pose-incremental-cfg04.md`
- `docs/ai/experiments/EXP-20260702-yolo26x-detection-suw-frames.md`

Script e configurazioni:

- `script_old/yolo26x_Detection-Training/yolo26x_detector_grid2x2.py`
- `script_old/yolo_training/yolo26x_detector_grid2x2.py`
- `script_old/yolo_training/yolo26x_pose_grid2x2.py`
- `script_old/hparam_search/vitposepp_huge_grid2x2.py`
- `script/yolo26x_detection_training/train_yolo26x_detection_frame.sh`
- `script/yolo26x_detection_training/evaluate_yolo_detection_split.py`
- `script/yolo26x_detection_training/export_yolo_detection_training_report.py`
- `script/yolo26x_detection_training/prune_yolo_epoch_checkpoints.py`
- `script/yolo26x_pose_training/train_yolo26x_pose_frame.sh`
- `script/yolo26x_pose_training/evaluate_yolo_pose_split.py`
- `script/yolo26x_pose_training/export_yolo_pose_training_report.py`
- `script/vitpose_training/train_vitpose_frame.sh`
- `script/vitpose_training/export_vitpose_val_metrics.py`
- `data/intermediate/SUW_frames/_Yolo26x_detection/swimxyz_side_above_water_yolo26x_detection.yaml`
- `data/intermediate/SUW_frames/_Yolo26x_pose/swimxyz_side_above_water_yolo26x_pose.yaml`
- `data/intermediate/Side_above_water/_VitPosePP/generated_configs/swimxyz_vitposepp_huge.py`
- `runs/hparam_search/yolo26x_detector_v2/summary.csv`
- `runs/hparam_search/yolo26x_pose/summary.csv`
- `runs/hparam_search/vitposepp_huge/summary.csv`

## Struttura repository ricostruita

- `script/`: root operativo corrente.
- `script_old/`: archivio legacy; contiene prototipi grid gia utili ma non piu root operativo.
- `docs/ai/`: memoria persistente del progetto.
- `data/intermediate/`: dataset derivati con split e formati per VitPose++, YOLO detection e YOLO pose.
- `models/detection/`: checkpoint detector pretrained, inclusi `yolo26x.pt`.
- `models/pose/`: checkpoint pose pretrained, inclusi `yolo26x-pose.pt`, `wholebody.pth`, `vitpose_huge.pth`.
- `runs/`: training, checkpoint, report e grid search.
- `data/output/experiments/`: output di prediction/evaluation finali, non da usare per la scelta iperparametri.

## Dataset e split

Dataset principali ricostruiti:

- `data/intermediate/Side_above_water/`: train `18181`, val `5195`, test `2597`.
- `data/intermediate/Side_above_water_EntireSwim/`: train `27732`, val `7949`, test `4001`.
- `data/intermediate/SAW_frames_EntireSwim/`: train `10489`, val `2997`, test `1498`.
- `data/intermediate/SUW_frames/`: train `6044`, val `1727`, test `863`.

Ogni dataset consolidato espone tipicamente:

- `_train_canonical/`: immagini canoniche e annotazioni COCO keypoint.
- `_VitPosePP/`: annotazioni COCO per VitPose++.
- `_Yolo26x_detection/`: dataset YOLO bbox one-class `swimmer`.
- `_Yolo26x_pose/`: dataset YOLO pose COCO17.

Gli script devono accettare `--dataset-dir` e risolvere internamente il sottoformato necessario. Il default consigliato per compatibilita storica delle grid e `data/intermediate/Side_above_water`; esempi e test operativi recenti devono anche supportare `data/intermediate/SUW_frames`.

## Formati annotazioni

- VitPose++: COCO-style JSON `person_keypoints_{train,val,test}.json`, keypoint COCO17, bbox GT padded.
- YOLO26x-Detection: label `.txt` con 5 campi per riga: `class x_center y_center width height`, normalizzati.
- YOLO26x-Pose: label `.txt` con 56 campi: `class bbox(4) 17*(x y visibility)`, `kpt_shape: [17, 3]`.
- Bbox GT consolidate: padding anisotropico `x=0.20`, `y=0.25`, minimo `15 px`.

Nota critica: Ultralytics puo non trovare label se `images/{split}` e una directory symlink che risolve fuori da `_Yolo26x_detection`. La grid detection deve validare cache/label e, se necessario, usare una dataset view con directory reali e symlink per-file.

## Comandi e configurazioni gia usati

### YOLO26x-Detection

Launcher corrente:

```bash
bash script/yolo26x_detection_training/train_yolo26x_detection_frame.sh \
  --dataset-dir data/intermediate/SUW_frames \
  --pretrained-checkpoint models/detection/yolo26x.pt \
  --run-name yolo26x-detection_SUW_frames_20260701 \
  --max-epochs 100 \
  --lr 0.00067 \
  --imgsz 768
```

Pattern da grid legacy:

```bash
conda run -n vitpose yolo detect train \
  model=models/detection/yolo26x.pt \
  data=<train_val_yaml> \
  epochs=5 imgsz=<imgsz> lr0=<lr0> batch=2 device=0 workers=2 \
  project=<output_root> name=<cfg_name> val=True split=val save=True
```

Risultato storico validato: `runs/hparam_search/yolo26x_detector_v2/cfg_03_lr0_0.00067_imgsz_768/weights/best.pt`, selezionato validation-only per recall.

### YOLO26x-Pose

Launcher corrente:

```bash
bash script/yolo26x_pose_training/train_yolo26x_pose_frame.sh \
  --dataset-dir data/intermediate/SUW_frames \
  --pretrained-checkpoint models/pose/yolo26x-pose.pt \
  --run-name <run_name> \
  --max-epochs 100 \
  --lr 0.00100 \
  --imgsz 768
```

Pattern da grid legacy:

```bash
conda run -n vitpose yolo pose train \
  model=models/pose/yolo26x-pose.pt \
  data=<pose_yaml> \
  epochs=5 imgsz=<imgsz> lr0=<lr0> batch=1 device=0 workers=2 \
  optimizer=AdamW project=<output_root> name=<cfg_name> val=True split=val \
  save=True save_period=-1
```

Nota: `optimizer=auto` puo ignorare `lr0`; per la grid pose il riferimento storico usa `optimizer=AdamW`.

### ViTPose++ / ViTPose++ huge

Launcher corrente:

```bash
bash script/vitpose_training/train_vitpose_frame.sh \
  --dataset-dir data/intermediate/SUW_frames \
  --pretrained-checkpoint models/pose/wholebody.pth \
  --run-name <run_name> \
  --max-epochs 100 \
  --lr 0.00100 \
  --crop-size 384x128
```

Pattern da grid legacy:

```bash
conda run -n vitpose python src/vitpose_base/tools/train.py \
  <effective_config.py> \
  --work-dir <cfg_dir> \
  --gpu-id 0
```

La grid legacy genera `effective_config.py` dal base config, imposta `total_epochs=5`, `optimizer.lr`, `checkpoint_config`, `evaluation`, crop size e `use_gt_bbox=True`.

## Checkpoint pretrained

- YOLO26x-Detection: `models/detection/yolo26x.pt`.
- YOLO26x-Pose: `models/pose/yolo26x-pose.pt`.
- ViTPose++ huge: riferimento storico `models/pose/wholebody.pth`; esiste anche `models/pose/vitpose_huge.pth`.

## Convenzioni output proposte

Root comune:

- `runs/grid_eval/yolo26x_detection/`
- `runs/grid_eval/yolo26x_pose/`
- `runs/grid_eval/vitposepp_huge/`

Ogni configurazione:

- `cfg_01_lr0_0.00067_imgsz_640/`
- `cfg_02_lr0_0.00100_imgsz_640/`
- `cfg_03_lr0_0.00067_imgsz_768/`
- `cfg_04_lr0_0.00100_imgsz_768/`
- `cfg_01_lr_0.00067_crop_384x128/` per VitPose++

Artifact per configurazione:

- `command.txt`: comando effettivo.
- `args.json`: argomenti effettivi e parametri risolti.
- `status.json`: `pending`, `running`, `completed`, `failed`, `skipped`.
- `stdout_stderr.log`: log del subprocess.
- checkpoint riusabili: solo `best` e `last/latest`.
- metriche validation: `results.csv` per YOLO o `*.log.json` / export CSV per VitPose++.
- `effective_config.py` solo per VitPose++.

Artifact root:

- `summary.csv`
- `summary.json`
- `report.md`
- `best_config.json`
- `grid_effective.json`

Il test set non deve comparire negli output di selezione.

## Griglie default

YOLO26x-Detection:

```yaml
grid:
  - name: cfg_01_lr0_0.00067_imgsz_640
    params: {lr0: 0.00067, imgsz: 640}
  - name: cfg_02_lr0_0.00100_imgsz_640
    params: {lr0: 0.00100, imgsz: 640}
  - name: cfg_03_lr0_0.00067_imgsz_768
    params: {lr0: 0.00067, imgsz: 768}
  - name: cfg_04_lr0_0.00100_imgsz_768
    params: {lr0: 0.00100, imgsz: 768}
```

YOLO26x-Pose: stessa griglia detection con `lr0` e `imgsz`.

ViTPose++:

```yaml
grid:
  - name: cfg_01_lr_0.00067_crop_384x128
    params: {lr: 0.00067, crop_size: "384x128"}
  - name: cfg_02_lr_0.00100_crop_384x128
    params: {lr: 0.00100, crop_size: "384x128"}
  - name: cfg_03_lr_0.00067_crop_512x128
    params: {lr: 0.00067, crop_size: "512x128"}
  - name: cfg_04_lr_0.00100_crop_512x128
    params: {lr: 0.00100, crop_size: "512x128"}
```

## Interfaccia CLI comune proposta

Argomenti comuni:

```bash
--dataset-dir PATH
--output-root PATH
--epochs 5
--patience 5
--device 0
--workers 2
--conda-env vitpose
--grid-json PATH
--grid-yaml PATH
--grid "lr0=0.00067,imgsz=640;lr0=0.001,imgsz=768"
--dry-run
--launch-tmux
--tmux-session NAME
--resume auto
--rerun-failed
--rerun-running
--overwrite
```

Argomenti YOLO:

```bash
--pretrained-checkpoint models/detection/yolo26x.pt
--batch-size 2
--optimizer AdamW
```

Per YOLO pose il batch default resta `1`; per detection il batch default resta `2`.

Argomenti ViTPose++:

```bash
--pretrained-checkpoint models/pose/wholebody.pth
--base-config data/intermediate/Side_above_water/_train_canonical/generated_configs/swimxyz_vitposepp_huge.py
--batch-size 1
--crop-size 384x128
```

Il parser deve rifiutare griglie che contengono split `test` o opzioni di evaluation test.

## Schema JSON/YAML proposto

```yaml
schema_version: 1
name: yolo26x_detection_grid2x2
selection_split: val
defaults:
  epochs: 5
  patience: 5
  save_policy: best_last
  dataset_dir: data/intermediate/Side_above_water
  pretrained_checkpoint: models/detection/yolo26x.pt
grid:
  - name: cfg_01_lr0_0.00067_imgsz_640
    params:
      lr0: 0.00067
      imgsz: 640
  - name: cfg_02_lr0_0.00100_imgsz_640
    params:
      lr0: 0.00100
      imgsz: 640
```

Per ViTPose++:

```yaml
schema_version: 1
name: vitposepp_huge_grid2x2
selection_split: val
defaults:
  epochs: 5
  patience: 5
  use_gt_bbox: true
  save_policy: best_latest
grid:
  - name: cfg_01_lr_0.00067_crop_384x128
    params:
      lr: 0.00067
      crop_size: 384x128
```

## Helper comuni da creare

File proposto: `script/grid-eval/grid_eval_common.py`.

Responsabilita:

- parsing CLI condiviso;
- parsing JSON/YAML con fallback chiaro se PyYAML non e disponibile;
- generazione e validazione griglia;
- normalizzazione nomi configurazione;
- risoluzione path relativi al repo;
- creazione directory output;
- gestione atomica `status.json`;
- scrittura `summary.csv`, `summary.json`, `report.md`, `best_config.json`;
- gestione subprocess con log stdout/stderr;
- supporto `--dry-run`;
- supporto `--launch-tmux`;
- resume `auto` da checkpoint della stessa configurazione;
- parser metriche YOLO da `results.csv`;
- parser metriche ViTPose++ da `*.log.json` o export CSV;
- pruning/checkpoint policy per mantenere solo best e last/latest.

## Architettura script

### `grid_eval_Yolo26x-Detection.py`

Adapter specifico:

- risolve `_Yolo26x_detection/*.yaml`;
- opzionalmente crea `dataset_view/` train+val robusta con per-file symlink;
- valida presenza immagini/label train e val;
- lancia `yolo detect train`;
- imposta `epochs=5`, `patience=5`, `save_period=-1`, `split=val`;
- non lancia test;
- usa `weights/best.pt` e `weights/last.pt`.

Criterio best:

1. massima recall bbox su validation;
2. AP75 o IoU medio se disponibili;
3. mAP50-95 bbox;
4. preferire `imgsz=640`.

Le metriche AP75, IoU medio e keypoint coverage vanno lasciate `null` se non calcolabili senza evaluator aggiuntivo.

### `grid_eval_Yolo26x-Pose.py`

Adapter specifico:

- risolve `_Yolo26x_pose/*.yaml`;
- valida label a 56 campi e split train/val;
- lancia `yolo pose train`;
- usa `optimizer=AdamW` per rendere effettivo `lr0`;
- imposta `epochs=5`, `patience=5`, `save_period=-1`, `split=val`;
- non lancia test;
- usa `weights/best.pt` e `weights/last.pt`.

Criterio best proposto:

1. maggiore Pose mAP50-95 su validation;
2. a parita, maggiore Pose mAP50 o Pose recall se disponibili;
3. a parita, maggiore Box mAP50-95;
4. a parita, preferire `imgsz=640`.

### `grid_eval_VitPose.py`

Adapter specifico:

- risolve `_VitPosePP/annotations/person_keypoints_{train,val}.json`;
- genera un `effective_config.py` per configurazione;
- imposta `use_gt_bbox=True` e `bbox_file=''` per train/val;
- imposta crop-size in convenzione `width x height`;
- imposta `total_epochs=5`;
- checkpoint policy: `max_keep_ckpts=1`, `save_best='AP'`, mantenendo `latest.pth`;
- non lancia test e non usa YOLO detector;
- lancia `src/vitpose_base/tools/train.py`.

Criterio best:

1. maggiore COCO keypoint AP validation;
2. a parita, maggiore AP75;
3. a parita, maggiore AR;
4. a parita, preferire crop `384x128`.

## Codice riusabile

Riusare:

- logica status/summary/report dai tre prototipi in `script_old/`;
- risoluzione dataset e convenzioni path dai launcher correnti in `script/`;
- parser YOLO `results.csv` dai prototipi grid e dagli exporter;
- generazione config VitPose++ da `script_old/hparam_search/vitposepp_huge_grid2x2.py`;
- pruning checkpoint da `script/yolo26x_detection_training/prune_yolo_epoch_checkpoints.py`.

Riscrivere:

- duplicazione tra grid legacy, portandola in `grid_eval_common.py`;
- supporto a griglia da CLI, JSON e YAML;
- checkpoint retention, per salvare solo best e last/latest;
- validazione preflight dataset/cache Ultralytics;
- semantica `status.json`, distinguendo `completed` da training terminato con codice non zero;
- disabilitazione completa del test set nei grid script.

## Assunzioni

- Gli script verranno eseguiti nell'ambiente conda `vitpose`.
- I comandi pesanti potranno essere lanciati in tmux tramite `--launch-tmux`.
- Gli iperparametri non specificati dalla griglia restano quelli del framework o della configurazione di riferimento.
- Per ViTPose++ il checkpoint pretrained di riferimento resta `models/pose/wholebody.pth`, salvo override CLI.
- `crop_size` e espresso come `width x height`.
- La selezione si basa solo su validation.

## Punti critici e ambigui

- `docs/ai/chat-index.md` contiene contenuto da task board; non e affidabile come indice chat in questa snapshot.
- Le grid storiche usano `runs/hparam_search/...`; per i nuovi script si propone `runs/grid_eval/...` per separare analisi nuove da storiche.
- La grid VitPose++ storica mostra metriche e checkpoint ma anche errori nei summary per alcuni run; la nuova semantica `status.json` deve essere piu rigorosa.
- Per YOLO detection va confermato se impostare esplicitamente `optimizer=AdamW`; altrimenti `optimizer=auto` potrebbe ignorare `lr0` come gia osservato per YOLO pose.
- AP75 bbox, IoU medio detector e keypoint coverage non sono direttamente disponibili nei `results.csv` YOLO; vanno implementati con evaluator leggero o dichiarati `null`.
- Prima del lancio bisogna controllare che non ci siano training GPU attivi.

## File da creare nella fase successiva

- `script/grid-eval/grid_eval_common.py`
- `script/grid-eval/grid_eval_Yolo26x-Detection.py`
- `script/grid-eval/grid_eval_Yolo26x-Pose.py`
- `script/grid-eval/grid_eval_VitPose.py`
- `script/grid-eval/default_grids/yolo26x_detection_grid2x2.yaml`
- `script/grid-eval/default_grids/yolo26x_pose_grid2x2.yaml`
- `script/grid-eval/default_grids/vitposepp_huge_grid2x2.yaml`
- `script/grid-eval/README.md`

## Piano implementativo

1. Creare `grid_eval_common.py` con CLI, grid loader, status, subprocess, summary e report.
2. Implementare adapter YOLO detection con preflight dataset, train val-only e parser `results.csv`.
3. Implementare adapter YOLO pose con parser metriche pose e `optimizer=AdamW`.
4. Implementare adapter ViTPose++ con generazione config e parsing `*.log.json`.
5. Aggiungere default grid YAML per i tre modelli.
6. Eseguire dry-run per verificare directory, comandi e blocco test set.
7. Lanciare training pesanti solo via tmux quando richiesto.
