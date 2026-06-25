# Tutorial `prepare_swimxyz_frames_dataset.py`

## Scopo
`script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py` costruisce un dataset canonico COCO-style a partire da frame SwimXYZ standalone e relative label `__COCO__2D_cam.txt`, poi rigenera i dataset derivati per:

- `VitPose++`
- `Yolo26x-Detection`
- `Yolo26x-Pose`

Il flusso produce quindi un dataset pronto sia per training pose sia per training detection.

## Cosa si aspetta in input
La directory di input deve contenere coppie:

- immagine frame: `.png`, `.jpg`, `.jpeg`
- label associata: stesso nome file + suffisso `__COCO__2D_cam.txt`

Esempi validi:

```text
swimmerA__frame_000001.jpg
swimmerA__frame_000001__COCO__2D_cam.txt

swimmerB__frm_000245.png
swimmerB__frm_000245__COCO__2D_cam.txt

FSAW_clip_01_pos_1_75_000002.jpg
FSAW_clip_01_pos_1_75_000002__COCO__2D_cam.txt
```

Lo script supporta due famiglie di naming:

- token espliciti `__frame_` e `__frm_`
- suffisso finale numerico, come `..._000002`

## Come funziona
Il flusso esegue questi passaggi:

1. Scansiona la directory input e trova tutte le immagini supportate.
2. Per ogni immagine, cerca la label `__COCO__2D_cam.txt` corrispondente.
3. Legge la label e accetta solo i campioni con una singola riga annotata.
4. Carica l’immagine per ricavare `width` e `height`.
5. Costruisce keypoint e bounding box con le regole canoniche del progetto.
6. Scarta i campioni non validi, ad esempio:
   - label mancanti
   - immagini illeggibili
   - keypoint insufficienti o non validi
7. Divide i campioni validi in `train`, `val`, `test`.
8. Materializza le immagini nel dataset canonico come `symlink` o `copy`.
9. Genera i JSON COCO:
   - `person_keypoints_train.json`
   - `person_keypoints_val.json`
   - `person_keypoints_test.json`
10. Se non disattivato, rigenera anche:
   - `_VitPosePP`
   - `_Yolo26x_detection`
   - `_Yolo26x_pose`
11. Scrive report e manifest finali.

## Struttura output
Con output root:

```text
data/intermediate/Side_above_water_frames
```

lo script crea:

```text
data/intermediate/Side_above_water_frames/
├── _train_canonical/
│   ├── train2017/
│   ├── val2017/
│   ├── test2017/
│   ├── annotations/
│   │   ├── person_keypoints_train.json
│   │   ├── person_keypoints_val.json
│   │   └── person_keypoints_test.json
│   ├── manifest.json
│   └── reports/
│       └── swimxyz_frames_preparation_report.json
├── _VitPosePP/
├── _Yolo26x_detection/
├── _Yolo26x_pose/
└── manifest.json
```

## Parametri

### `--input-root`
Directory sorgente con frame e label.

Default:

```text
data/input/subset_xyz/Side_above_water_frames
```

### `--output-root`
Directory root del dataset generato.

Default:

```text
data/intermediate/Side_above_water_frames
```

### `--val-ratio`
Quota del dataset validazione.

Default:

```text
0.2
```

Con il default, il `20%` dei campioni validi va in validation.

### `--test-ratio`
Quota del dataset test.

Default:

```text
0.1
```

Con il default, il `10%` dei campioni validi va in test.

### `--bbox-padding-x-ratio`
Padding orizzontale del bounding box, espresso come rapporto.

Default:

```text
0.20
```

Serve a dare più contesto laterale al nuotatore prima di salvare il bbox canonico.

### `--bbox-padding-y-ratio`
Padding verticale del bounding box, espresso come rapporto.

Default:

```text
0.25
```

Nel progetto è più alto di quello orizzontale per conservare più contesto sopra e sotto il corpo.

### `--bbox-min-padding-px`
Padding minimo per lato in pixel.

Default:

```text
15.0
```

Anche se il padding percentuale è piccolo, il bbox viene comunque espanso di almeno `15 px` per lato.

### `--min-visible-keypoints`
Numero minimo di keypoint validi richiesti per accettare il campione.

Default:

```text
4
```

Se un frame ha meno di `4` keypoint visibili/validi, viene scartato.

### `--copy-mode`
Modalità di materializzazione immagini nel dataset canonico.

Valori ammessi:

- `symlink`
- `copy`

Default:

```text
symlink
```

Con `symlink` non duplichi i byte delle immagini. Con `copy` le immagini vengono copiate fisicamente.

### `--vitpose-work-dir`
Working directory usata dall’exporter VitPose++.

Default:

```text
runs/vitposepp_side_above_water_frames
```

Serve alla generazione degli artifact/config dell’export VitPose++.

### `--yolo-pose-link-mode`
Modalità con cui il dataset `_Yolo26x_pose` materializza le immagini.

Valori ammessi:

- `symlink`
- `hardlink`
- `copy`

Default:

```text
symlink
```

### `--skip-model-exports`
Se presente, crea solo il dataset canonico `_train_canonical` e non rigenera:

- `_VitPosePP`
- `_Yolo26x_detection`
- `_Yolo26x_pose`

Default implicito:

```text
disattivato
```

Quindi normalmente gli export modello vengono creati.

### `--overwrite`
Se presente, elimina l’output root esistente e ricostruisce tutto da zero.

Default implicito:

```text
disattivato
```

Se l’output esiste e non passi `--overwrite`, lo script si ferma con errore.

## Comando minimo
Esempio base:

```bash
python script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py \
  --input-root data/input/subset_xyz/Side_above_water_frames \
  --output-root data/intermediate/Side_above_water_frames \
  --copy-mode symlink
```

## Esempi pratici

### Esempio 1: ricostruzione completa standard
```bash
python script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py \
  --input-root data/input/subset_xyz/Side_above_water_frames \
  --output-root data/intermediate/Side_above_water_frames \
  --copy-mode symlink \
  --overwrite
```

### Esempio 2: solo dataset canonico, senza export modello
```bash
python script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py \
  --input-root data/input/subset_xyz/Side_above_water_frames \
  --output-root data/intermediate/Side_above_water_frames \
  --skip-model-exports \
  --overwrite
```

### Esempio 3: split personalizzato
```bash
python script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py \
  --input-root data/input/subset_xyz/Side_above_water_frames \
  --output-root data/intermediate/Side_above_water_frames \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --overwrite
```

### Esempio 4: copia fisica delle immagini
```bash
python script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py \
  --input-root data/input/subset_xyz/Side_above_water_frames \
  --output-root data/intermediate/Side_above_water_frames_copy \
  --copy-mode copy \
  --overwrite
```

### Esempio 5: export YOLO pose con hardlink
```bash
python script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py \
  --input-root data/input/subset_xyz/Side_above_water_frames \
  --output-root data/intermediate/Side_above_water_frames \
  --yolo-pose-link-mode hardlink \
  --overwrite
```

## Lancio con tmux
Per esecuzioni lunghe:

```bash
tmux new-session -d -s prepare_swimxyz_frames_dataset \
"cd /home/albertosco/HPE && \
conda run -n vitpose python script/dataset_preparation-cleaning/prepare_swimxyz_frames_dataset.py \
  --input-root data/input/subset_xyz/Side_above_water_frames \
  --output-root data/intermediate/Side_above_water_frames \
  --val-ratio 0.2 \
  --test-ratio 0.1 \
  --copy-mode symlink \
  --vitpose-work-dir runs/vitposepp_side_above_water_frames \
  --yolo-pose-link-mode symlink \
  --overwrite \
  2>&1 | tee logs/prepare_swimxyz_frames_dataset.log"
```

Gestione sessione:

```bash
bash -lc 'tmux ls'
bash -lc 'tmux attach -t prepare_swimxyz_frames_dataset'
bash -lc 'tmux kill-session -t prepare_swimxyz_frames_dataset'
```

Per staccarti senza fermare il job:

```text
Ctrl-b poi d
```

## Report finale
Lo script produce un report JSON con:

- input usato
- output generato
- split ratios
- parametri bbox
- numero totale di coppie trovate
- numero di campioni accettati
- motivi di scarto
- conteggi train/val/test
- stato degli export modello

File principali:

- `data/intermediate/.../_train_canonical/reports/swimxyz_frames_preparation_report.json`
- `data/intermediate/.../_train_canonical/manifest.json`
- `data/intermediate/.../manifest.json`

## Default operativi consigliati
Per il workflow corrente del repository, i default hanno questo significato pratico:

- `--copy-mode symlink`: evita duplicazione delle immagini
- `--val-ratio 0.2` e `--test-ratio 0.1`: split standard `70/20/10`
- `--bbox-padding-x-ratio 0.20` e `--bbox-padding-y-ratio 0.25`: bbox coerenti con la convenzione canonica del progetto
- `--bbox-min-padding-px 15.0`: evita bbox troppo stretti
- `--min-visible-keypoints 4`: filtra frame troppo poveri di informazione
- `--yolo-pose-link-mode symlink`: export leggero e consistente con gli altri flussi

## Note utili
- Lo script accetta solo frame con label associate.
- Se una label contiene piu di una riga, il frame viene scartato.
- Se l’immagine non e leggibile, il frame viene scartato.
- Se i keypoint validi sono insufficienti, il frame viene scartato.
- `_Yolo26x_detection` usa symlink di directory verso le immagini canoniche.
- `_Yolo26x_pose` usa la modalita definita da `--yolo-pose-link-mode`.
