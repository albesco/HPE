# Note sulla pipeline consolidata SwimXYZ -> VitPose++

## Obiettivo

La pipeline consolidata serve a costruire dataset di addestramento per
VitPose++ single-head a partire dai video e dalle label originali SwimXYZ.

La versione approvata fino a questo momento adotta la variante con:

- inversione dell'asse Y
- scambio left/right di orecchie e occhi rispetto ai dati SwimXYZ
- produzione automatica delle immagini overlay accanto alle immagini finali
- logging dei salti anomali della bounding box

## Script mantenuti nella root `script`

- `CSV-2-JSON_Keypoint-Conversion.py`
  Normalizza un dataset SwimXYZ originale dentro `data/intermediate/.../_converted`
  e genera il manifest usato dagli step successivi.

- `prepare_swimxyz_vitposepp_utils.py`
  Modulo di utility condivise. Contiene parsing delle righe label, ricodifica
  SwimXYZ -> COCO17, calcolo delle bbox, logica di visibility, split
  train/val/test e generazione degli overlay.

- `prepare_swimxyz_vitposepp_single_head.py`
  Script principale della pipeline consolidata. Prepara il dataset VitPose++
  single-head applicando swap di occhi e orecchie prima della ricodifica.

- `prepare_swimxyz_vitposepp_train.py`
  Wrapper CLI corto da usare come entrypoint operativo della pipeline
  consolidata.

- `kp_check_swimxyz_video_frames.py`
  Script di verifica visiva. Serve per controllare frame specifici o interi
  video con overlay di skeleton e keypoint, senza passare da uno split di
  training completo.

## Script spostati in `script/old_trials`

I seguenti file sono stati archiviati perche non fanno parte della pipeline
consolidata attuale:

- `prepare_swimxyz_vitpose_train_label.py`
- `run_pipeline.py`
- `visualize_vitpose_keypoints.py`

Questi file non sono stati cancellati: restano disponibili in `old_trials`
come riferimento storico o per confronti futuri.

Il vecchio modulo `prepare_swimxyz_vitpose_train.py` non e stato archiviato:
e stato rinominato in `prepare_swimxyz_vitposepp_utils.py` per riflettere il
suo ruolo attuale nella pipeline VitPose++ consolidata.

## Logica consolidata

### 1. Inversione di Y

SwimXYZ usa coordinate 2D con origine in basso, mentre le immagini consumate
da OpenCV e da VitPose++ usano l'origine in alto a sinistra.

Per questo la pipeline applica:

- `image_y = image_height - y`

Questa inversione viene applicata in modo coerente sia nella preparazione del
dataset sia negli script di verifica visiva.

### 2. Scambio di occhi e orecchie

La variante consolidata nasce dall'ipotesi che in SwimXYZ i keypoint della
testa laterali siano invertiti rispetto alla semantica attesa da COCO17.

Per questo, prima della ricodifica finale, vengono scambiati:

- `LEye <-> REye`
- `LEar <-> REar`

Lo scambio avviene sulla riga label SwimXYZ e non nel codice di parsing di base.
Questo permette di mantenere una separazione netta tra:

- implementazione baseline
- implementazione ipotetica consolidata

### 3. Visibility dei keypoint

Nella pipeline consolidata il valore `z` di SwimXYZ non viene interpretato come
confidence. Viene trattato come coordinata spaziale, coerentemente con
l'origine 3D del dataset.

La visibility `v` viene quindi ricostruita solo dalla posizione orizzontale:

- se `x == 0` oppure `x == max_x`, allora `v = 0`
- se `0 < x < max_x`, allora `v = 2`

Questa e la regola attualmente consolidata. Se in vecchie note compare una
formulazione diversa come `v=0 per 0<x<Max`, va considerata un refuso: la
logica effettiva usata nel codice e quella sopra.

### 4. Bounding box

La bbox viene derivata dai keypoint validi visibili e poi allargata con un
padding percentuale. Non viene letta direttamente da SwimXYZ.

Questo rende la bbox coerente con i keypoint finali che verranno dati in input
al training.

### 5. Logging delle anomalie bbox

Per ogni video, la pipeline controlla se una bbox compie un salto troppo ampio
rispetto alla bbox valida precedente.

Il criterio consolidato usa due condizioni contemporanee:

- distanza tra i centri troppo grande rispetto alla diagonale della bbox
  precedente
- overlap (IoU) molto basso

Quando questo accade:

- il nome del video viene segnalato nel log
- viene registrato anche il frame con `previous_bbox` e `current_bbox`

Il log si trova in:

- `.../annotations/dataset_exceptions.log`

### 6. Overlay delle immagini finali

Per ogni immagine finale in:

- `train2017`
- `val2017`
- `test2017`

viene generata automaticamente anche la corrispondente immagine:

- `*_with-KP.jpg`

Gli overlay contengono:

- skeleton
- punti keypoint

Gli overlay non contengono etichette testuali.

## Struttura output della pipeline consolidata

Per un dataset o per un singolo video, la pipeline scrive in una directory
dedicata:

- immagini split
- overlay `*_with-KP`
- file COCO JSON di train/val/test
- log delle eccezioni
- report di preparazione
- config di training generato

Nel caso della variante consolidata, la directory termina tipicamente con:

- `_train_vitposepp_swap_ears`

Il nome storico e rimasto per compatibilita, ma la variante attuale scambia sia
orecchie sia occhi.

## Comando operativo consolidato

Per eseguire la pipeline consolidata, usare il wrapper:

```bash
conda run -n vitpose python script/prepare_swimxyz_vitposepp_train.py
```

Per verifiche puntuali su video/frame, usare:

```bash
conda run -n vitpose python script/kp_check_swimxyz_video_frames.py
```
