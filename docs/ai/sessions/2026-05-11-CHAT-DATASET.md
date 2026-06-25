# 2026-05-11 | CHAT-DATASET

## 1. Obiettivo della sessione
- Definire e implementare la conversione del dataset SwimXYZ nel formato richiesto da VitPose++/MMPose.
- Chiarire lo schema dei file immagine+label di SwimXYZ e il formato target di training.

## 2. Attività svolte
- Lettura dei documenti AI condivisi:
  - `AGENTS.md`
  - `docs/ai/context.md`
  - `docs/ai/chat-index.md`
  - `docs/ai/chat-roles.md`
  - `docs/ai/decision-log.md`
  - `docs/ai/task-board.md`
- Verifica dello stato corrente del repository rispetto alla conversione SwimXYZ -> VitPose++.
- Verifica dello stack di script attualmente presenti per:
  - normalizzazione del dataset sorgente;
  - parsing delle label SwimXYZ;
  - generazione dataset COCO17 / MMPose;
  - controllo qualitativo frame/label.
- Creazione di questa nota di sessione sotto `docs/ai/sessions/`.

## 3. File letti/modificati

### Letti
- `AGENTS.md`
- `docs/ai/context.md`
- `docs/ai/chat-index.md`
- `docs/ai/chat-roles.md`
- `docs/ai/decision-log.md`
- `docs/ai/task-board.md`
- `script/dataset_preparation/prepare_swimxyz_vitposepp_utils.py`
- `script_old/prepare_swimxyz_vitposepp.py`
- `script_old/kp_check_swimxyz_video_frames.py`

### Modificati
- `docs/ai/sessions/2026-05-11-CHAT-DATASET.md`

## 4. Decisioni prese
- Non risultano decisioni dataset-format aggiunte in `docs/ai/decision-log.md` per questa sessione specifica.
- Dalla repository corrente si ricava che il workflow attivo usa una pipeline consolidata SwimXYZ -> COCO17 -> VitPose++ standard.
- L’adozione esatta di questa pipeline durante la sessione originaria del `2026-05-11` è `not reconstructible from this session`.

## 5. Schema SwimXYZ ricostruito o ancora incerto

### Ricostruito
- I dati SwimXYZ sorgente sono organizzati come video + file label.
- La repository corrente usa `data/input/subset_xyz` come sorgente normalizzata e `data/intermediate/<dataset>/_converted/` come layout intermedio.
- I file label “COCO” di SwimXYZ:
  - mantengono un header BODY25;
  - ma possono essere serializzati nel formato compatto `SWIMXYZ_COCO18_ORDER` definito in `script/dataset_preparation/prepare_swimxyz_vitposepp_utils.py`.
- In `script/dataset_preparation/prepare_swimxyz_vitposepp_utils.py` sono esplicitati:
  - `EXPECTED_TRAILING_MISSING_HEADERS`
  - `SWIMXYZ_COCO18_ORDER`
  - `BODY25_TO_COCO`
- Il parser corrente gestisce due casi:
  - righe complete coerenti con l’header;
  - righe compatte a 18 keypoint (`SWIMXYZ_COCO18_ORDER`) quando l’header è più lungo.
- Il workflow corrente prevede opzionalmente il flip dell’asse Y:
  - da origine in basso (SwimXYZ) a origine in alto (immagine MMPose).

### Ancora incerto
- Lo schema sorgente completo “canonico” di SwimXYZ al di fuori del subset normalizzato in repo è `not reconstructible from this session`.
- Non è ricostruibile da questa sessione se tutte le varianti label (`base`, `body25`, `COCO`, `2D`, `pelvis`, `cam`) siano già state pienamente consolidate o se siano rimaste ipotesi intermedie.

## 6. Formato target VitPose++/MMPose
- Il target corrente del repository è un dataset top-down COCO17 compatibile con MMPose/VitPose++.
- Elementi ricostruibili dai file correnti:
  - split `train / val / test`;
  - immagini in directory stile `train2017`, `val2017`, `test2017`;
  - annotazioni JSON COCO-style con:
    - `images`
    - `annotations`
    - `categories`
    - `bbox`
    - `keypoints`
    - `num_keypoints`
    - `area`
  - 17 keypoint target COCO;
  - training config standard basata su:
    - `src/vitpose_base/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitPose+_huge_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py`
- La repository corrente indica anche una variante attiva “swap ears / swap eyes” in:
  - `script_old/prepare_swimxyz_vitposepp.py`
- Se questa ipotesi facesse già parte della sessione originaria del `2026-05-11` è `not reconstructible from this session`.

## 7. Script di conversione creati o modificati

### Ricostruibili nella repository corrente
  - oggi funge da step di normalizzazione/riorganizzazione del dataset sorgente verso `data/intermediate`.
- `script/dataset_preparation/prepare_swimxyz_vitposepp_utils.py`
  - utility condivise per parsing label, mapping SwimXYZ -> COCO17, bbox, split, export JSON.
- `script_old/prepare_swimxyz_vitposepp.py`
  - entrypoint di preparazione dataset/train config per la pipeline attiva standard.
- `script_old/prepare_swimxyz_vitposepp_train.py`
  - wrapper/entrypoint del flusso di preparazione training; nello stato corrente della repository ingloba anche la normalizzazione del dataset sorgente quando richiesto.

### Non ricostruibile con certezza
- L’elenco esatto degli script creati o modificati durante la sessione originaria del `2026-05-11` è `not reconstructible from this session`.
- Nella chat di lavoro esistono riferimenti a script precedenti o percorsi oggi non presenti come file attivi; la loro cronologia esatta non è ricostruibile dal solo stato corrente della repository.

## 8. Script di validazione
- `script_old/kp_check_swimxyz_video_frames.py`
  - valida l’allineamento frame/label renderizzando keypoint e, opzionalmente, bbox.
- `script_old/preview_test_predictions.py`
  - presente in repo come strumento di preview qualitativa su dataset preparato.
- `script_old/compare_test_overlays.py`
  - presente in repo per confronto overlay.
- Non esiste, nello stato corrente, un file chiaramente nominato `validate_vitpose_dataset.py`; quindi il nome preciso di un eventuale validatore originario è `not reconstructible from this session`.

## 9. Test eseguiti
- Test eseguiti e persistenti/referenziati in repository per questa sessione specifica: `not reconstructible from this session`.
- Dai file letti si ricava solo che esistono script di preview/validation qualitativa, ma non la lista completa dei test effettivamente lanciati nel `2026-05-11`.

## 10. Risultati
- Stato corrente ricostruibile:
  - esiste una pipeline consolidata SwimXYZ -> dataset COCO17 -> VitPose++ standard;
  - il parsing dei file SwimXYZ include gestione esplicita di righe compatte / trailing missing headers;
  - è disponibile almeno uno script per verifica qualitativa frame/label.
- Il risultato preciso conseguito al termine della sessione originaria del `2026-05-11` è `not reconstructible from this session`.

## 11. Problemi aperti
- `docs/ai/task-board.md` mantiene ancora in backlog:
  - definizione esatta dello schema input SwimXYZ;
  - definizione esatta del formato target VitPose++;
  - implementazione script di conversione;
  - aggiunta di uno script di validazione;
  - creazione config di training;
  - creazione launcher di training;
  - documentazione del workflow riproducibile.
- Questo backlog non è pienamente allineato allo stato corrente degli script presenti in repo; la discrepanza va risolta.
- Mancano nel `decision-log` decisioni dataset-specifiche esplicitamente registrate per la fase CHAT-DATASET.
- La cronologia precisa di migrazione dagli script “trial” agli script consolidati non è ricostruibile con certezza.

## 12. Prossimi passi
- Allineare `docs/ai/task-board.md` allo stato reale della repository per la parte dataset conversion.
- Registrare in `docs/ai/decision-log.md` le decisioni di schema ormai consolidate, se considerate definitive:
  - rappresentazione SwimXYZ accettata;
  - mapping verso COCO17;
  - convenzione asse Y;
  - policy di split train/val/test;
  - eventuale ipotesi “swap ears / swap eyes”.
- Valutare se creare una documentazione dataset dedicata, ad esempio `docs/dataset-format.md`, se richiesta dal team.
- Verificare se serva una nota separata per distinguere:
  - script legacy / `old_trials`
  - pipeline attiva consolidata.

---

# Aggiornamento 2026-05-15 | CHAT-DATASET

Questa sezione aggiorna la ricostruzione della sessione usando i file del
repository e gli artefatti locali verificabili. Dove la cronologia originaria
non e ricostruibile, viene indicato esplicitamente.

## 1. Obiettivo della sessione

Documentare e consolidare il lavoro del ruolo `CHAT-DATASET` relativo alla
conversione del subset SwimXYZ nel formato di training VitPose++/MMPose.

Obiettivo tecnico ricostruibile:

- normalizzare i dataset SwimXYZ da `data/input/subset_xyz`;
- produrre dataset COCO17 top-down compatibili con VitPose++/MMPose;
- validare allineamento frame/label e qualita delle annotazioni;
- mantenere tracciate le decisioni dataset in `docs/ai/`.

La cronologia completa della sessione originaria del 2026-05-11 e `not reconstructible from this session`.

## 2. Attivita svolte

Ricostruibile dai file correnti e dalla documentazione AI:

- Lettura dei documenti AI condivisi:
  - `AGENTS.md`
  - `docs/ai/context.md`
  - `docs/ai/chat-index.md`
  - `docs/ai/chat-roles.md`
  - `docs/ai/decision-log.md`
  - `docs/ai/task-board.md`
  - `docs/ai/tests-and-results.md`
- Verifica degli script attivi per conversione dataset, preparazione VitPose++ e validazione.
- Verifica della directory dataset locale `data/intermediate/Side_above_water/_train_vitposepp_swap_ears/`.
- Aggiornamento di questa nota di sessione.

Attivita precise della chat originaria del 2026-05-11: `not reconstructible from this session`.

## 3. File letti/modificati

### Letti

- `AGENTS.md`
- `docs/ai/context.md`
- `docs/ai/chat-index.md`
- `docs/ai/chat-roles.md`
- `docs/ai/decision-log.md`
- `docs/ai/task-board.md`
- `docs/ai/tests-and-results.md`
- `script/dataset_preparation/prepare_swimxyz_vitposepp_utils.py`
- `script_old/prepare_swimxyz_vitposepp.py`
- `script_old/remove_anomalous_frames_from_dataset.py`
- `data/intermediate/Side_above_water/_train_vitposepp_swap_ears/reports/preparation_report.json`

### Modificati

- `docs/ai/sessions/2026-05-11-CHAT-DATASET.md`

Altri file modificati nella working tree al momento della lettura risultano presenti, ma non sono stati modificati da questa operazione di documentazione: `not reconstructible from this session`.

## 4. Decisioni prese

Decisioni o convenzioni dataset ricostruibili dallo stato corrente:

- Il formato target attivo e COCO17 top-down compatibile con MMPose/VitPose++.
- La pipeline attiva usa VitPose++ standard.
- Le coordinate Y SwimXYZ vengono convertite con `flip_y=True`.
- `z` nelle label SwimXYZ non viene trattata come confidence.
- La visibility viene derivata dalla posizione X:
  - `v = 0` se il punto e sul bordo orizzontale (`x == 0` oppure `x == max_x`);
  - `v = 2` se `0 < x < max_x`.
- La pipeline attiva scambia `LEye/REye` e `LEar/REar` prima della conversione a COCO17.
- I frame con brusco salto della bbox rispetto al frame precedente vengono esclusi dal flusso dataset e loggati.
- Le bbox GT preparate nello stato corrente usano padding anisotropico:
  - `bbox_padding_x_ratio = 0.20`;
  - `bbox_padding_y_ratio = 0.25`;
  - `bbox_min_padding_px = 15.0`.

Queste decisioni non risultano tutte registrate in `docs/ai/decision-log.md`; la data esatta in cui ciascuna e stata presa e `not reconstructible from this session`.

## 5. Schema SwimXYZ ricostruito o ancora incerto

### Ricostruito

- Dataset sorgente: `data/input/subset_xyz/<dataset>/`.
- Ogni dataset sorgente contiene video `.webm` e label `.txt`.
- Sono presenti piu rappresentazioni/varianti label, tra cui `base`, `body25`, `COCO`, `2D`, `3D`, `cam`, `pelvis`.
- Lo step `_converted` riorganizza i dati in:
  - `data/intermediate/<dataset>/_converted/<video_stem>/<representation>/<label_kind>_<representation>.txt`;
  - `data/intermediate/<dataset>/_converted/manifest.json`.
- I file SwimXYZ COCO possono usare header BODY25 ma righe compatte con ordine `SWIMXYZ_COCO18_ORDER`.
- Le utility attive dichiarano:
  - `EXPECTED_TRAILING_MISSING_HEADERS`;
  - `SWIMXYZ_COCO18_ORDER`;
  - `BODY25_TO_COCO`.
- Il parser gestisce righe complete e righe COCO compatte a 18 keypoint.

### Ancora incerto

- Schema canonico completo di SwimXYZ fuori dal subset locale: `not reconstructible from this session`.
- Significato completo e ufficiale di tutte le varianti `base/body25/COCO`, `2D/3D`, `cam/pelvis`: `not reconstructible from this session`.
- Motivazione originale dello scambio occhi/orecchie: ricostruibile come ipotesi consolidata nel codice, ma prova esterna definitiva non presente nei documenti letti.

## 6. Formato target VitPose++/MMPose

Formato target corrente ricostruibile:

- Dataset COCO-style top-down per MMPose/VitPose++.
- Split:
  - `train2017`
  - `val2017`
  - `test2017`
- Annotazioni:
  - `annotations/person_keypoints_train.json`
  - `annotations/person_keypoints_val.json`
  - `annotations/person_keypoints_test.json`
- Struttura JSON:
  - `images`
  - `annotations`
  - `categories`
  - `bbox`
  - `area`
  - `num_keypoints`
  - `keypoints`
- Keypoint target: COCO17.
- Config generato: VitPose++ standard basato su config MMPose/VitPose++ sotto `src/vitpose_base/configs/.../vitPose+_huge_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py`.
- Overlay di verifica:
  - generati come `*_with-KP.jpg`;
  - contengono skeleton e punti;
  - non contengono label testuali.

## 7. Script di conversione creati o modificati

Script attivi ricostruibili:

  - normalizza dataset SwimXYZ sorgenti in `_converted`;
  - copia video e label;
  - genera `manifest.json`;
  - genera sidecar `*_coco.json` per label COCO 2D.
- `script/dataset_preparation/prepare_swimxyz_vitposepp_utils.py`
  - parsing label SwimXYZ;
  - mapping SwimXYZ -> COCO17;
  - calcolo visibility;
  - calcolo bbox;
  - split train/val/test;
  - export JSON COCO;
  - overlay `*_with-KP.jpg`.
- `script_old/prepare_swimxyz_vitposepp.py`
  - prepara dataset VitPose++ standard;
  - applica swap occhi/orecchie;
  - rileva e scarta frame con bbox anomala;
  - genera config di training.
- `script_old/prepare_swimxyz_vitposepp_train.py`
  - wrapper CLI della pipeline ufficiale.
- `script_old/remove_anomalous_frames_from_dataset.py`
  - rimuove da un dataset gia costruito i frame indicati dal log `anomalous_bbox_shift`;
  - aggiorna immagini, overlay e JSON COCO.

Quali script siano stati creati esattamente durante la sessione originaria del 2026-05-11 e `not reconstructible from this session`.

## 8. Script di validazione

Script di validazione o controllo qualitativo ricostruibili:

- `script_old/kp_check_swimxyz_video_frames.py`
  - renderizza frame specifici o tutti i frame di un video;
  - usa label SwimXYZ e mapping COCO17;
  - puo mostrare skeleton, punti e bbox;
  - usato per controllare allineamento frame/label.
- `script_old/compare_test_overlays.py`
  - script presente nel repository per confronto overlay.
- `script_old/preview_test_predictions.py`
  - script presente nel repository per preview predizioni.
- `script_old/visualize_gt_bboxes.py`
  - script presente nel repository per visualizzare bbox GT.
- `script_old/visualize_gt_vs_pred_keypoints.py`
  - script presente nel repository per confronto GT vs predizioni.

Un validatore formale chiamato `validate_vitpose_dataset.py` non risulta presente nei file letti; se sia esistito nella sessione originaria e `not reconstructible from this session`.

## 9. Test eseguiti

Test e verifiche ricostruibili da repository e dati locali:

- Verifica del dataset locale `Side_above_water` tramite `preparation_report.json`.
- Conteggio attuale JSON COCO locali:
  - train: `18181` immagini / `18181` annotazioni;
  - val: `5195` immagini / `5195` annotazioni;
  - test: `2597` immagini / `2597` annotazioni.
- Verifica documentale dei risultati registrati in `docs/ai/tests-and-results.md`:
  - YOLO anisotropic-padding smoke training completato per 5 epoche;
  - VitPose++ anisotropic GT training fermato dopo `epoch_4.pth`.

Comandi esatti dei test dataset eseguiti nella sessione originaria del 2026-05-11: `not reconstructible from this session`.

## 10. Risultati

Risultati dataset ricostruibili dallo stato corrente:

- Dataset attivo:
  - `data/intermediate/Side_above_water/_train_vitposepp_swap_ears/`
- Dataset sorgente convertito:
  - `data/intermediate/Side_above_water/_converted/`
- Report locale:
  - `data/intermediate/Side_above_water/_train_vitposepp_swap_ears/reports/preparation_report.json`
- Config generato:
  - `data/intermediate/Side_above_water/_train_vitposepp_swap_ears/generated_configs/swimxyz_vitposepp_huge_vitposepp_swap_ears.py`
- Conteggi correnti dopo esclusione anomalie:
  - train: `18181`
  - val: `5195`
  - test: `2597`
  - totale: `25973`
- Skips registrati nel report:
  - `missing_frame_for_label`: `100`
  - `missing_label_for_frame`: `0`
  - `no_valid_keypoints`: `4099`
  - `anomalous_bbox_shift`: `28`
- Dataset `Side_above_water` processato su `100` video.

Il risultato preciso conseguito al termine della sola sessione originaria del 2026-05-11 e `not reconstructible from this session`.

## 11. Problemi aperti

- `docs/ai/task-board.md` contiene ancora backlog dataset non pienamente allineato allo stato corrente degli script.
- Alcune decisioni dataset consolidate nel codice non risultano ancora registrate esplicitamente in `docs/ai/decision-log.md`.
- Il nome directory `_train_vitposepp_swap_ears` resta storico: la pipeline attuale scambia sia occhi sia orecchie.
- Validatore formale completo per dataset COCO/VitPose++ non identificato come script dedicato unico.
- Provenienza ufficiale della convenzione di swap occhi/orecchie: `not reconstructible from this session`.
- Schema SwimXYZ completo fuori dal subset locale: `not reconstructible from this session`.

## 12. Prossimi passi

- Allineare `docs/ai/task-board.md` allo stato reale della pipeline dataset.
- Registrare in `docs/ai/decision-log.md` le decisioni dataset ormai consolidate:
  - mapping SwimXYZ -> COCO17;
  - flip Y;
  - visibility derivata da X;
  - swap occhi/orecchie;
  - esclusione frame con bbox anomala;
  - padding bbox anisotropico.
- Valutare rinomina futura degli output storici `_train_vitposepp_swap_ears` se si vuole eliminare ambiguita terminologica.
- Creare o consolidare uno script formale di validazione dataset COCO/VitPose++ se serve riproducibilita automatica.
- Aggiornare `docs/ai/tests-and-results.md` con comandi esatti quando vengono rieseguite validazioni dataset.
