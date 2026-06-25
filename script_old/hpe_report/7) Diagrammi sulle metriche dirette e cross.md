  
Questo tutorial spiega a cosa serve e come usare lo script:  
  
```text  
script/report/build_hpe_report_tables.py  
```  
  
Lo script produce un file Excel con le tabelle del report HPE, a partire da:  
  
- GT COCO keypoints dei dataset `SAW_frames` e `SAW_frames_EntireSwim`;  
- JSON delle predizioni YOLO26x-Pose;  
- JSON delle predizioni della pipeline YOLO26x-Detection -> VitPose++;  
- CSV delle metriche di validation per ricostruire la tabella Val/Test.  
  
Output di default:  
  
```text  
data/output/experiments/hpe_report/hpe_report_tables.xlsx  
```  
  
Lo script e' pensato per server Linux headless: non richiede Excel, LibreOffice, notebook, GUI o Codex. La GPU V100 non e' necessaria per questi calcoli, perche' lo script fa solo analisi/evaluation su JSON e CSV gia' prodotti.  
  
---  
  
## 1. Cosa produce  
  
Lo script genera un unico file XLSX con una scheda per slide:  
  
```text  
Slide 8  
Slide 9  
Slide 10  
Slide 16  
Slide 17  
Slide 18  
Slide 19  
Slide 20  
```  
  
Le tabelle vengono formattate in modo simile al report PPTX. Se una slide contiene piu' tabelle, vengono scritte nello stesso foglio, una dopo l'altra partendo da sinistra.  
  
Le tabelle coperte sono:  
  
| Slide | Contenuto principale |  
|---:|---|  
| 8 | AP/AR Val e Test sui due dataset |  
| 9 | accuratezza geometrica globale: mean, median, P90, PCK |  
| 10 | affidabilita': frames con predizioni, missing visible KP, invisible KP |  
| 16 | cross-test AP/AR con e senza threshold |  
| 17 | errore mean / median / P90 per singolo KP nel cross-test |  
| 18 | direct vs cross per singolo KP |  
| 19 | difficolta' KP con P90 combinato |  
| 20 | vantaggio VitPose++ vs YOLO sui singoli KP |  
  
---  
  
## 2. Comando base  
  
Esempio dal root del progetto HPE:  
  
```bash  
python script/report/build_hpe_report_tables.py \  
--config script/report/hpe_report_config.example.json \  
--output data/output/experiments/hpe_report/hpe_report_tables.xlsx  
```  
  
Con esportazione anche dei CSV intermedi:  
  
```bash  
python script/report/build_hpe_report_tables.py \  
--config script/report/hpe_report_config.example.json \  
--output data/output/experiments/hpe_report/hpe_report_tables.xlsx \  
--export-intermediate-csv  
```  
  
Solo validazione dei file e dei calcoli, senza creare XLSX:  
  
```bash  
python script/report/build_hpe_report_tables.py \  
--config script/report/hpe_report_config.example.json \  
--validate-only  
```  
  
Generare solo alcune slide:  
  
```bash  
python script/report/build_hpe_report_tables.py \  
--config script/report/hpe_report_config.example.json \  
--slides 8,16,17,18,19,20  
```  
  
---  
  
## 3. Config JSON  
  
Lo script usa un file JSON per evitare problemi con path Linux/Windows e per non dover passare molti path da riga di comando.  
  
Esempio minimo:  
  
```json  
{  
"project_root": "/home/albertosco/HPE",  
"paths": {  
"gt": {  
"saw_frames": "data/intermediate/SAW_frames/_train_canonical/annotations/person_keypoints_test.json",  
"saw_frames_entireswim": "data/intermediate/SAW_frames_EntireSwim/_train_canonical/annotations/person_keypoints_test.json"  
},  
"predictions": {  
"direct": {  
"saw_frames": {  
"yolo": "data/output/experiments/yolo26x-pose_SAW_frames_20260614/kp_Test.json",  
"vitpose": "data/output/experiments/vitpose_SAW_frames_20260615/kp_Test.json"  
},  
"saw_frames_entireswim": {  
"yolo": "data/output/experiments/yolo26x-pose_SAW_frames_EntireSwim_20260612/kp_Test.json",  
"vitpose": "data/output/experiments/vitpose_SAW_frames_EntireSwim_20260612/kp_Test.json"  
}  
},  
"cross": {  
"train_saw_frames_test_saw_frames_entireswim": {  
"yolo": "data/output/experiments/pred_cross-train_20260618/yolo_train-SAW_frames_test-SAW_frames_EntireSwim_T-0/kp_Test.json",  
"vitpose": "data/output/experiments/pred_cross-train_20260618/vitpose_train-SAW_frames_test-SAW_frames_EntireSwim_T-0/kp_Test.json"  
}  
}  
},  
"val_metrics": {  
"saw_frames": {  
"yolo": "runs/yolo26x-pose_SAW_frames_20260614/reports/val_metrics_by_epoch.csv",  
"vitpose": "runs/vitpose_SAW_frames_20260615/reports/val_metrics_by_epoch.csv"  
},  
"saw_frames_entireswim": {  
"yolo": "runs/yolo26x-pose_SAW_frames_EntireSwim_20260612/reports/val_metrics_by_epoch.csv",  
"vitpose": "runs/vitpose_SAW_frames_EntireSwim_20260612/reports/val_metrics_by_epoch.csv"  
}  
}  
}  
}  
```  
  
### Path relativi e assoluti  
  
- Se `project_root` e' impostato, i path relativi vengono risolti rispetto a `project_root`.  
- I path assoluti vengono usati direttamente.  
- Lo script espande anche `~` e variabili ambiente come `$HOME`.  
- I path in JSON vanno scritti come stringhe. Su Windows, usare `\\` oppure preferire `/`.  
  
Esempio Windows-style valido in JSON:  
  
```json  
{  
"project_root": "C:/Users/alberto/HPE"  
}  
```  
  
---  
  
## 4. Parametri CLI  
  
| Parametro | Default | Descrizione |  
|---|---:|---|  
| `--config` | obbligatorio | Path del file JSON di configurazione. |  
| `--output` | `data/output/experiments/hpe_report/hpe_report_tables.xlsx` | File XLSX finale. |  
| `--yolo-threshold` | `0.30` | Soglia keypoint confidence per YOLO nelle tabelle thresholded. |  
| `--vitpose-threshold` | `0.20` | Soglia keypoint confidence per VitPose++ nelle tabelle thresholded. |  
| `--min-delta` | `0.007` | `min_delta` usato per ricostruire la selezione del best epoch Val. |  
| `--patience` | `3` | `patience` usata per ricostruire la selezione del best epoch Val. |  
| `--slides` | `all` | Lista slide da esportare, per esempio `8,16,17,18,19,20`. |  
| `--validate-only` | `false` | Calcola e valida senza scrivere l'XLSX. |  
| `--export-intermediate-csv` | `false` | Esporta CSV intermedi accanto al file XLSX. |  
| `--difficulty-easy-max` | `6.0` | Soglia massima P90 combinato per gruppo `Easy`. |  
| `--difficulty-medium-max` | `9.0` | Soglia massima P90 combinato per gruppo `Medium`. |  
| `--difficulty-high-max` | `12.0` | Soglia massima P90 combinato per gruppo `High`; oltre e' `Challenging`. |  
  
---  
  
## 5. Threshold operative  
  
Default usati nelle analisi:  
  
| Modello | Threshold default |  
|---|---:|  
| YOLO26x-Pose | `0.30` |  
| VitPose++ | `0.20` |  
  
Queste soglie vengono applicate solo nelle tabelle thresholded. Le tabelle without-threshold usano `0` / nessun filtro sullo score.  
  
Esempio con soglie diverse:  
  
```bash  
python script/report/build_hpe_report_tables.py \  
--config script/report/hpe_report_config.example.json \  
--yolo-threshold 0.25 \  
--vitpose-threshold 0.30  
```  
  
Nota metodologica: le confidence dei due modelli non sono necessariamente calibrate nello stesso modo. Quindi le soglie sono parametri operativi della pipeline, non valori direttamente confrontabili in senso assoluto.  
  
---  
  
## 6. Best epoch Val  
  
Per la slide 8, le metriche Val vengono lette dai CSV `val_metrics_by_epoch.csv`.  
  
Default:  
  
```text  
patience = 3  
min_delta = 0.007  
```  
  
Lo script ricostruisce il best epoch secondo questa logica, non necessariamente prendendo il massimo AP assoluto se la procedura di early stopping ha selezionato un epoch diverso.  
  
Esempio con parametri diversi:  
  
```bash  
python script/report/build_hpe_report_tables.py \  
--config script/report/hpe_report_config.example.json \  
--patience 5 \  
--min-delta 0.001  
```  
  
---  
  
## 7. Difficolta' dei keypoint  
  
Per le slide 19 e 20, la difficolta' dei KP e' calcolata sul caso:  
  
```text  
Train SAW_frames -> Test SAW_frames_EntireSwim, thresholded  
```  
  
Il P90 combinato viene calcolato unendo le distribuzioni di errore dei due modelli per lo stesso KP:  
  
```text  
errors_comb(KP) = errors_YOLO(KP) union errors_VitPose(KP)  
P90_comb(KP) = percentile_90(errors_comb(KP))  
```  
  
Classificazione default:  
  
| Gruppo | Regola default |  
|---|---|  
| Easy | `P90_comb <= 6.0` |  
| Medium | `6.0 < P90_comb <= 9.0` |  
| High | `9.0 < P90_comb <= 12.0` |  
| Challenging | `P90_comb > 12.0` |  
  
Esempio con soglie custom:  
  
```bash  
python script/report/build_hpe_report_tables.py \  
--config script/report/hpe_report_config.example.json \  
--difficulty-easy-max 5.5 \  
--difficulty-medium-max 8.5 \  
--difficulty-high-max 11.5  
```  
  
---  
  
## 8. CSV intermedi  
  
Con:  
  
```bash  
--export-intermediate-csv  
```  
  
lo script salva, accanto all'XLSX, file di controllo come:  
  
```text  
scenario_summary.csv  
per_kp_metrics.csv  
val_best_epoch_summary.csv  
```  
  
Questi file servono per:  
  
- controllare i numeri prima di inserirli nel report;  
- confrontare versioni diverse dei threshold;  
- debuggare eventuali differenze tra script e PPTX.  
  
---  
  
## 9. Esempi pratici  
  
### 9.1 Generazione completa del report tables XLSX  
  
```bash  
python script/report/build_hpe_report_tables.py \  
--config script/report/hpe_report_config.example.json  
```  
  
### 9.2 Generazione completa con path output esplicito  
  
```bash  
python script/report/build_hpe_report_tables.py \  
--config script/report/hpe_report_config.example.json \  
--output data/output/experiments/hpe_report/hpe_report_tables.xlsx  
```  
  
### 9.3 Validazione prima della generazione  
  
```bash  
python script/report/build_hpe_report_tables.py \  
--config script/report/hpe_report_config.example.json \  
--validate-only  
```  
  
### 9.4 Rigenerare solo le slide cross-test e KP  
  
```bash  
python script/report/build_hpe_report_tables.py \  
--config script/report/hpe_report_config.example.json \  
--slides 16,17,18,19,20  
```  
  
### 9.5 Rigenerare con CSV intermedi  
  
```bash  
python script/report/build_hpe_report_tables.py \  
--config script/report/hpe_report_config.example.json \  
--export-intermediate-csv  
```  
  
---  
  
## 10. Requisiti Python  
  
Lo script usa:  
  
```text  
numpy  
openpyxl  
pycocotools  
```  
  
Installazione tipica:  
  
```bash  
python -m pip install numpy openpyxl pycocotools  
```  
  
`pycocotools` e' necessario per AP/AR OKS in stile COCO. Se manca, le parti che dipendono da AP/AR COCO non possono essere riprodotte correttamente.  
  
---  
  
## 11. Controlli consigliati  
  
Prima di usare i risultati nel report:  
  
1. lanciare `--validate-only`;  
2. lanciare con `--export-intermediate-csv`;  
3. confrontare `scenario_summary.csv` con le metriche gia' note;  
4. verificare in particolare:  
- numero immagini GT;  
- numero immagini con predizione;  
- missing visible KP;  
- AP/AR Val e Test;  
- soglie effettivamente usate.  
  
---  
  
## 12. Nota sulla valutazione a 12 KP senza testa  
  
Le analisi che escludono i 5 KP testa:  
  
```text  
nose, left_eye, right_eye, left_ear, right_ear  
```  
  
sono valutazioni post-hoc su modelli addestrati a 17 KP. Non equivalgono a valutare un modello addestrato senza quei KP, perche' durante il training la loss e l'aggiornamento dei pesi sono stati influenzati anche dai KP della testa.  
  
Questa distinzione e' importante per interpretare correttamente eventuali confronti 17 KP vs 12 KP.