# Dataset Preparation Tutorial: prepare_entire_swim_dataset.py

## Obiettivo
Questo script prepara il dataset completo per l'analisi dei nuotatori ("Entire Swim") per il training del modello di pose estimation.

## Struttura del Dataset
```
data/
├── input/            # Dati grezzi (video, annotazioni)
│   └── entire_swim/  # Sotto-directory specifica per Entire Swim
├── intermediate/     # Dati processati temporanei
│   └── pose_dataset/ # Output di questo script
└── output/           # Dati pronti per il training finale
```

## Prerequisiti
1. Video registrati in formato MP4 nella directory `data/input/entire_swim/videos/`
2. File di annotazioni in COCO format in `data/input/entire_swim/annotations/`
3. Configurazione in `configs/pipeline/pose/dataset_config.yaml`

## Esecuzione
```bash
python script/prepare_entire_swim_dataset.py \
  --input_dir=data/input/entire_swim \
  --output_dir=data/intermediate/pose_dataset \
  --config=configs/pipeline/pose/dataset_config.yaml
```

## Parametri Chiave
| Parametro | Valore Default | Descrizione |
|-----------|----------------|-------------|
| `--input_dir` | `data/input/entire_swim` | Directory contenente i file grezzi |
| `--output_dir` | `data/intermediate/pose_dataset` | Directory per i file processati |
| `--config` | `configs/pipeline/pose/dataset_config.yaml` | File di configurazione per il preprocessing |

## Fasi di Elaborazione
1. **Sincronizzazione**:
   - Allinea annotazioni temporali con i frame video
   - Correzione di offset tra annotazioni e registrazione

2. **Cropping**:
   ```python
   # Esempio di processing da dataset_config.yaml
   crop_params:
     region_of_interest:
       x: 0.3  # 30% sinistra dello schermo
       y: 0.2  # 20% in alto
       width: 0.5  # 50% della larghezza
       height: 0.6  # 60% dell'altezza
   ```

3. **Normalization**:
   - Ridimensionamento a 640x480 risoluzione standard
   - Conversione a formato BGR -> RGB

4. **Split Dataset**:
   - 70% training, 15% validation, 15% test
   - Salvataggio in formato YOLOv8 (images + labels directory)

## Output Generato
- `images/train/` - Immagini per training
- `images/val/` - Immagini per validation
- `images/test/` - Immagini per testing
- `labels/train/` - File di annotazioni corrispondenti
- `labels/val/` - File di annotazioni corrispondenti
- `labels/test/` - File di annotazioni corrispondenti

## Debug e Troubleshooting
- Per visualizzare i frame processati:
  ```bash
  python script/kp_check_swimxyz_video_frames.py \
    --dataset_dir=data/intermediate/pose_dataset
  ```
- Per forzare il ricampionamento:
  ```bash
  python script/prepare_entire_swim_dataset.py \
    --input_dir=data/input/entire_swim \
    --output_dir=data/intermediate/pose_dataset \
    --force_resample
  ```