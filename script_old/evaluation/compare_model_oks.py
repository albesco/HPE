
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict

# --- Costanti COCO ---
# Queste costanti sono standard per il dataset COCO con 17 keypoint.
# Vengono usate nella formula OKS per pesare diversamente l'errore su ciascun keypoint.
COCO_KEYPOINT_SIGMAS = np.array([
    .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89
])

COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

def calculate_oks(gt_kps, pred_kps, gt_area, sigmas):
    """
    Calcola l'Object Keypoint Similarity (OKS) per ogni keypoint.
    
    Args:
        gt_kps (list): Keypoints del ground truth [x1, y1, v1, ...].
        pred_kps (list): Keypoints predetti [x1, y1, c1, ...].
        gt_area (float): Area del bounding box del ground truth.
        sigmas (np.array): Array delle deviazioni standard per ogni keypoint.

    Returns:
        np.array: Un array con il valore OKS per ciascuno dei 17 keypoint.
    """
    oks = np.zeros(len(sigmas))
    gt_kps = np.array(gt_kps).reshape(-1, 3)
    pred_kps = np.array(pred_kps).reshape(-1, 3)

    for i in range(len(sigmas)):
        gt_kp = gt_kps[i]
        pred_kp = pred_kps[i]
        
        # Considera solo i keypoint visibili nel ground truth (v=1 o v=2)
        if gt_kp[2] > 0:
            d_squared = (gt_kp[0] - pred_kp[0])**2 + (gt_kp[1] - pred_kp[1])**2
            # L'area del bbox non può essere 0 per evitare divisioni per zero
            # s (scala) è la radice quadrata dell'area del segmento
            s_squared = gt_area if gt_area > 0 else 1.0
            k_squared = sigmas[i]**2
            
            # Formula OKS
            oks[i] = np.exp(-d_squared / (2 * s_squared * k_squared))
        else:
            # Se il keypoint non è visibile nel GT, l'OKS non è definito per quel punto
            oks[i] = np.nan

    return oks

def process_model_predictions(gt_annotations, pred_annotations, sigmas):
    """
    Processa le predizioni di un modello per calcolare l'OKS medio per keypoint.
    
    Args:
        gt_annotations (dict): Annotazioni del ground truth (formato COCO).
        pred_annotations (dict): Predizioni del modello (formato COCO).
        sigmas (np.array): Sigmas dei keypoint.

    Returns:
        np.array: Array con l'OKS medio per ciascun keypoint.
    """
    gt_by_image = defaultdict(list)
    for ann in gt_annotations['annotations']:
        gt_by_image[ann['image_id']].append(ann)

    pred_by_image = defaultdict(list)
    for ann in pred_annotations:
        pred_by_image[ann['image_id']].append(ann)

    all_oks_scores = [[] for _ in range(len(sigmas))]

    # Itera su ogni immagine che ha un'annotazione ground truth
    for image_id, gt_anns in gt_by_image.items():
        pred_anns = pred_by_image.get(image_id, [])
        if not pred_anns:
            continue

        # Per ogni persona nel ground truth...
        for gt_ann in gt_anns:
            # ...trova la migliore predizione corrispondente
            best_match_oks = None
            
            # Scegli la predizione che massimizza l'OKS medio sulla persona
            # (strategia di matching comune)
            best_oks_avg = -1

            for pred_ann in pred_anns:
                oks_per_kp = calculate_oks(gt_ann['keypoints'], pred_ann['keypoints'], gt_ann['area'], sigmas)
                
                # Calcola OKS medio per la persona, ignorando i NaN
                avg_oks = np.nanmean(oks_per_kp)

                if avg_oks > best_oks_avg:
                    best_oks_avg = avg_oks
                    best_match_oks = oks_per_kp
            
            if best_match_oks is not None:
                for i in range(len(sigmas)):
                    if not np.isnan(best_match_oks[i]):
                        all_oks_scores[i].append(best_match_oks[i])

    # Calcola la media per ciascun keypoint
    mean_oks_per_keypoint = [np.mean(scores) if scores else 0 for scores in all_oks_scores]
    return np.array(mean_oks_per_keypoint)

def plot_comparison_histogram(model_results, keypoint_names, output_path):
    """
    Crea e salva un istogramma che confronta l'OKS medio per keypoint tra i modelli.
    
    Args:
        model_results (dict): Dizionario con {nome_modello: array_oks_medi}.
        keypoint_names (list): Nomi dei keypoint per l'asse X.
        output_path (str): Path dove salvare il file PNG.
    """
    n_keypoints = len(keypoint_names)
    n_models = len(model_results)
    
    index = np.arange(n_keypoints)
    bar_width = 0.8 / n_models
    
    fig, ax = plt.subplots(figsize=(20, 10))

    for i, (model_name, oks_scores) in enumerate(model_results.items()):
        pos = index - (bar_width * (n_models - 1) / 2) + (i * bar_width)
        ax.bar(pos, oks_scores, bar_width, label=model_name)

    ax.set_ylabel('Average OKS', fontsize=14)
    ax.set_title('Model Comparison: Average OKS per Keypoint', fontsize=16)
    ax.set_xticks(index)
    ax.set_xticklabels(keypoint_names, rotation=45, ha="right", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Grafico di comparazione salvato in: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compara le performance di modelli di pose estimation tramite OKS.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--gt-file', 
        required=True, 
        help="Path al file JSON delle annotazioni ground truth (formato COCO)."
    )
    parser.add_argument(
        '--model-preds', 
        nargs='+', 
        required=True,
        help="""Lista di predizioni dei modelli.
Formato per ogni modello: 'NomeModello:path/al/file_predizioni.json'
Esempio: --model-preds 'Yolo-Pose:preds/yolo.json' 'VitPose++:preds/vitpose.json'"""
    )
    parser.add_argument(
        '--output-png', 
        default='oks_comparison.png', 
        help="Path dove salvare l'istogramma di comparazione."
    )

    args = parser.parse_args()

    # Carica il ground truth
    with open(args.gt_file, 'r') as f:
        gt_annotations = json.load(f)

    model_results = {}

    # Processa le predizioni per ogni modello
    for model_arg in args.model_preds:
        try:
            model_name, pred_path = model_arg.split(':', 1)
        except ValueError:
            print(f"Argomento malformato: {model_arg}. Usare il formato 'NomeModello:path/al/file.json'")
            continue
        
        print(f"Processing modello: {model_name}...")
        with open(pred_path, 'r') as f:
            pred_annotations = json.load(f)
        
        mean_oks = process_model_predictions(gt_annotations, pred_annotations, COCO_KEYPOINT_SIGMAS)
        model_results[model_name] = mean_oks
        
        print(f"OKS medi per {model_name}:")
        for i, name in enumerate(COCO_KEYPOINT_NAMES):
            print(f"  - {name}: {mean_oks[i]:.4f}")

    # Genera il grafico
    if model_results:
        plot_comparison_histogram(model_results, COCO_KEYPOINT_NAMES, args.output_png)

if __name__ == '__main__':
    main()
