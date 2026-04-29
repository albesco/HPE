import json
import os

# --- COSTANTI ---
INPUT_FILE = "/home/albertosco/HPE/data/input/SwimXYZ_Initial/position_1,75/COCO/2D_cam_COCO.txt"  # Il tuo file CSV/TXT con i dati
IMAGE_WIDTH = 1920  # Imposta le dimensioni reali del tuo input
IMAGE_HEIGHT = 1080

def convert_to_vitpose_coco(input_path):
    base_name = os.path.splitext(input_path)[0]
    output_path = f"{base_name}_coco.json"
    
    # Mapping indici: da OpenPose BODY_25 a COCO 17 (ViTPose++)
    # Ordine COCO: 0:Nose, 1:LEye, 2:REye, 3:LEar, 4:REar, 5:LShoulder, 
    # 6:RShoulder, 7:LElbow, 8:RElbow, 9:LWrist, 10:RWrist, 11:LHip, 
    # 12:RHip, 13:LKnee, 14:RKnee, 15:LAnkle, 16:RAnkle
    mapping = [0, 15, 16, 17, 18, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}]
    }

    if not os.path.exists(input_path):
        print(f"Errore: File {input_path} non trovato.")
        return

    with open(input_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # Salto l'intestazione se la prima riga contiene testo (Nose)
    start_idx = 1 if "Nose" in lines[0] else 0
    data_lines = lines[start_idx:]

    print(f"Rilevate {len(data_lines)} righe di dati. Elaborazione in corso...")

    for i, line in enumerate(data_lines):
        # 1. Parsing: virgola -> punto e split
        raw_vals = line.replace(',', '.').split(';')
        coords = [float(v) for v in raw_vals if v]
        
        # Raggruppo in triplette (x, y, confidence/z)
        all_points = [coords[j:j+3] for j in range(0, len(coords), 3)]
        
        # 2. Generazione dati Immagine
        img_id = i + 1
        img_filename = f"{base_name}_{i}.png"
        coco_data["images"].append({
            "id": img_id,
            "file_name": img_filename,
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT
        })

        # 3. Mapping Keypoints e calcolo BBox
        coco_kpts = []
        x_coords, y_coords = [], []

        for idx in mapping:
            # Controllo se l'indice esiste (sicurezza per file troncati)
            if idx < len(all_points):
                x, y, conf = all_points[idx]
                v = 2 if conf > 0 else 0 # 2=visibile, 0=non presente
                coco_kpts.extend([round(x, 2), round(y, 2), v])
                if v > 0:
                    x_coords.append(x)
                    y_coords.append(y)
            else:
                coco_kpts.extend([0, 0, 0])

        # Calcolo Bounding Box [x, y, w, h]
        if x_coords and y_coords:
            xmin, ymin = min(x_coords), min(y_coords)
            w, h = max(x_coords) - xmin, max(y_coords) - ymin
            # Aggiunta padding del 10% (standard per modelli top-down come ViTPose)
            bbox = [round(xmin - w*0.05, 2), round(ymin - h*0.05, 2), 
                    round(w * 1.1, 2), round(h * 1.1, 2)]
        else:
            bbox = [0, 0, 0, 0]

        # 4. Creazione Annotazione
        coco_data["annotations"].append({
            "id": img_id,
            "image_id": img_id,
            "category_id": 1,
            "keypoints": coco_kpts,
            "bbox": bbox,
            "area": round(bbox[2] * bbox[3], 2),
            "iscrowd": 0,
            "num_keypoints": sum(1 for v in coco_kpts[2::3] if v > 0)
        })

    # Salvataggio finale
    with open(output_path, 'w') as out_f:
        json.dump(coco_data, out_f, indent=4)
    
    print(f"File salvato con successo: {output_path}")

if __name__ == "__main__":
    convert_to_vitpose_coco(INPUT_FILE)