import cv2
import os

# Costanti
VIDEO_PATH = "/home/albertosco/HPE/data/input/SwimXYZ_Initial/position_3,75.webm"
FRAME_INDEX = 10  # Indica qui l'indice del frame desiderato (0 è il primo)

def salva_frame_specifico(video_input, frame_to_save):
    # 1. Verifica se il file esiste
    if not os.path.exists(video_input):
        print(f"Errore: Il file '{video_input}' non esiste.")
        return

    # 2. Inizializza la cattura del video (Video Capture)
    cap = cv2.VideoCapture(video_input)
    
    if not cap.isOpened():
        print(f"Errore: Impossibile aprire il file video '{video_input}'.")
        return

    # Recupera il numero totale di frame per evitare errori di out-of-bounds
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_to_save >= total_frames:
        print(f"Errore: Il frame richiesto ({frame_to_save}) eccede il totale dei frame ({total_frames}).")
        cap.release()
        return

    # 3. Imposta la posizione del frame (Seek)
    # CAP_PROP_POS_FRAMES imposta l'indice del prossimo frame da catturare
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_save)

    # 4. Legge il frame specifico
    success, frame = cap.read()

    if success:
        # Costruisce il nome del file di output includendo l'indice del frame
        base_path = os.path.splitext(video_input)[0]
        output_path = f"{base_path}_frame_{frame_to_save}.png"

        # 5. Salva l'immagine
        cv2.imwrite(output_path, frame)
        print(f"Successo: Frame {frame_to_save} salvato come '{output_path}'")
    else:
        print(f"Errore: Impossibile leggere il frame {frame_to_save}.")

    # Rilascia la risorsa
    cap.release()

if __name__ == "__main__":
    salva_frame_specifico(VIDEO_PATH, FRAME_INDEX)