import cv2
import os

# Costante per il percorso del file video
VIDEO_PATH = "/home/albertosco/HPE/data/input/SwimXYZ_Initial/position_3,75.webm"

def salva_primo_frame(video_input):
    # 1. Verifica se il file esiste
    if not os.path.exists(video_input):
        print(f"Errore: Il file '{video_input}' non esiste.")
        return

    # 2. Inizializza la cattura del video
    cap = cv2.VideoCapture(video_input)
    
    if not cap.isOpened():
        print(f"Errore: Impossibile aprire il file video '{video_input}'.")
        return

    # 3. Legge il primo frame
    success, frame = cap.read()

    if success:
        # Costruisce il nome del file di output
        # splitext separa il path dall'estensione (.webm)
        base_path = os.path.splitext(video_input)[0]
        output_path = f"{base_path}_0.png"

        # 4. Salva l'immagine in formato PNG
        cv2.imwrite(output_path, frame)
        print(f"Successo: Primo frame salvato come '{output_path}'")
    else:
        print("Errore: Impossibile leggere il frame dal video.")

    # Rilascia la risorsa
    cap.release()

if __name__ == "__main__":
    salva_primo_frame(VIDEO_PATH)