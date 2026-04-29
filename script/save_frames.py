import cv2
import os

# Costanti
VIDEO_PATH = "/home/albertosco/HPE/data/input/A00049_1_2_001_L2R_resized_(verticale).mp4"
FRAME_PATH = "/home/albertosco/HPE/data/output/A00049_1_2_001_L2R_resized_(verticale)"  # Directory di destinazione

def salva_tutti_i_frame(video_input, output_dir):
    # 1. Verifica se il file video esiste
    if not os.path.exists(video_input):
        print(f"Errore: Il file '{video_input}' non esiste.")
        return

    # 2. Crea la directory di destinazione se non esiste (Directory Creation)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Cartella creata: {output_dir}")

    # 3. Inizializza la cattura del video
    cap = cv2.VideoCapture(video_input)
    
    if not cap.isOpened():
        print(f"Errore: Impossibile aprire il file video '{video_input}'.")
        return

    # Recupera metadati per il log
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Inizio estrazione: {total_frames} frame totali stimati.")

    frame_count = 0
    while True:
        # Legge il frame successivo
        success, frame = cap.read()

        # Se 'success' è False, il video è terminato o c'è un errore
        if not success:
            break

        # Costruisce il nome del file (es: frame_0000.png, frame_0001.png)
        # zfill(5) aggiunge zeri a sinistra per mantenere l'ordinamento alfabetico (Padding)
        filename = f"frame_{str(frame_count).zfill(5)}.png"
        output_file_path = os.path.join(output_dir, filename)

        # 4. Salva il frame su disco
        cv2.imwrite(output_file_path, frame)
        
        frame_count += 1
        
        # Feedback ogni 100 frame per monitorare il processo
        if frame_count % 100 == 0:
            print(f"Progresso: {frame_count} frame salvati...")

    # Rilascia la risorsa (Release resources)
    cap.release()
    print(f"Completato: {frame_count} frame estratti in '{output_dir}'.")

if __name__ == "__main__":
    salva_tutti_i_frame(VIDEO_PATH, FRAME_PATH)