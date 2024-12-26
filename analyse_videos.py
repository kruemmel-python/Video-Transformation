import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import pandas as pd

def evaluate_video_quality(video_path: str, output_csv="video_analysis.csv", step: int = 30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Video konnte nicht geöffnet werden.")
        return

    results = []
    frame_index = 0

    ret, prev_frame = cap.read()
    if not ret:
        print("Kein Frame im Video gefunden.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_index % step == 0:
            # PSNR-Berechnung
            psnr_value = cv2.PSNR(prev_frame, frame)
            
            # SSIM-Berechnung
            gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ssim_value, _ = ssim(gray_prev, gray_curr, full=True)

            # Durchschnittliche Helligkeit
            avg_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            # Rauschintensität (Varianz)
            noise_variance = np.var(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            # Ergebnisse speichern
            results.append({
                "Frame_Index": frame_index,
                "PSNR (dB)": psnr_value,
                "SSIM": ssim_value,
                "Durchschnittliche Helligkeit": avg_brightness,
                "Rauschintensität (Varianz)": noise_variance
            })

        prev_frame = frame
        frame_index += 1

    cap.release()

    if results:
        # Ergebnisse in DataFrame umwandeln
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Analyse abgeschlossen. Ergebnisse in '{output_csv}' gespeichert.")
        print(df.describe())  # Statistikübersicht anzeigen
    else:
        print("Keine Frames analysiert.")

# Beispielaufruf:
evaluate_video_quality("2k.mp4", step=30)
