import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import pandas as pd

def evaluate_video_quality(video_path: str, output_csv="video_analysis.csv", step: int = 30):
    """
    Analysiert die Videoqualität anhand verschiedener Metriken und speichert die Ergebnisse in einer CSV-Datei.

    Args:
        video_path (str): Der Pfad zum zu analysierenden Video.
        output_csv (str, optional): Der Name der CSV-Datei, in der die Ergebnisse gespeichert werden.
                                     Standardmäßig "video_analysis.csv".
        step (int, optional): Die Schrittweite, in der Frames analysiert werden.  Ein Wert von 30 bedeutet,
                                dass jeder 30. Frame analysiert wird. Standardmäßig 30.

    Returns:
        None. Die Ergebnisse werden in einer CSV-Datei gespeichert und eine Statistikübersicht wird auf der Konsole ausgegeben.

    Raises:
        IOError: Wenn das Video nicht geöffnet werden kann.

    Berechnete Metriken:
        - PSNR (Peak Signal-to-Noise Ratio):  Ein Maß für die Qualität der Rekonstruktion im Vergleich zum Originalsignal.
          Höhere Werte deuten auf eine bessere Qualität hin.
        - SSIM (Structural Similarity Index): Ein Maß für die wahrgenommene Ähnlichkeit zwischen zwei Bildern. Werte liegen zwischen -1 und 1,
          wobei 1 perfekte Ähnlichkeit bedeutet.
        - Durchschnittliche Helligkeit: Die durchschnittliche Helligkeit des Frames in Graustufen.
        - Rauschintensität (Varianz): Die Varianz der Pixelwerte in Graustufen, die als Schätzung der Rauschintensität dient.

    Beispiel:
        >>> evaluate_video_quality("my_video.mp4", output_csv="my_analysis.csv", step=60)
        Analyse abgeschlossen. Ergebnisse in 'my_analysis.csv' gespeichert.
             Frame_Index      PSNR (dB)        SSIM  Durchschnittliche Helligkeit  Rauschintensität (Varianz)
        count   101.000000     101.000000  101.000000                     101.000000                   101.000000
        mean   1500.000000      32.543210    0.954321                     120.567890                    45.678901
        std    1741.843722       2.345678    0.023457                      10.123456                     5.123456
        min       0.000000      28.123456    0.901234                      100.000000                    35.000000
        25%     750.000000      30.000000    0.930000                     112.000000                    41.000000
        50%    1500.000000      32.000000    0.950000                     120.000000                    45.000000
        75%    2250.000000      34.000000    0.970000                     128.000000                    50.000000
        max    3000.000000      37.000000    0.990000                     140.000000                    55.000000
    """

    # VideoCapture-Objekt erstellen, um das Video zu lesen
    cap = cv2.VideoCapture(video_path)

    # Überprüfen, ob das Video erfolgreich geöffnet wurde
    if not cap.isOpened():
        print("Video konnte nicht geöffnet werden.")
        return

    # Liste zum Speichern der Analyseergebnisse für jeden Frame
    results = []
    # Frame-Index, um den aktuellen Frame zu verfolgen
    frame_index = 0

    # Den ersten Frame lesen, um ihn als Referenz für Vergleiche zu verwenden
    ret, prev_frame = cap.read()
    # Wenn das Video leer ist (kein erster Frame vorhanden), beenden
    if not ret:
        print("Kein Frame im Video gefunden.")
        return

    # Schleife durch jeden Frame im Video
    while True:
        # Aktuellen Frame lesen
        ret, frame = cap.read()
        # Wenn kein Frame mehr vorhanden ist, Schleife beenden
        if not ret:
            break
        
        # Nur jeden 'step'-ten Frame analysieren
        if frame_index % step == 0:
            # PSNR-Berechnung zwischen dem vorherigen Frame und dem aktuellen Frame
            psnr_value = cv2.PSNR(prev_frame, frame)
            
            # SSIM-Berechnung: Konvertiere zuerst beide Frames in Graustufen
            gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ssim_value, _ = ssim(gray_prev, gray_curr, full=True, data_range=gray_curr.max() - gray_curr.min())  # data_range explizit angeben

            # Durchschnittliche Helligkeit des aktuellen Frames berechnen
            avg_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            # Rauschintensität (Varianz) des aktuellen Frames berechnen
            noise_variance = np.var(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            # Ergebnisse für diesen Frame in der Liste speichern
            results.append({
                "Frame_Index": frame_index,
                "PSNR (dB)": psnr_value,
                "SSIM": ssim_value,
                "Durchschnittliche Helligkeit": avg_brightness,
                "Rauschintensität (Varianz)": noise_variance
            })

        # Aktuellen Frame als vorherigen Frame für die nächste Iteration speichern
        prev_frame = frame
        # Frame-Index inkrementieren
        frame_index += 1

    # VideoCapture-Objekt freigeben
    cap.release()

    # Wenn Ergebnisse vorhanden sind, diese in eine CSV-Datei schreiben
    if results:
        # Ergebnisse in einen Pandas DataFrame konvertieren
        df = pd.DataFrame(results)
        # DataFrame in eine CSV-Datei speichern
        df.to_csv(output_csv, index=False)
        print(f"Analyse abgeschlossen. Ergebnisse in '{output_csv}' gespeichert.")
        # Statistikübersicht der Ergebnisse anzeigen
        print(df.describe())
    else:
        print("Keine Frames analysiert.")

# Beispielaufruf:
# Hier wird die Funktion mit dem Pfad zu einem Video ("2k.mp4") und einer Schrittweite von 30 aufgerufen.
# Dies bedeutet, dass jeder 30. Frame des Videos analysiert wird.
evaluate_video_quality("2k.mp4", step=30)
