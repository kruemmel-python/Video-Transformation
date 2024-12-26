# Video Transformation with PyTorch

Dieses Projekt bietet ein neuronales Netzwerk, das Videos transformiert, indem es Helligkeit, Kontrast und andere Eigenschaften anpasst. Das Modell wurde auf PyTorch implementiert und ist in der Lage, Videos in verschiedenen Auflösungen wie HD, Full HD, 2K und 4K zu generieren.

## Funktionen
- **Trainierbares Neuronales Netzwerk**: Transformiert Video-Frames basierend auf Zielwerten.
- **Flexible Auflösungen**: Unterstützung für HD, Full HD, 2K und 4K-Ausgaben.
- **GUI-Unterstützung**: Einfache Bedienung über eine grafische Benutzeroberfläche (GUI).
- **Qualitätsanalyse**: Berechnung von PSNR, SSIM, Helligkeit und Rauschintensität für die Videoanalyse.

## Voraussetzungen
- Python 3.8 oder höher
- PyTorch
- OpenCV
- NumPy
- scikit-image
- tqdm
- pandas
- tkinter (für die GUI)

## Installation
1. Klone das Repository:
   ```bash
   git clone https://github.com/kruemmel-python/Video-Transformation.git
   cd video-transformer
   ```

2. Installiere die benötigten Python-Pakete:
   ```bash
   pip install torch torchvision opencv-python-headless numpy scikit-image tqdm pandas
   ```

## Verwendung

### Training des Modells
1. Stelle ein Eingabevideo bereit (z. B. `input_video.mp4`).
2. Führe das Hauptskript aus, um Trainingsdaten zu generieren, das Modell zu trainieren und ein transformiertes Video zu erstellen:
   ```bash
   python vidtrain.py
   ```
3. Das trainierte Modell wird im Standardpfad `frame_transformer.pth` gespeichert.

### Verwendung des Modells mit GUI
1. Starte die GUI:
   ```bash
   python vid_ki_gui.py
   ```
2. Funktionen der GUI:
   - **Modell laden**: Lade ein trainiertes Modell (.pth-Datei).
   - **Eingabevideo auswählen**: Wähle das Video aus, das transformiert werden soll.
   - **Anpassungsfaktor setzen**: Passe die Stärke der Transformation an (Standard: 1.0).
   - **Auflösung auswählen**: Wähle die Ausgabeauflösung (HD, Full HD, 2K, 4K).
   - **Video generieren**: Speichere das transformierte Video.

### Qualitätsanalyse
Das Skript `analyse_videos.py` bietet eine Analyse der Videoqualität. Es berechnet Metriken wie PSNR, SSIM, durchschnittliche Helligkeit und Rauschintensität.

Führe die Analyse mit folgendem Befehl aus:
```bash
python evaluate_video_quality.py --video_path <video_path> --output_csv <output_csv>
```
Beispiel:
```bash
python evaluate_video_quality.py --video_path output_video.mp4 --output_csv video_analysis.csv
```

## Dateistruktur
- `vidtrain.py`: Trainings- und Testskript.
- `vid_ki_gui.py`: GUI für einfache Bedienung.
- `analyse_videos.py`: Analyse der Videoqualität.


## Beispiele
### Transformationsergebnisse
1. **Eingabevideo**: Ein 17-Sekunden-Video in HD-Auflösung.
2. **Ausgabevideo**: 4K-Video, generiert in 1 Minute und 2 Sekunden auf CPU.
3. **Qualitätsanalyse**:
   - **PSNR**: Durchschnitt von 45.00 dB (hohe Qualität).
   - **SSIM**: Durchschnitt von 0.990 (nahezu identisch mit dem Original).


## Verbesserungspotenzial
- **Erweiterte Transformationen**: Hinzufügen weiterer Bildverbesserungen wie Schärfen oder Farbkorrektur.
- **Optimierung**: Nutzung von GPUs für schnellere Verarbeitung.
- **Modellarchitektur**: Experimentieren mit komplexeren Netzwerken wie U-Net oder ResNet.

## Lizenz
Dieses Projekt ist unter der MIT-Lizenz veröffentlicht. Weitere Details findest du in der [LICENSE](LICENSE).

## Autoren
- **Ralf Krümmel** - Entwicklung & Dokumentation

Bei Fragen oder Feedback gerne ein Issue erstellen oder mich kontaktieren!
