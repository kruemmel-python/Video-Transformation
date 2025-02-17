

# Video Frame Transformer

Dieses Repository enthält ein Python-Projekt zur Transformation von Video-Frames mithilfe eines neuronalen Netzwerks. Das Projekt umfasst mehrere Module:

- **train_video.py** – Bereitet Trainingsdaten vor, definiert das CNN-Modell, führt das Training durch und verarbeitet Videos.
- **analyse_videos.py** – Analysiert die Qualität von Videos anhand verschiedener Metriken (PSNR, SSIM, durchschnittliche Helligkeit und Rauschintensität) und speichert die Ergebnisse in einer CSV-Datei.
- **vid_ki_gui.py** – Bietet eine grafische Benutzeroberfläche (GUI) mit CustomTkinter, um Videos interaktiv zu verarbeiten. Dabei können Parameter wie Helligkeit, Kontrast, Schärfe und Farbanpassungen eingestellt werden.

![image](https://github.com/user-attachments/assets/292737a0-3d55-4438-89fb-5526668ae929)

![image](https://github.com/user-attachments/assets/20195738-9eaf-4ff6-9375-4ff66bc8dc8a)

![image](https://github.com/user-attachments/assets/adbb37fb-c2a4-4224-a2b0-40299718dad2)

## Repository klonen

Um eine lokale Kopie des Repositories zu erstellen, führe folgenden Befehl in deinem Terminal aus:

```bash
git clone https://github.com/kruemmel-python/Video-Transformation.git
```

Dadurch wird das Repository in dein aktuelles Verzeichnis geklont. Anschließend kannst du in das Projektverzeichnis wechseln und die README.md für weitere Anweisungen einsehen.

## Inhalte des Repositories

- **train_video.py**  
  Enthält:
  - Eine **Dataset-Klasse** (`VideoFrameDataset`), die Eingabe- und Ziel-Frames eines Videos lädt.
  - Das **neuronale Netzwerk** (`VideoFrameTransformer`), welches einen Encoder-Decoder-Ansatz verwendet, um Frames zu transformieren.
  - Funktionen zur **Datenvorbereitung** (`generate_training_data`, `load_training_data`), zum **Training** (`train_model`) und zur **Videoverarbeitung** (`process_video_with_model`).
  - Eine **Hauptfunktion** (`main`), die das komplette Training und die Videoverarbeitung steuert.

- **analyse_videos.py**  
  Analysiert ein Video, indem:
  - Mit Hilfe von OpenCV und scikit-image Metriken wie **PSNR** und **SSIM** berechnet werden.
  - Zusätzlich werden die **durchschnittliche Helligkeit** und die **Rauschintensität (Varianz)** der Frames ermittelt.
  - Die Ergebnisse in einer CSV-Datei abgespeichert und eine statistische Übersicht auf der Konsole ausgegeben wird.

- **vid_ki_gui.py**  
  Stellt eine GUI zur Verfügung, die:
  - Den Anwendern ermöglicht, Eingabe- und Ausgabepfade auszuwählen.
  - Über Schieberegler Parameter wie Helligkeit, Kontrast, Schärfe und Farbanpassungen intuitiv eingestellt werden können.
  - Eine Vorschau des angepassten Frames anzeigt.
  - Das verarbeitete Video erstellt und dabei den originalen Ton mit ffmpeg integriert.

## Voraussetzungen

Stelle sicher, dass folgende Pakete installiert sind:

- Python 3.12
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/) (cv2)
- NumPy
- tqdm
- scikit-image (für SSIM)
- Pandas
- customtkinter
- Pillow (PIL)
- ffmpeg (für die Audiointegration in Videos)

Installation über pip:

```bash
pip install torch torchvision opencv-python numpy tqdm scikit-image pandas customtkinter pillow
```

Für ffmpeg:

- **Ubuntu/Debian:**  
  ```bash
  sudo apt-get install ffmpeg
  ```
- **Windows:**  
  Lade ffmpeg von der [offiziellen Website](https://ffmpeg.org/) herunter und füge es deinem PATH hinzu.

## Nutzung

### 1. Training und Videoverarbeitung

1. Lege dein Eingabevideo (standardmäßig `input_video.mp4`) in das Projektverzeichnis oder passe den Pfad in der Datei `train_video.py` an.
2. Starte das Training und die Videoverarbeitung über:
   ```bash
   python train_video.py
   ```
   Das Skript generiert zunächst Trainingsdaten, trainiert das Modell und speichert das transformierte Video unter `output_video.mp4`.

### 2. Videoanalyse

Um die Qualität eines Videos zu analysieren, führe das Skript `analyse_videos.py` aus:

```bash
python analyse_videos.py
```

Dabei werden Metriken für jeden 30. Frame berechnet und in einer CSV-Datei (`video_analysis.csv`) abgelegt. Anschließend wird eine statistische Übersicht auf der Konsole angezeigt.

### 3. Grafische Benutzeroberfläche (GUI)

Starte die GUI-Anwendung, um Videos interaktiv zu verarbeiten:

```bash
python vid_ki_gui.py
```

Die GUI ermöglicht es dir:

- Ein Eingabevideo auszuwählen.
- Den Zielpfad für das verarbeitete Video festzulegen.
- Über Schieberegler Parameter wie Helligkeit, Kontrast, Schärfe und Farbanpassungen einzustellen.
- Eine Vorschau des angepassten Frames zu betrachten.
- Das Video zu verarbeiten – inklusive Integration des Originaltons mithilfe von ffmpeg.

## Code-Details und Funktionsweise

### train_video.py

- **Datensatz & DataLoader:**  
  Die Klasse `VideoFrameDataset` lädt Frame-Paare (Eingabe und Ziel) und ermöglicht deren Verwendung in einem PyTorch DataLoader.
  
- **Modellarchitektur:**  
  Das Modell `VideoFrameTransformer` verwendet einen Encoder (mit Convolutional Layers und ReLU-Aktivierungen) und einen Decoder (mit transponierten Convolutions und einem abschließenden Sigmoid) zur Transformation der Frames. Dies stellt sicher, dass die Ausgabewerte im Bereich [0, 1] bleiben.

- **Trainingsprozess:**  
  Die Funktion `train_model` durchläuft mehrere Epochen, berechnet den mittleren Verlust und aktualisiert die Modellparameter mittels Adam-Optimierer.

- **Videoverarbeitung:**  
  Die Funktion `process_video_with_model` liest Frames aus einem Video, transformiert diese mit dem trainierten Modell und führt zusätzliche Anpassungen (Helligkeit, Kontrast, Schärfe, Farbanpassungen) durch, bevor das Video gespeichert wird.

### analyse_videos.py

- **Metriken:**  
  Mittels OpenCV und scikit-image werden PSNR und SSIM berechnet. Zudem werden Durchschnittshelligkeit und Varianz (als Rauschintensität) ermittelt.

- **Ergebnisexport:**  
  Die Ergebnisse werden in einer CSV-Datei gespeichert und eine statistische Zusammenfassung wird ausgegeben.

### vid_ki_gui.py

- **GUI-Design:**  
  Die GUI basiert auf CustomTkinter und bietet ein modernes, scrollbares Layout. Benutzer können Parameter über intuitive Schieberegler einstellen.

- **Modellintegration:**  
  Das vortrainierte Modell wird geladen, sodass Anpassungen in Echtzeit auf einen Vorschau-Frame angewendet werden.

- **Audiointegration:**  
  Nach der Verarbeitung des Videos wird der Ton des Originalvideos extrahiert und in das transformierte Video integriert. Hierbei kommen ffmpeg-Befehle zum Einsatz.

## Persönliche Meinung und Best Practices

Ich bin überzeugt, dass dieses Projekt ein gutes Beispiel für die Kombination von Deep Learning und Computer Vision darstellt. Besonders positiv hervorzuheben sind:

- **Modularität:** Die klare Trennung in Trainings-, Analyse- und GUI-Komponenten erleichtert Erweiterungen und Anpassungen.
- **Modernste Techniken:** Die Nutzung aktueller Sprachfeatures von Python 3.12 sowie Best Practices bei der Nutzung von PyTorch und OpenCV führt zu einem lesbaren und wartbaren Code.
- **Benutzerfreundlichkeit:** Durch die Integration einer GUI wird der Zugang zu den Funktionen auch für Anwender erleichtert, die nicht direkt im Code arbeiten möchten.
- **Ausführliche Dokumentation:** Umfassende Kommentare und Erklärungen in den Quellcodes erleichtern das Verständnis und fördern den Lerneffekt – besonders im Lehrkontext.


## Lizenz
Dieses Projekt ist unter der MIT-Lizenz veröffentlicht. Weitere Details findest du in der [LICENSE](LICENSE).

## Autoren
- **Ralf Krümmel** - Entwicklung & Dokumentation

Bei Fragen oder Feedback gerne ein Issue erstellen oder mich kontaktieren!



