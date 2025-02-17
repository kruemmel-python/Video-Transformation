import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from tqdm import tqdm
import os

# -----------------------------
# Dataset-Klasse
# -----------------------------
class VideoFrameDataset(Dataset):
    """
    Dataset-Klasse für das Laden von Video-Frames.

    Diese Klasse erbt von `torch.utils.data.Dataset` und dient dazu,
    einen Datensatz von Video-Frames effizient zu laden und zu verarbeiten.
    Sie speichert die Eingabe- und Ziel-Frames und ermöglicht den Zugriff
    auf einzelne Frame-Paare über einen Index.

    Attribute:
        input_frames (list): Eine Liste von Eingabe-Frames. Jeder Frame ist ein NumPy-Array.
        target_frames (list): Eine Liste von Ziel-Frames. Jeder Frame ist ein NumPy-Array.

    """
    def __init__(self, input_frames, target_frames):
        """
        Initialisiert das VideoFrameDataset.

        Args:
            input_frames (list): Eine Liste von Eingabe-Frames.
            target_frames (list): Eine Liste von Ziel-Frames.
        """
        self.input_frames = input_frames
        self.target_frames = target_frames

    def __len__(self):
        """
        Gibt die Anzahl der Frames im Datensatz zurück.

        Returns:
            int: Die Anzahl der Frames im Datensatz.
        """
        return len(self.input_frames)

    def __getitem__(self, idx):
        """
        Gibt ein Frame-Paar (Eingabe und Ziel) an einem bestimmten Index zurück.

        Args:
            idx (int): Der Index des gewünschten Frame-Paares.

        Returns:
            tuple: Ein Tupel, das den Eingabe-Frame und den Ziel-Frame enthält.
        """
        input_frame = self.input_frames[idx]
        target_frame = self.target_frames[idx]
        return input_frame, target_frame

# -----------------------------
# Neuronales Netzwerk
# -----------------------------
class VideoFrameTransformer(nn.Module):
    """
    Ein einfaches Convolutional Neural Network (CNN) für die Video-Frame-Transformation.

    Diese Klasse erbt von `torch.nn.Module` und definiert ein CNN, das dazu dient,
    Eingabe-Frames in Ziel-Frames zu transformieren.  Es besteht aus einem Encoder-Teil,
    der die Eingabe komprimiert, und einem Decoder-Teil, der die komprimierte Darstellung
    wieder in ein Bild umwandelt.  Sigmoid wird im letzten Layer verwendet um Werte
    zwischen 0 und 1 zu garantieren.

    Attribute:
        encoder (nn.Sequential): Ein sequentielles Modul, das die Encoder-Schichten enthält.
        decoder (nn.Sequential): Ein sequentielles Modul, das die Decoder-Schichten enthält.
    """
    def __init__(self):
        """
        Initialisiert das VideoFrameTransformer-Modell.
        """
        super(VideoFrameTransformer, self).__init__()
        # Encoder-Teil des Netzwerks
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Convolutional Layer 1: 3 Eingabekanäle, 64 Ausgabekanäle
            nn.ReLU(),  # ReLU-Aktivierungsfunktion
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Convolutional Layer 2: 64 Eingabekanäle, 128 Ausgabekanäle
            nn.ReLU(),  # ReLU-Aktivierungsfunktion
            nn.MaxPool2d(2)  # Max Pooling Layer zur Reduzierung der räumlichen Dimensionen
        )
        # Decoder-Teil des Netzwerks
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # Transponierte Convolutional Layer 1: 128 Eingabekanäle, 64 Ausgabekanäle
            nn.ReLU(),  # ReLU-Aktivierungsfunktion
            nn.Conv2d(64, 3, kernel_size=3, padding=1),  # Convolutional Layer 2: 64 Eingabekanäle, 3 Ausgabekanäle
            nn.Sigmoid()  # Sigmoid-Aktivierungsfunktion, um Werte zwischen 0 und 1 zu erhalten
        )

    def forward(self, x):
        """
        Führt einen Vorwärtsdurchlauf durch das Netzwerk durch.

        Args:
            x (torch.Tensor): Der Eingabe-Tensor (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Der Ausgabe-Tensor (Batch, Channels, Height, Width).
        """
        x = self.encoder(x)  # Eingabe durch den Encoder leiten
        x = self.decoder(x)  # Ausgabe des Encoders durch den Decoder leiten
        return x

# -----------------------------
# Funktionen zum Laden und Verarbeiten von Daten
# -----------------------------
def generate_training_data(video_path, transform_function, output_path):
    """
    Generiert Trainingsdaten aus einem Video, indem Frames extrahiert und transformiert werden.

    Diese Funktion liest ein Video ein, extrahiert einzelne Frames, wendet eine
    Transformationsfunktion auf jeden Frame an und speichert die ursprünglichen
    und transformierten Frames als NumPy-Arrays in einer .npz-Datei.

    Args:
        video_path (str): Der Pfad zum Eingabevideo.
        transform_function (callable): Eine Funktion, die auf jeden Frame angewendet wird.
        output_path (str): Der Pfad, in dem die Trainingsdaten gespeichert werden sollen.
    """
    cap = cv2.VideoCapture(video_path)  # VideoCapture-Objekt zum Lesen des Videos erstellen
    input_frames = []  # Liste zum Speichern der Eingabe-Frames
    target_frames = []  # Liste zum Speichern der Ziel-Frames

    success, frame = cap.read()  # Den ersten Frame lesen
    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Konvertiere Frame von BGR zu RGB
        input_frames.append(frame / 255.0)  # Normalisiere die Eingabedaten: Wertebereich [0, 1]

        # Transformationen auf das Frame anwenden
        transformed_frame = transform_function(frame)
        target_frames.append(transformed_frame / 255.0)  # Normalisiere das Ziel

        success, frame = cap.read()  # Nächsten Frame lesen
    cap.release()  # VideoCapture freigeben

    # Speichere die Daten
    np.savez(output_path, inputs=np.array(input_frames), targets=np.array(target_frames))


def load_training_data(data_path):
    """
    Lädt Trainingsdaten aus einer .npz-Datei.

    Diese Funktion lädt die in einer .npz-Datei gespeicherten Trainingsdaten,
    konvertiert die NumPy-Arrays in PyTorch-Tensoren und transponiert die
    Dimensionen, um das Format [Batch, Channels, Height, Width] zu erhalten.

    Args:
        data_path (str): Der Pfad zur .npz-Datei mit den Trainingsdaten.

    Returns:
        tuple: Ein Tupel, das die Eingabe- und Ziel-Tensoren enthält.
    """
    data = np.load(data_path)  # Lade die Daten aus der .npz-Datei
    inputs = torch.tensor(data['inputs'], dtype=torch.float32).permute(0, 3, 1, 2)  # [Batch, Channels, Height, Width]
    targets = torch.tensor(data['targets'], dtype=torch.float32).permute(0, 3, 1, 2)
    return inputs, targets

# -----------------------------
# Trainingsprozess
# -----------------------------
def train_model(model, dataloader, criterion, optimizer, device, save_path, epochs=5):
    """
    Trainiert das Modell mit den bereitgestellten Daten.

    Diese Funktion durchläuft den Trainingsdatensatz für eine bestimmte Anzahl von Epochen,
    berechnet den Verlust, führt die Backpropagation durch und aktualisiert die Modellparameter.

    Args:
        model (nn.Module): Das zu trainierende Modell.
        dataloader (DataLoader): Der DataLoader für den Trainingsdatensatz.
        criterion (nn.Module): Die Verlustfunktion.
        optimizer (torch.optim.Optimizer): Der Optimierer.
        device (torch.device): Das Gerät, auf dem das Training durchgeführt werden soll (CPU oder GPU).
        save_path (str): Der Pfad, in dem das trainierte Modell gespeichert werden soll.
        epochs (int): Die Anzahl der Trainingsepochen.
    """
    model.train()  # Setze das Modell in den Trainingsmodus
    for epoch in range(epochs):  # Iteriere über die Epochen
        epoch_loss = 0  # Initialisiere den Epochenverlust
        for input_batch, target_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):  # Iteriere über die Batches
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # Verschiebe Daten auf das Gerät

            optimizer.zero_grad()  # Setze die Gradienten auf Null
            outputs = model(input_batch)  # Vorwärtsdurchlauf: Eingabe durch das Modell leiten
            loss = criterion(outputs, target_batch)  # Berechne den Verlust
            loss.backward()  # Rückwärtsdurchlauf: Gradienten berechnen
            optimizer.step()  # Aktualisiere die Modellparameter

            epoch_loss += loss.item()  # Addiere den Batch-Verlust zum Epochenverlust

        print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(dataloader)}")  # Gib den durchschnittlichen Epochenverlust aus

    torch.save(model.state_dict(), save_path)  # Speichere den Modellzustand
    print(f"Modell gespeichert unter {save_path}")

# -----------------------------
# Testfunktion
# -----------------------------
def process_video_with_model(
    video_path,
    model,
    device,
    output_path,
    adjustment_factor=1.0,
    brightness_factor=0.0,
    contrast_factor=1.0,
    sharpness_factor=0.0,
    color_adjustments=None
):
    """
    Verarbeitet ein Video mit einem trainierten Modell und steuert die Stärke der Anpassungen.

    Diese Funktion liest ein Video ein, verarbeitet jeden Frame mit dem angegebenen Modell,
    führt optionale Anpassungen der Helligkeit, des Kontrasts, der Schärfe und der Farben durch
    und speichert das verarbeitete Video.

    Args:
        video_path (str): Pfad zum Eingabevideo.
        model (nn.Module): Das trainierte Modell.
        device (torch.device): CPU oder GPU
        output_path (str): Speicherort für das Ausgabevideo.
        adjustment_factor (float): Skalar zur Anpassung der Transformationsstärke.
        brightness_factor (float): Wert für die Helligkeitsanpassung (-1 bis 1).
        contrast_factor (float): Wert für die Kontrasterhöhung (0.5 bis 2).
        sharpness_factor (float): Wert für die Schärfenerhöhung (0 bis 1).
        color_adjustments (dict): Dictionary mit Farbfaktoren {'rot', 'grün', 'blau', 'gelb', 'cyan', 'magenta'}.
    """
    cap = cv2.VideoCapture(video_path)  # VideoCapture-Objekt zum Lesen des Videos erstellen
    frames = []  # Liste zum Speichern der Frames

    success, frame = cap.read()  # Den ersten Frame lesen
    while success:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Konvertiere Frame von BGR zu RGB und füge ihn der Liste hinzu
        success, frame = cap.read()  # Nächsten Frame lesen
    cap.release()  # VideoCapture freigeben

    frames = torch.tensor(np.array(frames) / 255.0, dtype=torch.float32).permute(0, 3, 1, 2).to(device)  # Konvertiere Frames in einen Tensor, normalisiere und verschiebe ihn auf das Gerät
    model.eval()  # Setze das Modell in den Evaluierungsmodus

    processed_frames = []  # Liste zum Speichern der verarbeiteten Frames
    with torch.no_grad():  # Deaktiviere die Gradientenberechnung
        for frame in tqdm(frames, desc="Frames verarbeiten"):  # Iteriere über die Frames
            output = model(frame.unsqueeze(0))  # Eingabe durch das Modell leiten

            # Transformationen anwenden
            adjusted_output = frame + (output - frame) * adjustment_factor  # Wende den Anpassungsfaktor an
            adjusted_output = torch.clamp(adjusted_output, 0, 1)  # Wertebereich auf [0, 1] beschränken

            # Helligkeit
            adjusted_output = adjusted_output + brightness_factor  # Passe die Helligkeit an
            adjusted_output = torch.clamp(adjusted_output, 0, 1)  # Wertebereich auf [0, 1] beschränken

            # Kontrast
            adjusted_output = (adjusted_output - 0.5) * contrast_factor + 0.5  # Passe den Kontrast an
            adjusted_output = torch.clamp(adjusted_output, 0, 1)  # Wertebereich auf [0, 1] beschränken

            # Schärfe
            out_np = adjusted_output.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # Konvertiere Tensor in ein NumPy-Array
            out_np_255 = (out_np * 255).astype(np.uint8)  # Skaliere Werte auf [0, 255] und konvertiere sie in uint8
            if sharpness_factor > 0:
                blurred = cv2.GaussianBlur(out_np_255, (3, 3), 0)  # Wende einen Gaußschen Weichzeichner an
                unsharp = cv2.addWeighted(out_np_255, 1 + sharpness_factor, blurred, -sharpness_factor, 0)  # Wende eine Unscharfmaske an
                out_np_255 = np.clip(unsharp, 0, 255).astype(np.uint8)  # Wertebereich auf [0, 255] beschränken und in uint8 konvertieren

            # Farbanpassungen
            if color_adjustments:
                hsv_img = cv2.cvtColor(out_np_255, cv2.COLOR_RGB2HSV).astype(np.float32)
                H, S, V = cv2.split(hsv_img)
                hue_shift = (
                    color_adjustments.get("rot", 0) * 10 +
                    color_adjustments.get("gelb", 0) * 5 +
                    color_adjustments.get("magenta", 0) * 8
                )
                sat_shift = (
                    color_adjustments.get("grün", 0) * 0.1 +
                    color_adjustments.get("cyan", 0) * 0.1 +
                    color_adjustments.get("blau", 0) * 0.1
                )
                H = (H + hue_shift) % 180
                S = np.clip(S * (1 + sat_shift), 0, 255)
                hsv_modified = cv2.merge([H, S, V]).astype(np.uint8)
                rgb_modified = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2RGB)
                out_np_255 = rgb_modified

            processed_frames.append(out_np_255)  # Füge den verarbeiteten Frame der Liste hinzu

    processed_frames = np.array(processed_frames)  # Konvertiere die Liste der verarbeiteten Frames in ein NumPy-Array

    # Speichere das Video
    height, width = processed_frames[0].shape[:2]  # Hole Höhe und Breite des Frames
    fps = 30  # Standard FPS
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))  # VideoWriter-Objekt erstellen
    for frame in processed_frames:  # Iteriere über die verarbeiteten Frames
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Konvertiere Frame von RGB zu BGR und schreibe ihn in das Video
    out.release()  # VideoWriter freigeben
    print(f"Video gespeichert unter {output_path}")


# -----------------------------
# Erweiterte transform_function
# -----------------------------
def transform_function(frame: np.ndarray) -> np.ndarray:
    """
    Erzeugt ein Zielbild, bei dem Farbe, Helligkeit, Kontrast, Schärfe
    unabhängig und zufällig angepasst werden.

    Args:
        frame (np.ndarray): Das Originalframe in RGB [H, W, 3], Werte 0..255.

    Returns:
        np.ndarray: Das transformierte Frame (np.ndarray) in RGB [H, W, 3], Werte 0..255.
    """
    # 1) Umwandlung in [0..1], damit wir leichter rechnen können
    img = frame.astype(np.float32) / 255.0

    # 2) Zufällige Faktoren für alle 4 Eigenschaften
    brightness_factor = np.random.uniform(-0.2, 0.2)  # -0.2 .. +0.2
    contrast_factor   = np.random.uniform(-0.2, 0.2)
    hue_shift         = np.random.uniform(-0.1, 0.1)  # -0.1 .. +0.1
    sharpness_factor  = np.random.uniform(0.0, 0.8)   # 0..0.8 (immer >= 0)

    # 3) Farbe (Hue) anpassen:
    img_hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = cv2.split(img_hsv)
    hue_shift_val = hue_shift * 180
    H = (H + hue_shift_val) % 180
    img_hsv = cv2.merge([H, S, V])
    img_rgb = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    img = img_rgb.astype(np.float32) / 255.0

    # 4) Helligkeit
    img = np.clip(img + brightness_factor, 0, 1)

    # 5) Kontrast
    alpha = 1.0 + contrast_factor
    img = np.clip((img - 0.5) * alpha + 0.5, 0, 1)

    # 6) Schärfen
    ksize = 3
    blur = cv2.GaussianBlur((img*255).astype(np.uint8), (ksize, ksize), 0)
    blur = blur.astype(np.float32)/255.0
    sharpened = np.clip(img*(1+sharpness_factor) - blur*sharpness_factor, 0, 1)
    img = sharpened

    # 7) Endergebnis
    out_frame = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return out_frame

# -----------------------------
# Hauptprogramm
# -----------------------------
def main():
    """
    Das Hauptprogramm, das das Training und die Verarbeitung des Videos durchführt.

    Diese Funktion führt die folgenden Schritte aus:
    1. Definiert die Pfade für das Eingabevideo, die Trainingsdaten, das Modell und das Ausgabevideo.
    2. Bestimmt das Gerät, auf dem das Training durchgeführt werden soll (CPU oder GPU).
    3. Generiert die Trainingsdaten, falls sie noch nicht vorhanden sind.
    4. Lädt die Trainingsdaten.
    5. Initialisiert das Modell.
    6. Lädt ein vortrainiertes Modell, falls vorhanden.
    7. Definiert den Optimierer und die Verlustfunktion.
    8. Trainiert das Modell.
    9. Verarbeitet das Eingabevideo mit dem trainierten Modell.
    """
    video_path = "input_video.mp4"  # Eingabevideo
    data_path = "training_data.npz"  # Pfad für Trainingsdaten
    model_path = "frame_transformer.pth"  # Modellpfad
    output_video_path = "output_video.mp4"  # Ausgabevideo

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Bestimme das Gerät

    # Trainingsdaten generieren
    if not os.path.exists(data_path):  # Überprüfe, ob die Trainingsdaten existieren
        generate_training_data(video_path, transform_function, data_path)  # Generiere die Trainingsdaten

    inputs, targets = load_training_data(data_path)  # Lade die Trainingsdaten
    dataset = VideoFrameDataset(inputs, targets)  # Erstelle ein Dataset-Objekt
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # Erstelle einen DataLoader

    # Modell initialisieren
    model = VideoFrameTransformer().to(device)  # Erstelle ein Modell und verschiebe es auf das Gerät
    if os.path.exists(model_path):  # Überprüfe, ob ein vortrainiertes Modell existiert
        model.load_state_dict(torch.load(model_path, map_location=device))  # Lade den Modellzustand
        print("Vortrainiertes Modell geladen.")
    else:
        print("Kein Modell gefunden. Neues Modell wird erstellt.")

    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Erstelle einen Optimierer
    criterion = nn.MSELoss()  # Erstelle eine Verlustfunktion

    # Training
    train_model(model, dataloader, criterion, optimizer, device, model_path, epochs=5)  # Trainiere das Modell

    # Test
    process_video_with_model(video_path, model, device, output_video_path)  # Verarbeite das Video

if __name__ == "__main__":
    main()
