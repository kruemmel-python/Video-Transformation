import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from tqdm import tqdm
import os
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as transforms
from torchvision.models import vgg19
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import torchmetrics

# -----------------------------
# Dataset-Klasse
# -----------------------------
class VideoFrameDataset(Dataset):
    def __init__(self, input_frames, target_frames):
        self.input_frames = input_frames
        self.target_frames = target_frames

    def __len__(self):
        return len(self.input_frames)

    def __getitem__(self, idx):
        input_frame = self.input_frames[idx]
        target_frame = self.target_frames[idx]
        return input_frame, target_frame

# -----------------------------
# Neuronales Netzwerk
# -----------------------------
class VideoFrameTransformer(nn.Module):
    def __init__(self):
        super(VideoFrameTransformer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Werte zwischen 0 und 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# -----------------------------
# Funktionen zum Laden und Verarbeiten von Daten
# -----------------------------
def generate_training_data(video_path, transform_function, output_path):
    cap = cv2.VideoCapture(video_path)
    input_frames = []
    target_frames = []

    success, frame = cap.read()
    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_frames.append(frame / 255.0)  # Normalisiere die Eingabedaten

        # Transformationen auf das Frame anwenden
        transformed_frame = transform_function(frame)
        target_frames.append(transformed_frame / 255.0)  # Normalisiere das Ziel

        success, frame = cap.read()
    cap.release()

    # Speichere die Daten
    np.savez(output_path, inputs=np.array(input_frames), targets=np.array(target_frames))

def load_training_data(data_path):
    data = np.load(data_path)
    inputs = torch.tensor(data['inputs'], dtype=torch.float32).permute(0, 3, 1, 2)  # [Batch, Channels, Height, Width]
    targets = torch.tensor(data['targets'], dtype=torch.float32).permute(0, 3, 1, 2)
    return inputs, targets

# -----------------------------
# Trainingsprozess
# -----------------------------
def calculate_accuracy(output, target):
    """
    Berechnet die Genauigkeit basierend auf einer einfachen SSIM-Metrik.
    Höhere Werte deuten auf eine größere Ähnlichkeit hin.
    """
    output_np = output.detach().permute(0, 2, 3, 1).cpu().numpy()  # [Batch, H, W, C]
    target_np = target.detach().permute(0, 2, 3, 1).cpu().numpy()

    batch_size = output_np.shape[0]
    accuracy = 0
    for i in range(batch_size):
        # Setze die Fenstergröße explizit auf 3 und füge den Parameter `data_range` hinzu
        accuracy += ssim(output_np[i], target_np[i], win_size=3, multichannel=True, channel_axis=-1, data_range=1.0)
    return accuracy / batch_size

def train_model(model, dataloader, criterion, optimizer, device, save_path, epochs=5):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        for input_batch, target_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            optimizer.zero_grad()
            outputs = model(input_batch)
            loss = criterion(outputs, target_batch)
            loss.backward()
            optimizer.step()

            # Berechnung von Verlust und Genauigkeit
            epoch_loss += loss.item()
            epoch_accuracy += calculate_accuracy(outputs, target_batch)

        # Durchschnittswerte der Epoche berechnen
        avg_loss = epoch_loss / len(dataloader)
        avg_accuracy = epoch_accuracy / len(dataloader)

        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {avg_accuracy:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Modell gespeichert unter {save_path}")

# -----------------------------
# Testfunktion
# -----------------------------
def process_video_with_model(video_path, model, device, output_path, adjustment_factor=1.0):
    """
    Verarbeitet ein Video mit einem trainierten Modell und steuert die Stärke der Anpassungen.

    :param video_path: Pfad zum Eingabevideo
    :param model: Das trainierte Modell
    :param device: CPU oder GPU
    :param output_path: Speicherort für das Ausgabenvideo
    :param adjustment_factor: Skalar zur Anpassung der Transformationsstärke
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    success, frame = cap.read()
    while success:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        success, frame = cap.read()
    cap.release()

    frames = torch.tensor(np.array(frames) / 255.0, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
    model.eval()

    processed_frames = []
    with torch.no_grad():
        for frame in tqdm(frames, desc="Frames verarbeiten"):
            output = model(frame.unsqueeze(0))
            # Skaliere die Ausgabe des Modells mit dem Adjustment-Faktor
            adjusted_output = frame + (output - frame) * adjustment_factor
            adjusted_output = torch.clamp(adjusted_output, 0, 1)  # Sicherstellen, dass die Werte im Bereich [0, 1] bleiben
            processed_frames.append(adjusted_output.squeeze(0).cpu().numpy())

    processed_frames = (np.array(processed_frames) * 255).astype(np.uint8).transpose(0, 2, 3, 1)

    # Speichere das Video
    height, width = processed_frames[0].shape[:2]
    fps = 30  # Standard FPS
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in processed_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Video gespeichert unter {output_path}")

# -----------------------------
# Hauptprogramm
# -----------------------------
def main():
    video_path = "input_video.mp4"  # Eingabevideo
    data_path = "training_data.npz"  # Pfad für Trainingsdaten
    model_path = "frame_transformer.pth"  # Modellpfad
    output_video_path = "output_video.mp4"  # Ausgabevideo

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Trainingsdaten generieren
    def transform_function(frame):
        # Beispiel: Transformationen (Helligkeit erhöhen)
        return np.clip(frame * 1.2, 0, 255)  # Erhöhe Helligkeit

    # Neugenerierung der Trainingsdaten
    if os.path.exists(data_path):
        os.remove(data_path)
    generate_training_data(video_path, transform_function, data_path)

    inputs, targets = load_training_data(data_path)
    dataset = VideoFrameDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Modell initialisieren
    model = VideoFrameTransformer().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Vortrainiertes Modell geladen.")
    else:
        print("Kein Modell gefunden. Neues Modell wird erstellt.")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training
    train_model(model, dataloader, criterion, optimizer, device, model_path, epochs=10)

    # Test
    process_video_with_model(video_path, model, device, output_video_path)

    # Verarbeitung mit geringerer Anpassungsstärke (50%)
    process_video_with_model("test_video.mp4", model, device, "output_low_adjustment.mp4", adjustment_factor=0.0)

    # Verarbeitung mit geringerer Anpassungsstärke (50%)
    process_video_with_model("test_video.mp4", model, device, "output_low_adjustment.mp4", adjustment_factor=0.5)

    # Verarbeitung mit verstärkter Anpassungsstärke (200%)
    process_video_with_model("test_video.mp4", model, device, "output_high_adjustment.mp4", adjustment_factor=2.0)

if __name__ == "__main__":
    main()
