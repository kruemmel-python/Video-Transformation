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
def train_model(model, dataloader, criterion, optimizer, device, save_path, epochs=5):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for input_batch, target_batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            optimizer.zero_grad()
            outputs = model(input_batch)
            loss = criterion(outputs, target_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(dataloader)}")

    torch.save(model.state_dict(), save_path)
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
    
    :param video_path: Pfad zum Eingabevideo
    :param model: Das trainierte Modell
    :param device: CPU oder GPU
    :param output_path: Speicherort für das Ausgabenvideo
    :param adjustment_factor: Skalar zur Anpassung der Transformationsstärke
    :param brightness_factor: Wert für die Helligkeitsanpassung (-1 bis 1)
    :param contrast_factor: Wert für die Kontrasterhöhung (0.5 bis 2)
    :param sharpness_factor: Wert für die Schärfenerhöhung (0 bis 1)
    :param color_adjustments: Dictionary mit Farbfaktoren {'rot', 'grün', 'blau', 'gelb', 'cyan', 'magenta'}
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

            # Transformationen anwenden
            adjusted_output = frame + (output - frame) * adjustment_factor
            adjusted_output = torch.clamp(adjusted_output, 0, 1)

            # Helligkeit
            adjusted_output = adjusted_output + brightness_factor
            adjusted_output = torch.clamp(adjusted_output, 0, 1)

            # Kontrast
            adjusted_output = (adjusted_output - 0.5) * contrast_factor + 0.5
            adjusted_output = torch.clamp(adjusted_output, 0, 1)

            # Schärfe
            out_np = adjusted_output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            out_np_255 = (out_np * 255).astype(np.uint8)
            if sharpness_factor > 0:
                blurred = cv2.GaussianBlur(out_np_255, (3, 3), 0)
                unsharp = cv2.addWeighted(out_np_255, 1 + sharpness_factor, blurred, -sharpness_factor, 0)
                out_np_255 = np.clip(unsharp, 0, 255).astype(np.uint8)

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

            processed_frames.append(out_np_255)

    processed_frames = np.array(processed_frames)

    # Speichere das Video
    height, width = processed_frames[0].shape[:2]
    fps = 30  # Standard FPS
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in processed_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Video gespeichert unter {output_path}")


# -----------------------------
# Erweiterte transform_function
# -----------------------------
def transform_function(frame: np.ndarray) -> np.ndarray:
    """
    Erzeugt ein Zielbild, bei dem Farbe, Helligkeit, Kontrast, Schärfe
    unabhängig und zufällig angepasst werden.
    Parameter:
        frame (np.ndarray): Das Originalframe in RGB [H, W, 3], Werte 0..255.
    Rückgabe:
        Das transformierte Frame (np.ndarray) in RGB [H, W, 3], Werte 0..255.
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
    video_path = "input_video.mp4"  # Eingabevideo
    data_path = "training_data.npz"  # Pfad für Trainingsdaten
    model_path = "frame_transformer.pth"  # Modellpfad
    output_video_path = "output_video.mp4"  # Ausgabevideo

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Trainingsdaten generieren
    if not os.path.exists(data_path):
        generate_training_data(video_path, transform_function, data_path)

    inputs, targets = load_training_data(data_path)
    dataset = VideoFrameDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Modell initialisieren
    model = VideoFrameTransformer().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Vortrainiertes Modell geladen.")
    else:
        print("Kein Modell gefunden. Neues Modell wird erstellt.")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training
    train_model(model, dataloader, criterion, optimizer, device, model_path, epochs=5)

    # Test
    process_video_with_model(video_path, model, device, output_video_path)

if __name__ == "__main__":
    main()
