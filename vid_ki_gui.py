import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import cv2
import numpy as np
from tqdm import tqdm
import os

# -----------------------------
# Neuronales Netzwerk
# -----------------------------
class VideoFrameTransformer(torch.nn.Module):
    def __init__(self):
        super(VideoFrameTransformer, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 3, kernel_size=3, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# -----------------------------
# Funktionen
# -----------------------------
def process_video_with_model(video_path, model, device, output_path, adjustment_factor=1.0, resolution=(1920, 1080)):
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
            adjusted_output = frame + (output - frame) * adjustment_factor
            adjusted_output = torch.clamp(adjusted_output, 0, 1)
            processed_frames.append(adjusted_output.squeeze(0).cpu().numpy())

    processed_frames = (np.array(processed_frames) * 255).astype(np.uint8).transpose(0, 2, 3, 1)

    height, width = resolution
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in processed_frames:
        resized_frame = cv2.resize(frame, (width, height))
        out.write(cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR))
    out.release()
    messagebox.showinfo("Erfolg", f"Video gespeichert unter {output_path}")

# -----------------------------
# GUI
# -----------------------------
def start_gui():
    def load_model():
        global model
        model_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pth")])
        if not model_path:
            return
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            messagebox.showinfo("Erfolg", "Modell erfolgreich geladen!")
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Laden des Modells: {e}")

    def select_input_video():
        video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])
        if video_path:
            input_video_var.set(video_path)

    def generate_output_video():
        video_path = input_video_var.get()
        if not os.path.exists(video_path):
            messagebox.showerror("Fehler", "Bitte wählen Sie ein gültiges Eingabevideo aus.")
            return

        output_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 Video", "*.mp4")])
        if not output_path:
            return

        try:
            adjustment_factor = float(adjustment_factor_var.get())
            resolution = resolution_var.get()
            resolution_map = {
                "HD": (1280, 720),
                "Full HD": (1920, 1080),
                "2K": (1080, 2048),
                "4K": (2160, 3840)
            }
            process_video_with_model(video_path, model, device, output_path, adjustment_factor, resolution_map[resolution])
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler bei der Videoverarbeitung: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global model
    model = VideoFrameTransformer().to(device)

    root = tk.Tk()
    root.title("Video Transformation GUI")
    root.geometry("400x400")

    tk.Label(root, text="Eingabevideo").pack(pady=5)
    input_video_var = tk.StringVar()
    tk.Entry(root, textvariable=input_video_var, width=40).pack(pady=5)
    tk.Button(root, text="Video auswählen", command=select_input_video).pack(pady=5)

    tk.Label(root, text="Anpassungsfaktor (z. B. 1.0)").pack(pady=5)
    adjustment_factor_var = tk.StringVar(value="1.0")
    tk.Entry(root, textvariable=adjustment_factor_var, width=10).pack(pady=5)

    tk.Label(root, text="Auflösung auswählen").pack(pady=5)
    resolution_var = tk.StringVar(value="Full HD")
    tk.OptionMenu(root, resolution_var, "HD", "Full HD", "2K", "4K").pack(pady=5)

    tk.Button(root, text="Modell laden", command=load_model).pack(pady=5)
    tk.Button(root, text="Video erstellen", command=generate_output_video).pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    start_gui()
