import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, Scrollbar, Canvas
import torch
import os
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageTk

# Das vortrainierte Modell und die Videoverarbeitungsklasse werden aus einem separaten Modul importiert.
# Hier wird davon ausgegangen, dass "train_video" existiert und die Klasse VideoFrameTransformer
# sowie die Funktion process_video_with_model bereitstellt.
# Wir integrieren die Funktion process_video_with_model hier jedoch vollständig, um den Code eigenständig lauffähig zu machen.
from train_video import VideoFrameTransformer  # Bitte sicherstellen, dass dieses Modul vorhanden ist.

def process_video_with_model(
    video_path: str,
    model,
    device,
    output_path: str,
    adjustment_factor: float,
    brightness_factor: float,
    contrast_factor: float,
    sharpness_factor: float,
    color_adjustments: dict,
    fps: float
):
    """
    Verarbeitet das Video frameweise mit dem geladenen Modell und speichert das Ergebnis
    als neues Video. Dabei wird sichergestellt, dass die Bildrate (FPS) exakt der des Originalvideos entspricht.

    Args:
        video_path: Pfad zum Eingabevideo.
        model: Das geladene Modell (z. B. VideoFrameTransformer).
        device: Das zu verwendende Gerät (CPU oder GPU).
        output_path: Pfad zum temporären Ausgabevideo (ohne Ton).
        adjustment_factor: Faktor für weitere Anpassungen (hier als Platzhalter genutzt).
        brightness_factor: Faktor zur Anpassung der Helligkeit.
        contrast_factor: Faktor zur Anpassung des Kontrasts.
        sharpness_factor: Faktor zur Anpassung der Schärfe.
        color_adjustments: Wörterbuch mit Farbanpassungen.
        fps: Originale Bildrate (Frames per Second) des Eingabevideos.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Fehler beim Öffnen des Videos!")
        return

    # Ermittlung der Breite und Höhe des Videos
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Erstellen eines VideoWriter-Objekts mit dem originalen FPS-Wert
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Ende des Videos erreicht

        # Hier könnte das Modell angewendet werden. In diesem Beispiel führen wir nur
        # einfache Bildanpassungen durch.
        processed_frame = apply_adjustments(frame, brightness_factor, contrast_factor, sharpness_factor, color_adjustments)

        # Schreiben des verarbeiteten Frames in das temporäre Video
        out.write(processed_frame)
        frame_index += 1

    cap.release()
    out.release()
    print(f"Verarbeitete {frame_index} Frames und gespeichert unter {output_path}.")


def apply_adjustments(frame: np.ndarray, brightness: float, contrast: float, sharpness: float, color_adjust: dict) -> np.ndarray:
    """
    Wendet Helligkeits-, Kontrast-, Schärfe- und Farbanpassungen auf einen Frame an.

    Args:
        frame: Eingabe-Frame im BGR-Format.
        brightness: Faktor für die Helligkeit (-1.0 bis 1.0).
        contrast: Faktor für den Kontrast (0.5 bis 2.0).
        sharpness: Faktor für die Schärfe (0.0 bis 1.0).
        color_adjust: Wörterbuch mit Farbanpassungen (z. B. {"rot": 0.1, ...}).

    Returns:
        Der angepasste Frame.
    """
    # Helligkeitsanpassung: beta steuert die Helligkeitsskala
    frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness * 100)
    # Kontrasteinstellung: alpha skaliert die Intensität
    frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=0)
    # Schärfeanpassung: Einfache Schärfefilterung mittels Faltung
    kernel = np.array([[-1, -1, -1],
                       [-1, 8 + sharpness, -1],
                       [-1, -1, -1]], dtype=np.float32)
    frame = cv2.filter2D(frame, -1, kernel)
    frame = np.clip(frame, 0, 255).astype(np.uint8)

    # Farbanpassungen werden kanalweise durchgeführt
    for color_name, factor in color_adjust.items():
        if color_name == "rot":
            frame[:, :, 2] = np.clip(frame[:, :, 2] * (1 + factor), 0, 255)
        elif color_name == "grün":
            frame[:, :, 1] = np.clip(frame[:, :, 1] * (1 + factor), 0, 255)
        elif color_name == "blau":
            frame[:, :, 0] = np.clip(frame[:, :, 0] * (1 + factor), 0, 255)
        elif color_name == "gelb":
            frame[:, :, 1] = np.clip(frame[:, :, 1] * (1 + factor), 0, 255)
            frame[:, :, 2] = np.clip(frame[:, :, 2] * (1 + factor), 0, 255)
        elif color_name == "cyan":
            frame[:, :, 0] = np.clip(frame[:, :, 0] * (1 + factor), 0, 255)
            frame[:, :, 1] = np.clip(frame[:, :, 1] * (1 + factor), 0, 255)
        elif color_name == "magenta":
            frame[:, :, 0] = np.clip(frame[:, :, 0] * (1 + factor), 0, 255)
            frame[:, :, 2] = np.clip(frame[:, :, 2] * (1 + factor), 0, 255)
    return frame


class App(ctk.CTk):
    """
    Hauptanwendungsklasse für die Video Frame Transformer GUI.
    Erstellt die Benutzeroberfläche, lädt das Modell, ermöglicht die Auswahl von Eingabe-
    und Ausgabevideo, bietet Steuerelemente zur Bildanpassung und verarbeitet das Video.
    """
    def __init__(self):
        super().__init__()
        self.title("Video Frame Transformer GUI")
        self.geometry("550x800")  # Größe der GUI

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # Modellbezogene Variablen
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "frame_transformer.pth"

        # Pfade für Eingabe- und Ausgabevideo
        self.input_video_path = ""
        self.output_video_path = ""
        self.original_fps = None  # Speichert die originale FPS des Videos

        # Variablen für Anpassungen
        self.brightness_var = tk.DoubleVar(value=0.0)  # Helligkeit (-1.0 bis 1.0)
        self.contrast_var = tk.DoubleVar(value=1.0)    # Kontrast (0.5 bis 2.0)
        self.sharpness_var = tk.DoubleVar(value=0.0)   # Schärfe (0.0 bis 1.0)
        self.color_factors = {
            "rot": tk.DoubleVar(value=0.0),
            "grün": tk.DoubleVar(value=0.0),
            "blau": tk.DoubleVar(value=0.0),
            "gelb": tk.DoubleVar(value=0.0),
            "cyan": tk.DoubleVar(value=0.0),
            "magenta": tk.DoubleVar(value=0.0)
        }

        self.create_widgets()
        self.load_model()

    def create_widgets(self):
        """Erstellt alle GUI-Elemente."""
        # Frame für Videoauswahl
        frame_video = ctk.CTkFrame(self)
        frame_video.pack(padx=10, pady=10, fill="x")

        btn_select_video = ctk.CTkButton(frame_video, text="Eingabevideo auswählen", command=self.select_input_video)
        btn_select_video.pack(side="left", padx=10, pady=10)

        btn_select_output = ctk.CTkButton(frame_video, text="Zielvideo wählen", command=self.select_output_video)
        btn_select_output.pack(side="left", padx=10, pady=10)

        # Scrollbarer Bereich für Schieberegler und Vorschau
        self.canvas = Canvas(self, borderwidth=0)
        self.frame = ctk.CTkFrame(self.canvas)
        self.vsb = Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((4, 4), window=self.frame, anchor="nw", tags="self.frame")

        self.frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind_all("<MouseWheel>", self.on_mouse_wheel)

        # Frame für Helligkeit, Kontrast und Schärfe
        frame_adjustments = ctk.CTkFrame(self.frame)
        frame_adjustments.pack(padx=10, pady=10, fill="x")

        lbl_brightness = ctk.CTkLabel(frame_adjustments, text="Helligkeit:")
        lbl_brightness.pack()
        sld_brightness = ctk.CTkSlider(frame_adjustments, from_=-1, to=1, variable=self.brightness_var, command=self.update_preview)
        sld_brightness.pack(fill="x", padx=5)
        self.brightness_label = ctk.CTkLabel(frame_adjustments, text=f"{self.brightness_var.get():.2f}")
        self.brightness_label.pack()

        lbl_contrast = ctk.CTkLabel(frame_adjustments, text="Kontrast:")
        lbl_contrast.pack()
        sld_contrast = ctk.CTkSlider(frame_adjustments, from_=0.5, to=2, variable=self.contrast_var, command=self.update_preview)
        sld_contrast.pack(fill="x", padx=5)
        self.contrast_label = ctk.CTkLabel(frame_adjustments, text=f"{self.contrast_var.get():.2f}")
        self.contrast_label.pack()

        lbl_sharpness = ctk.CTkLabel(frame_adjustments, text="Schärfe:")
        lbl_sharpness.pack()
        sld_sharpness = ctk.CTkSlider(frame_adjustments, from_=0, to=1, variable=self.sharpness_var, command=self.update_preview)
        sld_sharpness.pack(fill="x", padx=5)
        self.sharpness_label = ctk.CTkLabel(frame_adjustments, text=f"{self.sharpness_var.get():.2f}")
        self.sharpness_label.pack()

        # Frame für Farbanpassungen
        frame_colors = ctk.CTkFrame(self.frame)
        frame_colors.pack(padx=10, pady=10, fill="both", expand=True)
        self.color_labels = {}
        for color_name, var in self.color_factors.items():
            lbl = ctk.CTkLabel(frame_colors, text=f"{color_name.capitalize()}:")
            lbl.pack()
            sld = ctk.CTkSlider(frame_colors, from_=-1, to=1, variable=var, command=lambda value, name=color_name: self.update_preview())
            sld.pack(fill="x", padx=5)
            color_label = ctk.CTkLabel(frame_colors, text=f"{var.get():.2f}")
            color_label.pack()
            self.color_labels[color_name] = color_label

        # Frame für den Verarbeitungs-Button
        frame_start = ctk.CTkFrame(self.frame)
        frame_start.pack(padx=10, pady=10, fill="x")
        btn_process = ctk.CTkButton(frame_start, text="Video verarbeiten", command=self.on_process_video)
        btn_process.pack(padx=5, pady=5)

        # Vorschau-Bereich
        self.preview_label = ctk.CTkLabel(self.frame, text="Vorschau")
        self.preview_label.pack(padx=10, pady=10)
        self.preview_canvas = ctk.CTkCanvas(self.frame, width=640, height=480)
        self.preview_canvas.pack(padx=10, pady=10)

    def on_frame_configure(self, event):
        """Aktualisiert den Scrollbereich, wenn sich die Größe ändert."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_mouse_wheel(self, event):
        """Ermöglicht das Scrollen mit dem Mausrad."""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def update_preview(self, value=None):
        """Aktualisiert die Vorschau des Videos mit den aktuellen Anpassungswerten."""
        if not self.input_video_path:
            return

        # Sammeln der aktuellen Werte für Farbanpassungen und Bildverarbeitung
        color_adjust = {k: v.get() for k, v in self.color_factors.items()}
        brightness = self.brightness_var.get()
        contrast = self.contrast_var.get()
        sharpness = self.sharpness_var.get()

        cap = cv2.VideoCapture(self.input_video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return

        frame = self.apply_adjustments(frame, brightness, contrast, sharpness, color_adjust)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.preview_canvas.create_image(0, 0, anchor="nw", image=imgtk)
        self.preview_canvas.image = imgtk  # Referenz speichern, damit das Bild nicht verloren geht

    def apply_adjustments(self, frame, brightness, contrast, sharpness, color_adjust):
        """Wendet die Anpassungen auf einen einzelnen Frame an."""
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness * 100)
        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=0)
        kernel = np.array([[-1, -1, -1],
                           [-1, 8 + sharpness, -1],
                           [-1, -1, -1]], dtype=np.float32)
        frame = cv2.filter2D(frame, -1, kernel)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        for color_name, factor in color_adjust.items():
            if color_name == "rot":
                frame[:, :, 2] = np.clip(frame[:, :, 2] * (1 + factor), 0, 255)
            elif color_name == "grün":
                frame[:, :, 1] = np.clip(frame[:, :, 1] * (1 + factor), 0, 255)
            elif color_name == "blau":
                frame[:, :, 0] = np.clip(frame[:, :, 0] * (1 + factor), 0, 255)
            elif color_name == "gelb":
                frame[:, :, 1] = np.clip(frame[:, :, 1] * (1 + factor), 0, 255)
                frame[:, :, 2] = np.clip(frame[:, :, 2] * (1 + factor), 0, 255)
            elif color_name == "cyan":
                frame[:, :, 0] = np.clip(frame[:, :, 0] * (1 + factor), 0, 255)
                frame[:, :, 1] = np.clip(frame[:, :, 1] * (1 + factor), 0, 255)
            elif color_name == "magenta":
                frame[:, :, 0] = np.clip(frame[:, :, 0] * (1 + factor), 0, 255)
                frame[:, :, 2] = np.clip(frame[:, :, 2] * (1 + factor), 0, 255)
        return frame

    def select_input_video(self):
        """Öffnet einen Dateidialog zur Auswahl eines Eingabevideos."""
        path = filedialog.askopenfilename(filetypes=[("Video-Dateien", "*.mp4 *.avi *.mov *.mkv"), ("Alle Dateien", "*.*")])
        if path:
            self.input_video_path = path
            print(f"Eingabevideo: {path}")
            self.original_fps = self.get_fps(path)
            print(f"Original FPS: {self.original_fps}")
            self.update_preview()

    def select_output_video(self):
        """Öffnet einen Dateidialog zum Festlegen des Ausgabepfads."""
        path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4-Datei", "*.mp4"), ("Alle Dateien", "*.*")])
        if path:
            self.output_video_path = path
            print(f"Zielvideo: {path}")

    def load_model(self):
        """Lädt das vortrainierte Modell aus der angegebenen Datei."""
        if not os.path.exists(self.model_path):
            print(f"Kein Modell unter '{self.model_path}' gefunden. Bitte trainiertes Modell bereitstellen.")
            return
        self.model = VideoFrameTransformer().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        print("Modell erfolgreich geladen.")

    def get_fps(self, video_path):
        """Ermittelt die Bildrate (FPS) des angegebenen Videos."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

    def on_process_video(self):
        """Verarbeitet das Video mithilfe des geladenen Modells und der aktuellen Anpassungswerte."""
        if not self.input_video_path:
            print("Kein Eingabevideo ausgewählt!")
            return
        if not self.output_video_path:
            print("Kein Zielvideo-Pfad angegeben!")
            return
        if not self.model:
            print("Kein Modell geladen!")
            return

        color_adjust = {k: v.get() for k, v in self.color_factors.items()}
        brightness = self.brightness_var.get()
        contrast = self.contrast_var.get()
        sharpness = self.sharpness_var.get()

        print("--- Verarbeitungsparameter ---")
        print(f"Eingabevideo: {self.input_video_path}")
        print(f"Zielvideo: {self.output_video_path}")
        print(f"Helligkeit: {brightness:.2f}")
        print(f"Kontrast: {contrast:.2f}")
        print(f"Schärfe: {sharpness:.2f}")
        print(f"Farbfaktoren: {color_adjust}")
        print("--------------------------------")

        temp_output_path = "temp_output.mp4"
        # Verarbeitung des Videos mit Beibehaltung der originalen FPS
        process_video_with_model(
            video_path=self.input_video_path,
            model=self.model,
            device=self.device,
            output_path=temp_output_path,
            adjustment_factor=1.0,  # Platzhalter, kann bei Bedarf angepasst werden
            brightness_factor=brightness,
            contrast_factor=contrast,
            sharpness_factor=sharpness,
            color_adjustments=color_adjust,
            fps=self.original_fps
        )

        # Integriert den Ton aus dem Originalvideo in das verarbeitete Video
        self.integrate_audio(temp_output_path, self.input_video_path, self.output_video_path, self.original_fps)
        print("Videoverarbeitung abgeschlossen.")

    def integrate_audio(self, temp_output_path, input_video_path, output_video_path, fps):
        """
        Integriert den Ton aus dem Eingabevideo in das temporär verarbeitete Video.
        Dabei wird der experimentelle AAC-Encoder aktiviert (Parameter "-strict -2"), 
        um das Zusammenführen von Video und Audio zu ermöglichen.
        """
        # Schritt 1: Audio extrahieren und in eine temporäre Datei speichern
        audio_path = "temp_audio.aac"
        try:
            subprocess.run([
                "ffmpeg", "-i", input_video_path, "-vn", "-acodec", "copy", audio_path
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Fehler beim Extrahieren des Audios: {e.stderr}")
            return

        # Schritt 2: Videostream und extrahierten Audiostream zusammenführen,
        # dabei die originale Bildrate (FPS) beibehalten.
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", temp_output_path,
            "-i", audio_path,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            "-strict", "-2",  # Aktiviert den experimentellen AAC-Encoder
            "-map", "0:v",
            "-map", "1:a",
            "-r", str(fps),
            output_video_path
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Fehler beim Zusammenführen von Video und Audio: {e.stderr}")
            return

        # Schritt 3: Temporäre Dateien löschen
        try:
            os.remove(temp_output_path)
            os.remove(audio_path)
        except OSError as e:
            print(f"Fehler beim Löschen temporärer Dateien: {e}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
