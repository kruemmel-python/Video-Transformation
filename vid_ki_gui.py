import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, Scrollbar, Canvas
import torch
import os
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageTk
from train_video import VideoFrameTransformer, process_video_with_model

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Video Frame Transformer GUI")
        self.geometry("550x800")  # Hier können Sie die Größe der GUI festlegen

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "frame_transformer.pth"

        self.input_video_path = ""
        self.output_video_path = ""

        # Variablen für Helligkeit, Kontrast, Schärfe und Farben
        self.brightness_var = tk.DoubleVar(value=0.0)  # Werte: -1.0 bis 1.0
        self.contrast_var = tk.DoubleVar(value=1.0)    # Werte: 0.5 bis 2.0
        self.sharpness_var = tk.DoubleVar(value=0.0)  # Werte: 0.0 bis 1.0

        self.color_factors = {
            "rot": tk.DoubleVar(value=0.0),     # Werte: -1.0 bis 1.0
            "grün": tk.DoubleVar(value=0.0),   # Werte: -1.0 bis 1.0
            "blau": tk.DoubleVar(value=0.0),   # Werte: -1.0 bis 1.0
            "gelb": tk.DoubleVar(value=0.0),   # Werte: -1.0 bis 1.0
            "cyan": tk.DoubleVar(value=0.0),   # Werte: -1.0 bis 1.0
            "magenta": tk.DoubleVar(value=0.0) # Werte: -1.0 bis 1.0
        }

        self.create_widgets()
        self.load_model()

    def create_widgets(self):
        # Videoauswahl
        frame_video = ctk.CTkFrame(self)
        frame_video.pack(padx=10, pady=10, fill="x")

        btn_select_video = ctk.CTkButton(frame_video, text="Eingabevideo auswählen", command=self.select_input_video)
        btn_select_video.pack(side="left", padx=10, pady=10)

        btn_select_output = ctk.CTkButton(frame_video, text="Zielvideo wählen", command=self.select_output_video)
        btn_select_output.pack(side="left", padx=10, pady=10)

        # Scrollbares Fenster für Schieberegler und Vorschau
        self.canvas = Canvas(self, borderwidth=0)
        self.frame = ctk.CTkFrame(self.canvas)
        self.vsb = Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((4,4), window=self.frame, anchor="nw", tags="self.frame")

        self.frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind_all("<MouseWheel>", self.on_mouse_wheel)

        # Helligkeit, Kontrast und Schärfe
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

        # Farbsteuerung
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

        # Verarbeiten-Button
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
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_mouse_wheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def update_preview(self, value=None):
        if not self.input_video_path:
            return

        # Farb- und Anpassungswerte sammeln
        color_adjust = {k: v.get() for k, v in self.color_factors.items()}
        brightness = self.brightness_var.get()
        contrast = self.contrast_var.get()
        sharpness = self.sharpness_var.get()

        # Ein Frame aus dem Video lesen
        cap = cv2.VideoCapture(self.input_video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return

        # Anpassungen anwenden
        frame = self.apply_adjustments(frame, brightness, contrast, sharpness, color_adjust)

        # Frame in der Vorschau anzeigen
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.preview_canvas.create_image(0, 0, anchor="nw", image=imgtk)
        self.preview_canvas.image = imgtk

    def apply_adjustments(self, frame, brightness, contrast, sharpness, color_adjust):
        # Helligkeit anpassen
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness * 100)

        # Kontrast anpassen
        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=0)

        # Schärfe anpassen
        kernel = np.array([[-1, -1, -1], [-1, 8 + sharpness, -1], [-1, -1, -1]], dtype=np.float32)
        frame = cv2.filter2D(frame, -1, kernel)

        # Pixelwerte begrenzen
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Farbanpassungen anwenden
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
        path = filedialog.askopenfilename(filetypes=[("Video-Dateien", "*.mp4 *.avi *.mov *.mkv"), ("Alle Dateien", "*.*")])
        if path:
            self.input_video_path = path
            print(f"Eingabevideo: {path}")
            self.update_preview()

    def select_output_video(self):
        path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4-Datei", "*.mp4"), ("Alle Dateien", "*.*")])
        if path:
            self.output_video_path = path
            print(f"Zielvideo: {path}")

    def load_model(self):
        if not os.path.exists(self.model_path):
            print(f"Kein Modell unter '{self.model_path}' gefunden. Bitte trainiertes Modell bereitstellen.")
            return

        self.model = VideoFrameTransformer().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        print("Modell erfolgreich geladen.")

    def on_process_video(self):
        if not self.input_video_path:
            print("Kein Eingabevideo ausgewählt!")
            return
        if not self.output_video_path:
            print("Kein Zielvideo-Pfad angegeben!")
            return
        if not self.model:
            print("Kein Modell geladen!")
            return

        # Farb- und Anpassungswerte sammeln
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
        process_video_with_model(
            video_path=self.input_video_path,
            model=self.model,
            device=self.device,
            output_path=temp_output_path,
            adjustment_factor=1.0,  # Dies könnte angepasst werden
            brightness_factor=brightness,
            contrast_factor=contrast,
            sharpness_factor=sharpness,
            color_adjustments=color_adjust
        )

        # Ton extrahieren und in das neue Video integrieren
        self.integrate_audio(temp_output_path, self.input_video_path, self.output_video_path)

        print("Videoverarbeitung abgeschlossen.")

    def integrate_audio(self, temp_output_path, input_video_path, output_video_path):
        # Ton aus dem Eingabevideo extrahieren
        audio_path = "temp_audio.aac"
        subprocess.run([
            "ffmpeg", "-i", input_video_path, "-q:a", "0", "-map", "a", audio_path
        ], check=True)

        # Ton in das neue Video integrieren
        subprocess.run([
            "ffmpeg", "-i", temp_output_path, "-i", audio_path, "-c", "copy", "-map", "0:v", "-map", "1:a", output_video_path
        ], check=True)

        # Temporäre Dateien löschen
        os.remove(temp_output_path)
        os.remove(audio_path)

if __name__ == "__main__":
    app = App()
    app.mainloop()
