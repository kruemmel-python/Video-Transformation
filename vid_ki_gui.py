import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import torch
import os
import subprocess
from train_video import VideoFrameTransformer, process_video_with_model

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Video Frame Transformer GUI")
        self.geometry("800x600")

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

        # Helligkeit, Kontrast und Schärfe
        frame_adjustments = ctk.CTkFrame(self)
        frame_adjustments.pack(padx=10, pady=10, fill="x")

        lbl_brightness = ctk.CTkLabel(frame_adjustments, text="Helligkeit:")
        lbl_brightness.pack()
        sld_brightness = ctk.CTkSlider(frame_adjustments, from_=-1, to=1, variable=self.brightness_var, command=self.update_brightness_label)
        sld_brightness.pack(fill="x", padx=5)
        self.brightness_label = ctk.CTkLabel(frame_adjustments, text=f"{self.brightness_var.get():.2f}")
        self.brightness_label.pack()

        lbl_contrast = ctk.CTkLabel(frame_adjustments, text="Kontrast:")
        lbl_contrast.pack()
        sld_contrast = ctk.CTkSlider(frame_adjustments, from_=0.5, to=2, variable=self.contrast_var, command=self.update_contrast_label)
        sld_contrast.pack(fill="x", padx=5)
        self.contrast_label = ctk.CTkLabel(frame_adjustments, text=f"{self.contrast_var.get():.2f}")
        self.contrast_label.pack()

        lbl_sharpness = ctk.CTkLabel(frame_adjustments, text="Schärfe:")
        lbl_sharpness.pack()
        sld_sharpness = ctk.CTkSlider(frame_adjustments, from_=0, to=1, variable=self.sharpness_var, command=self.update_sharpness_label)
        sld_sharpness.pack(fill="x", padx=5)
        self.sharpness_label = ctk.CTkLabel(frame_adjustments, text=f"{self.sharpness_var.get():.2f}")
        self.sharpness_label.pack()

        # Farbsteuerung
        frame_colors = ctk.CTkFrame(self)
        frame_colors.pack(padx=10, pady=10, fill="both", expand=True)

        self.color_labels = {}
        for color_name, var in self.color_factors.items():
            lbl = ctk.CTkLabel(frame_colors, text=f"{color_name.capitalize()}:")
            lbl.pack()
            sld = ctk.CTkSlider(frame_colors, from_=-1, to=1, variable=var, command=lambda value, name=color_name: self.update_color_label(name))
            sld.pack(fill="x", padx=5)
            color_label = ctk.CTkLabel(frame_colors, text=f"{var.get():.2f}")
            color_label.pack()
            self.color_labels[color_name] = color_label

        # Verarbeiten-Button
        frame_start = ctk.CTkFrame(self)
        frame_start.pack(padx=10, pady=10, fill="x")

        btn_process = ctk.CTkButton(frame_start, text="Video verarbeiten", command=self.on_process_video)
        btn_process.pack(padx=5, pady=5)

    def update_brightness_label(self, value):
        self.brightness_label.configure(text=f"{self.brightness_var.get():.2f}")

    def update_contrast_label(self, value):
        self.contrast_label.configure(text=f"{self.contrast_var.get():.2f}")

    def update_sharpness_label(self, value):
        self.sharpness_label.configure(text=f"{self.sharpness_var.get():.2f}")

    def update_color_label(self, color_name):
        var = self.color_factors[color_name]
        self.color_labels[color_name].configure(text=f"{var.get():.2f}")

    def select_input_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video-Dateien", "*.mp4 *.avi *.mov *.mkv"), ("Alle Dateien", "*.*")])
        if path:
            self.input_video_path = path
            print(f"Eingabevideo: {path}")

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
