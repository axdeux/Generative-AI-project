import os
import tkinter as tk
from tkinter import Entry
from PIL import ImageTk, Image
import pygame

from image_model import generate_image
from prompt_model import generate_prompt
from sound_model import generate_audio
from scipy.io.wavfile import write as wav_write
import numpy as np

class DexApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Sound Player")
        self.current_folder = 0
        self.folders = sorted(os.listdir("pokemons/"))  # assuming your folders are in a directory named "data"
        self.load_data()
        self.create_widgets()

    def load_data(self):
        folder_path = os.path.join("pokemons", self.folders[self.current_folder])
        self.name = self.folders[self.current_folder]
        self.image_path = os.path.join(folder_path, "pokemon.png")
        self.description_path = os.path.join(folder_path, "type_description.txt")
        self.sound_path = os.path.join(folder_path, "sound.wav")

    def create_widgets(self):
        self.image_label = tk.Label(self.master)
        self.image_label.grid(row=0, column=0)


        self.description_label = tk.Label(self.master, wraplength=300, justify="left")
        self.description_label.grid(row=0, column=1)

        self.play_button = tk.Button(self.master, text="Play Sound", command=self.play_sound)
        self.play_button.grid(row=2, column=1)

        self.next_button = tk.Button(self.master, text="Next", command=self.next_object)
        self.next_button.grid(row=2, column=0, sticky="E")

        self.previous_button = tk.Button(self.master, text="Previous", command=self.previous_object)
        self.previous_button.grid(row=2, column=0, sticky="W")

        self.prompt_entry = Entry(self.master)
        self.prompt_entry.grid(row=3, column=1, sticky="W")

        self.generate_button = tk.Button(self.master, text="Generate New Pokemon", command= lambda: self.generate_new_pokemon(self.prompt_entry.get()))
        self.generate_button.grid(row=3, column=0, sticky="E")

        self.update_display()

    def update_display(self):
        # Update image
        img = Image.open(self.image_path)
        img = img.resize((300, 300))
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img

        # Update description
        with open(self.description_path, "r") as f:
            description_text = f.read()

        self.type_description = description_text.split("\n")[0]
        self.description = description_text.split("\n")[1]
        self.rw_compare = description_text.split("\n")[2]

        display_text = f"Name: {self.name}\n\n\nType: {self.type_description}\n\n\nDescription: {self.description}\n\nReal-world comparison: {self.rw_compare}"

        self.description_label.config(text=display_text)

    def play_sound(self):
        pygame.mixer.init()
        pygame.mixer.music.load(self.sound_path)
        pygame.mixer.music.play()

    def next_object(self):
        self.current_folder = (self.current_folder + 1) % len(self.folders)
        self.load_data()
        self.update_display()

    def previous_object(self):
        self.current_folder = (self.current_folder - 1) % len(self.folders)
        self.load_data()
        self.update_display()

    def generate_new_pokemon(self, prompt=None):
        folder = "pokemons/"
        if not os.path.exists(folder):
            os.makedirs(folder)

        name, ptype, desc, rw_compar, visual_prompt, audio_prompt = generate_prompt(
            prompt
        )

        pokemon_folder = folder + name +"/"
        if not os.path.exists(pokemon_folder):
            os.makedirs(pokemon_folder)

        type_description = f"{ptype}\n {desc}\n {rw_compar}"
        np.savetxt(pokemon_folder+"type_description.txt", [type_description], fmt="%s")


        # print(f"Name: {name}\nType: {ptype}\nDescription: {desc}\nR/W Comparison: {rw_compar}")

        image = generate_image(visual_prompt)
        image.save(pokemon_folder+"pokemon.png")

        audio = generate_audio(audio_prompt)
        wav_write(pokemon_folder+"sound.wav", 16000, audio)

        self.folders = sorted(os.listdir("pokemons/"))


if __name__ == "__main__":
    root = tk.Tk()
    app = DexApp(root)
    root.mainloop()
