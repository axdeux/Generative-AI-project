import tkinter as tk
from tkinter import Text, Label, Entry, Button
from PIL import Image, ImageTk
from image_model import generate_image
from prompt_model import generate_prompt
from sound_model import generate_audio
from scipy.io.wavfile import write as wav_write
import pygame
import numpy as np
import io
import os

# Initialize pygame mixer for playing sound
pygame.mixer.init()
current_directory = os.path.dirname(os.path.abspath(__file__))+"/"
# Lists to store the history of generated Pokémon
pokemon_path = current_directory+"pokemons/"
pokemon_list = os.listdir(pokemon_path)
current_index = -1

# Function to generate new Pokémon details
def generate_pokemon(prompt):
    global current_index
    name, ptype, desc, rw_compar, visual_prompt, audio_prompt = generate_prompt(prompt)
    pokemon_folder = pokemon_path + name +"/"
    if not os.path.exists(pokemon_folder):
       os.makedirs(pokemon_folder)
    image = generate_image(visual_prompt)
    sound = generate_audio(audio_prompt)
    type_description = f"{ptype}\n {desc}\n {rw_compar}"
    np.savetxt(pokemon_path+"name"+"type_description.txt", [type_description], fmt="%s")
    image.save(pokemon_folder+"pokemon.png")
    wav_write(pokemon_folder+"sound.wav", 16000, sound)

    current_index = len(pokemon_list) - 1
    update_display(pokemon_data)

def update_display(pokemon_data):
    # Update the image
    print(pokemon_data)
    img = Image.open(pokemon_data + '/pokemon.png')
    img = img.resize((250, 250))
    img = ImageTk.PhotoImage(img)
    pokemon_image_label.config(image=img)
    pokemon_image_label.image = img  # Keep a reference to avoid garbage collection
    
    # Update the description
    with open(pokemon_data + '/type_description.txt', 'r') as f:
        description = f.read()
        
    pokemon_description = description.split("\n")[1]
    pokemon_type = description.split("\n")[0]
    pokemon_name = pokemon_data.split("/")[-1]
    #Show description below the image
    pokemon_description_text.delete('1.0', tk.END)
    pokemon_description_text.insert(tk.END, f"Name: {pokemon_name}\nType: {pokemon_type}\nDescription: {pokemon_description}")
    
    # Play the sound
    global current_index
    if current_index != -1:
        pygame.mixer.music.load(pokemon_path+pokemon_list[current_index]+'/sound.wav')
        pygame.mixer.music.play()


def play_sound():
    global current_index
    if current_index != -1:
        pygame.mixer.music.load(pokemon_path+pokemon_list[current_index]+'/sound.wav')
        pygame.mixer.music.play()

# Function to navigate to the previous Pokémon
def show_previous_pokemon():
    global current_index
    if current_index > 0:
        current_index -= 1
        update_display(pokemon_path+pokemon_list[current_index])

# Function to navigate to the next Pokémon
def show_next_pokemon():
    global current_index
    if current_index < len(pokemon_list) - 1:
        current_index += 1
        update_display(pokemon_path+pokemon_list[current_index])

# Create the main window
root = tk.Tk()
root.title("AI Pokedex")

# Pokémon image display

# Entry box for name
name_label = Label(root, text="Name:")
name_label.grid(row=0, column=0, padx=10, pady=5)
name_entry = Entry(root, width=30)
name_entry.grid(row=0, column=1, padx=10, pady=5)

# Entry box for type
type_label = Label(root, text="Type:")
type_label.grid(row=1, column=0, padx=10, pady=5)
type_entry = Entry(root, width=30)
type_entry.grid(row=1, column=1, padx=10, pady=5)

# Pokémon image display
pokemon_image_label = Label(root)
pokemon_image_label.grid(row=2, column=0, columnspan=2, padx=10, pady=5)

# Pokémon description display
pokemon_description_text = Text(root, height=10, width=50)
pokemon_description_text.grid(row=3, column=0, columnspan=2, padx=10, pady=5)

# Entry box for prompts
prompt_entry = Entry(root, width=50)
prompt_entry.grid(row=5, column=0, columnspan=2, padx=10, pady=5)

# Button to generate new Pokémon
generate_button = Button(root, text="Generate Pokémon", command=lambda: generate_pokemon(prompt_entry.get()))
generate_button.grid(row=5, column=1, columnspan=2, padx=10, pady=5)

# Navigation buttons
navigation_frame = tk.Frame(root)
navigation_frame.grid(row=6, column=0, columnspan=2, padx=10, pady=5)

prev_button = Button(navigation_frame, text="Previous", command=show_previous_pokemon)
prev_button.grid(row=0, column=0, padx=5)

next_button = Button(navigation_frame, text="Next", command=show_next_pokemon)
next_button.grid(row=0, column=1, padx=5)

# Button to play sound
play_sound_button = Button(root, text="Play Sound", command=play_sound)
play_sound_button.grid(row=7, column=0, columnspan=2, padx=10, pady=5)

# Start the GUI event loop
root.mainloop()