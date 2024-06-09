from image_model import generate_image
from prompt_model import generate_prompt
from sound_model import generate_audio
from scipy.io.wavfile import write as wav_write
import os
import numpy as np

folder = "pokemons/"
if not os.path.exists(folder):
    os.makedirs(folder)

name, ptype, desc, rw_compar, visual_prompt, audio_prompt = generate_prompt(
    input("Pokemon decription: ")
)

pokemon_folder = folder + name +"/"
if not os.path.exists(pokemon_folder):
    os.makedirs(pokemon_folder)

type_description = f"{ptype}\n {desc}\n {rw_compar}"
np.savetxt(pokemon_folder+"type_description.txt", [type_description], fmt="%s")


print(f"Name: {name}\nType: {ptype}\nDescription: {desc}\nR/W Comparison: {rw_compar}")

image = generate_image(visual_prompt)
image.save(pokemon_folder+"pokemon.png")

audio = generate_audio(audio_prompt)
wav_write(pokemon_folder+"sound.wav", 16000, audio)
