from image_model import generate_image
from prompt_model import generate_prompt
from sound_model import generate_audio
from scipy.io.wavfile import write as wav_write

name, ptype, desc, rw_compar, visual_prompt, audio_prompt = generate_prompt(
    input("Pokemon decription: ")
)

print(f"Name: {name}\nType: {ptype}\nDescription: {desc}\nR/W Comparison: {rw_compar}")

image = generate_image(visual_prompt)
image.save("pokemon.png")

audio = generate_audio(audio_prompt)
wav_write("sound.wav", 16000, audio)
