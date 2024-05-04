from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from scipy.io.wavfile import write as wav_write


def get_diffusion_pipeline():
    # Load the pretrained model
    pipe = (
        DiffusionPipeline()
        .from_pretrained("cvssp/audioldm2")
        .to("cuda" if torch.cuda.is_available() else "cpu")
    )

    return pipe


def generate_audio(pipe, text):

    prompt_template = f"""The following is a specification for a Pokemon. You are to create the sound of the pokemon according to the description.

    Description:
    {text}
    """

    # Generate the audio
    output = pipe(prompt_template, num_inference_steps=50, guidance_scale=7.5)

    return output.audios[0]

if __name__ == "__main__":
    audio_file = Path("output.wav")
    pipe = get_diffusion_pipeline()
    text = "A small, cute, and fluffy Pokemon with a high-pitched voice."
    audio = generate_audio(pipe, text)
    wav_write(audio_file, 16000, audio)
    print(f"Audio saved to {audio_file}")
