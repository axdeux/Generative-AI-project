from pathlib import Path

import torch
from diffusers import AudioLDM2Pipeline
from scipy.io.wavfile import write as wav_write


def get_pipeline():
    """Get the pretrained AudioLDM2 pipeline."""

    pipe = (
        AudioLDM2Pipeline
        .from_pretrained("cvssp/audioldm2", torch_dtype=torch.float16)
        .to("cuda" if torch.cuda.is_available() else "cpu")
    )

    return pipe


def generate_audio(pipe, auditory_prompt, negative_prompt=None, generator=None):
    """Generate audio from the given prompt.

    Args:
        pipe (AudioLDM2Pipeline): The pretrained AudioLDM2 pipeline.
        prompt (str): The prompt to generate audio from.
        negative_prompt (str): The negative prompt to generate audio from.

    Returns:
        numpy.ndarray: The generated audio.
    """

    if generator is None:
        generator = torch.Generator("cuda").manual_seed(69)
    
    positive_prompt = "Vocalization from an animal."
    if negative_prompt is None:
        negative_prompt = "Hum, background, noise, enviroment, Static, flat, monotonous, hums, lifeless, dull, repetitive, droning"

    positive_prompt += auditory_prompt

    # Generate the audio
    audio = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=100,
        audio_length_in_s=4.0,
        num_waveforms_per_prompt=3,
        generator=generator,
    ).audios

    return audio


if __name__ == "__main__":
    pipe = get_pipeline()

    prompt = "A small, cute, and fluffy Pokemon with a high-pitched voice."
    neg_prompt = "A large, scary, and intimidating Pokemon with a deep voice."

    audio = generate_audio(pipe, prompt, neg_prompt)

    audio_file = Path("output.wav")
    wav_write(audio_file, 16000, audio)
    print(f"Audio saved to {audio_file}")
