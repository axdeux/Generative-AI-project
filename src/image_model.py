from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline


def get_pipeline():
    """
    Function for getting the StableDiffusionXL model pipeline.

    Returns:
        pipe (StableDiffusionXLPipeline): The StableDiffusionXL model pipeline.
    """
    pipe = (
        StableDiffusionXLPipeline
        .from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        )
        .to("cuda" if torch.cuda.is_available() else "cpu")
    )

    return pipe


def generate_image(pipe, visual_prompt, negative_promt=None, generator=None):
    """
    Function for generating an image given a visual prompt, based on StableDiffusionXL model.

    Args:
        pipe (StableDiffusionXLPipeline): The StableDiffusionXL model pipeline.
        visual_prompt (str): A visual prompt for generating the image.
        negative_promt (str): A negative prompt for the image generation, defaults to None.
        generator (torch.Generator): A torch random generator, defaults to None.

    Returns:
        image (PIL.Image): The generated image.
    """

    positive_promt = "Pokemon and cartoon style."
    negative_promt = "3d, realistic"

    positive_promt += visual_prompt

    #Enable attention slicing for better performance, uses less GPU memory. 
    pipe.enable_attention_slicing()

    if generator is None:
        generator = torch.Generator("cuda").manual_seed(69)

    image = pipe(
        positive_promt,
        negative_promt=negative_promt,
        num_inference_steps=20,
        generator=generator,
    ).images[0]

    return image

#Enable running the script directly to generate an image with the given prompt.
if __name__ == "__main__":
    pipe = get_pipeline()

    prompt = "A small, cute, and fluffy Pokemon with a high-pitched voice."
    neg_prompt = "A large, scary, and intimidating Pokemon with a deep voice."

    image = generate_image(pipe, prompt, neg_prompt)
    image.save("output.png")
    print("Image saved to output.png")
