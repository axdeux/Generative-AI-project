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
        StableDiffusionXLPipeline()
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
        visual_prompt (str): A visual prompt for generating the image.

    Returns:
        image (PIL.Image): The generated image.
    """
    # print("--------------------------")
    # print("Prompt: ", visual_prompt)
    # print("--------------------------")

    positive_promt = "Pokemon and cartoon style."
    negative_promt = "3d, realistic"

    positive_promt += visual_prompt

    pipe.enable_attention_slicing()

    image = pipe(
        visual_prompt,
        negative_promt=negative_promt,
        num_inference_steps=20,
        generator=generator,
        num_images_per_prompt=1,
    ).images

    return image


if __name__ == "__main__":
    pipe = get_pipeline()

    prompt = "A small, cute, and fluffy Pokemon with a high-pitched voice."
    neg_prompt = "A large, scary, and intimidating Pokemon with a deep voice."

    image = generate_image(pipe, prompt, neg_prompt)
    image.save("output.png")
    print("Image saved to output.png")
