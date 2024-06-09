import torch
from diffusers import StableDiffusionXLPipeline


def generate_image(visual_prompt):
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

    visual_prompt = positive_promt + visual_prompt

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device: ", device) 

    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
    pipe = pipe.to(device)


    pipe.enable_attention_slicing()

    
    image = pipe(visual_prompt, num_inference_steps=20, negative_promt=negative_promt).images[0]

    return image

