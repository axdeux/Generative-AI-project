import torch
from diffusers import AutoPipelineForText2Image


def generate_image(visual_prompt):
    """Takes a text decsription and returns an image using the wuerstchen model.

    Args:
        visual_prompt (str): The text description of the image to be generated.

    Returns:
        PIL Image: The generated image.
    
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline =  AutoPipelineForText2Image.from_pretrained(
        "warp-diffusion/wuerstchen"
    ).to(device)


    description = "Pokemon 3d animation style. Entire body and realistic detailed. " + visual_prompt
    negative_prompt = "portrait, HD, drawing, painting, pixel art, text"

    output = pipeline(
        prompt=description,
        height=256,
        width=256,
        prior_guidance_scale=4.0,
        decoder_guidance_scale=0.0,
        negative_prompt=negative_prompt,
    ).images

    return output[0]