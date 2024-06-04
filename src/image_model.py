import torch
from diffusers import StableDiffusionXLPipeline


def generate_image(visual_prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"


    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, added_cond_kwargs={})
    pipe = pipe.to(device)

    
    image = pipe(visual_prompt, num_inference_steps=20).images[0]

    return image


