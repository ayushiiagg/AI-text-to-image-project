# AI-text-to-image-project
!pip install diffusers transformers accelerate --upgrade
!pip install safetensors

import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from huggingface_hub import login

# Login to Hugging Face
login("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")

# Load model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

# Prompt enhancer
user_prompt = input("Enter your story prompt: ")
enhanced_prompt = f"A magical illustration of: {user_prompt}, digital art, vibrant colors, fantasy style"

# Generate image
image = pipe(enhanced_prompt).images[0]
# images[0] extracts the first image (the model can return multiple images).
plt.imshow(image)
plt.axis('off')
plt.title("StorySketch AI Output")
plt.show()
