import torch
from diffusers import StableDiffusionXLPipeline
import os

# Create folder for unseen test data
os.makedirs("unseen_sdxl_test", exist_ok=True)

print("Loading SDXL Pipeline (this may take a moment)...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True
)

# CRITICAL FOR 4GB VRAM: Move parts of the model to CPU RAM when not in use
pipe.enable_model_cpu_offload()

prompts = [
    "A highly detailed portrait of a person, realistic face, 8k resolution",
    "A professional headshot of a man, studio lighting, photorealistic",
    "A close-up photo of a woman's face, natural lighting, highly detailed",
    "A cinematic shot of a human face, realistic skin texture",
    "A selfie of a person, realistic lighting, photorealistic"
]

print("Generating 100 SDXL images... (This will be slow on 4GB VRAM, approx 1-2 images per minute)")
for i in range(100):
    prompt = prompts[i % len(prompts)]
    # Use different seeds to get different faces
    image = pipe(prompt=prompt, num_inference_steps=20, guidance_scale=7.5, generator=torch.Generator(device="cuda").manual_seed(i)).images[0]
    image.save(f"unseen_sdxl_test/sdxl_face_{i}.jpg")
    if (i+1) % 5 == 0:
        print(f"Generated {i+1}/100 images...", flush=True)

print("Done! SDXL images saved to unseen_sdxl_test folder.")