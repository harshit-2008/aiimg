import torch
from diffusers import StableDiffusionPipeline

# Load the pre-trained Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4-original"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = StableDiffusionPipeline.from_pretrained(model_id)
pipeline.to(device)

def generate_image(prompt, output_path):
    # Generate image
    with torch.no_grad():
        image = pipeline(prompt).images[0]
    
    # Save the image
    image.save(output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    prompt = "A futuristic city skyline at sunset"
    output_path = "output_image.png"
    generate_image(prompt, output_path)
  
