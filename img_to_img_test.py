from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import torch
import numpy as np

# Load the Fine-Tuned Components
model_id = "runwayml/stable-diffusion-v1-5"
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained("finetuned_unet")  # Fine-tuned UNet
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")


device = "cuda" if torch.cuda.is_available() else "cpu"
vae.to(device)
unet.to(device)
text_encoder.to(device)


prompt = "A colorful artistic version of this jacket"
input_image_path = "test_images/jack.jpg"


input_image = Image.open(input_image_path).convert("RGB").resize((512, 512))
input_image = torch.tensor(np.array(input_image)).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
input_image = input_image.unsqueeze(0).to(device)  # Add batch dimension

# Encode Input Image into Latents
with torch.no_grad():
    latents = vae.encode(input_image).latent_dist.sample()
latents = latents * 0.18215  # Scaling factor


inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
input_ids = inputs["input_ids"].to(device)
with torch.no_grad():
    text_embeddings = text_encoder(input_ids)["last_hidden_state"]

# Add Noise to Latents
scheduler.set_timesteps(50)  # Number of denoising steps
timestep = scheduler.timesteps[0]  # Initial timestep
noise = torch.randn_like(latents)
noisy_latents = scheduler.add_noise(latents, noise, timestep)

# Denoising Process
for t in scheduler.timesteps:
    with torch.no_grad():
        pred_noise = unet(
            sample=noisy_latents,
            timestep=t,
            encoder_hidden_states=text_embeddings
        ).sample
    noisy_latents = scheduler.step(pred_noise, t, noisy_latents)["prev_sample"]

# Decode Latents to Image
noisy_latents = 1 / 0.18215 * noisy_latents  # Scaling factor
with torch.no_grad():
    generated_image = vae.decode(noisy_latents).sample

# Convert Image to PIL Format
generated_image = (generated_image / 2 + 0.5).clamp(0, 1)  # Rescale to [0, 1]
generated_image = generated_image.cpu().permute(0, 2, 3, 1).numpy()  # (batch, height, width, channels)
generated_image = (generated_image[0] * 255).astype(np.uint8)  # Convert to uint8
generated_image = Image.fromarray(generated_image)

# Save or Display Image
generated_image.save("image_to_image_result.png")
generated_image.show()
