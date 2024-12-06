from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import numpy as np

# Define Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, image_size=512):
        self.data = data
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.transform = lambda x: x.resize((self.image_size, self.image_size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = sample["image"]
        text = sample["text"]

        # Load image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]

        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=77)

        return {
            "pixel_values": image,
            "input_ids": inputs["input_ids"].squeeze(),
        }

# Load Components
model_id = "runwayml/stable-diffusion-v1-5"

vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

# Dummy Dataset
dummy_data = [
    {"image": "sample_imges/redshirt.jpg", "text": "A red shirt on a hanger"},
    {"image": "sample_imges/jeans.jpg", "text": "A pair of blue jeans on a rack"},
    {"image": "sample_imges/jacket.jpg", "text": "A stylish black jacket on display"},
]

dataset = CustomDataset(dummy_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Optimizer
optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5)

# Training Loop
device = "cuda" if torch.cuda.is_available() else "cpu"
vae.to(device)
unet.to(device)
text_encoder.to(device)

num_epochs = 3
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    unet.train()

    for batch in tqdm(dataloader):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)

        # Encode text
        text_embeddings = text_encoder(input_ids)["last_hidden_state"]

        # Encode images into latents
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * 0.18215  # Scaling factor

        # Sample timesteps
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, 
            (latents.shape[0],), 
            device=device
        ).long()

        # Add noise to the latents using the scheduler
        noise = torch.randn_like(latents)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        # Predict noise with UNet
        pred_noise = unet(
            sample=noisy_latents, 
            timestep=timesteps, 
            encoder_hidden_states=text_embeddings
        ).sample

        # Compute loss (MSE loss)
        loss = F.mse_loss(pred_noise, noise)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} finished. Loss: {loss.item()}")

# Save Fine-tuned UNet
unet.save_pretrained("finetuned_unet")
