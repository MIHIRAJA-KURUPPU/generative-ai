import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from PIL import Image

# Define the same model architecture that was used during training
image_size = 64
latent_dim = 100
label_dim = 50
num_classes = 102

# Define the ConditionalGenerator class again
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, label_dim):
        super(ConditionalGenerator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, label_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + label_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * image_size * image_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_emb = self.label_embedding(labels)
        gen_input = torch.cat((noise, label_emb), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), 3, image_size, image_size)
        return img

def generate_and_save_image(label, generator, latent_dim, num_classes, save_path="generated.png", device='cpu'):
    generator.eval()
    
    if not (0 <= label < num_classes):
        raise ValueError(f'Label should be between 0 and {num_classes - 1}')
    
    # Generate noise and label
    noise = torch.randn(1, latent_dim).to(device)
    label_tensor = torch.tensor([label], dtype=torch.long).to(device)
    
    # Generate image
    with torch.no_grad():
        generated_image = generator(noise, label_tensor)
    
    # Prepare image for saving
    img = generated_image.squeeze().cpu().permute(1, 2, 0)  # (C, H, W) → (H, W, C)
    img = (img + 1) / 2  # Rescale from [-1, 1] to [0, 1]
    
    # Plot and save image
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Generated class: {label}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Image saved to {save_path}")

# Initialize the generator
generator = ConditionalGenerator(latent_dim, label_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.load_state_dict(torch.load('basic_model/models/generator.pth', map_location=device))
generator.to(device)

# Generate image for class 25
generate_and_save_image(label=35, generator=generator, latent_dim=100, num_classes=102, save_path="basic_model/outputs/flower_35.png", device=device)