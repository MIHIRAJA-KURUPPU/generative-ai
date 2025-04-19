import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Import from our modules
from config import (
    device, data_dir, image_size, batch_size, latent_dim, 
    label_dim, num_classes, n_epochs
)
from data import download_and_extract_dataset, FlowerDataset
from models import ConditionalGenerator, ConditionalDiscriminator
from training import GanTrainer
from utils import generate_and_save_image, generate_grid

def main():
    print(f"Using device: {device}")
    
    # Step 1: Download and extract dataset
    image_dir, label_file = download_and_extract_dataset()
    
    # Step 2: Create dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = FlowerDataset(image_dir=image_dir, label_file=label_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Dataset loaded with {len(dataset)} images")
    
    # Step 3: Initialize models
    generator = ConditionalGenerator(latent_dim, label_dim).to(device)
    discriminator = ConditionalDiscriminator(label_dim).to(device)
    
    print("Models initialized")
    
    # Step 4: Train the model
    trainer = GanTrainer(generator, discriminator, dataloader)
    trainer.train(n_epochs)
    
    # Step 5: Generate some sample images
    print("Generating sample images...")
    for class_idx in [10, 25, 35, 50, 75]:
        generate_and_save_image(class_idx, generator)
    
    # Step 6: Generate a grid of images
    print("Generating image grid...")
    generate_grid(generator, class_range=(0, 16))
    
    print("Done!")

if __name__ == "__main__":
    main()