import os
import torch
import matplotlib.pyplot as plt
from config import device, latent_dim, num_classes, output_dir

def generate_and_save_image(label, generator, save_path=None):
    """
    Generate an image for a specific class label and save it.
    
    Args:
        label (int): Class label to generate
        generator: Trained generator model
        save_path (str, optional): Path to save the image. If None, a default path is used.
    """
    # Set generator to evaluation mode
    generator.eval()
    
    # Validate label
    if not (0 <= label < num_classes):
        raise ValueError(f'Label should be between 0 and {num_classes - 1}')
    
    # Default save path if not provided
    if save_path is None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"flower_class_{label}.png")
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Generate noise and label
    noise = torch.randn(1, latent_dim, 1, 1, device=device)
    
    # Convert label to tensor
    label_tensor = torch.tensor([label], dtype=torch.long, device=device)
    
    # Generate image
    with torch.no_grad():
        generated_image = generator(noise, label_tensor)
    
    # Prepare image for saving
    img = generated_image.squeeze().cpu().permute(1, 2, 0).numpy()  # (C, H, W) → (H, W, C)
    img = (img + 1) / 2  # Rescale from [-1, 1] to [0, 1]
    
    # Plot and save image
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Generated flower class: {label}")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Image saved to {save_path}")

def generate_grid(generator, class_range=None, rows=4, cols=4):
    """
    Generate a grid of images for different classes.
    
    Args:
        generator: Trained generator model
        class_range (tuple, optional): Range of classes to generate (start, end)
        rows (int): Number of rows in the grid
        cols (int): Number of columns in the grid
    """
    # Set generator to evaluation mode
    generator.eval()
    
    # Determine class range
    if class_range is None:
        start_class = 0
        end_class = min(rows * cols, num_classes)
    else:
        start_class, end_class = class_range
        end_class = min(end_class, num_classes)
    
    # Prepare figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = axes.flatten()
    
    # Generate images for each class
    for i, class_idx in enumerate(range(start_class, end_class)):
        if i >= rows * cols:
            break
            
        # Generate noise and label
        noise = torch.randn(1, latent_dim, 1, 1, device=device)
        label_tensor = torch.tensor([class_idx], dtype=torch.long, device=device)
        
        # Generate image
        with torch.no_grad():
            generated_image = generator(noise, label_tensor)
        
        # Prepare image for plotting
        img = generated_image.squeeze().cpu().permute(1, 2, 0).numpy()
        img = (img + 1) / 2  # Rescale from [-1, 1] to [0, 1]
        
        # Plot image
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Class {class_idx}")
    
    # Hide unused subplots
    for i in range(end_class - start_class, rows * cols):
        axes[i].axis('off')
    
    # Save the grid
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "flower_grid.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Grid image saved to {save_path}")