import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Make sure to import the model classes from your main script
# Or redefine them here if needed
from flower_cgan_train import ConditionalGenerator, device, latent_dim, label_dim, num_classes, image_size

def generate_from_trained_model(num_samples=16, class_idx=0, model_path='improved_models/generator.pth'):
    """
    Generate samples using a trained generator model
    
    Args:
        num_samples: Number of samples to generate
        class_idx: Class index to condition on
        model_path: Path to the saved generator model
    """
    # Create a new generator instance
    generator = ConditionalGenerator(latent_dim, label_dim, num_classes).to(device)
    
    # Load the trained weights
    print(f"Loading model from {model_path}")
    generator.load_state_dict(torch.load(model_path, map_location=device))
    
    # Set to evaluation mode
    generator.eval()
    
    with torch.no_grad():
        # Create fixed class labels
        fixed_class = torch.LongTensor([class_idx] * num_samples).to(device)
        
        # Generate random noise
        z = torch.randn(num_samples, latent_dim, device=device)
        
        # Generate images
        samples = generator(z, fixed_class)
        samples = (samples + 1) / 2.0  # Rescale to [0, 1]
        
        # Create output directory if it doesn't exist
        os.makedirs("generated_samples", exist_ok=True)
        
        # Display the images
        rows = int(np.sqrt(num_samples))
        cols = int(num_samples / rows)
        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
        
        for i, ax in enumerate(axes.flatten()):
            if i < num_samples:
                img = samples[i].cpu().permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.axis('off')
                
                # Save individual images
                img_pil = Image.fromarray((img * 255).astype(np.uint8))
                img_pil.save(f"generated_samples/class_{class_idx}_sample_{i}.png")
        
        plt.suptitle(f"Generated Flowers - Class {class_idx}")
        plt.tight_layout()
        plt.savefig(f"generated_samples/grid_class_{class_idx}.png")
        plt.close()
    
    print(f"Generated {num_samples} samples for class {class_idx}")

# Generate samples for a few classes
if __name__ == "__main__":
    # Create directory for improved models if it doesn't exist
    os.makedirs("improved_models", exist_ok=True)
    
    # Save the trained models from the main script to this directory
    # (Only run this if the models haven't been saved yet)
    if not os.path.exists("improved_models/generator.pth"):
        print("Saving current models to improved_models directory...")
        from flower_cgan_train import generator, discriminator, optimizer_G, optimizer_D
        
        # Save the models
        torch.save(generator.state_dict(), "improved_models/generator.pth")
        torch.save(discriminator.state_dict(), "improved_models/discriminator.pth")
        
        # Save a checkpoint
        torch.save({
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': optimizer_G.state_dict(),
            'd_optimizer_state_dict': optimizer_D.state_dict(),
            'epoch': 100  # Assuming training is complete
        }, "improved_models/checkpoint_final.pth")
        
        print("Models saved successfully!")
    
    # Generate samples for a few different classes
    for class_idx in [0, 25, 50, 75]:
        generate_from_trained_model(num_samples=16, class_idx=class_idx)