import os
import tarfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.io
from PIL import Image
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset directory
data_dir = './content'
os.makedirs(data_dir, exist_ok=True)

# Define image and label paths
image_dir = os.path.join(data_dir, 'jpg')
label_file = os.path.join(data_dir, 'imagelabels.mat')
archive_path = os.path.join(data_dir, '102flowers.tgz')

# Download dataset if not already present
dataset_url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
labels_url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'

# Function to download files with progress
def download_file(url, filepath):
    if not os.path.exists(filepath):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Download complete: {filepath}")
    else:
        print(f"File already exists: {filepath}")

# Download dataset files
download_file(dataset_url, archive_path)
download_file(labels_url, label_file)

# Extract the tar file if images don't exist yet
if not os.path.exists(image_dir):
    print("Extracting archive...")
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=data_dir)
    print("Extraction complete!")
else:
    print("Images directory already exists. Skipping extraction.")

# Define custom dataset class
class FlowerDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        """
        Custom Dataset for loading flower images and labels.

        Args:
            image_dir (str): Directory with all the flower images.
            label_file (str): Path to the .mat file containing labels.
            transform (callable, optional): Transform to apply to images.
        """
        self.image_dir = image_dir
        self.transform = transform

        # Load labels and convert from 1-indexed (MATLAB) to 0-indexed (Python)
        mat = scipy.io.loadmat(label_file)
        self.labels = mat['labels'].flatten() - 1

        # Get list of .jpg files and ensure consistent order
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

        # Check that the number of labels matches the number of images
        assert len(self.image_files) == len(self.labels), \
            f"Mismatch: {len(self.image_files)} images vs {len(self.labels)} labels"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Open image and convert to RGB
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder if image can't be loaded
            image = Image.new('RGB', (64, 64), color='black')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        # Retrieve label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label

# Hyperparameters
image_size = 64
batch_size = 64
latent_dim = 100
label_dim = 50
num_classes = 102
n_epochs = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999

# Image transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Create dataset and split into train/test
dataset = FlowerDataset(image_dir=image_dir, label_file=label_file, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

print(f"Dataset size: {len(dataset)}, Train: {len(train_dataset)}, Test: {len(test_dataset)}")

class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, label_dim, num_classes):
        super(ConditionalGenerator, self).__init__()
        # label_dim is the dimension of the embedded label vector
        self.label_embedding = nn.Embedding(num_classes, label_dim)
        
        # Input processing
        input_dim = latent_dim + label_dim
        
        # Generator network
        self.model = nn.Sequential(
            # First layer
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            
            # Second layer
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            
            # Third layer
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            # Output layer
            nn.Linear(1024, 3 * image_size * image_size),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        # Convert labels to embeddings
        label_emb = self.label_embedding(labels)
        
        # Concatenate noise and label embeddings
        gen_input = torch.cat((noise, label_emb), dim=1)
        
        # Generate flattened image
        img_flat = self.model(gen_input)
        
        # Reshape to image format (batch_size, channels, height, width)
        img = img_flat.view(img_flat.size(0), 3, image_size, image_size)
        
        return img

class ConditionalDiscriminator(nn.Module):
    def __init__(self, label_dim, num_classes):
        super(ConditionalDiscriminator, self).__init__()
        
        self.label_embedding = nn.Embedding(num_classes, label_dim)
        
        # Input dimensions: flattened image + label embedding
        input_dim = 3 * image_size * image_size + label_dim
        
        # Discriminator network
        self.model = nn.Sequential(
            # First layer
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            # Second layer
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            # Output layer - single value for validity
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        # Flatten image
        img_flat = img.view(img.size(0), -1)
        
        # Get label embeddings
        label_emb = self.label_embedding(labels)
        
        # Concatenate image and label embeddings
        d_input = torch.cat((img_flat, label_emb), dim=1)
        
        # Determine validity
        validity = self.model(d_input)
        
        return validity

# Initialize weights for better training
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Model saving functions
def save_models(generator, discriminator, g_optimizer, d_optimizer, epoch, path='improved_models'):
    """
    Save the trained models and optimization states
    
    Args:
        generator: The generator model
        discriminator: The discriminator model
        g_optimizer: Generator optimizer
        d_optimizer: Discriminator optimizer
        epoch: Current epoch number
        path: Directory to save models
    """
    os.makedirs(path, exist_ok=True)
    
    # Save complete models with optimizers (for resuming training)
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'epoch': epoch
    }, os.path.join(path, f'checkpoint_epoch_{epoch}.pth'))
    
    # Save just the models (for inference)
    torch.save(generator.state_dict(), os.path.join(path, 'generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(path, 'discriminator.pth'))
    
    print(f"Models saved to {path} directory!")

def load_models(generator, discriminator, g_optimizer=None, d_optimizer=None, path='improved_models', filename=None):
    """
    Load trained models and optimization states
    
    Args:
        generator: The generator model
        discriminator: The discriminator model
        g_optimizer: Generator optimizer (optional)
        d_optimizer: Discriminator optimizer (optional)
        path: Directory where models are saved
        filename: Specific checkpoint file to load (if None, loads latest)
        
    Returns:
        Start epoch number
    """
    # Default to latest checkpoint if filename not specified
    if filename is None:
        checkpoints = [f for f in os.listdir(path) if f.startswith('checkpoint_epoch_')]
        if not checkpoints:
            print("No checkpoints found. Starting from scratch.")
            return 0
        
        # Get the latest checkpoint
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        filename = latest_checkpoint
    
    checkpoint_path = os.path.join(path, filename)
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file {checkpoint_path} not found. Starting from scratch.")
        return 0
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model states
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    # Load optimizer states if provided
    if g_optimizer is not None:
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    
    if d_optimizer is not None:
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    print(f"Models loaded from {checkpoint_path}. Resuming from epoch {start_epoch}")
    
    return start_epoch

# Function to load just the generator for inference
def load_generator_for_inference(generator, path='improved_models/generator.pth'):
    """
    Load only the generator model for inference
    
    Args:
        generator: The generator model instance
        path: Path to the saved generator model
        
    Returns:
        Loaded generator model
    """
    generator.load_state_dict(torch.load(path, map_location=device))
    generator.eval()  # Set to evaluation mode
    return generator

# Function to generate samples using a trained generator
def generate_samples(generator, num_samples=16, class_idx=0, save_path=None):
    """
    Generate samples using a trained generator
    
    Args:
        generator: The generator model
        num_samples: Number of samples to generate
        class_idx: Class index to condition on
        save_path: Path to save the generated samples image
    """
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
        
        # Display the images
        rows = int(np.sqrt(num_samples))
        cols = int(num_samples / rows)
        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
        
        for i, ax in enumerate(axes.flatten()):
            if i < num_samples:
                img = samples[i].cpu().permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.axis('off')
        
        plt.suptitle(f"Generated Flowers - Class {class_idx}")
        plt.tight_layout()
        
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Samples saved to {save_path}")
        
        plt.close()
    
    # Set back to training mode
    generator.train()
    
    return samples

# Function to save sample images during training
def save_sample_images(generator, fixed_noise, fixed_labels, epoch):
    """
    Save sample images during training
    
    Args:
        generator: The generator model
        fixed_noise: Fixed noise vectors for consistent samples
        fixed_labels: Fixed labels for consistent samples
        epoch: Current epoch number
    """
    # Set generator to evaluation mode
    generator.eval()
    
    with torch.no_grad():
        # Generate images
        gen_imgs = generator(fixed_noise, fixed_labels)
        
        # Rescale images from [-1, 1] to [0, 1]
        gen_imgs = (gen_imgs + 1) / 2.0
        
        # Create grid of images
        plt.figure(figsize=(10, 4))
        for i in range(10):
            plt.subplot(2, 5, i+1)
            img = gen_imgs[i].cpu().permute(1, 2, 0).numpy()
            plt.imshow(img)
            plt.title(f"Class {fixed_labels[i].item()}")
            plt.axis('off')
        
        # Save grid
        os.makedirs("samples", exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"samples/epoch_{epoch}.png")
        plt.close()
    
    # Set generator back to training mode
    generator.train()

# Training function
def train_model(generator, discriminator, train_loader, n_epochs, optimizer_G, optimizer_D, 
               scheduler_G=None, scheduler_D=None, resume_training=False, save_interval=10):
    """
    Train the conditional GAN model
    
    Args:
        generator: The generator model
        discriminator: The discriminator model
        train_loader: DataLoader for the training data
        n_epochs: Number of training epochs
        optimizer_G: Generator optimizer
        optimizer_D: Discriminator optimizer
        scheduler_G: Learning rate scheduler for generator
        scheduler_D: Learning rate scheduler for discriminator
        resume_training: Whether to resume from a saved checkpoint
        save_interval: Interval for saving model checkpoints
    """
    # Loss function
    adversarial_loss = nn.BCELoss()
    
    # Lists to store losses
    g_losses = []
    d_losses = []
    
    # Sample latent vectors to visualize progress
    fixed_noise = torch.randn(10, latent_dim, device=device)
    fixed_labels = torch.LongTensor([i % num_classes for i in range(10)]).to(device)
    
    # Starting epoch
    start_epoch = 0
    
    # Resume training if requested
    if resume_training:
        start_epoch = load_models(generator, discriminator, optimizer_G, optimizer_D)
    
    # Training Loop
    print(f"Starting training from epoch {start_epoch + 1}...")
    
    for epoch in range(start_epoch, n_epochs):
        generator.train()
        discriminator.train()
        
        # Track losses for this epoch
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        # Use tqdm for progress bar
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{n_epochs}")
            
            for i, (real_imgs, labels) in enumerate(tepoch):
                # Move data to device
                real_imgs = real_imgs.to(device)
                labels = labels.to(device)
                batch_size = real_imgs.size(0)
                
                # Create labels for real and fake images
                real_labels = torch.ones(batch_size, 1, device=device)
                fake_labels = torch.zeros(batch_size, 1, device=device)
                
                # Add some noise to the labels for stability
                real_labels = real_labels * 0.9 + 0.1 * torch.rand_like(real_labels)
                fake_labels = fake_labels * 0.9 + 0.1 * torch.rand_like(fake_labels)
                
                #------------------------
                # Train Discriminator
                #------------------------
                optimizer_D.zero_grad()
                
                # Loss on real images
                real_output = discriminator(real_imgs, labels)
                d_real_loss = adversarial_loss(real_output, real_labels)
                
                # Generate fake images
                z = torch.randn(batch_size, latent_dim, device=device)
                fake_imgs = generator(z, labels)
                
                # Loss on fake images
                fake_output = discriminator(fake_imgs.detach(), labels)
                d_fake_loss = adversarial_loss(fake_output, fake_labels)
                
                # Combined loss
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()
                
                #------------------------
                # Train Generator
                #------------------------
                optimizer_G.zero_grad()
                
                # Generate new fake images (since we've updated D)
                z = torch.randn(batch_size, latent_dim, device=device)
                fake_imgs = generator(z, labels)
                
                # Try to fool the discriminator
                fake_output = discriminator(fake_imgs, labels)
                g_loss = adversarial_loss(fake_output, real_labels)
                
                g_loss.backward()
                optimizer_G.step()
                
                # Update running loss values
                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()
                
                # Update progress bar
                tepoch.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item())
        
        # Average losses for the epoch
        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_g_loss = epoch_g_loss / len(train_loader)
        
        # Store losses
        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        
        # Step the learning rate schedulers if provided
        if scheduler_G is not None:
            scheduler_G.step()
        if scheduler_D is not None:
            scheduler_D.step()
        
        # Print epoch results
        print(f"Epoch [{epoch+1}/{n_epochs}] | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")
        
        # Save sample images every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            save_sample_images(generator, fixed_noise, fixed_labels, epoch + 1)
        
        # Save model checkpoints at specified intervals
        if (epoch + 1) % save_interval == 0:
            save_models(generator, discriminator, optimizer_G, optimizer_D, epoch + 1)
    
    # Save final models
    save_models(generator, discriminator, optimizer_G, optimizer_D, n_epochs)
    
    # Plot the training losses
    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Training Losses')
    plt.savefig('training_losses.png')
    plt.close()
    
    print("Training complete!")
    
    return g_losses, d_losses

# Function to evaluate the generator on the test set
def evaluate_generator(generator, test_loader):
    """
    Evaluate the generator on the test set
    
    Args:
        generator: The generator model
        test_loader: DataLoader for the test set
    """
    generator.eval()
    
    # Use FID or other evaluation metrics if available
    print("Evaluating generator...")
    
    # Generate samples for each class in the test set
    class_samples = {}
    
    with torch.no_grad():
        for i, (real_imgs, labels) in enumerate(test_loader):
            # Get unique classes in the batch
            unique_labels = torch.unique(labels)
            
            for label in unique_labels:
                label_val = label.item()
                
                # Skip if we already have samples for this class
                if label_val in class_samples:
                    continue
                
                # Generate samples for this class
                z = torch.randn(16, latent_dim, device=device)
                fixed_class = torch.LongTensor([label_val] * 16).to(device)
                
                # Generate images
                samples = generator(z, fixed_class)
                samples = (samples + 1) / 2.0
                
                # Store samples
                class_samples[label_val] = samples
                
                # Only generate for a few classes
                if len(class_samples) >= 5:
                    break
            
            # Stop after a few batches
            if len(class_samples) >= 5:
                break
    
    # Visualize samples for each class
    for class_idx, samples in class_samples.items():
        # Create a directory for the samples
        os.makedirs("eval_samples", exist_ok=True)
        
        # Display the images
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i, ax in enumerate(axes.flatten()):
            if i < 16:
                img = samples[i].cpu().permute(1, 2, 0).numpy()
                ax.imshow(img)
                ax.axis('off')
        
        plt.suptitle(f"Generated Flowers - Test Class {class_idx}")
        plt.tight_layout()
        plt.savefig(f"eval_samples/test_class_{class_idx}.png")
        plt.close()
    
    generator.train()
    print("Evaluation complete!")


# Main execution
if __name__ == "__main__":
    # Initialize models and move to device
    generator = ConditionalGenerator(latent_dim, label_dim, num_classes).to(device)
    discriminator = ConditionalDiscriminator(label_dim, num_classes).to(device)
    
    # Apply weight initialization
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Initialize optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    
    # Initialize learning rate schedulers
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.5)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=0.5)
    
    # Check if we should resume training
    resume_training = False  # Set to True if you want to resume training
    
    # Train the model
    g_losses, d_losses = train_model(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        n_epochs=n_epochs,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        scheduler_G=scheduler_G,
        scheduler_D=scheduler_D,
        resume_training=resume_training,
        save_interval=10
    )
    
    # Evaluate the generator on the test set
    evaluate_generator(generator, test_loader)
    
    # Generate final samples
    for class_idx in range(0, num_classes, 25):
        if class_idx >= num_classes:
            break
        
        generate_samples(
            generator=generator,
            num_samples=16,
            class_idx=class_idx,
            save_path=f"final_samples/class_{class_idx}.png"
        )