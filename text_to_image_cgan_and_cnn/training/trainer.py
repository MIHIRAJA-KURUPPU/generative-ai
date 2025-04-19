import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import device, model_dir, latent_dim

class GanTrainer:
    def __init__(self, generator, discriminator, dataloader, lr=0.0002, beta1=0.5):
        """
        Trainer for Conditional GAN.
        
        Args:
            generator: The generator model
            discriminator: The discriminator model
            dataloader: DataLoader for the training data
            lr: Learning rate for Adam optimizer
            beta1: Beta1 parameter for Adam optimizer
        """
        self.generator = generator
        self.discriminator = discriminator
        self.dataloader = dataloader
        
        # Loss function
        self.adversarial_loss = nn.BCELoss()
        
        # Optimizers
        self.optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        
    def train(self, n_epochs):
        """
        Train the GAN for the specified number of epochs.
        
        Args:
            n_epochs: Number of epochs to train for
        """
        for epoch in range(n_epochs):
            for i, (imgs, labels) in enumerate(self.dataloader):
                # Move data to device
                real_imgs = imgs.to(device)
                labels = labels.to(device)
                
                # Create target labels (real=1, fake=0)
                valid = torch.ones(imgs.size(0), 1, device=device)
                fake = torch.zeros(imgs.size(0), 1, device=device)

                # -----------------
                # Train Discriminator
                # -----------------
                self.optimizer_D.zero_grad()
                
                # Loss on real images
                real_loss = self.adversarial_loss(self.discriminator(real_imgs, labels), valid)
                
                # Generate fake images
                noise = torch.randn(imgs.size(0), latent_dim, 1, 1, device=device)
                fake_imgs = self.generator(noise, labels).detach()
                
                # Loss on fake images
                fake_loss = self.adversarial_loss(self.discriminator(fake_imgs, labels), fake)
                
                # Total discriminator loss
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()

                # -----------------
                # Train Generator
                # -----------------
                self.optimizer_G.zero_grad()
                
                # Generate new fake images
                gen_imgs = self.generator(noise, labels)
                
                # Generator loss
                g_loss = self.adversarial_loss(self.discriminator(gen_imgs, labels), valid)
                g_loss.backward()
                self.optimizer_G.step()

            # Print epoch results
            print(f"Epoch [{epoch+1}/{n_epochs}] | Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")
            
        # Save models after training
        self.save_models()
            
    def save_models(self):
        """Save the trained models."""
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.generator.state_dict(), os.path.join(model_dir, "generator.pth"))
        torch.save(self.discriminator.state_dict(), os.path.join(model_dir, "discriminator.pth"))
        print(f"Models saved to {model_dir}")