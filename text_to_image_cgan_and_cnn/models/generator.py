import torch
import torch.nn as nn
from config import num_classes

class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, label_dim, image_size=64):
        """
        Conditional Generator for a GAN.
        
        Args:
            latent_dim (int): Dimension of the latent space (noise)
            label_dim (int): Dimension of the label embedding
            image_size (int): Size of the output image (height=width)
        """
        super(ConditionalGenerator, self).__init__()
        self.image_size = image_size
        
        # Label embedding layer
        self.label_embedding = nn.Embedding(num_classes, label_dim)
        
        # Main generator architecture
        self.model = nn.Sequential(
            # Input is concatenated latent vector (noise + label embedding)
            # First upsampling: 1x1 -> 4x4
            nn.ConvTranspose2d(latent_dim + label_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # Second upsampling: 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Third upsampling: 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # Fourth upsampling: 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # Final upsampling: 32x32 -> 64x64
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        """
        Forward pass of the generator.
        
        Args:
            noise (torch.Tensor): Batch of noise vectors
            labels (torch.Tensor): Batch of class labels
            
        Returns:
            torch.Tensor: Generated images
        """
        # Embed labels and reshape for concatenation
        label_emb = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)  # [batch_size, label_dim, 1, 1]
        
        # Concatenate noise and label embeddings
        gen_input = torch.cat((noise, label_emb), 1)
        
        # Generate images
        img = self.model(gen_input)
        
        return img  # Already in shape [batch_size, 3, H, W]