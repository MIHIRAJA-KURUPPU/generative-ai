import torch
import torch.nn as nn
from config import num_classes

class ConditionalDiscriminator(nn.Module):
    def __init__(self, label_dim, image_size=64):
        """
        Conditional Discriminator for a GAN.
        
        Args:
            label_dim (int): Dimension of the label embedding
            image_size (int): Size of the input image (height=width)
        """
        super(ConditionalDiscriminator, self).__init__()
        
        # Label embedding layer
        self.label_embedding = nn.Embedding(num_classes, label_dim)
        
        # Main discriminator architecture
        self.model = nn.Sequential(
            # First downsampling: 64x64 -> 32x32
            # No BatchNorm in first layer as it processes raw input data
            nn.Conv2d(3 + label_dim, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Second downsampling: 32x32 -> 16x16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Third downsampling: 16x16 -> 8x8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Fourth downsampling: 8x8 -> 4x4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final layer: 4x4 -> 1x1
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        """
        Forward pass of the discriminator.
        
        Args:
            img (torch.Tensor): Batch of real or generated images
            labels (torch.Tensor): Batch of class labels
            
        Returns:
            torch.Tensor: Probability that the image is real
        """
        # Embed labels and expand to image dimensions
        label_emb = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)  # [batch_size, label_dim, 1, 1]
        label_emb = label_emb.expand(label_emb.size(0), label_emb.size(1), img.size(2), img.size(3))
        
        # Concatenate image and label embeddings
        d_in = torch.cat((img, label_emb), 1)
        
        # Discriminate
        validity = self.model(d_in)
        
        return validity.view(-1, 1)  # [batch_size, 1]