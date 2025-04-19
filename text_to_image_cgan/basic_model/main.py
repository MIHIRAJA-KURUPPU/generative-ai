import os
import tarfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.io
from PIL import Image

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
        image = Image.open(img_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        # Retrieve label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label


image_size = 64
batch_size=64
latent_dim = 100
label_dim = 50
num_classes = 102 
n_epochs = 100

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

dataset = FlowerDataset(image_dir=image_dir, label_file=label_file, transform=transform)
dataloader= DataLoader(dataset, batch_size=batch_size, shuffle=True)

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


class ConditionalDiscriminator(nn.Module):
    def __init__(self, label_dim):
        super(ConditionalDiscriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, label_dim)
        self.model = nn.Sequential(
            nn.Linear(3 * image_size * image_size + label_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        label_emb = self.label_embedding(labels)
        d_in = torch.cat((img_flat, label_emb), -1)
        validity = self.model(d_in)
        return validity


#Initialize the models

generator = ConditionalGenerator(latent_dim, label_dim)
discriminator= ConditionalDiscriminator(label_dim)


# Loss function & Optimizer
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr= 0.0002)



#Training Loop
for epoch in range(n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        label_emb = labels
        real_imgs = imgs
        valid = torch.ones(imgs.size(0),1)
        fake = torch.zeros(imgs.size(0),1)

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs,label_emb),valid)
        fake_imgs = generator(torch.randn(imgs.size(0),latent_dim),label_emb).detach()
        fake_loss = adversarial_loss(discriminator(fake_imgs,label_emb),fake)
        d_loss=  (real_loss+fake_loss)/2
        d_loss.backward()
        optimizer_D.step()

        #Train Generator
        optimizer_G.zero_grad()
        gen_imgs =  generator(torch.randn(imgs.size(0),latent_dim),label_emb)
        g_loss = adversarial_loss(discriminator(gen_imgs, label_emb),valid)
        g_loss.backward()
        optimizer_G.step()

    # Visualizing the results of the training
    print(f"Epoch [{epoch+1}/{n_epochs}] | Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")


os.makedirs("basic_model/models", exist_ok=True)
torch.save(generator.state_dict(), "basic_model/models/generator.pth")
torch.save(discriminator.state_dict(), "basic_model/models/discriminator.pth")


