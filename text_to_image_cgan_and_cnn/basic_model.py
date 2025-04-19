import os
import tarfile
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scipy.io
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset directory
data_dir = './content'
os.makedirs(data_dir, exist_ok=True)

# Define image and label paths
image_dir = os.path.join(data_dir, 'jpg')
label_file = os.path.join(data_dir, 'imagelabels.mat')
archive_path = os.path.join(data_dir, '102flowers.tgz')

# Dataset URLs
dataset_url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
labels_url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'

def download_file(url, filepath):
    if not os.path.exists(filepath):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Download complete: {filepath}")
    else:
        print(f"File already exists: {filepath}")

# Download files
download_file(dataset_url, archive_path)
download_file(labels_url, label_file)

# Extract if not already
if not os.path.exists(image_dir):
    print("Extracting archive...")
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=data_dir)
    print("Extraction complete!")
else:
    print("Images directory already exists. Skipping extraction.")

# Dataset
class FlowerDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        mat = scipy.io.loadmat(label_file)
        self.labels = mat['labels'].flatten() - 1  # 0-indexed
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        assert len(self.image_files) == len(self.labels), \
            f"Mismatch: {len(self.image_files)} images vs {len(self.labels)} labels"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# Hyperparameters
image_size = 64
batch_size = 64
latent_dim = 100
label_dim = 50
num_classes = 102
n_epochs = 100

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = FlowerDataset(image_dir, label_file, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, label_dim):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, label_dim)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + label_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_emb = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        gen_input = torch.cat((noise, label_emb), 1)
        return self.model(gen_input)

# Discriminator
class ConditionalDiscriminator(nn.Module):
    def __init__(self, label_dim):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, label_dim)
        self.model = nn.Sequential(
            nn.Conv2d(3 + label_dim, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_emb = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        label_map = label_emb.expand(img.size(0), label_emb.size(1), img.size(2), img.size(3))
        d_in = torch.cat((img, label_map), 1)
        return self.model(d_in).view(-1, 1)

# Initialize
generator = ConditionalGenerator(latent_dim, label_dim).to(device)
discriminator = ConditionalDiscriminator(label_dim).to(device)

# Loss and Optimizers
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training
for epoch in range(n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        real_imgs = imgs.to(device)
        labels = labels.to(device)
        batch_size = real_imgs.size(0)

        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
        fake_imgs = generator(noise, labels).detach()
        fake_loss = adversarial_loss(discriminator(fake_imgs, labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        gen_imgs = generator(noise, labels)
        g_loss = adversarial_loss(discriminator(gen_imgs, labels), valid)
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{n_epochs}] | Loss D: {d_loss.item():.4f} | Loss G: {g_loss.item():.4f}")

# Save models
os.makedirs("basic_models/models", exist_ok=True)
torch.save(generator.state_dict(), "basic_models/models/generator.pth")
torch.save(discriminator.state_dict(), "basic_models/models/discriminator.pth")

# Generate and save image
def generate_and_save_image(label, generator, latent_dim, save_path):
    generator.eval()
    if not (0 <= label < num_classes):
        raise ValueError(f'Label must be between 0 and {num_classes - 1}')
    noise = torch.randn(1, latent_dim, 1, 1, device=device)
    label_tensor = torch.tensor([label], dtype=torch.long, device=device)
    with torch.no_grad():
        gen_img = generator(noise, label_tensor)
    img = gen_img.squeeze().cpu().permute(1, 2, 0).numpy()
    img = (img + 1) / 2  # [-1,1] → [0,1]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Generated class: {label}")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Image saved to {save_path}")

# Example usage
generate_and_save_image(
    label=35,
    generator=generator,
    latent_dim=latent_dim,
    save_path="basic_models/outputs/flower_35.png"
)
