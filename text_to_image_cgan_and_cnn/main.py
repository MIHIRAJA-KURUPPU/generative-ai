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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            # ConvTranspose2d used for upsampling 
            # out_size = (input_size-1)* stride + kernal -2* kernal_size
            
            nn.ConvTranspose2d(latent_dim + label_dim, 512,4,1,0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Output = (1-1)*1+4-2*0 =4
           
            nn.ConvTranspose2d(512,256,4,2,1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Output = (4-1)*2+4-2*1 =8
                   
            nn.ConvTranspose2d(256,128,4,2,1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
             # Output = 16 
            nn.ConvTranspose2d(128,64,4,2,1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),          

            nn.ConvTranspose2d(64,3,4,2,1, bias=False),
            nn.Tanh()
            # Output = 64 
        )

    def forward(self, noise, labels):
        label_emb = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)  #(batch_size*labell_dim*1*1)
        gen_input = torch.cat((noise, label_emb), 1)
        img = self.model(gen_input)
        img = img.view(img.size(0), 3, image_size, image_size)
        return img


class ConditionalDiscriminator(nn.Module):
    def __init__(self, label_dim):
        super(ConditionalDiscriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, label_dim)
        self.model = nn.Sequential(
        # So here we have not used the BatchNorm layer just like the generator in the first layer.
        # But we are going to use that in the rest of the layers.
        # But in a first layer we didn't use it because the first layer of the discriminator processes a raw input data, which can be noisy.
        # And applying BatchNorm directly to this raw input could potentially normalize away useful features that are essential for distinguishing real image from generated ones.
            

        # REason for increasing number of channels in discriminatorfrom each layer to each layer, because as the discriminator processed the image, it needs to extract and analyze various levels of features.
        # In early layers it captures simple features like edges, textures.
        # Then deeper layers it capture more complex structures and patterns from the image, 
        # and increasing the number of channels allows the network to learn and represent a richer set of features necessary for effectively discriminate between real and fake images.
        
        # Con2d for downsample(downscaling)    
            nn.Conv2d(3 + label_dim, 64,4,2,1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64,128,4,2,1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128,256,4,2,1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256,512,4,2,1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512,1,4,1,0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_emb = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)  #(batch_size*labell_dim*1*1)
        label_emb=label_emb.expand(label_emb.size(0), label_emb.size(1), img.size(2), img.size(3)) #(batch_size, label_dim, height, width))
        d_in = torch.cat((label_emb), 1)
        validity = self.model(d_in)
        return validity.view(-1,1)  # [batch_size,1]


#Initialize the models

generator = ConditionalGenerator(latent_dim, label_dim).to(device)
discriminator= ConditionalDiscriminator(label_dim).to(device)


# Loss function & Optimizer
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr= 0.0002)



#Training Loop
for epoch in range(n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        label_emb = labels.to(device)
        real_imgs = imgs.to(device)
        valid = torch.ones(imgs.size(0),1,  device=device)
        fake = torch.zeros(imgs.size(0),1,  device=device)

        # Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs,label_emb),valid)
        noise = torch.randn(imgs.size(0),latent_dim,1,1)
        fake_imgs = generator(noise,label_emb).detach()
        fake_loss = adversarial_loss(discriminator(fake_imgs,label_emb),fake)
        d_loss=  (real_loss+fake_loss)/2
        d_loss.backward()
        optimizer_D.step()

        #Train Generator
        optimizer_G.zero_grad()
        gen_imgs =  generator(noise,label_emb)
        g_loss = adversarial_loss(discriminator(gen_imgs, label_emb),valid)
        g_loss.backward()
        optimizer_G.step()

    # Visualizing the results of the training
    print(f"Epoch [{epoch+1}/{n_epochs}] | Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")


os.makedirs("basic_models/models", exist_ok=True)
torch.save(generator.state_dict(), "basic_models/models/generator.pth")
torch.save(discriminator.state_dict(), "basic_models/models/discriminator.pth")



def generate_and_save_image(label, generator, latent_dim, num_classes, save_path):
    generator.eval()

    if not (0 <= label < num_classes):
        raise ValueError(f'Label should be between 0 and {num_classes - 1}')

    # Generate noise and label
    noise = torch.randn(1, latent_dim,1,1,device=device)

    #Convert label to tensor
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
    plt.title(f"Generated class: {label}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Image saved to {save_path}")


label_to_generate =35
generate_and_save_image(label_to_generate, generator=generator, latent_dim=100, save_path="basic_model/outputs/flower_25.png")
