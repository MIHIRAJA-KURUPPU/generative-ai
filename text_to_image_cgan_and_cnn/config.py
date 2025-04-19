# Configuration parameters for the GAN model and training
import torch

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data directory
data_dir = './content'

# URLs
dataset_url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
labels_url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'

# Model parameters
image_size = 64
batch_size = 64
latent_dim = 100  # Size of the noise vector
label_dim = 50    # Size of the label embedding dimension
num_classes = 102  # Number of flower classes

# Training parameters
n_epochs = 100
learning_rate = 0.0002
beta1 = 0.5  # Adam optimizer beta parameter

# Output directories
model_dir = "models"
output_dir = "outputs"