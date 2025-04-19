import os
import torch
from torch.utils.data import Dataset
import scipy.io
from PIL import Image
from torchvision import transforms
from config import image_size

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
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

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