import os
import tarfile
import urllib.request
from config import data_dir, dataset_url, labels_url

def download_file(url, filepath):
    """
    Download a file with progress indication.
    
    Args:
        url (str): URL to download from
        filepath (str): Path where the file should be saved
    """
    if not os.path.exists(filepath):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Download complete: {filepath}")
    else:
        print(f"File already exists: {filepath}")

def download_and_extract_dataset():
    """
    Download and extract the 102 Oxford Flowers dataset.
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Define image and label paths
    image_dir = os.path.join(data_dir, 'jpg')
    label_file = os.path.join(data_dir, 'imagelabels.mat')
    archive_path = os.path.join(data_dir, '102flowers.tgz')
    
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
    
    return image_dir, label_file