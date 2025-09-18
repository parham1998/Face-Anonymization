import os
import cv2
import torch
from ignite.metrics import PSNR
import numpy as np

# Define the paths to the two folders
folder1 = 'assets/datasets/CelebA-HQ_align/all'
folder2 = 'results/mobile_face/new 15/all'

# Get a list of image filenames in each folder
images_folder1 = sorted(os.listdir(folder1))
images_folder2 = sorted(os.listdir(folder2))

# Check if the number of images is the same
assert len(images_folder1) == len(
    images_folder2), "The two folders must contain the same number of images."

# Initialize a list to store PSNR values
psnr_values = []

# PSNR Metric from PyTorch Ignite, assuming a data range of 1.0 for normalized image values (0-1)
psnr_metric = PSNR(data_range=1.0)

# Loop through both image folders and calculate PSNR for each pair of images
for img1_name, img2_name in zip(images_folder1, images_folder2):
    # Construct full file paths
    img1_path = os.path.join(folder1, img1_name)
    img2_path = os.path.join(folder2, img2_name)

    # Read the images (assuming they are grayscale; if not, handle the channels appropriately)
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # Ensure both images have the same dimensions
    assert img1.shape == img2.shape, f"Image shapes are not the same: {img1_name} vs {img2_name}"

    # Convert the images to PyTorch tensors, normalize to range [0, 1], and add batch and channel dimensions
    img1_tensor = torch.tensor(
        img1, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    img2_tensor = torch.tensor(
        img2, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

    # Update the PSNR metric
    psnr_metric.update((img1_tensor, img2_tensor))

    # Compute the PSNR for this pair and add to the list
    psnr_value = psnr_metric.compute()
    psnr_values.append(psnr_value)

    print(f"PSNR between {img1_name} and {img2_name}: {psnr_value:.4f} dB")

# Optionally, calculate the average PSNR for all image pairs
average_psnr = np.mean(psnr_values)
print(f"Average PSNR: {average_psnr:.4f} dB")
