import os
import cv2
import numpy as np
from tqdm import tqdm

def load_images_from_folder(folder_path):
    """
    Load all jpg images from a folder using OpenCV.

    Args:
        folder_path (str): The path to the folder containing jpg images.

    Returns:
        images (list): A list of images as numpy arrays.
        filenames (list): A list of filenames corresponding to the images.
    """
    images = []
    filenames = []

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]

    for file in tqdm(image_files, desc=f"Loading images from {folder_path}"):
        full_path = os.path.join(folder_path, file)
        try:
            img = cv2.imread(full_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)
            filenames.append(file)
        except Exception as e:
            print(f"Error loading {full_path}: {e}")

    return images, filenames

