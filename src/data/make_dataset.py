import os
import cv2
import numpy as np

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
    
    for file in os.listdir(folder_path):
        if file.lower().endswith('.jpg'):
            full_path = os.path.join(folder_path, file)
            try:
                img = cv2.imread(full_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img_rgb)
                filenames.append(file)
            except Exception as e:
                print(f"Error loading {full_path}: {e}")
                
    return images, filenames

train_folder = "../../data/raw/train"
test_folder = "../../data/raw/test"
valid_folder = "../../data/raw/valid"

train_images, train_filenames = load_images_from_folder(train_folder)
test_images, test_filenames = load_images_from_folder(test_folder)
valid_images, valid_filenames = load_images_from_folder(valid_folder)

print(f"Loaded {len(train_images)} training images.")
