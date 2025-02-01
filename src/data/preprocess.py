# src/data/preprocess.py

import cv2
import numpy as np
import os

def load_images_from_directory(directory: str):
    """
    Load all images from the given directory.
    Returns a list of images.
    """
    # Implement loading images here

def resize_images(images, target_size=(256, 256)):
    """
    Resize all images to a consistent size.
    """
    # Implement resizing here

def normalize_images(images):
    """
    Normalize image pixels (e.g., 0-255 to 0-1 range).
    """
    # Implement normalization here

def augment_data(images):
    """
    Apply augmentations like rotation, flipping, etc.
    """
    # Implement augmentation here
