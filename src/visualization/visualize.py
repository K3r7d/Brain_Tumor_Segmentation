import matplotlib.pyplot as plt
import os
import json
import numpy as np
def visualize(image, mask):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[1].imshow(mask)
    plt.show()

def visualize_random_images(image_folder, mask_folder):
    image_files = os.listdir(image_folder)
    random_image = np.random.choice(image_files)
    image_path = os.path.join(image_folder, random_image)
    mask_path = os.path.join(mask_folder, random_image.replace('.jpg', '_mask.tif'))
    image = plt.imread(image_path)
    mask = plt.imread(mask_path)
    visualize(image, mask)

if __name__ == '__main__':
    image_folder = 'data/processed/images/'
    mask_folder = 'data/processed/masks/'
    visualize_random_images(image_folder, mask_folder)
    