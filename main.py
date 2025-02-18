import matplotlib.pyplot as plt
import os
import json
from data.make_dataset import create_mask
import shutil

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE =1e-3

def main(json_file, mask_output_folder, image_output_folder, original_image_dir):
    annotation_file = 'data/raw/train/_annotations.coco.json'
    images_dir = 'data/raw/train/'

    with open(annotation_file, 'r') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']

    if not os.path.exists(mask_output_folder):
        os.makedirs(mask_output_folder)
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)

    for img in images:
        create_mask(img, annotations, mask_output_folder)
        original_image_path = os.path.join(original_image_dir, img['file_name'])
        new_image_path = os.path.join(image_output_folder, os.path.basename(original_image_path))
        shutil.copy2(original_image_path, new_image_path)


if __name__ == '__main__':
    json_file = 'data/raw/train/_annotations.coco.json'
    mask_output_folder = 'data/processed/masks/'
    image_output_folder = 'data/processed/images/'
    original_image_dir = 'data/raw/train/'
    main(json_file, mask_output_folder, image_output_folder, original_image_dir)
