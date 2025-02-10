# src/data/make_dataset.py

import json
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

def load_data(annotation_file, images_dir):
    """
    Loads the dataset and creates masks for each image based on COCO annotations.
    :param annotation_file: Path to the annotation file (COCO format)
    :param images_dir: Path to the directory containing images
    :return: List of images and corresponding masks
    """
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    image_ids = [img['id'] for img in coco_data['images']]
    annotations = coco_data['annotations']

    def create_mask(image_id):
        # Get the image metadata
        image_info = next(img for img in coco_data['images'] if img['id'] == image_id)
        image_filename = image_info['file_name']
        img_path = f'{images_dir}/{image_filename}'

        # Open the image using Pillow
        image = Image.open(img_path)

        # Create an empty mask (background)
        mask = np.zeros((image.height, image.width), dtype=np.uint8)

        # Get the annotations for this image
        img_annotations = [ann for ann in annotations if ann['image_id'] == image_id]

        # Loop through each annotation and update the mask
        for ann in img_annotations:
            if 'segmentation' in ann:
                if type(ann['segmentation']) == list:  # Polygon format
                    for poly in ann['segmentation']:
                        poly = np.array(poly).reshape((int(len(poly) / 2), 2))  # Reshape to (N, 2)
                        poly_mask = coco_mask.frPyObjects(poly, image.height, image.width)

                        # Decode the mask and ensure it's a 2D binary mask (remove extra channels)
                        decoded_mask = coco_mask.decode(poly_mask)
                        if decoded_mask.ndim == 3:  # If the mask has multiple channels, take the first one
                            decoded_mask = decoded_mask[:, :, 0]

                        # Combine the decoded mask with the existing mask
                        mask |= decoded_mask  # Bitwise OR to add the mask to the existing one

                elif type(ann['segmentation']) == dict:  # RLE format
                    rle = ann['segmentation']
                    rle_mask = coco_mask.decode(rle)

                    # If the mask has multiple channels, take the first one
                    if rle_mask.ndim == 3:
                        rle_mask = rle_mask[:, :, 0]

                    mask |= rle_mask  # Combine the decoded mask with the existing mask

        return image, mask

    # Create masks for all images
    dataset = []
    for image_id in image_ids:
        img, mask = create_mask(image_id)
        dataset.append((img, mask))
    
    return dataset
