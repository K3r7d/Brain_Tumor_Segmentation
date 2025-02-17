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



    dataset = []
    
    return dataset
