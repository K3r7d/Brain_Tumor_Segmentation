# tests/test_data.py

import unittest
from src.data.preprocess import resize_images
from src.data.make_dataset import load_images_from_folder

class TestDataProcessing(unittest.TestCase):

    def test_load_images(self):
        train_folder = "../../data/raw/train"
        test_folder = "../../data/raw/test"
        valid_folder = "../../data/raw/valid"

        train_images, train_filenames = load_images_from_folder(train_folder)
        print(f"Loaded {len(train_images)} training images.")

        test_images, test_filenames = load_images_from_folder(test_folder)
        print(f"Loaded {len(train_images)} training images.")

        valid_images, valid_filenames = load_images_from_folder(valid_folder)
        print(f"Loaded {len(valid_images)} training images.")


    def test_resize_images(self):
        pass

    def test_augment_images(self):
        pass

if __name__ == "__main__":
    unittest.main()