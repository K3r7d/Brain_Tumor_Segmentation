# main.py

from src.data.make_dataset import load_data
import matplotlib.pyplot as plt

def main():
    annotation_file = 'data/raw/train/_annotations.coco.json'
    images_dir = 'data/raw/train/'

    # Load dataset and create masks
    dataset = load_data(annotation_file, images_dir)

    # Example: Visualize the first image and its mask
    img, mask = dataset[0]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Image")
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask")
    plt.show()

if __name__ == '__main__':
    main()
