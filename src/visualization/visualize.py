import matplotlib.pyplot as plt
from src.data.preprocess import resize_images, normalize_images
def visualize_image(image):
    """
    Display an image.
    """
    # Implement image visualization here
    pass
def visualize_segmentation(image, prediction):
    """
    Visualize the original image and the predicted tumor regions.
    """
    # Implement segmentation visualization here
    pass
def visualize_images(images, size=(256, 256)):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Display original images
    axes[0].imshow(images[0])
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Resize and display the resized images
    resized_image = resize_images([images[0]], size)[0]
    axes[1].imshow(resized_image)
    axes[1].set_title("Resized Image")
    axes[1].axis('off')

    # Display normalized image
    normalized_image = normalize_images([images[0]])[0]
    axes[2].imshow(normalized_image)
    axes[2].set_title("Normalized Image")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()