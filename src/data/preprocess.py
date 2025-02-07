import cv2
import numpy as np
import os
import random


def resize_images(images, target_size=(256, 256)):
    """
    Resize images using OpenCV's resize function with Nearest Neighbor interpolation.
    """
    resized_images = [cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST) for image in images]
    return resized_images


def normalize_images(images):
    """
    normalize images to [0, 1] range.
    """
    return [image / 255.0 for image in images]

def augment_data(images):
    """
    augment images by applying basic methods of data augmentation:
        + Geometric Transformations : Flip, Rotate, Translate
        + Color-Based Augmentations : Brightness, Contrast, Hue
        + Noise Injection : Gaussian Noise
        + Blur and Sharpening : Gaussian Blur, Sharpening
    """

    augmented_images = []

    # Geometric Transformations

    def random_flip(image):
        if random.random() > 0.5:
            image = np.flip(image, axis=1)  # Horizontal flip
        if random.random() > 0.5:
            image = np.flip(image, axis=0)  # Vertical flip
        return image

    def random_rotate(image):
        """
        Using rotation matrix to rotate the image.
            [cos(theta) -sin(theta)]
            [sin(theta)  cos(theta)]
        theta is the angle of rotation.
        """
        center = tuple(np.array(image.shape[1::-1]) / 2)
        angle = random.randint(-30, 30)  # Random angle between -30 to 30 degrees
        rotation_matrix = np.array([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
                                    [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]])

        # Apply the rotation matrix
        rows, cols = image.shape[:2]
        rotated_image = np.zeros_like(image)
        for i in range(rows):
            for j in range(cols):
                x_new, y_new = np.dot(rotation_matrix, np.array([j - center[1], i - center[0]])) + center
                x_new, y_new = int(x_new), int(y_new)

                if 0 <= x_new < cols and 0 <= y_new < rows:
                    rotated_image[i, j] = image[y_new, x_new]
        return rotated_image

    def random_translate(image):
        """
        Translate the image by a random number of pixels in both x and y directions.
        """
        tx = random.randint(-20, 20)
        ty = random.randint(-20, 20)
        translated_image = np.roll(image, (tx, ty), axis=(1, 0))
        return translated_image

    # Color-Based Augmentations

    def random_brightness(image):
        """
        Randomly adjust the brightness of the image 
        by multiplying each pixel value by a random factor.
        """
        factor = random.uniform(0.5, 1.5) 
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
        return image

    def random_contrast(image):
        """
        Randomly adjust the contrast of the image
        by scaling the pixel values around the mean.
        """
        factor = random.uniform(0.5, 1.5)  # Random factor between 0.5 and 1.5
        mean = np.mean(image)
        image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        return image

    def random_hue(image):
        hue_shift = random.randint(-10, 10)
        image = image.astype(np.int32)
        image += hue_shift
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    # Noise Injection

    def add_gaussian_noise(image):
        """
        Add Gaussian noise to the image
        with a mean of 0 and a standard deviation of 25.
        """
        mean = 0
        sigma = 25  
        noise = np.random.normal(mean, sigma, image.shape)
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy_image

    # Blur and Sharpening

    def apply_gaussian_blur(image):
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16  # 3x3 Gaussian kernel
        image_padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
        blurred_image = np.zeros_like(image)

        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                region = image_padded[i - 1:i + 2, j - 1:j + 2, :]
                blurred_image[i, j] = np.sum(region * kernel[..., np.newaxis], axis=(0, 1))

        return blurred_image

    def apply_sharpening(image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # 3x3 Sharpening kernel
        image_padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
        sharpened_image = np.zeros_like(image)

        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                region = image_padded[i - 1:i + 2, j - 1:j + 2, :]
                sharpened_image[i, j] = np.sum(region * kernel[..., np.newaxis], axis=(0, 1))

        return sharpened_image

    # Augmentations
    for image in images:
        augmented_image = image

        #geometric transformations
        if random.random() > 0.5:
            augmented_image = random_flip(augmented_image)
        if random.random() > 0.5:
            augmented_image = random_rotate(augmented_image)
        if random.random() > 0.5:
            augmented_image = random_translate(augmented_image)

        #color-based augmentations
        if random.random() > 0.5:
            augmented_image = random_brightness(augmented_image)
        if random.random() > 0.5:
            augmented_image = random_contrast(augmented_image)
        if random.random() > 0.5:
            augmented_image = random_hue(augmented_image)

        #noise injection
        if random.random() > 0.5:
            augmented_image = add_gaussian_noise(augmented_image)

        #blur or sharpening
        if random.random() > 0.5:
            augmented_image = apply_gaussian_blur(augmented_image)
        if random.random() > 0.5:
            augmented_image = apply_sharpening(augmented_image)

        augmented_images.append(augmented_image)

    return augmented_images
