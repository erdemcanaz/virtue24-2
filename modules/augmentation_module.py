# augmentattion:
# input: an image where the object is placed in known orientation and position
#
# rotate
# flip
# duplicate
# crop
# color filter whole image
# contrast filter whole image
# brightness filter whole image
# new background
#
# output: image with augmented object will be returned

import cv2
import numpy as np
import random
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

def display_image(image: np.ndarray, title: str = ""):
    # Convert the image from BGR (OpenCV default) to RGB for display in matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

class ImageAugmentor:
    def __init__(self, frame: np.ndarray, labels: List[Tuple[int, float, float, float, float]]):
        self.frame = frame
        self.labels = labels

    def rotate_image(self, angle: float) -> np.ndarray:
        (h, w) = self.frame.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(self.frame, M, (w, h))
        return rotated

    def flip_image(self, flip_code: int) -> np.ndarray:
        return cv2.flip(self.frame, flip_code)

    def duplicate_image(self) -> np.ndarray:
        return self.frame.copy()

    def crop_image(self, crop_factor: float) -> np.ndarray:
        h, w = self.frame.shape[:2]
        new_h, new_w = int(h * crop_factor), int(w * crop_factor)
        start_h, start_w = (h - new_h) // 2, (w - new_w) // 2
        return self.frame[start_h:start_h + new_h, start_w:start_w + new_w]

    def apply_color_filter(self) -> np.ndarray:
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.5, 1.5)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def adjust_contrast_brightness(self, contrast: float, brightness: int) -> np.ndarray:
        return cv2.convertScaleAbs(self.frame, alpha=contrast, beta=brightness)

    def replace_background(self) -> np.ndarray:
        # Create a background with the color (122, 207, 246)
        background = np.full(self.frame.shape, (122, 207, 246), dtype=np.uint8)
        
        # Assuming the object is segmented, and black (0, 0, 0) is the background
        mask = (self.frame == [0, 0, 0]).all(axis=2)
        
        # Replace the black background with the specified color background
        new_frame = self.frame.copy()
        new_frame[mask] = background[mask]
        
        return new_frame

    def augment(self, apply_rotate: bool = False, angle: float = 0,
                apply_flip: bool = False, flip_code: int = 1,
                apply_duplicate: bool = False,
                apply_crop: bool = False, crop_factor: float = 0.9,
                apply_color_filter: bool = False,
                apply_contrast_brightness: bool = False, contrast: float = 1.0, brightness: int = 0,
                apply_replace_background: bool = False, background: np.ndarray = None) -> List[Dict[str, Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]]]:

        augmented_images = []

        # Rotate
        if apply_rotate:
            rotated_image = self.rotate_image(angle)
            augmented_images.append({"rotate": (rotated_image, self.labels)})

        # Flip
        if apply_flip:
            flipped_image = self.flip_image(flip_code)
            augmented_images.append({"flip": (flipped_image, self.labels)})

        # Duplicate
        if apply_duplicate:
            duplicated_image = self.duplicate_image()
            augmented_images.append({"duplicate": (duplicated_image, self.labels)})

        # Crop
        if apply_crop:
            cropped_image = self.crop_image(crop_factor)
            augmented_images.append({"crop": (cropped_image, self.labels)})

        # Color Filter
        if apply_color_filter:
            color_filtered_image = self.apply_color_filter()
            augmented_images.append({"color_filter": (color_filtered_image, self.labels)})

        # Contrast and Brightness
        if apply_contrast_brightness:
            contrast_brightness_image = self.adjust_contrast_brightness(contrast, brightness)
            augmented_images.append({"contrast_brightness": (contrast_brightness_image, self.labels)})

        # Replace Background with specified RGB (122, 207, 246)
        if apply_replace_background:
            replaced_bg_image = self.replace_background()
            augmented_images.append({"replace_background": (replaced_bg_image, self.labels)})

        return augmented_images

# Example usage
# Load the PNG image from file
image_path = "borek.png"
frame = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Check if the image was loaded correctly
if frame is None:
    print("Error: Could not load the image.")
else:
    print("Image loaded successfully.")

labels = [[0, 0.5, 0.5, 0.5, 0.5], [1, 0.5, 0.5, 0.5, 0.5]]

# Create an instance of ImageAugmentor
augmentor = ImageAugmentor(frame, labels)

# Perform augmentations
augmented_images = augmentor.augment(
    apply_rotate=True, angle=45,
    apply_flip=True, flip_code=1,
    apply_duplicate=True,
    apply_crop=True, crop_factor=0.8,
    apply_color_filter=True,
    apply_contrast_brightness=True, contrast=1.2, brightness=30,
    apply_replace_background=True
)

# Example of processing the results
for augment in augmented_images:
    for augmentation_type, (image, labels) in augment.items():
        print(f"Augmentation: {augmentation_type}, Image Shape: {image.shape}, Labels: {labels}")
        display_image(image, f"Augmentation: {augmentation_type}")

