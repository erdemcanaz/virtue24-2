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

def draw_bounding_boxes(image: np.ndarray, labels: List[Tuple[int, float, float, float, float]], color=(0, 255, 0), thickness=2) -> np.ndarray:
    """
    Draws bounding boxes on the image using labels in YOLO format.
    :param image: The image on which to draw the bounding boxes.
    :param labels: List of labels in YOLO format [class, x_center, y_center, width, height].
    :param color: The color of the bounding box.
    :param thickness: Thickness of the bounding box lines.
    :return: The image with drawn bounding boxes.
    """
    h, w = image.shape[:2]
    
    for label in labels:
        class_id, x_center, y_center, bbox_width, bbox_height = label
        
        # Convert normalized coordinates to pixel coordinates
        x_center_pixel = int(x_center * w)
        y_center_pixel = int(y_center * h)
        bbox_width_pixel = int(bbox_width * w)
        bbox_height_pixel = int(bbox_height * h)
        
        # Calculate the top-left and bottom-right corners of the bounding box
        x1 = int(x_center_pixel - bbox_width_pixel / 2)
        y1 = int(y_center_pixel - bbox_height_pixel / 2)
        x2 = int(x_center_pixel + bbox_width_pixel / 2)
        y2 = int(y_center_pixel + bbox_height_pixel / 2)
        
        # Draw the rectangle on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Optionally, draw the class label on the bounding box
        label_text = f"Class {class_id}"
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image

class ImageAugmentor:
    def __init__(self, frame: np.ndarray, labels: List[Tuple[int, float, float, float, float]]):
        self.frame = frame
        self.labels = labels
        self.h, self.w = self.frame.shape[:2]  # Store image height and width

    def rotate_image(self, angle: float) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        # Get rotation matrix
        center = (self.w // 2, self.h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(self.frame, M, (self.w, self.h))
        
        # Update labels: rotate the (x_center, y_center) of each label
        new_labels = []
        for cls, x_center, y_center, bbox_width, bbox_height in self.labels:
            # Convert normalized coordinates to pixel coordinates
            x_center_pixel = x_center * self.w
            y_center_pixel = y_center * self.h

            # Apply the rotation matrix to the center of the bounding box
            new_x_center_pixel = M[0, 0] * x_center_pixel + M[0, 1] * y_center_pixel + M[0, 2]
            new_y_center_pixel = M[1, 0] * x_center_pixel + M[1, 1] * y_center_pixel + M[1, 2]

            # Convert back to normalized coordinates
            new_x_center = new_x_center_pixel / self.w
            new_y_center = new_y_center_pixel / self.h

            # Append the new label with the rotated center, keeping width and height unchanged
            new_labels.append((cls, new_x_center, new_y_center, bbox_width, bbox_height))
        
        return rotated, new_labels

    def flip_image(self, flip_code: int) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        flipped = cv2.flip(self.frame, flip_code)
        
        # Update labels for horizontal flip (flip_code = 1)
        if flip_code == 1:
            new_labels = [
                (cls, 1 - x_center, y_center, width, height)
                for cls, x_center, y_center, width, height in self.labels
            ]
        # Update labels for vertical flip (flip_code = 0)
        elif flip_code == 0:
            new_labels = [
                (cls, x_center, 1 - y_center, width, height)
                for cls, x_center, y_center, width, height in self.labels
            ]
        # Update labels for both axes flip (flip_code = -1)
        else:
            new_labels = [
                (cls, 1 - x_center, 1 - y_center, width, height)
                for cls, x_center, y_center, width, height in self.labels
            ]
        
        return flipped, new_labels

    def duplicate_image(self) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        return self.frame.copy(), self.labels

    def crop_image(self, crop_factor: float) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        new_h, new_w = int(self.h * crop_factor), int(self.w * crop_factor)
        start_h, start_w = (self.h - new_h) // 2, (self.w - new_w) // 2
        cropped = self.frame[start_h:start_h + new_h, start_w:start_w + new_w]
        
        return cropped, self.labels

    def apply_color_filter(self) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.5, 1.5)
        filtered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return filtered, self.labels

    def adjust_contrast_brightness(self, contrast: float, brightness: int) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        adjusted = cv2.convertScaleAbs(self.frame, alpha=contrast, beta=brightness)
        return adjusted, self.labels

    def replace_background(self) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        # Create a background with the color (122, 207, 246)
        background = np.full(self.frame.shape, (122, 207, 246), dtype=np.uint8)
        
        # Assuming the object is segmented, and black (0, 0, 0) is the background
        mask = (self.frame == [0, 0, 0]).all(axis=2)
        
        # Replace the black background with the specified color background
        new_frame = self.frame.copy()
        new_frame[mask] = background[mask]
        
        return new_frame, self.labels

    def augment(self, apply_rotate: bool = False, angle: float = 0,
                apply_flip: bool = False, flip_code: int = 1,
                apply_duplicate: bool = False,
                apply_crop: bool = False, crop_factor: float = 0.9,
                apply_color_filter: bool = False,
                apply_contrast_brightness: bool = False, contrast: float = 1.0, brightness: int = 0,
                apply_replace_background: bool = False) -> List[Dict[str, Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]]]:

        augmented_images = []

        # Rotate
        if apply_rotate:
            rotated_image, rotated_labels = self.rotate_image(angle)
            augmented_images.append({"rotate": (rotated_image, rotated_labels)})

        # Flip
        if apply_flip:
            flipped_image, flipped_labels = self.flip_image(flip_code)
            augmented_images.append({"flip": (flipped_image, flipped_labels)})

        # Duplicate
        if apply_duplicate:
            duplicated_image, duplicated_labels = self.duplicate_image()
            augmented_images.append({"duplicate": (duplicated_image, duplicated_labels)})

        # Crop
        if apply_crop:
            cropped_image, cropped_labels = self.crop_image(crop_factor)
            augmented_images.append({"crop": (cropped_image, cropped_labels)})

        # Color Filter
        if apply_color_filter:
            color_filtered_image, color_filtered_labels = self.apply_color_filter()
            augmented_images.append({"color_filter": (color_filtered_image, color_filtered_labels)})

        # Contrast and Brightness
        if apply_contrast_brightness:
            contrast_brightness_image, contrast_brightness_labels = self.adjust_contrast_brightness(contrast, brightness)
            augmented_images.append({"contrast_brightness": (contrast_brightness_image, contrast_brightness_labels)})

        # Replace Background with specified RGB (122, 207, 246)
        if apply_replace_background:
            replaced_bg_image, replaced_bg_labels = self.replace_background()
            augmented_images.append({"replace_background": (replaced_bg_image, replaced_bg_labels)})

        return augmented_images

# Example usage
image_path = "borek.png"
frame = cv2.imread(image_path, cv2.IMREAD_COLOR)

labels = [[0, 0.5, 0.5, 1, 1]]
frame_with_boxes = draw_bounding_boxes(frame, labels)
display_image(frame_with_boxes, f"Augmentation: None")

# Check if the image was loaded correctly
if frame is None:
    print("Error: Could not load the image.")
else:
    print("Image loaded successfully.")

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
        
        # Draw bounding boxes on the image
        image_with_boxes = draw_bounding_boxes(image, labels)
        
        # Display the augmented image with bounding boxes
        display_image(image_with_boxes, f"Augmentation: {augmentation_type}")



