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
import copy
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
    image = copy.deepcopy(image)
    h, w = image.shape[:2]

    print("label: ",labels)
    
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
        self.original_frame = frame
        self.edited_frame = copy.deepcopy(frame)
        self.labels = labels
        self.h, self.w = self.original_frame.shape[:2]  # Store image height and width

    def rotate_image(self, angle: float) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        # Get rotation matrix
        center = (self.w // 2, self.h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(self.edited_frame, M, (self.w, self.h))
        
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
        
        image_with_boxes = draw_bounding_boxes(rotated, new_labels)
        
        return image_with_boxes, new_labels

    def flip_image(self, flip_code: int) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        flipped = cv2.flip(self.edited_frame, flip_code)
        
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
        return self.edited_frame.copy(), self.labels

    def crop_image(self, crop_factor: float) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        new_frame = self.edited_frame.copy()
        new_labels = []

        for label in self.labels:
            class_id, x_center, y_center, bbox_width, bbox_height = label
            
            # Convert normalized coordinates to pixel coordinates
            x_center_pixel = int(x_center * self.w)
            y_center_pixel = int(y_center * self.h)
            bbox_width_pixel = int(bbox_width * self.w)
            bbox_height_pixel = int(bbox_height * self.h)

            # Calculate the original top-left and bottom-right corners of the bounding box
            x1 = int(x_center_pixel - bbox_width_pixel / 2)
            y1 = int(y_center_pixel - bbox_height_pixel / 2)
            x2 = int(x_center_pixel + bbox_width_pixel / 2)
            y2 = int(y_center_pixel + bbox_height_pixel / 2)

            # Crop a portion of the bounding box region based on the crop_factor
            cropped_width_pixel = int(bbox_width_pixel * crop_factor)
            cropped_height_pixel = int(bbox_height_pixel * crop_factor)

            # Calculate the top-left and bottom-right of the cropped region (center remains the same)
            new_x1 = x_center_pixel - cropped_width_pixel // 2
            new_y1 = y_center_pixel - cropped_height_pixel // 2
            new_x2 = x_center_pixel + cropped_width_pixel // 2
            new_y2 = y_center_pixel + cropped_height_pixel // 2

            # Crop the region from the original frame
            cropped_region = self.edited_frame[new_y1:new_y2, new_x1:new_x2]

            # Paste the cropped region back to the original location
            new_frame[y1:y2, x1:x2] = cv2.resize(cropped_region, (bbox_width_pixel, bbox_height_pixel))

            # Update the label with the new dimensions of the cropped region
            new_bbox_width = cropped_width_pixel / self.w
            new_bbox_height = cropped_height_pixel / self.h
            new_labels.append((class_id, x_center, y_center, new_bbox_width, new_bbox_height))

        return new_frame, new_labels

    def apply_color_filter(self) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        hsv = cv2.cvtColor(self.edited_frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.5, 1.5)
        filtered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return filtered, self.labels

    def adjust_contrast_brightness(self, contrast: float, brightness: int) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        adjusted = cv2.convertScaleAbs(self.edited_frame, alpha=contrast, beta=brightness)
        return adjusted, self.labels

    def replace_background(self) -> np.ndarray:
        # Define the replacement color (RGB)
        replacement_color = (122, 207, 246)

        # Create a mask for all pixels that are black (0, 0, 0)
        black_mask = (self.edited_frame == [0, 0, 0]).all(axis=2)

        # Replace black pixels with the replacement color
        self.edited_frame[black_mask] = replacement_color
        
        return self.edited_frame

    def crop_and_paste_region(self) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
        # Initialize a new frame for pasting regions
        filled_frame = np.full_like(self.edited_frame, (246, 207, 122), dtype=np.uint8)
        new_labels_list = []
        
        for label in self.labels:
            class_id, x_center, y_center, bbox_width, bbox_height = label

            # Convert normalized coordinates to pixel coordinates
            x_center_pixel = int(x_center * self.w)
            y_center_pixel = int(y_center * self.h)
            bbox_width_pixel = int(bbox_width * self.w)
            bbox_height_pixel = int(bbox_height * self.h)

            # Calculate top-left and bottom-right corners of the bounding box
            x1 = int(x_center_pixel - bbox_width_pixel / 2)
            y1 = int(y_center_pixel - bbox_height_pixel / 2)
            x2 = int(x_center_pixel + bbox_width_pixel / 2)
            y2 = int(y_center_pixel + bbox_height_pixel / 2)

            # Crop the region from the frame
            cropped_region = self.edited_frame[y1:y2, x1:x2]

            # Randomly generate new_x_center and new_y_center, ensuring the cropped region fits within the frame
            new_x_center = random.uniform(bbox_width / 2, 1 - bbox_width / 2)
            new_y_center = random.uniform(bbox_height / 2, 1 - bbox_height / 2)

            # Convert new normalized coordinates to pixel coordinates
            new_x_center_pixel = int(new_x_center * self.w)
            new_y_center_pixel = int(new_y_center * self.h)

            # Calculate new top-left and bottom-right corners for the pasted region
            new_x1 = int(new_x_center_pixel - bbox_width_pixel / 2)
            new_y1 = int(new_y_center_pixel - bbox_height_pixel / 2)
            new_x2 = int(new_x_center_pixel + bbox_width_pixel / 2)
            new_y2 = int(new_y_center_pixel + bbox_height_pixel / 2)

            # Paste the cropped region at the new location in the filled frame
            filled_frame[new_y1:new_y2, new_x1:new_x2] = cropped_region

            # Create the new label for the pasted region in YOLO format
            new_label = (class_id, new_x_center, new_y_center, bbox_width, bbox_height)
            new_labels_list.append(new_label)

        return filled_frame, new_labels_list
    
    def augment(self, apply_rotate: bool = False, angle: float = 0,
            apply_flip: bool = False, flip_code: int = 1,
            apply_duplicate: bool = False,
            apply_crop: bool = False, crop_factor: float = 0.5,
            apply_color_filter: bool = False,
            apply_contrast_brightness: bool = False, contrast: float = 1.0, brightness: int = 0,
            apply_replace_background: bool = False,
            apply_crop_and_paste: bool = False) -> List[Dict[str, Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]]]:

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
            replaced_bg_image = self.replace_background()
            augmented_images.append({"replace_background": (replaced_bg_image, self.labels)})

        # Crop and Paste Region (New Feature)
        if apply_crop_and_paste:
            cropped_pasted_image, new_labels = self.crop_and_paste_region()
            augmented_images.append({"crop_and_paste": (cropped_pasted_image, new_labels)})

        return augmented_images

# Example usage
# Close any previous plots
plt.close('all')
image_path = "modules/borek_2.png"
frame = cv2.imread(image_path, cv2.IMREAD_COLOR)

borek_labels = [[0, 0.47, 0.48, 0.22, 0.62]]  # Example label for one object
frame_with_boxes = draw_bounding_boxes(frame, borek_labels)
display_image(frame_with_boxes, f"Original Image with Bounding Box")

# Create an instance of ImageAugmentor
augmentor = ImageAugmentor(frame, borek_labels)

# Perform augmentations
augmented_images = augmentor.augment(
    apply_rotate=True, angle=45,
    apply_flip=False, flip_code=1,
    apply_duplicate=False,
    apply_crop=False, crop_factor=0.8,
    apply_color_filter=False,
    apply_contrast_brightness=False, contrast=1.2, brightness=30,
    apply_replace_background=False,
    apply_crop_and_paste=False
)

# Display augmented images with their respective transformations
for augment in augmented_images:
    for augmentation_type, (image, labels) in augment.items():
        print(f"Augmentation: {augmentation_type}, Image Shape: {image.shape}, Labels: {labels}")
        #print(f"Augmentation: {augmentation_type}, Labels: {labels}")
        
        # Draw bounding boxes on the image
        #image_with_boxes = draw_bounding_boxes(image, labels)
        
        # Display the augmented image with bounding boxes
        display_image(image, f"Augmentation: {augmentation_type}")
