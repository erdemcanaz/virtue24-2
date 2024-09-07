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

# Close any previous plots
plt.close('all')

def display_image(image: np.ndarray, title: str = ""):
    # Convert the image from BGR (OpenCV default) to RGB for display in matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

def draw_bounding_boxes(image: np.ndarray, label: List[Tuple[int, float, float, float, float]], color=(0, 255, 0), thickness=2):
    copy_image = copy.deepcopy(image)
    h, w = copy_image.shape[:2]    
    
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
    cv2.rectangle(copy_image, (x1, y1), (x2, y2), color, thickness)
    
    # Optionally, draw the class label on the bounding box
    label_text = f"Class {class_id}"
    cv2.putText(copy_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return copy_image

class ImageAugmentor:
    def __init__(self, frame: np.ndarray, labels: List[Tuple[int, float, float, float, float]]):
        self.original_frame = frame
        self.edited_frame = copy.deepcopy(frame)
        self.labels = labels
        self.h, self.w = self.original_frame.shape[:2]  # Store image height and width
    
    def label_to_coordinates(self, label):
        x_center, y_center, bbox_width, bbox_height = label[1:5]
        # Convert normalized center coordinates and dimensions to pixel values
        x_center_pixel = x_center * self.w
        y_center_pixel = y_center * self.h
        bbox_width_pixel = bbox_width * self.w
        bbox_height_pixel = bbox_height * self.h

        # Calculate the original corners of the bounding box (before rotation)
        x1 = x_center_pixel - bbox_width_pixel / 2
        y1 = y_center_pixel - bbox_height_pixel / 2
        x2 = x_center_pixel + bbox_width_pixel / 2
        y2 = y_center_pixel + bbox_height_pixel / 2

        return x1, y1, x2, y2, bbox_width_pixel, bbox_height_pixel

    def center_image_with_yolo_labels(self, image: np.ndarray, target_size: Tuple[int, int]):
        target_height, target_width = target_size
        input_height, input_width = image.shape[:2]

        # Resize the input image if it's larger than the target size
        if input_height > target_height or input_width > target_width:
            image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            input_height, input_width = image.shape[:2]
        
        # Create new blank image with target size and fill it with black
        new_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        new_image[:] = (246, 207, 122)
        
        # Calculate padding to center the input image
        x_offset = (target_width - input_width) // 2
        y_offset = (target_height - input_height) // 2
        
        # Place the original image at the center of the new image
        new_image[y_offset:y_offset + input_height, x_offset:x_offset + input_width] = image
        
        x_center, y_center, width, height = 0.5, 0.5, 1.0, 1.0    
        # Adjust YOLO coordinates (they are normalized between 0 and 1, so we need to adjust them)
        new_x_center = (x_center * input_width + x_offset) / target_width
        new_y_center = (y_center * input_height + y_offset) / target_height
        new_width = width * input_width / target_width
        new_height = height * input_height / target_height
        
        new_yolo_label = [new_x_center, new_y_center, new_width, new_height]
        
        return new_image, new_yolo_label

    def rotate_image(self, angle: float):

        # Get rotation matrix for the entire frame
        center = (self.w // 2, self.h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(self.edited_frame, M, (self.w, self.h))

        # Updated images and labels after rotating the bounding box
        rotated_frames = []
        new_labels = []

        for label in self.labels:
            x1, y1, x2, y2 = self.label_to_coordinates(label)[0:4]

            # Define the four corners of the bounding box as a matrix
            box_corners = np.array([
                [x1, y1],  # top-left
                [x2, y1],  # top-right
                [x2, y2],  # bottom-right
                [x1, y2]   # bottom-left
            ])

            # Apply the rotation matrix to the bounding box corners
            rotated_corners = cv2.transform(np.array([box_corners]), M)[0]

            # Extract the new rotated coordinates by finding the min and max of the transformed corners
            x_min, y_min = np.min(rotated_corners, axis=0)
            x_max, y_max = np.max(rotated_corners, axis=0)

            # Draw the rotated bounding box on the image
            #cv2.rectangle(rotated, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            current_rotated = rotated[int(y_min):int(y_max), int(x_min):int(x_max)]
            rotated_frames.append(current_rotated)
            h, w = current_rotated.shape[0:2]  # Store image height and width

            # Convert the new bounding box back to YOLO format (normalized values)
            new_x_center = ((x_min + x_max) / 2) / w
            new_y_center = ((y_min + y_max) / 2) / h
            new_bbox_width = (x_max - x_min) / w
            new_bbox_height = (y_max - y_min) / h

            # Append the new label with the updated coordinates
            new_labels.append((label[0], new_x_center, new_y_center, new_bbox_width, new_bbox_height))

        return rotated_frames

    def flip_image(self, flip_code: int):
        flipped = cv2.flip(self.edited_frame, flip_code)
        
        # Update labels for horizontal flip (flip_code = 1)
        if flip_code == 1:
            new_labels = [
                (class_id, 1 - x_center, y_center, width, height)
                for class_id, x_center, y_center, width, height in self.labels
            ]
        # Update labels for vertical flip (flip_code = 0)
        elif flip_code == 0:
            new_labels = [
                (class_id, x_center, 1 - y_center, width, height)
                for class_id, x_center, y_center, width, height in self.labels
            ]
        # Update labels for both axes flip (flip_code = -1)
        else:
            new_labels = [
                (class_id, 1 - x_center, 1 - y_center, width, height)
                for class_id, x_center, y_center, width, height in self.labels
            ]
        
        flipped_frames = []
        for label in new_labels:
            x1, y1, x2, y2 = self.label_to_coordinates(label)[0:4]

            current_flipped = flipped[int(y1):int(y2), int(x1):int(x2)]
            flipped_frames.append(current_flipped)
        
        return flipped_frames

    def cut_image(self, cut_factor: float):
        
        cropped_frames = []
        for label in self.labels:
            x_center, y_center, bbox_width, bbox_height = label[1:5]

            # Convert normalized values to pixel values
            x_center_pixel = int(x_center * self.w)
            y_center_pixel = int(y_center * self.h)
            bbox_width_pixel = int(bbox_width * self.w)
            bbox_height_pixel = int(bbox_height * self.h)

            # Calculate the coordinates of the bounding box
            x1 = max(0, x_center_pixel - bbox_width_pixel // 2)
            y1 = max(0, y_center_pixel - bbox_height_pixel // 2)
            x2 = min(self.w, x_center_pixel + bbox_width_pixel // 2)
            y2 = min(self.h, y_center_pixel + bbox_height_pixel // 2)

            # Randomly select a portion of the labelled area
            crop_width = random.randint(int(bbox_width_pixel * cut_factor), bbox_width_pixel)
            crop_height = random.randint(int(bbox_height_pixel * cut_factor), bbox_height_pixel)

            # Ensure the cropped portion stays within the bounding box
            crop_x1 = random.randint(x1, x2 - crop_width)
            crop_y1 = random.randint(y1, y2 - crop_height)
            crop_x2 = crop_x1 + crop_width
            crop_y2 = crop_y1 + crop_height

            # Crop the randomly selected portion from the image
            cropped_area = self.edited_frame[crop_y1:crop_y2, crop_x1:crop_x2]

            # Append the cropped area and corresponding label to the list
            cropped_frames.append(cropped_area)

        return cropped_frames

    def duplicate_image(self):
        duplicated_frames = []
        for label in self.labels:
            x1, y1, x2, y2 = self.label_to_coordinates(label)[0:4]
            current_duplicated = self.edited_frame[int(y1):int(y2), int(x1):int(x2)]
            duplicated_frames.append(current_duplicated)
        return duplicated_frames
    
    def zoom_image(self, zoom_factor: float):
        duplicated_frames = self.duplicate_image()   
        zoomed_frames = []  
        for frame in duplicated_frames:
            h, w = frame.shape[0:2]
            # Compute the new dimensions
            new_width = int(w * zoom_factor)
            new_height = int(h * zoom_factor)
            # Resize the image using INTER_LINEAR interpolation for enlarging
            resized_image = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            zoomed_frames.append(resized_image)            

        return zoomed_frames

    def apply_color_filter(self, color_mode: int, custom: Tuple[int, int, int]):
        h, w, c = self.edited_frame.shape
        red_img  = np.full((h,w,c), (0,0,255), np.uint8)
        blue_img  = np.full((h,w,c), (0,255,0), np.uint8)
        green_img  = np.full((h,w,c), (255,0,0), np.uint8)
        custom_img = np.full((h,w,c), custom, np.uint8)
        # add the filter  with a weight factor of 20% to the target image
        if color_mode == 0: #red
            fused_img  = cv2.addWeighted(self.edited_frame, 0.8, red_img, 0.2, 0)    
        elif color_mode == 1: #green
            fused_img  = cv2.addWeighted(self.edited_frame, 0.8, blue_img, 0.2, 0)    
        elif color_mode == 2: #blue
            fused_img  = cv2.addWeighted(self.edited_frame, 0.8, green_img, 0.2, 0)   
        elif color_mode == 3: # custom 
            fused_img  = cv2.addWeighted(self.edited_frame, 0.8, custom_img, 0.2, 0)   
        filtered_frames = []
        for label in self.labels:
            x1, y1, x2, y2 = self.label_to_coordinates(label)[0:4]
            current_filtered = fused_img[int(y1):int(y2), int(x1):int(x2)]
            filtered_frames.append(current_filtered)
        return filtered_frames

    def adjust_contrast_brightness(self, contrast: float, brightness: int):
        adjusted = cv2.convertScaleAbs(self.edited_frame, alpha=contrast, beta=brightness)
        adjusted_frames = []
        for label in self.labels:
            x1, y1, x2, y2 = self.label_to_coordinates(label)[0:4]
            current_adjusted = adjusted[int(y1):int(y2), int(x1):int(x2)]
            adjusted_frames.append(current_adjusted)
        return adjusted_frames
    
    def augment(self, apply_rotate: bool = False, angle: float = 0,
            apply_flip: bool = False, flip_code: int = 1,
            apply_duplicate: bool = False,
            apply_cut: bool = False, cut_factor: float = 0.5,
            apply_zoom: bool = False, zoom_factor: float = 1.5,
            apply_color_filter: bool = False, color_mode: int = 1, custom = (0, 0, 0),
            apply_contrast_brightness: bool = False, contrast: float = 1.0, brightness: int = 0
            ):

        augmented_images = []
        target_size = (self.h, self.w)

        # Rotate
        if apply_rotate:
            rotated_image = self.rotate_image(angle)
            for image in rotated_image:
                rotated_images, labels = self.center_image_with_yolo_labels(image, target_size)
                augmented_images.append({"rotate": (rotated_images, labels)})

        # Flip
        if apply_flip:
            flipped_image = self.flip_image(flip_code)
            for image in flipped_image:
                flipped_images, labels = self.center_image_with_yolo_labels(image, target_size)
                augmented_images.append({"flip": (flipped_images, labels)})

        # Duplicate
        if apply_duplicate:
            duplicated_image = self.duplicate_image()
            for image in duplicated_image:
                duplicated_images, labels = self.center_image_with_yolo_labels(image, target_size)
                augmented_images.append({"duplicate": (duplicated_images, labels)})

        # Cut
        if apply_cut:
            cut_image = self.cut_image(cut_factor)
            for image in cut_image:
                cut_images, labels = self.center_image_with_yolo_labels(image, target_size)
                augmented_images.append({"cut": (cut_images, labels)})
        
        # Zoom
        if apply_zoom:
            zoom_image = self.zoom_image(zoom_factor)
            for image in zoom_image:
                zoom_images, labels = self.center_image_with_yolo_labels(image, target_size)
                augmented_images.append({"zoom": (zoom_images, labels)})

        # Color Filter
        if apply_color_filter:
            color_filtered_image = self.apply_color_filter(color_mode, custom)
            for image in color_filtered_image:
                color_filtered_images, labels = self.center_image_with_yolo_labels(image, target_size)
                augmented_images.append({"color_filter": (color_filtered_images, labels)})

        # Contrast and Brightness
        if apply_contrast_brightness:
            contrast_brightness_image = self.adjust_contrast_brightness(contrast, brightness)
            for image in contrast_brightness_image:
                contrast_brightness_images, labels = self.center_image_with_yolo_labels(image, target_size)
                augmented_images.append({"contrast_brightness": (contrast_brightness_images, labels)})

        return augmented_images

# Example usage
image_path = "modules/borek_2.png"
frame = cv2.imread(image_path, cv2.IMREAD_COLOR)

borek_labels = [[0, 0.47, 0.48, 0.22, 0.62]]  # Example label for one object
""" frame_with_boxes = draw_bounding_boxes(frame, borek_labels)
display_image(frame_with_boxes, f"Original Image with Bounding Box") """

# Create an instance of ImageAugmentor
augmentor = ImageAugmentor(frame, borek_labels)

# Perform augmentations
augmented_images = augmentor.augment(
    apply_rotate=True, angle=45,
    apply_flip=True, flip_code=1,
    apply_duplicate=True,
    apply_cut=True, cut_factor=0.8,
    apply_zoom=True, zoom_factor=1.5,
    apply_color_filter=True, color_mode=2, custom = (0, 0, 0), # color_mode=3 to use custom rgb
    apply_contrast_brightness=True, contrast=1.2, brightness=30
)

# Display augmented images with their respective transformations
for augment in augmented_images:
    for augmentation_type, (image, label) in augment.items():
        print(f"Augmentation: {augmentation_type}, Image: {image.shape}, Label: {label}")
        
        #image_with_boxes = image
        """ if augmentation_type!= 'rotate':
            # Draw bounding boxes on the image
            image_with_boxes = draw_bounding_boxes(image[i], labels[i]) """
        
        # Display the augmented image with bounding boxes
        display_image(image, f"Augmentation: {augmentation_type}")
