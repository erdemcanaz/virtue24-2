# defect generator:
# input: an image where the object is placed in known orientation and position
# 
# cutout:
# cutpaste:
# scar:
# cutpaste scar:
# color filter:
# 
# output: image with defect will be returned

import copy
import numpy as np
import cv2
from typing import List, Dict
import random

def show_labeled_image(original_frame:np.ndarray = None, labels:List[str]=None):
    """show an image with labels"""
    edited_frame = copy.deepcopy(original_frame)
    for label in labels:
        x_center, y_center, width, height = label[1:]
        x_center, y_center, width, height = int(x_center*edited_frame.shape[1]), int(y_center*edited_frame.shape[0]), int(width*edited_frame.shape[1]), int(height*edited_frame.shape[0])
        cv2.rectangle(edited_frame, (x_center-width//2, y_center-height//2), (x_center+width//2, y_center+height//2), (0, 255, 0), 2)

    cv2.imshow("labeled image", edited_frame)
    cv2.waitKey(0)
    return edited_frame, labels

def cutout_with_default_background(original_frame:np.ndarray = None, labels:List[str]=None, probability_per_label:float=None, max_normalized_bbox_size:List[float]= None, default_background_color:List[int]= None):
    """cutout a normalized rectangle from the image and replace it with a default background color
    the cutout rectangle center is placed randomly within the label bounding boxes and the size is randomly chosen within the maximum allowed size
    and the label bounding box size

    INPUT:
    original_frame: np.ndarray             : image to be edited
    labels: List[str]                      : list of labels in the format [class, nx_center, ny_center, nwidth, nheight]         
    probability_per_label: float           : probability of applying the cutout operation on each label  
    max_normalized_bbox_size: List[float]  : maximum size of the cutout rectangle in normalized coordinates [nwidth, nheight]
    default_background_color: List[int]    : default background color to replace the cutout rectangle with

    OUTPUT:
    returns the edited frame and the labels
    output: np.ndarray, List[str]    
    """
    edited_frame = copy.deepcopy(original_frame)
    
    max_normalized_bbox_size = max_normalized_bbox_size or [0.10, 0.05] # [nwidth, nheight]
    default_background_color = default_background_color or (122,207,246)
    probability_per_label = probability_per_label or 0.5

    for label in labels:
        #>>> Apply the cutout operation with a probability, if the probability is not met, skip the label
        if random.random() > probability_per_label:
            continue

        #>>> Determine the cutout rectangle bbox        
        #0:<class> 1:<nx_center> 2:<ny_center> 3:<nwidth> 4:<nheight>
        cut_nwidth  = min(random.uniform(0, max_normalized_bbox_size[0]), label[3])
        cut_nheight  = min(random.uniform(0, max_normalized_bbox_size[1]), label[4])

        topleft_nx = label[1] - label[3]/2
        topleft_ny = label[2] - label[4]/2

        cut_center_nx = random.uniform(topleft_nx, topleft_nx+label[3])
        cut_center_ny = random.uniform(topleft_ny, topleft_ny+label[4])

        cut_nbbox = [cut_center_nx-cut_nwidth/2, cut_center_ny-cut_nheight/2, cut_center_nx+cut_nwidth/2, cut_center_ny+cut_nheight/2] # [nx1, ny1, nx2, ny2]
        cut_bbox = [int(cut_nbbox[0]*edited_frame.shape[1]), int(cut_nbbox[1]*edited_frame.shape[0]), int(cut_nbbox[2]*edited_frame.shape[1]), int(cut_nbbox[3]*edited_frame.shape[0])] # [x1, y1, x2, y2]

        #>>> Cutout the rectangle and replace it with the default background color  
        edited_frame[cut_bbox[1]:cut_bbox[3], cut_bbox[0]:cut_bbox[2]] = default_background_color

    cv2.imshow("cutout_with_default_background", edited_frame)
    cv2.waitKey(0)

    return edited_frame, labels

def line_scar(original_frame: np.ndarray = None, labels: List[str] = None, probability_per_label: float = 0.5, max_scar_per_label: int = 5, scar_color: List[int] = None, opacity_range: List[float] = [0.2, 0.8], max_scar_nlength: float = 0.5):
    """
    Add a line scar to the image.
    The line scar is added to the image with a random orientation and length.
    The line start point is placed randomly within the label bounding boxes and the length is randomly chosen within the maximum allowed size.
    """
    edited_frame = copy.deepcopy(original_frame)    
    scar_color = scar_color or [random.randint(0, 255) for _ in range(3)]

    for label in labels:
        #>>> Probability check
        if random.random() > probability_per_label:
            continue
        
        number_of_scars = random.randint(1, max_scar_per_label)  # Random number of scars to add
        for _ in range(number_of_scars):

            #>>> Extract label bounding box (assuming format: [class, x_center, y_center, width, height])
            x_center, y_center, width, height = label[1], label[2], label[3], label[4]
            
            # Calculate bounding box coordinates
            bbox_x1 = int((x_center - width / 2) * frame.shape[1])
            bbox_y1 = int((y_center - height / 2) * frame.shape[0])
            bbox_x2 = int((x_center + width / 2) * frame.shape[1])
            bbox_y2 = int((y_center + height / 2) * frame.shape[0])
            
            # Random start point inside the bounding box
            start_point = (
                random.randint(bbox_x1, bbox_x2), 
                random.randint(bbox_y1, bbox_y2)
            )
            
            # Random line length and angle
            line_length = random.uniform(0, max_scar_nlength) * frame.shape[1]  # Max width
            angle = random.uniform(0, 360)  # Random angle
            
            # Calculate end point based on the random angle and length
            end_point = (
                int(start_point[0] + line_length * np.cos(np.radians(angle))),
                int(start_point[1] + line_length * np.sin(np.radians(angle)))
            )
            
            # Generate random opacity between 0 and 1
            opacity = random.uniform(opacity_range[0], opacity_range[1])
            
            # Draw the line on a temporary overlay
            overlay = edited_frame.copy()
            cv2.line(overlay, start_point, end_point, scar_color, thickness=2)
            
            print(f"Adding scar from {start_point} to {end_point} with opacity {opacity}")
            # Blend the line with the original image using the random opacity
            cv2.addWeighted(overlay, opacity, edited_frame, 1 - opacity, 0, edited_frame)
    
    cv2.imshow("line_scar", edited_frame)
    cv2.waitKey(0)

    return edited_frame, labels


def color_filter(frame:np.ndarray = None, labels:List[str]=None):
    pass

if __name__ == "__main__":
    frame = cv2.imread("borek_2.png")
    labels = [ [0, 0.47, 0.48, 0.22, 0.62] ]
    
    #show_labeled_image(frame, labels)

    edited_frame, labels = cutout_with_default_background(original_frame= frame, probability_per_label=0.25, labels= labels, max_normalized_bbox_size= [0.10, 0.10], default_background_color = (246,207,122))

    edited_frame, labels = line_scar(original_frame= frame, labels= labels, probability_per_label= 0.99, max_scar_per_label= 5, scar_color= [0, 0, 0], opacity_range= [0.2, 0.8], max_scar_nlength= 0.5)
