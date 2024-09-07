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

def cutout_with_default_background(original_frame:np.ndarray = None, labels:List[str]=None, max_normalized_bbox_size:List[float]= None, default_background_color:List[int]= None):
    """cutout a normalized rectangle from the image and replace it with a default background color"""
    edited_frame = copy.deepcopy(original_frame)
    
    max_normalized_bbox_size = max_normalized_bbox_size or [0.10, 0.05] # [nwidth, nheight]
    default_background_color = default_background_color or (122,207,246)

    for label in labels:
        print(label)
        print(max_normalized_bbox_size)
        print(default_background_color)
        # Determine the cutout rectangle bbox
        #0:<class> 1:<x_center> 2:<y_center> 3:<width> 4:<height>
        cut_nwidth  = min(random.uniform(0, max_normalized_bbox_size[0]), label[3])
        cut_nheight  = min(random.uniform(0, max_normalized_bbox_size[1]), label[4])
        cut_nx_center = random.uniform(label[1], label[1]+label[3])
        cut_ny_center = random.uniform(label[2], label[2]+label[4])
        print(cut_nwidth, cut_nheight, cut_nx_center, cut_ny_center)
        cut_nbbox = [cut_nx_center-cut_nwidth/2, cut_ny_center-cut_nheight/2, cut_nx_center+cut_nwidth/2, cut_ny_center+cut_nheight/2]
        cut_bbox = [int(cut_nbbox[0]*frame.shape[1]), int(cut_nbbox[1]*frame.shape[0]), int(cut_nbbox[2]*frame.shape[1]), int(cut_nbbox[3]*frame.shape[0])]
        
        # Cutout the rectangle and replace it with the default background color
        edited_frame[cut_bbox[1]:cut_bbox[3], cut_bbox[0]:cut_bbox[2]] = default_background_color

        cv2.imshow("cutout_with_default_background", edited_frame)
        cv2.waitKey(0)

    return edited_frame, labels





def cutpaste(frame:np.ndarray = None, labels:List[str]=None):
    pass

def scar(frame:np.ndarray = None, labels:List[str]=None):
    pass

def cutpaste_scar(frame:np.ndarray = None, labels:List[str]=None):
    pass

def color_filter(frame:np.ndarray = None, labels:List[str]=None):
    pass

if __name__ == "__main__":
    frame = cv2.imread("borek_2.png")
    labels = [ [0, 0.47, 0.48, 0.22, 0.62] ]
    # show_labeled_image(frame, labels)

    cutout_with_default_background(frame, labels)

