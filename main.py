# YoloV8 format: 
#<class> <x_center> <y_center> <width> <height>
#<class>: The class label of the object.
#<x_center>: The normalized x-coordinate of the bounding box center.
#<y_center>: The normalized y-coordinate of the bounding box center.
#<width>: The normalized width of the bounding box.
#<height>: The normalized height of the bounding box.

# functions( frame:np.ndarray = None, labels:List[str]=None) | labels = [[ 0, 0.5, 0.5, 0.5, 0.5], [ 1, 0.5, 0.5, 0.5, 0.5]] | frame = np.ndarray
# output: image with augmented object will be returned

import cv2, copy
from modules import defect_generator_module
from pathlib import Path
import uuid

###############################
# TODO: Tuna will implement augmentation functions
###############################

for i in range(500):
    original_frame = cv2.imread('borek_3.png')
    labels = [ [0, 0.47, 0.48, 0.22, 0.62], [0, 0.86, 0.50, 0.3, 0.35] ] 

    #>> Cutout with default background
    edited_frame, labels = defect_generator_module.cutout_with_default_background(
        original_frame= original_frame,
        probability_per_label=0.15,
        labels= labels,
        max_normalized_bbox_size= [0.20, 0.20],
        default_background_color = (246,207,122),
        show_edited_frame = False
    )

    #>> Line scar    
    edited_frame, labels = defect_generator_module.line_scar(
        original_frame= edited_frame,
        labels= labels,
        probability_per_label= 0.15,
        max_scar_per_label= 5,
        desired_scar_color= None,
        opacity_range= [0.3, 0.95],
        max_scar_nlength= 0.25,
        max_scar_thickness= 2,
        show_edited_frame = False
    )

    #>> Polygon color mask
    edited_frame, labels = defect_generator_module.polygon_color_mask(
        original_frame= edited_frame,
        labels= labels,
        probability_per_label= 0.15,
        desired_color= None,
        num_sides= 7,
        opacity_range= [0.2,0.8],
        scale_factor_range = [0.05,0.5],
        show_edited_frame = False
    )

    #>> save the data
    IMAGES_PATH = Path(__file__).parent / 'dataset' / 'images'
    LABELS_PATH = Path(__file__).parent / 'dataset' / 'labels'

    data_uuid = str(uuid.uuid4())
    image_data_name = f"borek_augmented_defected_{data_uuid}.png"
    label_data_name = f"borek_augmented_defected_{data_uuid}.txt"

    final_image_file_path = str(IMAGES_PATH / image_data_name)
    final_label_file_path = str(LABELS_PATH / label_data_name)
    cv2.imwrite(str(IMAGES_PATH / image_data_name), edited_frame)

    with open(final_label_file_path, 'w') as f:
        for label in labels:
            f.write(" ".join([str(i) for i in label]) + "\n")
        f.close()