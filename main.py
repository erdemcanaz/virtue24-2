# YoloV8 format: 
#<class> <x_center> <y_center> <width> <height>
#<class>: The class label of the object.
#<x_center>: The normalized x-coordinate of the bounding box center.
#<y_center>: The normalized y-coordinate of the bounding box center.
#<width>: The normalized width of the bounding box.
#<height>: The normalized height of the bounding box.

# functions( frame:np.ndarray = None, labels:List[str]=None) | labels = [[ 0, 0.5, 0.5, 0.5, 0.5], [ 1, 0.5, 0.5, 0.5, 0.5]] | frame = np.ndarray
# output: image with augmented object will be returned

