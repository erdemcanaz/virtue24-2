import numpy as np
import pprint
import cv2


class Kernel:
    def __init__(self, kernel_dimensions:tuple):
        self.KERNEL_DIMENSIONS = kernel_dimensions
        self.kernel = None

    def convolve_frame_with_kernel(self, frame:np.ndarray):
        filtered_frame = cv2.filter2D(frame, -1, self.kernel)
        return filtered_frame
    
    def print_kernel(self):
        pprint.pprint(self.kernel)

    def init_random_kernel(self, random_type:str, min_val:float, max_val:float):
        if random_type == 'uniform':
            self.kernel = np.random.uniform(min_val, max_val, self.KERNEL_DIMENSIONS)
        elif random_type == 'normal':
            self.kernel = np.random.normal(min_val, max_val, self.KERNEL_DIMENSIONS)
        else:
            raise ValueError('random_type should be either uniform or normal')
    
    def init_mean_filter(self):
        self.kernel = np.ones(self.KERNEL_DIMENSIONS) / (self.KERNEL_DIMENSIONS[0] * self.KERNEL_DIMENSIONS[1])

    def convert_frame_to_black_white(self, frame:np.ndarray):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  
if __name__ == '__main__':
    kernel = Kernel(kernel_dimensions=(10,10))

    frame = cv2.imread('borek.png')


    black_white_frame = kernel.convert_frame_to_black_white(frame)

    kernel.init_mean_filter()
    kernel.print_kernel()
    filtered_frame = kernel.convolve_frame_with_kernel(black_white_frame)
    cv2.imshow('filtered_frame', filtered_frame)
    cv2.waitKey(0)



  

    



    

