import numpy as np
import pprint, random
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

    def init_random_kernel(self, random_type:str, min_val:float, max_val:float, normalize:bool=False):
        if random_type == 'uniform':
            self.kernel = np.random.uniform(min_val, max_val, self.KERNEL_DIMENSIONS)
        elif random_type == 'normal':
            self.kernel = np.random.normal(min_val, max_val, self.KERNEL_DIMENSIONS)
        else:
            raise ValueError('random_type should be either uniform or normal')
        
        if normalize:
            self.kernel = self.kernel / np.sum(self.kernel)
    
    def init_mean_filter(self):
        self.kernel = np.ones(self.KERNEL_DIMENSIONS) / (self.KERNEL_DIMENSIONS[0] * self.KERNEL_DIMENSIONS[1])

    def average_of_grey_pixels(self, frame:np.ndarray):
        number_of_pixels = frame.shape[0] * frame.shape[1]
        return np.sum(frame) / number_of_pixels
    
    def average_of_kernel(self):
        return np.sum(self.kernel) / (self.KERNEL_DIMENSIONS[0] * self.KERNEL_DIMENSIONS[1])
    
    def init_removal_unity_kernel(self, number_of_changes:int, delta_change:float):
        # Initialize the kernel with ones
        self.kernel = np.ones(self.KERNEL_DIMENSIONS)/np.prod(self.KERNEL_DIMENSIONS)
        
        # Get the total number of pixels in the kernel
        total_pixels = np.prod(self.KERNEL_DIMENSIONS)
        
        # Flatten the kernel for easier pixel manipulation
        kernel_flat = self.kernel.flatten()
        
        for _ in range(number_of_changes):
            # Randomly select two different pixel indices
            idx1, idx2 = random.sample(range(total_pixels), 2)
            
            # Add delta_change to one pixel and subtract from the other
            kernel_flat[idx1] += delta_change
            kernel_flat[idx2] -= delta_change
        
        # Reshape the kernel back to its original dimensions
        self.kernel = kernel_flat.reshape(self.KERNEL_DIMENSIONS)


    def convert_frame_to_black_white(self, frame:np.ndarray):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


if __name__ == '__main__':

    frame = cv2.imread('borek.png')
    frame_test = cv2.imread('test_image.png')


    min_sum = float('inf')

    number_of_kernels = 8
    for n in range(2,7):

        kernels = [ Kernel(kernel_dimensions=(n,n)) for _ in range(number_of_kernels)]
        black_white_frame = kernels[0].convert_frame_to_black_white(frame)
        black_white_frame_test = kernels[0].convert_frame_to_black_white(frame_test)

        for i in range(1000):
            for kernel in kernels:
                kernel.init_random_kernel(random_type='uniform', min_val=-1, max_val=1, normalize=True)

            convolved_frames = [kernel.convolve_frame_with_kernel(black_white_frame) for kernel in kernels]
            convolved_frames_test = [kernel.convolve_frame_with_kernel(black_white_frame_test) for kernel in kernels]

            # Stack the result frames along a new axis to perform min pooling
            stacked_frames = np.stack(convolved_frames, axis=0)
            stacked_frames_test = np.stack(convolved_frames_test, axis=0)
            
            # Perform min pooling: take the element-wise minimum across all stacked frames
            min_pooled_frame = np.min(stacked_frames, axis=0)
            min_pooled_frame_test = np.min(stacked_frames_test, axis=0)

            average = kernels[0].average_of_grey_pixels(min_pooled_frame)
            if average < min_sum:
                min_sum = average
                min_kernel = kernel.kernel
                kernel.print_kernel()
                print(kernel.average_of_kernel())
                print(kernel.average_of_grey_pixels(min_pooled_frame))

                cv2.imshow('filtered_frame', min_pooled_frame)
                cv2.imshow('filtered_frame_test', min_pooled_frame_test)
                cv2.waitKey(50)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break



  

    



    

