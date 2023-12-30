import cv2
import numpy as np

def computeDifferenceOfGaussians(blurred_images):
    # The number of octaves can be inferred from the length of blurred_images
    DoG = []
    num_octaves = len(blurred_images)
    for oct in range(num_octaves):
        dog_stack = np.zeros(blurred_images[oct].shape - np.array([0,0,1]))
        for i in range(blurred_images[oct].shape[-1]-1):
            dog_stack[:,:, i] = np.abs(blurred_images[oct][:,:,i+1] - blurred_images[oct][:,:,i])
        DoG.append(dog_stack)
    return DoG



