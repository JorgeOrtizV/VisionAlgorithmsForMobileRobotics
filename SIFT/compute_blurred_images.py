import cv2
import numpy as np

def computeBlurredImages(image_pyramid, num_scales, sift_sigma):
    # The number of octaves can be inferred from the length of the image pyramid
    octaves = len(image_pyramid)
    blurred_images = []
    imgs_per_oct = num_scales+3
    for oct in range(octaves):
        octave = np.zeros(np.r_[image_pyramid[oct].shape, imgs_per_oct])
        for scale in range(-1, num_scales+2):
            sigma = (2**(scale/num_scales))*sift_sigma
            ksize = int(2*np.ceil(2*sigma)+1)
            octave[:,:, scale+1] = (cv2.GaussianBlur(image_pyramid[oct], (ksize,ksize), sigma))
        blurred_images.append(octave)
    return blurred_images

        
