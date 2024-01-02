import cv2
import numpy as np
import scipy

def extractKeypoints(diff_of_gaussians, contrast_threshold):
    # returns the keypoint locations
    num_octaves = len(diff_of_gaussians)
    keypoint_locations = []
    
    for oct_idx, dog in enumerate(diff_of_gaussians):
        # Obtain the maximum value in a 3x3x3 volume
        dog_max = scipy.ndimage.maximum_filter(dog, [3,3,3])
        # If the maximum value obtained is the same as the current DoG and it is greater than the threshold we keep it.
        is_keypoint = (dog == dog_max) & (dog >= contrast_threshold)
        # Since we compare the center value in a 3x3 volume, the first scale and the last scale will never be keypoints.
        is_keypoint[:, :, 0] = False
        is_keypoint[:, :, -1] = False
        keypoint_locations.append( np.array(is_keypoint.nonzero()).T )

    return keypoint_locations



