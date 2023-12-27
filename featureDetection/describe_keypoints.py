import numpy as np


def describeKeypoints(img, keypoints, r):
    """
    Returns a (2r+1)^2xN matrix of image patch vectors based on image img and a 2xN matrix containing the keypoint
    coordinates. r is the patch "radius".
    """
    descriptor = np.zeros(((2*r+1)**2, keypoints.shape[1]))
    temp_img = np.pad(img, [(r, r), (r, r)], mode='constant', constant_values=0)
    for i in range(keypoints.shape[1]):
        x, y = keypoints[:,i].astype(np.int) + r
        intensity = temp_img[(x-r):(x+r+1), (y-r):(y+r+1)]
        descriptor[:, i] = intensity.flatten()
    
    return descriptor


