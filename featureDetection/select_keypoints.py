import numpy as np


def selectKeypoints(scores, num, r):
    """
    Selects the num best scores as keypoints and performs non-maximum supression of a (2r + 1)*(2r + 1) box around
    the current maximum.
    """
    keypoints = np.zeros((2, num)) # matrix of x,y keypoints
    temp_scores = np.pad(scores, [(r, r), (r, r)], mode='constant', constant_values=0)

    for i in range(num):
        # Argmax returns the maxvalue as in a flatten array, we need unravel to get the original index
        kp = np.unravel_index(np.argmax(temp_scores), temp_scores.shape)
        keypoints[:, i] = np.array(kp)-r
        temp_scores[(kp[0]-r):(kp[0]+r+1), (kp[1]-r):(kp[1]+r+1)] = 0
    
    return keypoints
