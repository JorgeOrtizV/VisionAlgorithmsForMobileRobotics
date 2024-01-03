import numpy as np

from fundamental_eight_point import fundamentalEightPoint
from normalise_2D_pts import normalise2DPts

def fundamentalEightPointNormalized(p1, p2):
    """ Normalized Version of the 8 Point algorith
     Input: point correspondences
      - p1 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 1
      - p2 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 2

     Output:
      - F np.ndarray(3,3) : fundamental matrix
    """
    p1_xmean = np.mean(p1[1,:])
    p1_ymean = np.mean(p1[2,:])
    p1_std = np.std(p1)
    s1 = np.sqrt(2)/p1_std
    p2_xmean = np.mean(p2[1,:])
    p2_ymean = np.mean(p2[2,:])
    p2_std = np.std(p2)
    s2 = np.sqrt(2)/p2_std

    T1 = np.array([[s1, 0, -s1*p1_xmean],
                   [0, s1, -s1*p1_ymean],
                   [0, 0, 1]])
    T2 = np.array([[s2, 0, -s2*p2_xmean],
                   [0, s2, -s2*p2_ymean],
                   [0, 0, 1]])
    
    p1_norm = T1@p1
    p2_norm = T2@p2
    
    F_norm = fundamentalEightPoint(p1_norm, p2_norm)
    
    # Unormalize F
    F = T2.T @ F_norm @ T1

    return F
  
