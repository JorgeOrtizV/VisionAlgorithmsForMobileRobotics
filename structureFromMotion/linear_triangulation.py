import numpy as np

from utils import cross2Matrix

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def linearTriangulation(p1, p2, M1, M2):
    """ Linear Triangulation
     Input:
      - p1 np.ndarray(3, N): homogeneous coordinates of points in image 1
      - p2 np.ndarray(3, N): homogeneous coordinates of points in image 2
      - M1 np.ndarray(3, 4): projection matrix corresponding to first image
      - M2 np.ndarray(3, 4): projection matrix corresponding to second image

     Output:
      - P np.ndarray(4, N): homogeneous coordinates of 3-D points
    """
    P = np.zeros((4, p1.shape[1]))
    for n in range(p1.shape[1]):
        A = np.r_[skew(p1[:, n]) @ M1,
                  skew(p2[:, n]) @ M2]
        U, S, V_T = np.linalg.svd(A, full_matrices=False)
        P[:, n] = V_T.T[:, -1]
      
    # Dehomogenize P
    P /= P[-1, :]

    return P

        
