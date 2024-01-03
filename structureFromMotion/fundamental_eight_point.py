import numpy as np

def fundamentalEightPoint(p1, p2):
    """ The 8-point algorithm for the estimation of the fundamental matrix F

     The eight-point algorithm for the fundamental matrix with a posteriori
     enforcement of the singularity constraint (det(F)=0).
     Does not include data normalization.

     Reference: "Multiple View Geometry" (Hartley & Zisserman 2000), Sect. 10.1 page 262.

     Input: point correspondences
      - p1 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 1
      - p2 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 2

     Output:
      - F np.ndarray(3,3) : fundamental matrix
    """
    # QF = 0
    # Q = (p1_i x p2_i).T
    # (p1_i x p2_i).T * vec(F) = 0

    # Build Q
    Q = []
    for i in range(p1.shape[1]):
      row = np.kron(p1[:, i], p2[:,i]).T
      Q.append(row)
    Q = np.asarray(Q)
    
    U, S, V_T = np.linalg.svd(Q, full_matrices=False)
    # I don't fully understand why the last trasposition
    F = V_T.T[:,-1].reshape(3,3).T
  
    # Ensure F determinant to be 0
    U_f, S_f, V_f = np.linalg.svd(F)
    S_f[2] = 0
    F = U_f @ np.diag(S_f) @ V_f

    return F
