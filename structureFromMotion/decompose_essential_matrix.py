import numpy as np


def decomposeEssentialMatrix(E):
    """ Given an essential matrix, compute the camera motion, i.e.,  R and T such
     that E ~ T_x R
     
     Input:
       - E(3,3) : Essential matrix

     Output:
       - R(3,3,2) : the two possible rotations
       - u3(3,1)   : a vector with the translation information
    """
    U, S, V = np.linalg.svd(E)
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    rotation1 = U @ W @ V
    rotation2 = U @ W.T @ V

    if np.linalg.det(rotation1) < 0:
        rotation1 *= -1
    if np.linalg.det(rotation2) < 0:
        rotation2 *= -1
    # Possible rotations
    R = np.zeros((3,3,2))
    R[:,:,0] = rotation1
    R[:,:,1] = rotation2
    # Traslation
    u3 = U[:, -1]
    if np.linalg.norm(u3) != 0:
        u3 /= np.linalg.norm(u3)
    
    print(R.shape)

    return R, u3
