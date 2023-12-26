import numpy as np

def reprojectPoints(P, M_tilde, K):
    # Reproject 3D points given a projection matrix
    #
    # P         [n x 3] coordinates of the 3d points in the world frame
    # M_tilde   [3 x 4] projection matrix
    # K         [3 x 3] camera matrix
    #
    # Returns [n x 2] coordinates of the reprojected 2d points

    proj_mat = K@M_tilde
    lambda_u_v = np.matmul(proj_mat, np.concatenate([P.T, np.ones((1, P.shape[0]))], axis=0)).T
    lamb = lambda_u_v[:,2].reshape(-1,1)
    u_v = lambda_u_v[:,:2]/lamb
    
    return u_v
