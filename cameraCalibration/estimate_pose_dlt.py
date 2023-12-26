import numpy as np

def estimatePoseDLT(p, P, K):
    # Estimates the pose of a camera using a set of 2D-3D correspondences
    # and a given camera matrix.
    # 
    # p  [n x 2] array containing the undistorted coordinates of the 2D points
    # P  [n x 3] array containing the 3D point positions
    # K  [3 x 3] camera matrix
    #
    # Returns a [3 x 4] projection matrix of the form 
    #           M_tilde = [R_tilde | alpha * t] 
    # where R is a rotation matrix. M_tilde encodes the transformation 
    # that maps points from the world frame to the camera frame

    pass

    # Convert 2D to normalized coordinates
    # Since we know K, we multply times K^-1 both sides of the eq. In this step K^-1 * lambda * [u,v,1]T
    p_norm = (np.linalg.inv(K) @ (np.concatenate([p, np.ones((p.shape[0],1))], axis=1).T)).T

    # Build measurement matrix Q
    # Q = [PnT, 0T, -unPnT; 0T, PnT, -vnPnT; ...]
    Q = []
    for idx, point in enumerate(P):
        u = p_norm[idx][0]/p_norm[idx][-1]
        v = p_norm[idx][1]/p_norm[idx][-1]
        Pn1 = np.concatenate([np.concatenate([point, np.ones(1)]),np.zeros((1,4))[0], -1*u*np.concatenate([point, np.ones(1)])])
        Pn2 = np.concatenate([np.zeros((1,4))[0], np.concatenate([point, np.ones(1)]), -1*v*np.concatenate([point, np.ones(1)])])
        Q.append(Pn1)
        Q.append(Pn2)
    Q = np.asanyarray(Q)

    # Solve for Q.M_tilde = 0 subject to the constraint ||M_tilde||=1
    U, S, V = np.linalg.svd(Q, full_matrices=True)
    # The solution is the eigvec of the smallest eigval, which is last column of V if S is sorted.
    M_tilde = np.reshape(V.T[:,-1], (3,4))
    
    # Extract [R | t] with the correct scale
    # Since K was given, then M = [R|T], for this reason QR factorization is not needed.
    # Ensure the determinant is 1
    if np.linalg.det(M_tilde[:,:3]) < 0:
        M_tilde*=-1
    R = M_tilde[:,:3]

    # Find the closest orthogonal matrix to R
    # https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    u_r, s_r, v_r = np.linalg.svd(R)
    R_tilde = u_r @ v_r # Closest orthogonal matrix

    # Normalization scheme using the Frobenius norm:
    # recover the unknown scale using the fact that R_tilde is a true rotation matrix
    alpha = np.linalg.norm(R_tilde, "fro")/np.linalg.norm(R, "fro")

    # Build M_tilde with the corrected rotation and scale
    M_tilde[:,:3] = R_tilde
    M_tilde[:,-1] *= alpha

    return M_tilde
