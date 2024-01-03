import numpy as np

from linear_triangulation import linearTriangulation

def disambiguateRelativePose(Rots,u3,points0_h,points1_h,K1,K2):
    """ DISAMBIGUATERELATIVEPOSE- finds the correct relative camera pose (among
     four possible configurations) by returning the one that yields points
     lying in front of the image plane (with positive depth).

     Arguments:
       Rots -  3x3x2: the two possible rotations returned by decomposeEssentialMatrix
       u3   -  a 3x1 vector with the translation information returned by decomposeEssentialMatrix
       p1   -  3xN homogeneous coordinates of point correspondences in image 1
       p2   -  3xN homogeneous coordinates of point correspondences in image 2
       K1   -  3x3 calibration matrix for camera 1
       K2   -  3x3 calibration matrix for camera 2

     Returns:
       R -  3x3 the correct rotation matrix
       T -  3x1 the correct translation vector

       where [R|t] = T_C2_C1 = T_C2_W is a transformation that maps points
       from the world coordinate system (identical to the coordinate system of camera 1)
       to camera 2.
    """
    # Projection matrix of camera 1
    M1 = K1 @ np.eye(3,4) # K1 @ [I|0]

    selected_rot = None
    selected_tran = None
    max_count = 0

    for rot in range(Rots.shape[-1]):
        test_rot = Rots[:,:,rot]
        for tran in range(2):
            # Apparently the translation can be as given or in opposite direction, that builds up the 4 possibilities
            test_tran = u3 * (-1)**tran
            # Obtain projection for each convination
            M2 = K2 @ np.c_[test_rot, test_tran]
            P_C1 = linearTriangulation(points0_h, points1_h, M1, M2)
            P_C2 = np.c_[test_rot, test_tran] @ P_C1
            count = np.sum(P_C1[2,:] > 0) + np.sum(P_C2[2,:] > 0)
            if count > max_count:
              max_count = count
              selected_rot = test_rot
              selected_tran = test_tran

    return selected_rot, selected_tran
