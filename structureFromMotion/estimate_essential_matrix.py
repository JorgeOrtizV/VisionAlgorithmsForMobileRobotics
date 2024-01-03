import numpy as np

from fundamental_eight_point_normalized import fundamentalEightPointNormalized

def estimateEssentialMatrix(p1, p2, K1, K2):
    """ estimates the essential matrix given matching point coordinates,
        and the camera calibration K

     Input: point correspondences
      - p1 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 1
      - p2 np.ndarray(3,N): homogeneous coordinates of 2-D points in image 2
      - K1 np.ndarray(3,3): calibration matrix of camera 1
      - K2 np.ndarray(3,3): calibration matrix of camera 2

     Output:
      - E np.ndarray(3,3) : fundamental matrix
    """
    # Not doing it through F:
    """
    Q = []
    for i in range(p1.shape[1]):
      p1_norm = np.linalg.inv(K1) @ p1[:,i]
      p2_norm = np.linalg.inv(K2) @ p2[:,i]
      row = np.array([p1_norm[1]*p1_norm[0], p1_norm[1]*p2_norm[0], p1_norm[1], p2_norm[1]*p1_norm[0], p2_norm[1]*p2_norm[0], p2_norm[1], p1_norm[0], p2_norm[0], 1])
      Q.append(row)
    Q = np.asarray(Q)
    _, _, V = np.linalg.svd(Q, full_matrices=True)
    E = V.T[:, -1]
    return E.reshape((3,3))
    """
    # Using the fundamental matrix
    F = fundamentalEightPointNormalized(p1, p2)
    E = K2.T @ F @ K1

    return E
