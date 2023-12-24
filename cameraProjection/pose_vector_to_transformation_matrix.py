import numpy as np


def pose_vector_to_transformation_matrix(pose_vec: np.ndarray) -> np.ndarray:
    """
    Converts a 6x1 pose vector into a 4x4 transformation matrix.

    Args:
        pose_vec: 6x1 vector representing the pose as [wx, wy, wz, tx, ty, tz]

    Returns:
        T: 4x4 transformation matrix
    """

    # Use Rodrigues' rotation formula: R = I + sin(theta)*K + (1-cos(theta))*(K**2)
    w = pose_vec[0:3]
    t = pose_vec[3:]
    w_norm = np.linalg.norm(w)
    k = w/w_norm
    kx, ky, kz = k
    K = np.array([[0, -kz, ky],
                 [kz, 0, -kx],
                 [-ky, kx, 0]])
    theta = w_norm
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    #R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*K@K
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

