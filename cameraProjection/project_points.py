import numpy as np

from distort_points import distort_points


def project_points(points_3d: np.ndarray,
                   K: np.ndarray,
                   D: np.ndarray) -> np.ndarray:
    """
    Projects 3d points to the image plane, given the camera matrix,
    and distortion coefficients.

    Args:
        points_3d: 3d points (3xN)
        K: camera matrix (3x3)
        D: distortion coefficients (2x1)

    Returns:
        projected_points: 2d points (2xN)
    """
    lambda_u_v = np.matmul(K, points_3d[:,:,None]).squeeze(-1)
    lamb = lambda_u_v[:,2].reshape(-1,1)
    u_v = lambda_u_v[:,:2]/lamb
    # Apply distortion
    u_v_distorted = distort_points(u_v, D, K)
    return u_v_distorted
