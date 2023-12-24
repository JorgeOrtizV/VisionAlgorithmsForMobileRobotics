import numpy as np


def distort_points(x: np.ndarray,
                   D: np.ndarray,
                   K: np.ndarray) -> np.ndarray:
    """
    Applies lens distortion to 2D points xon the image plane.

    Args:
        x: 2d points (Nx2)
        D: distortion coefficients (4x1)
        K: camera matrix (3x3)
    """
    k1, k2 = D[0], D[1]
    u0 = K[0,2]
    v0 = K[1,2]

    # Apply formula [ud;vd] = (1+k1*r**2+k2*r**4)*[u-u0;v-v0]+[u0;v0]
    xp = x[:, 0] - u0
    yp = x[:, 1] - v0
    # Obtain r
    r2 = xp**2+yp**2 

    ud = u0+xp*(1+ k1*r2 + k2*r2**2)
    vd = v0+yp*(1+ k1*r2 + k2*r2**2)

    x_d = np.stack([ud, vd], axis=-1)

    return x_d
