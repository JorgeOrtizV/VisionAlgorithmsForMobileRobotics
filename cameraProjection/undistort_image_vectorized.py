import numpy as np

from distort_points import distort_points


def undistort_image_vectorized(img: np.ndarray,
                               K: np.ndarray,
                               D: np.ndarray) -> np.ndarray:

    """
    Undistorts an image using the camera matrix and distortion coefficients.

    Args:
        img: distorted image (HxW)
        K: camera matrix (3x3)
        D: distortion coefficients (4x1)

    Returns:
        und_img: undistorted image (HxW)
    """
    height, width = img.shape[-2:] # Remove the channel dimension (in case the image is rgb)
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    # Obtain the (x,y) location of all the pixels
    px_locations = np.stack([X, Y], axis=-1).reshape([height*width, 2])
    # Obtain the (x',y') locations (considering distortion)
    dist_px_locs = distort_points(px_locations, D, K)
    # Id(u',v') = Id(round(u'), round(v')) -> Neighbor interpolation
    intensity_vals = img[np.round(dist_px_locs[:,1]).astype(np.int),
                        np.round(dist_px_locs[:,0]).astype(np.int)]
    undist_img = intensity_vals.reshape(img.shape).astype(np.uint8)

    return undist_img
