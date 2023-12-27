import numpy as np
from scipy import signal


def harris(img, patch_size, kappa):
    """ Returns the harris scores for an image given a patch size and a kappa value
        The returned scores are of the same shape as the input image """

    # Define sobel filters, divide the filters for efficient
    sobel_1 = np.array([-1, 0, 1])
    sobel_2 = np.array([1, 2, 1])
    
    # Calculate intensity in X and Y
    I_x = signal.convolve2d(img, sobel_1[None, :], mode='valid')
    I_x = signal.convolve2d(I_x, sobel_2[:, None], mode="valid").astype(float)
    I_y = signal.convolve2d(img, sobel_1[:, None], mode='valid')
    I_y = signal.convolve2d(I_y, sobel_2[None, :], mode="valid").astype(float)

    # Coefficients of M matrix
    Ixx = I_x**2
    Iyy = I_y**2
    Ixy = I_x * I_y

    # Create the patch for corner evaluation
    patch = np.ones((patch_size, patch_size))
    pr = patch_size // 2

    # Obtain the sum variables. Since we have a patch size, and the value M has sum(Ixx), sum(Iyy), and sum(Ixy)
    # which determine if the pixel being analyzed we compute a matrix of the multiple coefficients using
    # convolution
    sum_Ixx = signal.convolve2d(Ixx, patch, mode="valid")
    sum_Iyy = signal.convolve2d(Iyy, patch, mode="valid")
    sum_Ixy = signal.convolve2d(Ixy, patch, mode="valid")

    # Now for each of these elements we should be able to obtain the eigen values
    trace = sum_Ixx + sum_Iyy
    determinant = sum_Ixx*sum_Iyy - sum_Ixy**2

    # Obtain R
    R = determinant - kappa*trace**2
    R[R<0] = 0
    R = np.pad(R, [(pr+1, pr+1), (pr+1, pr+1)], mode='constant', constant_values=0)

    return R

