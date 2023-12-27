import numpy as np
from scipy import signal


def shi_tomasi(img, patch_size):
    """ Returns the shi-tomasi scores for an image and patch size patch_size
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

    # the eigen values of a matrix M=[a,b;c,d] are lambda1/2 = (Tr(A)/2 +- ((Tr(A)/2)^2-det(A))^.5
    # The smaller one is the one with the negative sign, so we not calculate the eigenvalue obtained through the sum.
    scores = trace/2 - ((trace/2)**2 - determinant)**0.5
    scores[scores < 0] = 0

    scores = np.pad(scores, [(pr+1, pr+1), (pr+1, pr+1)], mode='constant', constant_values=0)

    return scores

