import cv2

def computeImagePyramid(img, num_octaves):
    # Calculates the SIFT pyramid, where each level the image downsamples by a factor of 2**i
    pyramid = []
    pyramid.append(img)
    for i in range(num_octaves-1):
        pyramid.append(cv2.resize(pyramid[i], (0,0), fx=0.5, fy=0.5))
    return pyramid