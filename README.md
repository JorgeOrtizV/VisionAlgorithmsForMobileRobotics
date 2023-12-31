# VisionAlgorithmsForMobileRobotics
Code exercises for the course Vision Algorithms for Mobile Robotics @ UZH

cameraProjection - In this exercise points are projected in the camera plane given known world coordinates. Additionally, the distortion of the lens is considered and based on the coefficients the image is undisorted using bilinear interpolation. Finally, an augmented reality cube is plotted.

cameraCalibration - Implementation of the DLT algorithm

featureDetection - Implements Harris and Shi-Tomasi detectors from scratch. Implements as well a keypoint identificator, a descriptor and a matcher from scratch, able to successfully follow keypoints through a sequence of images.

SIFT - Implements SIFT detector and descriptor from scratch

stereoRectification - Implements a 3D reconstruction based on sterio rectification of a simplified 2 cameras model.

structureFromMotion - Implements the eight point Algorithm for 3D reconstruction

RANSAC - Implementation of RANSAC outlier removal for keypoints tracking
