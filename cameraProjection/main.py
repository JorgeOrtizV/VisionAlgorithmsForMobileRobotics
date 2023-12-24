import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

from pose_vector_to_transformation_matrix import \
    pose_vector_to_transformation_matrix
from project_points import project_points
from undistort_image import undistort_image
from undistort_image_vectorized import undistort_image_vectorized


def main():
    # load camera poses

    # each row i of matrix 'poses' contains the transformations that transforms
    # points expressed in the world frame to
    # points expressed in the camera frame
    poses = np.loadtxt('data/poses.txt', usecols=range(6))

    # define 3D corner positions
    # [Nx3] matrix containing the corners of the checkerboard as 3D points
    # (X,Y,Z), expressed in the world coordinate system
    square_size = 0.04 # in meters
    num_corners_x = 9
    num_corners_y = 6
    num_corners = num_corners_x*num_corners_y
    X,Y = np.meshgrid(np.arange(num_corners_x), np.arange(num_corners_y))
    pW_corners = square_size*np.stack([X, Y], axis=-1).reshape([num_corners, 2])
    Z = np.zeros((num_corners, 1))
    # Corner positions in world reference frame
    pW_corners = np.concatenate((pW_corners, Z), axis=-1)

    # load camera intrinsics
    # Load K
    K = np.loadtxt('data/K.txt', usecols=range(3))
    # Load D
    D = np.loadtxt('data/D.txt', usecols=range(2))

    # load one image with a given index
    img_index = 1
    img = cv2.imread('data/images/img_000{}.jpg'.format(img_index))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #cv2.imshow('test',img)
    #cv2.waitKey(0)

    # project the corners on the image
    # compute the 4x4 homogeneous transformation matrix that maps points
    # from the world to the camera coordinate frame

    # Only index 1, since that is the index of our image 
    T_matrix = pose_vector_to_transformation_matrix(poses[img_index, :])
    # transform 3d points from world to current camera pose
    # We need to add an extra one to the pW points to have homogeneous coordinates (x,y,z,1)
    pW_homogeneous = np.concatenate((pW_corners, np.ones((num_corners,1))), axis=-1)

    # Naive approach - Using foor loop
    # pC_corners = []
    # for point in pW_homogeneous:
    #     pC_corners.append(np.matmul(T_matrix, point.T).T)
    # pC_corners = np.asarray(pC_corners)

    # More efficient approach - vectorization
    pC_corners = np.matmul(T_matrix[None, :, :], pW_homogeneous[:,:,None]).squeeze(-1)
    pC_corners = pC_corners[:, :3]
    
    # Projection
    projected_points = project_points(pC_corners, K, D)
    plt.imshow(img, cmap='gray')
    plt.plot(projected_points[:, 0], projected_points[:, 1], 'r+')
    plt.show()


    # undistort image with bilinear interpolation
    start_t = time.time()
    img_undistorted = undistort_image(img, K, D, bilinear_interpolation=True)
    print('Undistortion with bilinear interpolation completed in {}'.format(
        time.time() - start_t))

    # vectorized undistortion without bilinear interpolation
    start_t = time.time()
    img_undistorted_vectorized = undistort_image_vectorized(img, K, D)
    print('Vectorized undistortion completed in {}'.format(
        time.time() - start_t))
    
    plt.clf()
    plt.close()
    fig, axs = plt.subplots(2)
    axs[0].imshow(img_undistorted, cmap='gray')
    axs[0].set_axis_off()
    axs[0].set_title('With bilinear interpolation')
    axs[1].imshow(img_undistorted_vectorized, cmap='gray')
    axs[1].set_axis_off()
    axs[1].set_title('Without bilinear interpolation')
    plt.show()

    # calculate the cube points to then draw the image
    # Calculate the cube points
    offset_x = 0.04*3
    offset_y = 0.04
    s = 2*0.04
    # World coordinate
    X,Y,Z = np.meshgrid(np.arange(2), np.arange(2), np.arange(-1,1))
    pW_cube = np.stack([offset_x+X.flatten()*s, offset_y+Y.flatten()*s,Z.flatten()*s, np.ones([8])], axis=-1)
    # Camera coordinates
    pC_cube = np.matmul(T_matrix[None, :, :], pW_cube[:,:,None]).squeeze(-1)
    pC_cube = pC_cube[:,:3]
    cube_pts = project_points(pC_cube, K, np.zeros([4,1]))

    plt.clf()
    plt.close()
    plt.imshow(img_undistorted, cmap='gray')

    lw = 3

    # base layer of the cube
    plt.plot(cube_pts[[1, 3, 7, 5, 1], 0],
             cube_pts[[1, 3, 7, 5, 1], 1],
             'r-',
             linewidth=lw)

    # top layer of the cube
    plt.plot(cube_pts[[0, 2, 6, 4, 0], 0],
             cube_pts[[0, 2, 6, 4, 0], 1],
             'r-',
             linewidth=lw)

    # vertical lines
    plt.plot(cube_pts[[0, 1], 0], cube_pts[[0, 1], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[2, 3], 0], cube_pts[[2, 3], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[4, 5], 0], cube_pts[[4, 5], 1], 'r-', linewidth=lw)
    plt.plot(cube_pts[[6, 7], 0], cube_pts[[6, 7], 1], 'r-', linewidth=lw)

    plt.show()


if __name__ == "__main__":
    main()
