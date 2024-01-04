import cv2
import numpy as np

from code_previous_exercises.estimate_pose_dlt import estimatePoseDLT
from code_previous_exercises.projectPoints import projectPoints


def ransacLocalization(matched_query_keypoints, corresponding_landmarks, K):
    """
    best_inlier_mask should be 1xnum_matched and contain, only for the matched keypoints,
    False if the match is an outlier, True otherwise.
    """
    
    p3p = True
    tweaked_for_more = True
    adaptive = True

    if not p3p:
        iterations = 2000
        pixel_tolerance = 10
        k = 6
    else:
        iterations = 1000 if tweaked_for_more else 200
        pixel_tolerance = 10
        k = 3
    
    best_inlier_mask = np.zeros(matched_query_keypoints.shape[1])
    # (row, col) to (u,v)
    matched_query_keypoints = np.flip(matched_query_keypoints, axis=0) # equivalent to revers

    max_num_inliers_history = []
    num_iteration_history = []
    max_num_inliers = 0

    # Ransac
    i = 0
    while iterations > i:
        # Obtain k random samples
        sample = np.random.choice(corresponding_landmarks.shape[0], k, replace=False)
        landmark_sample = corresponding_landmarks[sample, :]
        keypoint_sample = matched_query_keypoints[:, sample]

        if p3p:
            success, rotation_vectors, translation_vectors = cv2.solveP3P(landmark_sample, keypoint_sample.T, K,
                                                                        None, flags=cv2.SOLVEPNP_P3P)
            t_C_W_guess = []
            R_C_W_guess = []
            for rotation_vector in rotation_vectors:
                rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
                for translation_vector in translation_vectors:
                    R_C_W_guess.append(rotation_matrix)
                    t_C_W_guess.append(translation_vector)
            
            is_inlier = np.zeros(corresponding_landmarks.shape[0])
            for rot_idx in range(len(R_C_W_guess)):
                C_points = np.matmul(R_C_W_guess[rot_idx], corresponding_landmarks[:,:,None]).squeeze(-1)+t_C_W_guess[rot_idx][None, :].squeeze(-1)
                projected_points = projectPoints(C_points, K)
                difference = matched_query_keypoints - projected_points.T
                errors = (difference ** 2).sum(0)
                alternative_is_inlier = errors < pixel_tolerance ** 2
                if alternative_is_inlier.sum() > is_inlier.sum():
                    is_inlier = alternative_is_inlier
        
        else:
            # Obtain model
            model = estimatePoseDLT(keypoint_sample.T, landmark_sample, K)
            rotation_guess = model[:3, :3]
            translation_guess = model[:3, -1]

            # Obtain predictions
            # pA = R_AB*pB + T_AB
            C_points = (rotation_guess @ corresponding_landmarks[:,:,None]).squeeze(-1) + translation_guess[None, :]
            projected_points = projectPoints(C_points, K)
            difference = matched_query_keypoints - projected_points.T
            errors = (difference**2).sum(0)
            is_inlier = errors < pixel_tolerance**2

        min_inlier_count = 30 if tweaked_for_more else 6

        if is_inlier.sum() > max_num_inliers and is_inlier.sum() >= min_inlier_count:
            max_num_inliers = is_inlier.sum()
            best_inlier_mask = is_inlier

        if adaptive:
            inlier_ratio = max_num_inliers/is_inlier.shape[0]
            outlier_ratio = 1-inlier_ratio
            # formula to compute number of iterations from estimated outlier ratio
            confidence = 0.95
            upper_bound_on_outlier_ratio = 0.90
            outlier_ratio = min(upper_bound_on_outlier_ratio, outlier_ratio)
            iterations = np.log(1-confidence)/np.log(1-(1-outlier_ratio)**k)
            # cap the number of iterations at 15000
            iterations = min(15000, iterations)

        num_iteration_history.append(iterations)
        max_num_inliers_history.append(max_num_inliers)

        i += 1
    
    if max_num_inliers == 0:
        R_C_W = None
        t_C_W = None
    else:
        M_C_W  = estimatePoseDLT(matched_query_keypoints[:, best_inlier_mask].T, corresponding_landmarks[best_inlier_mask, :], K)
        R_C_W = M_C_W[:, :3]
        t_C_W = M_C_W[:, -1]

        if adaptive:
            print("    Adaptive RANSAC: Needed {} iteration to converge.".format(i - 1))
            print("    Adaptive RANSAC: Estimated Ouliers: {} %".format(100 * outlier_ratio))

    return R_C_W, t_C_W, best_inlier_mask, max_num_inliers_history, num_iteration_history


