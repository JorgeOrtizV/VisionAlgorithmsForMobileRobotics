import numpy as np

def parabolaRansac(data, max_noise):
    """
    best_guess_history is 3xnum_iterations with the polynome coefficients  
    from polyfit of the BEST GUESS SO FAR at each iteration columnwise and max_num_inliers_history is
    1xnum_iterations, with the inlier count of the BEST GUESS SO FAR at each iteration.
    """
    iterations = 100
    num_points = 3
    best_guess_history = np.zeros((3, iterations))
    max_num_inliers_history = np.zeros((iterations))
    max_num_inliers = 0
    best_guess = np.zeros((3,1))
    num_inliers = 0
    np.random.seed(42)

    # NOTE: data is shape (2xN), where N is the number of points

    for i in range(iterations):
        # select random points
        sample = np.random.choice(data.shape[1], num_points, replace=False)
        X_sample = data[0, sample]
        Y_sample = data[1, sample]
        # Fit the model
        model = np.polyfit(X_sample, Y_sample, deg=2)

        # Avoiding for loop
        """
        for j in range(data.shape[0]):
            if np.abs(data[j,1]-np.polyval(model, data[j, 0])) <= max_noise:
                # Inlier if |y_i - m(x_i)| <= max_noise
                inliers+=1
        """

        # Errors
        errors = np.abs(data[1,:]-np.polyval(model, data[0, :]))
        inliers = errors <= max_noise + 1e-5
        num_inliers = (inliers).sum()
        if num_inliers > max_num_inliers:
            # Rerun the model with all the inliers
            model_opt = np.polyfit(data[0, inliers], data[1, inliers], deg=2)
            print(model_opt)
            max_num_inliers = num_inliers
            best_guess = model_opt
        # Add best stored result to history
        max_num_inliers_history[i] = max_num_inliers
        best_guess_history[:, i] = best_guess

    return best_guess_history, max_num_inliers_history


