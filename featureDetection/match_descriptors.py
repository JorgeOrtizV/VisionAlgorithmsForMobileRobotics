import numpy as np
from scipy.spatial.distance import cdist


def matchDescriptors(query_descriptors, database_descriptors, match_lambda):
    """
    Returns a 1xQ matrix where the i-th coefficient is the index of the database descriptor which matches to the
    i-th query descriptor. The descriptor vectors are MxQ and MxD where M is the descriptor dimension and Q and D the
    amount of query and database descriptors respectively. matches(i) will be -1 if there is no database descriptor
    with an SSD < lambda * min(SSD). No elements of matches will be equal except for the -1 elements.
    """
    # matches = -1*np.ones((1, query_descriptors.shape([1])))
    # for i in range(query_descriptors.shape[1]):
    #     SSD = np.sum((query_descriptors[:,i]-database_descriptors)**2, axis=0)
    #     min_SSD = np.min(SSD)
    #     idx = np.flatnonzero(SSD >= match_lambda*min_SSD)
    #     if len(matches) > 0:
    #         matches[i] = idx[0]
    # return matches

    # Calculate SSD among descriptors
    dists = cdist(query_descriptors.T, database_descriptors.T, 'euclidean')
    # For each calculation obtain the minimum match
    matches = np.argmin(dists, axis=1)
    dists = dists[np.arange(matches.shape[0]), matches]
    # Minimum overall match
    min_non_zero_dist = dists.min()

    # Remove all the features that are not worth tracking
    matches[dists >= match_lambda * min_non_zero_dist] = -1

    # remove double matches
    unique_matches = np.ones_like(matches) * -1
    _, unique_match_idxs = np.unique(matches, return_index=True)
    unique_matches[unique_match_idxs] = matches[unique_match_idxs]

    return unique_matches

