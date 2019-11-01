import numpy as np
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from scipy.sparse import csr_matrix
from pathos.multiprocessing import ProcessingPool as Pool


def mixed_metric(features_1, features_2, is_bool_feature=None):
    """
    A distance function that uses Jaccard distance for binary features and Euclidean distance for continuous features.
    The mixed metric is the root of sum of squares of the distance on the binary and continuous features.
    :param features_1: 2D numpy array of features
    :param features_2: 2D numpy array of features
    :param is_bool_feature: 1D numpy boolean array indexing which features are bool type (fingerprints)
    :return: 1D numpy array returning distances from rows of features_1 to rows of features_2
    """
    if not is_bool_feature:
        "If no binary features are flagged, assume all features are continuous."
        *_, dim = features_1.shape
        is_bool_feature = np.zeros(dim).astype(bool)
    if features_1.ndim == 1:
        features_1_ = features_1.reshape(1, -1)
    if features_2.ndim == 1:
        features_2_ = features_2.reshape(1, -1)
    categorical_features_1 = features_1[:, is_bool_feature].astype(bool)
    continuous_features_1 = features_1[:, ~is_bool_feature]
    categorical_features_2 = features_2[:, is_bool_feature].astype(bool)
    continuous_features_2 = features_2[:, ~is_bool_feature]
    cat_distances = pairwise_distances(categorical_features_1, categorical_features_2, metric="jaccard")
    cont_distances = pairwise_distances(continuous_features_1, continuous_features_2, metric="euclidean")
    hybrid_dist = np.sqrt(np.square(cat_distances) + np.square(cont_distances))
    return hybrid_dist


def min_distances_parallel(X, Y, bool_mask=None):
    """
    Parallel computation of minimum distances.
    :param X: 2D numpy array of features
    :param Y: 2D numpy array of features
    :param bool_mask: 1D numpy boolean array indexing which features are bool type (fingerprints)
    :return numpy array of minimum distance from each row of X to any row of Y
    """
    def grab_min(x):
        _, dsts = pairwise_distances_argmin_min(np.array([x]),
                                                Y,
                                                axis=1,
                                                metric=mixed_metric,
                                                metric_kwargs={"is_bool_feature": bool_mask}
                                                )
        return dsts
    min_dsts_ = np.hstack(Pool().map(grab_min, X.tolist()))
    return min_dsts_


def min_distances(X, Y, bool_mask=None):
    """
    Serial computation of minimum distances.
    :param X: 2D numpy array of features
    :param Y: 2D numpy array of features
    :param bool_mask: 1D numpy boolean array indexing which features are bool type (fingerprints)
    :return numpy array of minimum distance from each row of X to any row of Y
    """
    _, min_dsts_ = pairwise_distances_argmin_min(X, Y, axis=1, metric=mixed_metric,
                                                 metric_kwargs={"is_bool_feature": bool_mask}
                                                 )
    return min_dsts_


def min_dist_func_getter(parallel_, is_bool_feature_):
    """Get correct distance function according to 'parallel' keyword, and pre-feed it with is_bool_feature_ argument."""
    if parallel_:
        return lambda X, Y: min_distances_parallel(X, Y, is_bool_feature_)
    else:
        return lambda X, Y: min_distances(X, Y, is_bool_feature_)


def get_weights_and_bias(features_, labels_, is_validation_, is_bool_feature_=None, parallel=False):
    """
    Computes validation weights for later use in weighted model performance metrics. Also computes uky_score.
    :param features_: 2D numpy array of features
    :param is_bool_feature_: 1D boolean numpy array indicating which features are binary (fingerprints)
    :param labels_: 1D boolean numpy array of activity labels
    :param is_validation_: 1D boolean numpy array indicating which rows are in the validation set
    :param parallel: Optional keyword argument (Default: False) for parallel computation of distances
    :return: 1D numpy array of weights (length equal to number of validation rows), uky_score (float)
    """
    training_indices = np.where(~is_validation_)[0]
    validation_indices = np.where(is_validation_)[0]
    validation_size = validation_indices.size
    active_indices = np.where(labels_)[0]
    decoy_indices = np.where(~labels_)[0]
    training_active_indices = np.sort(np.array(list(set(training_indices).intersection(set(active_indices)))))
    training_decoy_indices = np.sort(np.array(list(set(training_indices).intersection(set(decoy_indices)))))
    validation_decoy_indices = np.sort(np.array(list(set(validation_indices).intersection(set(decoy_indices)))))
    validation_active_indices = np.sort(np.array(list(set(validation_indices).intersection(set(active_indices)))))
    # compute min distances
    min_dist_func = min_dist_func_getter(parallel, is_bool_feature_)
    active_active_dsts = min_dist_func(features_[validation_active_indices], features_[training_active_indices])
    active_decoy_dsts = min_dist_func(features_[validation_active_indices], features_[training_decoy_indices])
    decoy_active_dsts = min_dist_func(features_[validation_decoy_indices], features_[training_active_indices])
    decoy_decoy_dsts = min_dist_func(features_[validation_decoy_indices], features_[training_decoy_indices])
    # compute differences / ratios
    active_ratios = active_active_dsts / active_decoy_dsts
    active_spatial_biases = np.sum(active_decoy_dsts - active_active_dsts)
    decoy_ratios = decoy_decoy_dsts / decoy_active_dsts
    decoy_spatial_biases = np.sum(decoy_active_dsts - decoy_decoy_dsts)
    # compute spatial statistic uky_score
    uky_score = np.sqrt(active_spatial_biases**2 + decoy_spatial_biases**2)
    # compute validation_weights
    cols = np.hstack([validation_active_indices, validation_decoy_indices])
    data = np.hstack([active_ratios, decoy_ratios])
    data = (1 + np.argsort(data)) / validation_size
    validation_weights = csr_matrix((data, (np.zeros(validation_size).astype(int), np.argsort(cols)))
                                    ).toarray().flatten()
    return validation_weights, uky_score


