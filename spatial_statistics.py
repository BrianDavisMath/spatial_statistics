import numpy as np
from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
from scipy.sparse import csr_matrix


def mixed_metric(features_1_, features_2_, cat_mask=np.empty(0)):
    if features_1_.ndim == 1:
        features_1_ = features_1_.reshape(1, -1)
    if features_2_.ndim == 1:
        features_2_ = features_2_.reshape(1, -1)
    categorical_features_1, continuous_features_1 = features_1_[:, cat_mask].astype(bool), features_1_[:, ~cat_mask]
    categorical_features_2, continuous_features_2 = features_2_[:, cat_mask].astype(bool), features_2_[:, ~cat_mask]
    cat_distances = pairwise_distances(categorical_features_1, categorical_features_2, metric="jaccard")
    cont_distances = pairwise_distances(continuous_features_1, continuous_features_2, metric="euclidean")
    hybrid_dist = np.sqrt(np.square(cat_distances) + np.square(cont_distances))
    return hybrid_dist


def get_weights_and_bias(features_, cat_mask_, labels_, is_validation_):
    training_indices = np.where(~is_validation_)[0]
    validation_indices = np.where(is_validation_)[0]
    validation_size = validation_indices.size
    active_indices = np.where(labels_)[0]
    decoy_indices = np.where(~labels_)[0]
    training_active_indices = np.sort(np.array(list(set(training_indices).intersection(set(active_indices)))))
    training_decoy_indices = np.sort(np.array(list(set(training_indices).intersection(set(decoy_indices)))))
    validation_decoy_indices = np.sort(np.array(list(set(validation_indices).intersection(set(decoy_indices)))))
    validation_active_indices = np.sort(np.array(list(set(validation_indices).intersection(set(active_indices)))))
    ###############################################
    # validation actives
    _, active_active_dsts = pairwise_distances_argmin_min(features_[validation_active_indices],
                                                          features_[training_active_indices],
                                                          axis=1,
                                                          metric=mixed_metric,
                                                          metric_kwargs={"cat_mask": cat_mask_}
                                                          )
    _, active_decoy_dsts = pairwise_distances_argmin_min(features_[validation_active_indices],
                                                         features_[training_decoy_indices],
                                                         axis=1,
                                                         metric=mixed_metric,
                                                         metric_kwargs={"cat_mask": cat_mask_}
                                                         )
    active_ratios = active_active_dsts / active_decoy_dsts
    active_spatial_biases = np.sum(active_decoy_dsts - active_active_dsts)
    ###############################################
    # validation decoys
    _, decoy_active_dsts = pairwise_distances_argmin_min(features_[validation_decoy_indices],
                                                         features_[training_active_indices],
                                                         axis=1,
                                                         metric=mixed_metric,
                                                         metric_kwargs={"cat_mask": cat_mask_}
                                                         )
    _, decoy_decoy_dsts = pairwise_distances_argmin_min(features_[validation_decoy_indices],
                                                        features_[training_decoy_indices],
                                                        axis=1,
                                                        metric=mixed_metric,
                                                        metric_kwargs={"cat_mask": cat_mask_}
                                                        )
    decoy_ratios = decoy_decoy_dsts / decoy_active_dsts
    decoy_spatial_biases = np.sum(decoy_active_dsts - decoy_decoy_dsts)
    ###############################################
    uky_score = np.sqrt(active_spatial_biases**2 + decoy_spatial_biases**2)
    ###############################################
    cols = np.hstack([validation_active_indices, validation_decoy_indices])
    data = np.hstack([active_ratios, decoy_ratios])
    data = (1 + np.argsort(data)) / validation_size
    validation_weights = csr_matrix((data, (np.zeros(validation_size).astype(int), np.argsort(cols)))).toarray().flatten()
    return validation_weights, uky_score


"""
# synthetic data
num_data = 25000
cat_length = 10
cont_length = 10
class_balance = 0.25
categorical_features_ = np.random.choice(2, (num_data, cat_length))
continuous_features_ = np.random.sample((num_data, cont_length))
features = np.hstack([categorical_features_, continuous_features_])
cat_mask = np.hstack([np.ones(cat_length), np.zeros(cont_length)]).astype(bool)
labels = np.random.choice(2, num_data, p=(1 - class_balance, class_balance)).astype(bool)
###############################################
# generate a split randomly
training_ratio, validation_ratio = 0.8, 0.2

ratios_sum = training_ratio + validation_ratio
if ratios_sum == 0:
    raise ValueError("Invalid train/validation/test split ratios.")
if ratios_sum != 1:
    training_ratio /= ratios_sum
    validation_ratio /= ratios_sum
    warn(UserWarning("Ratios normalized (don't sum to 1.0)"))
is_validation = np.random.choice(2, num_data, p=(training_ratio, validation_ratio)).astype(bool)
###############################################
# usage
validation_weights, uky_score = get_weights_and_bias(features, cat_mask, labels, is_validation)
"""



