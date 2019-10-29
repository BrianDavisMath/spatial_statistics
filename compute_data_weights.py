import numpy as np
import pandas as pd
import warnings

"""
testing_data_size = 250
testing_neighbors_size = 20
testing_neighbors = np.random.choice(testing_data_size, size=(testing_data_size, testing_neighbors_size))
testing_distances = np.sort(np.random.sample((testing_data_size, testing_neighbors_size)), axis=1)
testing_sims = np.array([testing_neighbors, testing_distances])
testing_labels = np.random.choice(2, testing_data_size).astype(bool)
"""

training_ratio, validation_ratio = 0.8, 0.2


def training_split(sims_, labels_, training_ratio_, validation_ratio_):
    _, data_size, neighbors_size = sims_.shape
    ratios_sum = training_ratio_ + validation_ratio_
    if ratios_sum == 0:
        raise ValueError("Invalid train/validation/test split ratios.")
    if ratios_sum != 1:
        training_ratio_ /= ratios_sum
        validation_ratio_ /= ratios_sum
        warnings.warn(UserWarning("Ratios normalized (don't sum to 1.0)"))
    train_valid_split = np.random.choice(2, data_size, p=(training_ratio_, validation_ratio_)).astype(bool)
    training_indices = np.where(~train_valid_split)[0]
    validation_indices = np.where(train_valid_split)[0]
    active_indices = np.where(labels_)[0]
    decoy_indices = np.where(~labels_)[0]
    weights = []
    for index in validation_indices:
        validation_sim = sims_[:, index, :]
        neighbors = set(validation_sim[0, :].astype(int))
        training_neighbors = set(training_indices).intersection(neighbors)
        # label
        label = labels_[index]
        # closest neighbors
        training_active_closest_neighbors = set(active_indices).intersection(training_neighbors)
        training_decoy_closest_neighbors = set(decoy_indices).intersection(training_neighbors)
        if training_active_closest_neighbors and training_decoy_closest_neighbors:
            training_active_closest_neighbor = min(training_active_closest_neighbors)
            training_decoy_closest_neighbor = min(training_decoy_closest_neighbors)
            training_active_closest_distance = validation_sim[1, list(neighbors).index(training_active_closest_neighbor)]
            training_decoy_closest_distance = validation_sim[1, list(neighbors).index(training_decoy_closest_neighbor)]
            if label:
                weight = training_active_closest_distance / training_decoy_closest_distance
            else:
                weight = training_decoy_closest_distance / training_active_closest_distance
        elif training_decoy_closest_neighbors:
            weight = int(label)
        else:
            weight = int(~label)
        weights.append((index, weight))
    dsts = np.array(weights)[:, 1]
    order = np.argsort(dsts)
    normalizer = len(order)
    cdf = dict([(j, (i + 1) / (normalizer + 1)) for i, j in enumerate(order)])
    cdf = [cdf[i] for i in range(normalizer)]
    return training_indices, validation_indices, cdf


training_indices, validation_indices, weights = training_split(testing_sims, testing_labels, 0.8, 0.2)


"""
need nearest neighbors n large enough so that validation_ratio * min(class_balance, (1-class_balance)) * n is at least 
one, and preferably higher.
With class_balance 0.03 (3% active) and validation_ratio = 0.2, you want number of neighbors in similarity matrix to be
at least 150 or 200.
"""

