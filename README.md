# spatial_statistics
A repo for computing spatial statistics of training / validation splits
of binary-labeled data sets. The inspiration is [this paper](https://arxiv.org/pdf/1706.06619.pdf) which discusses how, for some binary-labeled datasets, random splits lead to overly optimistic performance estimates.
This suggests that, in order to combat overfitting, we may want to compare weighted and unweighted performance metrics, for a weighting which accounts for the spatial distribution of the binary classes between the training and validation sets.

The function ```spatial_statistics.get_weights_and_bias``` allows one to consider distances for features which have a mix of binary features and continuous features. It is recommended that the continuous features be normalized in pre-processing in order to avoid one feature dominating the distance computations.
## Usage
  ```
import numpy as np
from spatial_statistics import get_weights_and_bias

# synthetic data
data_size = 500
continuous_feature_dim = 3
binary_features_dim = 10
balance = 0.25
labels = np.random.choice(2, data_size, p=(1-balance, balance)).astype(bool)
continuous_features = np.random.sample((data_size, continuous_feature_dim))
binary_features = np.random.choice(2, size=(data_size, binary_features_dim)).astype(bool)
features = np.hstack([binary_features, continuous_features])
bool_features = np.hstack([np.ones(binary_features_dim), np.zeros(continuous_feature_dim)]).astype(bool)
is_training = np.random.choice(data_size, size=int(data_size * 0.8), replace=False)
is_validation = np.array([index not in is_training for index in range(data_size)])

validation_weights, uky_score = get_weights_and_bias(features, labels, is_validation, is_bool_feature=bool_features)
```


Keyword arguments for ```spatial_statistics.get_weights_and_bias``` include:
   * ```cat_weight``` for the weighting (between 0 and 1) for the binary features in the distance computation. The default value is the proportion of features that are binary.
   * ```parallel``` for using ```pathos.multiprocessing.ProcessingPool``` to parallelize the distance computations. Default is ```False```.
   * ```is_bool_feature``` for declaring which features are binary. Default is ```None```.