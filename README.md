# spatial_statistics
A repo for computing spatial statistics of training / validation splits
of binary labeled data sets.

## Usage
  ```
  uky_score, validation_weights = get_weights_and_bias(features, is_bool_feature, labels, is_validation, parallel=True)
  ```