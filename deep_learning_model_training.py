import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None
tf.logging.set_verbosity(tf.logging.FATAL)

import pandas as pd

###############################################################
# synthesize data
data_size = 1000
feature_dim = 3
balance = 0.25
labels = np.random.choice(2, data_size, p=(1-balance, balance)).astype(bool)
features = np.random.sample((data_size, feature_dim))
class_balance = np.mean(labels)
###############################################################
# Set hyperparameters
training_ratio = 0.8  # Fraction of total data size
batch_size = 20
num_epochs = 100
learning_rate = 1e-3
# balance_correction = 10
balance_correction = 1 / class_balance
hidden_size = 10
data_size, features_dim = features.shape
training_size = batch_size * int(training_ratio * data_size / batch_size)
# assumes data already in randomized order
validation_labels = labels[training_size:]
validation_features = features[training_size:]
training_labels = labels[:training_size]
training_features = features[:training_size]
batches_per_epoch = int(training_size / batch_size)
###############################################################
# Network structure
features_in = tf.placeholder(tf.float64, (None, features_dim))
labels_in = tf.placeholder(tf.float64, (None, 1))
hidden_layer = tf.contrib.layers.fully_connected(features_in, hidden_size)
predicted_labels = tf.contrib.layers.fully_connected(hidden_layer, 1, activation_fn=tf.math.sigmoid)
# Cost function
BCE = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=labels_in,
                                                              logits=predicted_labels,
                                                              pos_weight=balance_correction))
l2RegTerm = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
cost = BCE + l2RegTerm
# Optimizer
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
###############################################################
training_costs = []
validation_costs = []
validation_performances = []
validation_performances_weighted = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # initialize cost
    epoch = 0
    while epoch < num_epochs:
        epoch += 1
        # reshuffle training data each epoch
        batches = np.random.permutation(training_size).reshape(batches_per_epoch, batch_size)
        batch_count = 0
        for batch in batches:
            batch_count += 1
            batch_labels = training_labels[batch]
            batch_features = training_features[batch]
            # run optimizer on current batch
            sess.run(opt, feed_dict={features_in: batch_features, labels_in: batch_labels})
        training_cost = sess.run(cost, feed_dict={features_in: training_features, labels_in: training_labels})
        validation_cost = sess.run(cost, feed_dict={features_in: validation_features, labels_in: validation_labels})
        training_costs.append(training_cost)
        validation_costs.append(validation_cost)
        validation_performances.append()
        validation_performances_weighted.append()

