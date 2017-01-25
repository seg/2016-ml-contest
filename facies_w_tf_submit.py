################################################################################
#
#       Facies Classification using ML in (Google) TensorFlow
#
#         Russell A. Kappius
#         Kappius Consulting LLC
#         Supported by Sterling Seismic Services
#
#         January 18, 2017
#
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy  as np

#np.random.seed(0)

# Input from csv files
################################################################################
# 2. Current method: Use 'facies_vectors.csv' and 'validation_data_nofacies.csv'
################################################################################
training_data = pd.read_csv('facies_vectors.csv')
test_data     = pd.read_csv('validation_data_nofacies.csv')

# isolate training vectors & labels, and test vectors & create labels as just 1
all_vectors = training_data.drop(['Facies','Formation', 'Well Name', 'Depth'], axis=1)
all_labels  = training_data['Facies'].values

# Remove NaNs 
nan_idx = np.any(np.isnan(all_vectors), axis=1)
training_vectors = all_vectors[np.logical_not(nan_idx)]
training_labels  = all_labels [np.logical_not(nan_idx)]

test_vectors = test_data.drop(['Formation', 'Well Name', 'Depth'], axis=1)
test_labels  = np.ones(test_vectors.shape[0], dtype=np.int)
################################################################################

################################################################################
# Scale feature vectors
from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(training_vectors)
scaled_training_vectors = scaler.transform(training_vectors)

test_scaler = preprocessing.StandardScaler().fit(test_vectors)
scaled_test_vectors = test_scaler.transform(test_vectors)
################################################################################

################################################################################
# use (my) DataSet class to provide 'next_batch' functionality to TensorFlow
# Also changes labels to 'one-hot' 2D arrays

import DataSet
training_dataset = DataSet.load_dataset(scaled_training_vectors,training_labels)
test_dataset     = DataSet.load_dataset(scaled_test_vectors,test_labels)
################################################################################

################################################################################
# Solve with (Google) TensorFlow
import tensorflow as tf

# Create the model
# 7 elements in each feature vector, 9 possible facies

x = tf.placeholder(tf.float32, [None, 7])
W = tf.Variable(tf.zeros([7, 9]))
b = tf.Variable(tf.zeros([9]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None,9])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# create a session
sess = tf.InteractiveSession()
tf.initialize_all_variables().run()

# Train
for _ in range(1000):
  batch_xs, batch_ys = training_dataset.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# produce unknown labels
run_test = tf.argmax(y,1)
test_labels = \
  sess.run(run_test, \
           feed_dict={x: test_dataset.feature_vectors, y_: test_dataset.labels})

# save predicted labels
test_data['Facies'] = test_labels
test_data.to_csv('PredictedResults.csv')

#print(test_labels)
print('done')
################################################################################
