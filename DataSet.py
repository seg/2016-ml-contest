################################################################################
# 
#
#                    Module Name: DataSet
#
#                    Russell A. Kappius
#                    Kappius Consulting LLC 
#                    Supported by Sterling Seismic Services 
#
#            Copyright 2017 All Rights Reserved
#
#	Class to hold feature vectors and labels and aggregate with
#	"next_batch" fuction for tensorflow use.
#
#   $Id: $
################################################################################

"""Functions for loading well data into data structures"""

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

import numpy

################################################################################
def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()-1] = 1
  return labels_one_hot

################################################################################
def one_hot_to_dense(labels_one_hot):
  dense = numpy.where(labels_one_hot[:,:] == 1)
  dense = dense[1][:]
  dense = dense[:]+1
  return dense

################################################################################
class DataSet(object):
  def __init__(self,
               feature_vectors,
               labels):
    self._feature_vectors = feature_vectors
    self._labels = dense_to_one_hot(labels,9)
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._num_examples = feature_vectors.shape[0]

  @property
  def feature_vectors(self):
    return self._feature_vectors

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._feature_vectors = self._feature_vectors[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._feature_vectors[start:end], self._labels[start:end]
################################################################################

################################################################################
def load_dataset(features, labels):
  return DataSet(features, labels)
################################################################################
