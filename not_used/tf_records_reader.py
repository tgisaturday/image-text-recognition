# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Train and Eval the MNIST network.
This version is like fully_connected_feed.py but uses data converted
to a TFRecords file containing tf.train.Example protocol buffers.
See:
https://www.tensorflow.org/programmers_guide/reading_data#reading_from_files
for context.
YOU MUST run convert_to_records before running this (but you only need to
run it once).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
"""A generic module to read data."""
import numpy as np
import matplotlib.pyplot as plt
import collections
from tensorflow.python.framework import dtypes
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import random_seed

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import mnist

TRAIN_DIR = 'chars74k_data/'
NUM_EPOCHS = 2
BATCH_SIZE = 100

# Constants used for dealing with the files, matches convert_to_records.

TRAIN_FILE = 'chars74k_fnt_train.tfrecords'
VALIDATION_FILE = 'chars74k_fnt_validation.tfrecords'


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features=tf.parse_single_example(
          serialized_example,
          features={
          'image/encoded':  tf.FixedLenFeature([], tf.string, default_value=''),
          #'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
          'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
          #'image/height': tf.FixedLenFeature([],tf.int64),
          #'image/width': tf.FixedLenFeature([],tf.int64),
      })


  # Convert from a scalar string tensor (whose single string has
  # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
  # [mnist.IMAGE_PIXELS].
  #height = tf.cast(features['image/height'], tf.int32)
  #width = tf.cast(features['image/width'], tf.int32)
  image = tf.decode_raw(features['image/encoded'], tf.uint8)
  #image_shape=tf.stack([height,width,3])
  #image = tf.reshape(image,image_shape)
  #image.set_shape([HEIGHT, WIDTH, 3])

  image.set_shape([mnist.IMAGE_PIXELS])

  # OPTIONAL: Could reshape into a 28x28 image and apply distortions
  # here.  Since we are not applying any distortions in this
  # example, and the next step expects the image to be flattened
  # into a vector, we don't bother.
  #image.reshape([height,width,1])

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['image/class/label'], tf.int32)
  return image, label


def inputs(train_dir, train, batch_size, num_epochs):
  """Reads input data num_epochs times.
  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.
  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  if not num_epochs: num_epochs = None
  filename = os.path.join(train_dir,
                          TRAIN_FILE if train else VALIDATION_FILE)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)
#    images, sparse_labels = tf.train.batch([image, label], batch_size=batch_size,num_threads=2, capacity=1000 + 3 *batch_size,allow_smaller_final_batch=True )
    #change dimension according to number of labels.
    #return images, tf.one_hot(sparse_labels,10)
    return images, sparse_labels


