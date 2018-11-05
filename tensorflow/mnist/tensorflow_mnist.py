# Copyright 2017 Uber Technologies, Inc. All Rights Reserved.
# Modifications copyright (C) 2018 Engine ML
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
# !/usr/bin/env python
import argparse
import os

import engineml.tensorflow as eml
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import layers

# Training settings
parser = argparse.ArgumentParser(description='TensorFlow MNIST Example')
parser.add_argument('--epochs', type=int, default=2, metavar='N_EPOCHS',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N_SAMPLES',
                    help='input batch size for training (default: 64)')
parser.add_argument('--data-dir', type=str, default='/engine/data', metavar='DATA_DIR',
                    help='path to data directory')
parser.add_argument('--test-replica-weights', action='store_true',
                    help='test that weights are identical across all GPU devices')
parser.add_argument('--run-on-subset', action='store_true',
                    help='run on a subset of the data')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.INFO)


def _normalize_data(image, label):
  """Normalize image within range 0-1."""
  image = tf.cast(image, tf.float32)
  image = image / 255.0
  return image, label


def _parse_data(image_paths, labels):
  """Reads image and one-hot encodes label."""
  image_content = tf.read_file(image_paths)
  images = tf.image.decode_png(image_content, channels=1)
  labels = tf.one_hot(tf.cast(labels, tf.int32), 10, 1, 0)
  return images, labels


def data_batch(df, batch_size, data_dir, sub_dir='train', epochs=None, num_threads=32):
  """Reads, normalizes, shuffles, and batches data.

  :param df: dataframe containing labels and img paths
  :param data_dir: data directory
  :param sub_dir: subdirectory, train or test
  :param batch_size: int, batch size
  :param epochs: how many times to loop over data, if None loops indefinitely
  :param num_threads: how many threads to run to batch data in parallel
  :return: next element in dataset iterator and the initializer op
  """
  # Convert lists of paths to tensors for tensorflow
  num_prefetch = num_threads * batch_size
  image_list = [os.path.join(data_dir, sub_dir, fn) for fn in df['filenames'].values]
  label_list = list(df['labels'].values)
  num_sample = len(image_list)
  images = tf.convert_to_tensor(image_list, dtype=tf.string)
  labels = tf.convert_to_tensor(label_list, dtype=tf.int16)
  # Create dataset out of the 2 file lists:
  data = tf.data.Dataset.from_tensor_slices((images, labels))
  # Shuffle data
  data = data.shuffle(buffer_size=num_sample)
  # Parse images and label
  data = data.map(_parse_data, num_parallel_calls=num_threads).prefetch(num_prefetch)
  # Normalize
  data = data.map(_normalize_data, num_parallel_calls=num_threads).prefetch(num_prefetch)
  # Set batch and epochs
  data = data.batch(batch_size)
  data = data.repeat(epochs)
  # Create iterator
  iterator = data.make_one_shot_iterator()
  # Next element Op
  next_element = iterator.get_next()
  # Data set init. op
  init_op = iterator.make_initializer(data)
  return next_element, init_op


def conv_model(feature, target, mode):
  """2-layer convolution model."""
  # First conv layer will compute 32 features for each 5x5 patch
  with tf.variable_scope('conv_layer1'):
    h_conv1 = layers.conv2d(
      feature, 32, kernel_size=[5, 5], activation_fn=tf.nn.relu)
    h_pool1 = tf.nn.max_pool(
      h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  # Second conv layer will compute 64 features for each 5x5 patch.
  with tf.variable_scope('conv_layer2'):
    h_conv2 = layers.conv2d(
      h_pool1, 64, kernel_size=[5, 5], activation_fn=tf.nn.relu)
    h_pool2 = tf.nn.max_pool(
      h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # reshape tensor into a batch of vectors
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

  # Densely connected layer with 1024 neurons.
  h_fc1 = layers.dropout(
    layers.fully_connected(
      h_pool2_flat, 1024, activation_fn=tf.nn.relu),
    keep_prob=0.5,
    is_training=mode == tf.contrib.learn.ModeKeys.TRAIN)

  # Compute logits (1 per class) and compute loss.
  logits = layers.fully_connected(h_fc1, 10, activation_fn=None)
  loss = tf.losses.softmax_cross_entropy(target, logits)
  acc, acc_op = tf.metrics.accuracy(tf.argmax(target, 1), tf.argmax(logits, 1))

  # Create summaries to monitor loss, accuracy, and example input images
  tf.summary.scalar('loss', loss)
  tf.summary.scalar('accuracy', acc)
  tf.summary.image('images', feature, max_outputs=3)
  return tf.argmax(logits, 1), loss, acc, acc_op, tf.summary.merge_all()


def set_checkpoint_dir(test_replica_weights):
  """Set and create checkpoint directory

  :param test_replica_weights: boolean whether running replica weight test
  :return: checkpoint directory, log directory
  """
  # Set the output directory for saving event files and checkpoints
  # `eml.data.output_dir()` returns `None` when running locally
  checkpoint_dir = eml.data.output_dir() or './checkpoints'
  log_dir = eml.data.output_dir() or './logs'
  # If replica weight test, set manually.
  # THIS IS ONLY FOR TESTING! THERE IS NO REASON TO WRITE MULTIPLE CHECKPOINTS FOR EACH REPLICA.
  # MODEL WEIGHTS ARE UPDATED USING AVG. GRADIENTS ACROSS ALL REPLICAS; THEREFORE EVERY CHECKPOINT WOULD BE IDENTICAL.
  if test_replica_weights:
    checkpoint_dir = os.path.join('/engine/outputs/', str(eml.replica_id()))
  if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
  if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
  return checkpoint_dir, log_dir


def main(_):
  # Create dataframe with train paths and labels
  # If running integration tests, only use a subset of the data
  if args.run_on_subset:
    df = pd.read_csv(os.path.join(args.data_dir, 'train_labels.csv'))[:5000]
  else:
    df = pd.read_csv(os.path.join(args.data_dir, 'train_labels.csv'))
  # Partition the training data across replicas
  df = eml.data.distribute(df)
  # Reset indices to start from 0 for sliced data frame
  df.reset_index(drop=True, inplace=True)
  # Create multi-threaded data loader
  (image, label), init_op = data_batch(df, args.batch_size, args.data_dir, epochs=args.epochs)

  # Build model...
  predict, loss, acc, acc_op, summaries = conv_model(image, label, tf.contrib.learn.ModeKeys.TRAIN)

  # Scale the learning rate by the number of model replicas
  lr = eml.optimizer.scale_learning_rate(0.001)
  opt = tf.train.RMSPropOptimizer(lr)

  # Wrap your optimizer in the All Reduce Optimizer
  opt = eml.optimizer.distribute(opt)

  global_step = tf.contrib.framework.get_or_create_global_step()
  train_op = opt.minimize(loss, global_step=global_step)

  hooks = [
    # Synchronize all replica weights
    eml.session.init_op_hook(),

    tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': loss, 'accuracy': acc}, every_n_iter=50),
  ]

  # Enable NCCL communication between GPUs
  config = tf.ConfigProto()
  config = eml.session.distribute_config(config)
  config.gpu_options.allow_growth = True

  # Set the output directories for saving event files and checkpoints
  checkpoint_dir, log_dir = set_checkpoint_dir(args.test_replica_weights)

  # The MonitoredTrainingSession takes care of session initialization,
  # restoring from a checkpoint, saving to a checkpoint, and closing when done
  # or an error occurs.
  with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir, hooks=hooks, config=config) as mon_sess:
    writer = tf.summary.FileWriter(log_dir, mon_sess.graph)
    # Run init op for data loader
    # mon_sess.run(init_op)
    batch_cnt = 0
    while not mon_sess.should_stop():
      batch_cnt += 1
      # Run a training step synchronously.
      batch_summaries, _, _ = mon_sess.run([summaries, acc_op, train_op])
      if batch_cnt % 10 == 0:
        writer.add_summary(batch_summaries, batch_cnt)

  if args.test_replica_weights:
    a = tf.train.latest_checkpoint('/engine/outputs/0')
    b = tf.train.latest_checkpoint('/engine/outputs/1')
    assert eml.compare_checkpoints(a, b), 'Weights do not match across replicas!'


if __name__ == '__main__':
  args = parser.parse_args()
  tf.app.run()
