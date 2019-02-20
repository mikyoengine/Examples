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
import math
import os
import time

import engineml.tensorflow as eml
import numpy as np
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
parser.add_argument('--restore-checkpoint-path', type=str, default='', metavar='RESTORE_CHKPT_PATH',
                    help='path to checkpoint to load')

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


def data_batch(df_train, df_test, batch_size, data_dir, num_threads=32):
  """Reads, normalizes, shuffles, and batches data.

  :param df_train: dataframe containing train labels and img paths
  :param df_test: dataframe containing test labels and img paths
  :param data_dir: data directory
  :param batch_size: int, batch size
  :param num_threads: how many threads to run to batch data in parallel
  :return: next element in dataset iterator and the initializer op for train and test sets
  """
  # Convert lists of paths to tensors for tensorflow
  num_prefetch = num_threads * batch_size

  image_list_train = [os.path.join(data_dir, 'train', fn) for fn in df_train['filenames'].values]
  image_list_test = [os.path.join(data_dir, 'test', fn) for fn in df_test['filenames'].values]

  label_list_train = list(df_train['labels'].values)
  label_list_test = list(df_test['labels'].values)

  images_train = tf.convert_to_tensor(image_list_train, dtype=tf.string)
  images_test = tf.convert_to_tensor(image_list_test, dtype=tf.string)

  labels_train = tf.convert_to_tensor(label_list_train, dtype=tf.int16)
  labels_test = tf.convert_to_tensor(label_list_test, dtype=tf.int16)

  # Create dataset out of the 2 file lists:
  data_train = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
  data_test = tf.data.Dataset.from_tensor_slices((images_test, labels_test))

  # Shuffle training data only
  data_train = data_train.shuffle(buffer_size=len(image_list_train))

  # Parse images and label and normalize images
  data_train = data_train.map(
    _parse_data, num_parallel_calls=num_threads).map(
    _normalize_data, num_parallel_calls=num_threads).prefetch(num_prefetch)
  data_test = data_test.map(
    _parse_data, num_parallel_calls=num_threads).map(
    _normalize_data, num_parallel_calls=num_threads).prefetch(num_prefetch)

  # Set batch
  data_train = data_train.batch(batch_size)
  data_test = data_test.batch(batch_size)

  # Create iterator
  iterator = tf.data.Iterator.from_structure(data_train.output_types, data_train.output_shapes)

  # Next element op
  next_element = iterator.get_next()

  # Data set init_op
  train_init_op = iterator.make_initializer(data_train)
  test_init_op = iterator.make_initializer(data_test)

  return next_element, train_init_op, test_init_op


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
      h_pool2_flat, 512, activation_fn=tf.nn.relu),
    keep_prob=0.5,
    is_training=mode)

  # Compute logits (1 per class) and compute loss.
  logits = layers.fully_connected(h_fc1, 10, activation_fn=None)
  loss = tf.losses.softmax_cross_entropy(target, logits)
  acc, acc_op = tf.metrics.accuracy(tf.argmax(target, 1), tf.argmax(logits, 1))

  # Create summaries to monitor loss, accuracy, and example input images
  summaries = [
    tf.summary.scalar('loss', loss),
    tf.summary.scalar('acc', acc),
    tf.summary.image('images', feature, max_outputs=1)
  ]
  return tf.argmax(target, 1), tf.argmax(logits, 1), loss, acc, acc_op, tf.summary.merge(summaries)


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


def train(sess, epoch, batch_size, n_examples, writer, is_train, targets, summaries, loss, acc_op, train_op,
          train_init_op):
  """Train model

  :param sess: tensorflow session
  :param epoch: current epoch
  :param batch_size: batch_size
  :param n_examples: number of training examples
  :param writer: tensorboard event file writer
  :param is_train: tensor containing bool whether in train mode
  :param targets: tensor containing labels
  :param summaries: tensor of events to write to Tensorboard
  :param loss: tensor containing loss
  :param acc_op: operation that updates accuracy metrics
  :param train_op: operation that updates model weights
  :param train_init_op: operation that initializes train data generator
  """
  # Run init op for train data loader
  sess.run(train_init_op, feed_dict={is_train: True})
  samples_seen = epoch * n_examples
  batches_per_epoch = int(math.ceil(n_examples / batch_size))
  for batch_cnt in range(batches_per_epoch):
    # Run a training step synchronously.
    batch_targets, batch_summaries, batch_loss, _, _ = sess.run(
      [targets, summaries, loss, acc_op, train_op], feed_dict={is_train: True}
    )
    samples_seen += len(batch_targets)
    if batch_cnt % 10 == 0:
      print('Train Epoch: {}/{}\tLoss: {:.6f}'.format(epoch + 1, args.epochs, batch_loss))
      writer.add_summary(batch_summaries, samples_seen)


def test(sess, samples_seen, batch_size, n_examples, writer, is_train, targets, preds, loss, test_init_op):
  """Evaluate model on test set

  :param sess: tensorflow session
  :param samples_seen: train samples seen so far
  :param batch_size: batch_size
  :param n_examples: number of test examples
  :param writer: tensorboard event file writer
  :param is_train: tensor containing bool whether in train mode
  :param targets: tensor containing labels
  :param preds: tensor containing predicted labels
  :param loss: tensor containing loss
  :param test_init_op: operation that initializes test data generator
  """
  # Run init op for train data loader
  sess.run(test_init_op, feed_dict={is_train: False})
  batches_per_epoch = int(math.ceil(n_examples / batch_size))
  replica_test_loss = 0.
  replica_test_accuracy = 0.
  for batch_cnt in range(batches_per_epoch):
    # Run a training step synchronously.
    batch_targets, batch_preds, batch_loss = sess.run(
      [targets, preds, loss], feed_dict={is_train: False}
    )
    replica_test_loss += batch_loss * len(batch_targets)
    replica_test_accuracy += sum(batch_targets == batch_preds)

  # Gather total test size, total accuracy, and total loss across all replicas
  total_test_size = eml.sync.replica_sum(np.float32(n_examples))
  total_test_loss = eml.sync.replica_sum(np.float32(replica_test_loss)) / total_test_size
  total_test_accuracy = eml.sync.replica_sum(np.float32(replica_test_accuracy)) / total_test_size
  print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(total_test_loss, 100. * total_test_accuracy))
  writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='val_loss', simple_value=total_test_loss)]),
                     samples_seen)
  writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='val_acc', simple_value=total_test_accuracy)]),
                     samples_seen)


def wait_for_files(num_retries=10, delay_secs=5):
  """Wait for files to be written

  :param num_retries: number of retries before raising error
  :param delay_secs: delay time between retries
  :return: path to checkpoint a, path to checkpoint b
  """
  try:
    a = tf.train.latest_checkpoint('/engine/outputs/0')
    b = tf.train.latest_checkpoint('/engine/outputs/1')
    assert a is not None and b is not None, 'Missing checkpoints for some replicas!'
    assert a.split('/')[-1] == b.split('/')[-1], 'Checkpoints are from different iterations!'
    return a, b
  except AssertionError as e:
    if num_retries > 0:
      time.sleep(delay_secs)
      return wait_for_files(num_retries - 1, delay_secs)
    else:
      raise e


def main(_):
  # Write configuration from arguments to eml-cli
  eml.config.write_from_args(args)

  # Create dataframe with train paths and labels
  # If running integration tests, only use a subset of the data
  if args.run_on_subset:
    df_train = pd.read_csv(os.path.join(args.data_dir, 'train_labels.csv'))[:5000]
    df_test = pd.read_csv(os.path.join(args.data_dir, 'test_labels.csv'))[:500]
  else:
    df_train = pd.read_csv(os.path.join(args.data_dir, 'train_labels.csv'))
    df_test = pd.read_csv(os.path.join(args.data_dir, 'test_labels.csv'))
  # Partition the data across replicas
  df_train = eml.data.distribute(df_train)
  df_test = eml.data.distribute(df_test)
  # Reset indices to start from 0 for sliced data frame
  df_train.reset_index(drop=True, inplace=True)
  df_test.reset_index(drop=True, inplace=True)
  # Create multi-threaded data loader
  (image, label), train_init_op, test_init_op = data_batch(df_train, df_test, args.batch_size, args.data_dir)

  # Build model...
  is_train = tf.placeholder(tf.bool)
  targets, preds, loss, acc, acc_op, summaries = conv_model(image, label, is_train)

  # Scale the learning rate by the number of model replicas
  lr = eml.optimizer.scale_learning_rate(0.001)
  opt = tf.train.RMSPropOptimizer(lr)
  # Wrap your optimizer in the All Reduce Optimizer
  opt = eml.optimizer.distribute(opt)
  train_op = opt.minimize(loss)

  # Enable NCCL communication between GPUs
  config = tf.ConfigProto()
  config = eml.session.distribute_config(config)
  config.gpu_options.allow_growth = True

  # Set the output directories for saving event files and checkpoints
  checkpoint_dir, log_dir = set_checkpoint_dir(args.test_replica_weights)

  saver = tf.train.Saver(max_to_keep=2)
  # Wrap saver in eml wrapper if we aren't running the replica weight test.
  if not args.test_replica_weights:
    saver = eml.saver(saver)

  with tf.Session(config=config) as sess:
    # Set a handler to automatically save a model checkpoint if the job is preempted
    eml.preempted_handler(saver.save, sess, os.path.join(checkpoint_dir, 'preempted'))
    # Initialize variables and syncrhonize all replica weights
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(eml.session.init_op())
    # If there is a predefined checkpoint, check that it exists and load it
    if args.restore_checkpoint_path:
      if os.path.isfile('%s.meta' % args.restore_checkpoint_path):
        print('Loading model from checkpoint {}'.format(args.restore_checkpoint_path))
        saver.restore(sess, args.restore_checkpoint_path)
      else:
        raise IOError('No checkpoint found at %s.meta' % args.restore_checkpoint_path)
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    for e in range(args.epochs):
      eml.annotate(title='Train', comment='Start training', tags=[str(e)])
      train(sess=sess, epoch=e, batch_size=args.batch_size, n_examples=len(df_train), writer=writer, is_train=is_train,
            targets=targets, summaries=summaries, loss=loss, acc_op=acc_op, train_op=train_op,
            train_init_op=train_init_op)
      samples_seen = (e + 1) * len(df_train)
      eml.annotate(title='Validation', comment='Start validation', tags=[str(e)])
      test(sess=sess, samples_seen=samples_seen, batch_size=args.batch_size, n_examples=len(df_test), writer=writer,
           is_train=is_train, targets=targets, preds=preds, loss=loss, test_init_op=test_init_op)
      saver.save(sess, os.path.join(checkpoint_dir, 'checkpoint-%s' % (e + 1)))

  if args.test_replica_weights:
    # Sometimes replica 0 will reach the test_replica_weights phase before the other replicas have finished writing
    # the checkpoint file, causing an assertion error because checkpoints from different steps are loaded.
    # wait_for_files ensures that both checkpoints are ready to be accessed before comparing the weights.
    a, b = wait_for_files(num_retries=10, delay_secs=5)
    assert eml.compare_checkpoints(a, b), 'Weights do not match across replicas!'


if __name__ == '__main__':
  args = parser.parse_args()
  tf.app.run()
