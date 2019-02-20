from __future__ import print_function

import argparse
import os
import time

import engineml.keras as eml
import h5py
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from keras import backend as K
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential

# Training settings
parser = argparse.ArgumentParser(description='TensorFlow MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N_SAMPLES',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=2, metavar='N_EPOCHS',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--data-dir', type=str, default='/engine/data', metavar='DATA_DIR',
                    help='path to data directory')
parser.add_argument('--test-replica-weights', action='store_true',
                    help='test that weights are identical across all GPU devices')
parser.add_argument('--run-on-subset', action='store_true',
                    help='run on a subset of the data')
parser.add_argument('--restore-checkpoint-path', type=str, default='', metavar='RESTORE_CHKPT_PATH',
                    help='path to checkpoint to load')


class DataGenerator(keras.utils.Sequence):
  """Generates data for Keras"""
  def __init__(self, df, data_dir, batch_size, target_size=(28, 28), num_classes=10, is_train=True):
    """Initialization"""
    self.filenames = df['filenames'].values
    self.labels = df['labels'].values
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.target_size = target_size
    self.n_classes = num_classes
    self.indices = np.arange(len(self.filenames))
    if is_train:
      self.sub_dir = 'train'
      self.shuffle = True
    else:
      self.sub_dir = 'test'
      self.shuffle = False
    self.on_epoch_end()

  def __len__(self):
    """Denotes the number of batches per epoch"""
    return int(np.floor(len(self.filenames) / self.batch_size))

  def __getitem__(self, index):
    """Generate one batch of data"""
    # Generate shuffled indices of the batch
    indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
    # Pull list of randomly shuffled images
    batch_x = [self.load_mnist_img(os.path.join(self.data_dir, self.sub_dir, self.filenames[k])) for k in indices]
    batch_y = [self.labels[k] for k in indices]
    return np.array(batch_x), keras.utils.to_categorical(np.array(batch_y), num_classes=self.n_classes)

  def load_mnist_img(self, fn):
    """ Load an MNIST image

    :param fn: path to img
    :return: grayscale img array
    """
    return np.asarray(Image.open(fn), dtype=np.float32).reshape(self.target_size[0], self.target_size[1], 1) / 255.

  def on_epoch_end(self):
    """Updates indices after each epoch"""
    if self.shuffle:
      np.random.shuffle(self.indices)


def build_conv_model(input_shape=(28, 28, 1), num_classes=10):
  """2-layer convolution model

  :param input_shape: input data shape
  :param num_classes: number of classes to train on
  :return: CNN
  """
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(5, 5),
                   activation='relu',
                   input_shape=input_shape))
  model.add(Conv2D(64, (5, 5), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax'))
  return model


def get_data_generators(data_dir, batch_size, run_on_subset):
  """Convert class vectors to binary class matrices

  :param data_dir: path to data directory
  :param batch_size: int, batch_size
  :param run_on_subset: boolean whether running replica weight test
  :return: image train_generator, image test_generator
  """
  # If running integration tests, only use a subset of the data
  if run_on_subset:
    df_train = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))[:5000]
    df_test = pd.read_csv(os.path.join(data_dir, 'test_labels.csv'))[:500]
  else:
    df_train = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))
    df_test = pd.read_csv(os.path.join(data_dir, 'test_labels.csv'))

  # Partition your training data across replicas
  df_train = eml.data.distribute(df_train)

  # Reset indices to start from 0 for sliced data frame
  df_train.reset_index(drop=True, inplace=True)

  train_generator = DataGenerator(df_train, data_dir, batch_size)
  test_generator = DataGenerator(df_test, data_dir, batch_size, is_train=False)

  return train_generator, test_generator


def get_output_dirs(test_replica_weights):
  """Set and create checkpoint and log directories

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


def wait_for_files(a, b, num_retries=10, delay_secs=5):
  """Wait for files to be written

  :param a: file a
  :param b: file b
  :param num_retries: number of retries before raising error
  :param delay_secs: delay time between retries
  :return: True if files exist, else raise error
  """
  try:
    with h5py.File(a, mode='r'):
      with h5py.File(b, mode='r'):
        return True
  except IOError as e:
    if num_retries > 0:
      time.sleep(delay_secs)
      wait_for_files(a, b, num_retries - 1, delay_secs)
    else:
      raise e


def main(args):
  # Write configuration from arguments to eml-cli
  eml.config.write_from_args(args)

  # Set config
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  # Enable NCCL communication between GPUs
  config = eml.session.distribute_config(config)
  K.set_session(tf.Session(config=config))

  train_generator, test_generator = get_data_generators(args.data_dir, args.batch_size, args.run_on_subset)

  model = build_conv_model()

  # Scale the learning rate by the number of model replicas
  lr = eml.optimizer.scale_learning_rate(1.0)
  opt = keras.optimizers.Adadelta(lr)

  # Wrap your optimizer in the All Reduce Optimizer
  opt = eml.optimizer.distribute(opt)

  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])

  # Set the output directory and filepath for saving event files and checkpoints
  checkpoint_dir, log_dir = get_output_dirs(args.test_replica_weights)

  # If there is a predefined checkpoint, check that it exists and load it
  if args.restore_checkpoint_path:
    if os.path.isfile(args.restore_checkpoint_path):
      print('Loading model from checkpoint {}'.format(args.restore_checkpoint_path))
      model.load_weights(args.restore_checkpoint_path)
    else:
      raise IOError('No checkpoint found at %s' % args.restore_checkpoint_path)

  callbacks = [
    # Synchronize all replica weights
    eml.callbacks.init_op_callback(),

    # Set the callback to automatically save a model checkpoint if the job is preempted
    eml.callbacks.preempted_callback(os.path.join(checkpoint_dir, 'preempted.hdf5')),

    keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=args.batch_size, write_graph=False,
                                write_grads=True, write_images=False, update_freq=64 * 10),

    keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_dir, 'checkpoint-{epoch:d}.hdf5'), monitor='val_loss',
                                    verbose=1, save_best_only=False, save_weights_only=False, mode='auto',
                                    period=args.epochs),
  ]

  # Train model
  eml.annotate(title='Train', comment='Start training', tags=[str(args.epochs)])
  model.fit_generator(generator=train_generator,
                      validation_data=test_generator,
                      callbacks=callbacks,
                      epochs=args.epochs,
                      use_multiprocessing=False,
                      workers=8,
                      max_queue_size=128,
                      verbose=1)
  eml.annotate(title='Train', comment='Finished training', tags=[str(args.epochs)])

  # Run weight replica tests if flag is set
  if args.test_replica_weights and eml.replica_id() == 0:
    a = '/engine/outputs/0/checkpoint-%d.hdf5' % (args.epochs)
    b = '/engine/outputs/1/checkpoint-%d.hdf5' % (args.epochs)
    # Sometimes replica 0 will reach the test_replica_weights phase before the other replicas have finished writing
    # the checkpoint file, causing an IOError. wait_for_files ensures that both checkpoints are ready to be accessed
    # before comparing the weights.
    wait_for_files(a, b, num_retries=10, delay_secs=5)
    assert eml.compare_checkpoints(a, b), 'Weights do not match across replicas!'


if __name__ == '__main__':
  arguments = parser.parse_args()
  main(arguments)
