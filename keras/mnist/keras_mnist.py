from __future__ import print_function
import os
import argparse

import engineml.keras as eml
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf

# Training settings
parser = argparse.ArgumentParser(description='TensorFlow MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--test-replica-weights', type=bool, default=False,
                    help='test that weights are identical across all GPU devices')


def build_conv_model(input_shape, num_classes=10):
  """
  2-layer convolution model.

  :param input_shape: input data shape
  :param num_classes: number of classes to train on

  :return: CNN
  """
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3),
                   activation='relu',
                   input_shape=input_shape))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation='softmax'))
  return model


def one_hot_labels(y, num_classes=10):
  """
  Convert class vectors to binary class matrices

  :param y: labels
  :param num_classes: number of classes to train on

  :return: binary class matrices
  """
  return keras.utils.to_categorical(y, num_classes)


def scale_features(x):
  """
  Scale feature data

  :param x: feature data

  :return: scaled feature data
  """
  x = x.astype('float32')
  x /= 255.
  return x


def reshape_features(x, image_data_format, img_rows, img_cols):
  """
  Reshape feature data

  :param x: feature data
  :param image_data_format: either "channels_first" or "channels_last"
  :param img_rows: number of rows in image
  :param img_cols: number of columns in image

  :return: reshaped feature data
  """
  if image_data_format() == 'channels_first':
    x = x.reshape(x.shape[0], 1, img_rows, img_cols)
  else:
    x = x.reshape(x.shape[0], img_rows, img_cols, 1)
  return x


def prepare_data(image_data_format, img_rows=28, img_cols=28):
  """
  Convert class vectors to binary class matrices

  :param image_data_format: either "channels_first" or "channels_last"
  :param img_rows: number of rows in image
  :param img_cols: number of columns in image

  :return: data, scaled between 0 and 1, shuffled and split between train and test sets
  """
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  
  # Convert class vectors to binary class matrices
  y_train = one_hot_labels(y_train)
  y_test = one_hot_labels(y_test)
  
  # Reshape features
  x_train = reshape_features(x_train, image_data_format, img_rows, img_cols)
  x_test = reshape_features(x_test, image_data_format, img_rows, img_cols)
  
  # Scale features
  x_train = scale_features(x_train)
  x_test = scale_features(x_test)
  
  print('x_train shape:', x_train.shape)
  print(x_train.shape[0], 'train samples')
  print(x_test.shape[0], 'test samples')
  return x_train, y_train, x_test, y_test


def set_checkpoint_path(test_replica_weights):
  """
  Set and create checkpoint directory

  :param test_replica_weights: boolean whether running replica weight test

  :return: checkpoint directory
  """
  # Set the output directory for saving event files and checkpoints
  # `eml.data.output_dir()` returns `None` when running locally
  checkpoint_dir = eml.data.output_dir() or './checkpoints'
  
  # If replica weight test, set manually.
  # THIS IS ONLY FOR TESTING! THERE IS NO REASON TO WRITE MULTIPLE CHECKPOINTS FOR EACH REPLICA.
  # MODEL WEIGHTS ARE UPDATED USING AVG. GRADIENTS ACROSS ALL REPLICAS; THEREFORE EVERY CHECKPOINT WOULD BE IDENTICAL.
  if test_replica_weights:
    checkpoint_dir = os.path.join('/engine/outputs/', str(eml.replica_id()))
  
  if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)
  
  return os.path.join(checkpoint_dir, 'checkpoint.hdf5')


def main(args):
  # Set config
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  # Enable NCCL communication between GPUs
  config = eml.session.distribute_config(config)
  K.set_session(tf.Session(config=config))
  
  # Load and prepare data
  x_train, y_train, x_test, y_test = prepare_data(K.image_data_format)
  
  # Partition your training data across replicas
  x_train = eml.data.distribute(x_train)
  y_train = eml.data.distribute(y_train)
  
  model = build_conv_model(x_train.shape[1:])
  
  # Scale the learning rate by the number of model replicas
  lr = eml.optimizer.scale_learning_rate(1.0)
  opt = keras.optimizers.Adadelta(lr)
  
  # Wrap your optimizer in the All Reduce Optimizer
  opt = eml.optimizer.distribute(opt)
  
  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=opt,
                metrics=['accuracy'])
  
  # Set the output directory and filepath for saving event files and checkpoints
  filepath = set_checkpoint_path(args.test_replica_weights)
  
  callbacks = [
    # Synchronize all replica weights
    eml.callbacks.init_op_callback(),
    
    keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss',
                                    verbose=0, save_best_only=False,
                                    save_weights_only=False,
                                    mode='auto', period=1),
  ]
  
  # Train model
  model.fit(x_train, y_train,
            batch_size=args.batch_size,
            callbacks=callbacks,
            epochs=args.epochs,
            verbose=1,
            validation_split=0.0)
  
  # Evaluate on test set
  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', score[0])
  
  # Run weight replica tests if flag is set
  if args.test_replica_weights:
    a = '/engine/outputs/0/checkpoint.hdf5'
    b = '/engine/outputs/1/checkpoint.hdf5'
    assert eml.compare_checkpoints(a, b), "Weights don't match across replicas!"


if __name__ == "__main__":
  arguments = parser.parse_args()
  main(arguments)
