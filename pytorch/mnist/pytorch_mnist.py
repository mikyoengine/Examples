from __future__ import print_function

import argparse
import os

import engineml.torch as eml
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from PIL import Image
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import Dataset

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N_SAMPLES',
                    help='input batch size for training (default: 64)')
parser.add_argument('--data-dir', type=str, default='/engine/data', metavar='DATA_DIR',
                    help='path to data directory')
parser.add_argument('--epochs', type=int, default=2, metavar='N_EPOCHS',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--test-replica-weights', action='store_true',
                    help='test that weights are identical across all GPU devices')
parser.add_argument('--run-on-subset', action='store_true',
                    help='run on a subset of the data')
parser.add_argument('--restore-checkpoint-path', type=str, default='', metavar='RESTORE_CHKPT_PATH',
                    help='path to checkpoint to load')


class DataGenerator(Dataset):
  """Generates MNIST data for pytorch"""
  def __init__(self, df, data_dir, target_size=(28, 28), n_classes=10, is_train=True):
    """Initialization"""
    self.df = df
    self.data_dir = data_dir
    self.target_size = target_size
    self.n_classes = n_classes
    if is_train:
      self.sub_dir = 'train'
    else:
      self.sub_dir = 'test'

  def __len__(self):
    """Denotes the number of batches per epoch"""
    return len(self.df)

  def __getitem__(self, index):
    """Generate one sample of data"""
    x = self.load_mnist_img(os.path.join(self.data_dir, self.sub_dir, self.df[index][0]))
    y = self.df[index][1]
    sample = {'x': x, 'y': y}
    return sample

  def load_mnist_img(self, fn):
    """ Load MNIST image

    :param fn: path to img
    :return: grayscale img array
    """
    out = np.asarray(Image.open(fn), dtype=np.float32).reshape(1, self.target_size[0], self.target_size[1]) / 255.
    return out


def collate_fn(data):
  inputs = np.array([sample['x'] for sample in data])
  targets = np.array([sample['y'] for sample in data])
  return torch.tensor(inputs), torch.tensor(targets)


def create_data_loaders(data_dir, batch_size, run_on_subset):
  """Create data loaders for train and test sets

  :param data_dir: path to data directory
  :param batch_size: int, batch_size
  :param run_on_subset: boolean whether running replica weight test
  :return: train_loader, test_loader
  """
  # If running integration tests, only use a subset of the data
  if run_on_subset:
    df_train = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))[:5000]
    df_test = pd.read_csv(os.path.join(data_dir, 'test_labels.csv'))[:500]
  else:
    df_train = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))
    df_test = pd.read_csv(os.path.join(data_dir, 'test_labels.csv'))

  # Partition your training and test data across replicas
  df_train = eml.data.distribute(df_train.values, prefetch=True,
                                 prefetch_func=lambda x: os.path.join(eml.data.data_dir(), 'train', x[0]))
  df_test = eml.data.distribute(df_test.values, prefetch=True,
                                prefetch_func=lambda x: os.path.join(eml.data.data_dir(), 'test', x[0]))

  train_generator = DataGenerator(df_train, data_dir)
  test_generator = DataGenerator(df_test, data_dir, is_train=False)

  train_loader = torch.utils.data.DataLoader(train_generator,
                                             batch_size=batch_size,
                                             num_workers=16,
                                             collate_fn=collate_fn,
                                             shuffle=True)

  test_loader = torch.utils.data.DataLoader(test_generator,
                                            batch_size=batch_size,
                                            num_workers=16,
                                            collate_fn=collate_fn)

  return train_loader, test_loader


class Net(nn.Module):
  """A simple CNN in pytorch"""
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
    self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(512, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 512)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x)


def build_model():
  """Build and initialize model and optimizer

  :return: model, optimizer
  """
  model = Net()

  # Setup Model. init_devices pins your model to a single GPU while enabling NCCL communication
  eml.session.init_devices()

  # Move model to GPU.
  if torch.cuda.is_available():
    model.cuda()

  # Scale learning rate by the number of GPUs.
  optimizer = optim.Adadelta(model.parameters(), lr=eml.optimizer.scale_learning_rate(0.1))

  # Wrap optimizer with distributed optimizer.
  optimizer = eml.optimizer.distribute(optimizer)

  # Synchronize all replica weights
  eml.session.sync_model_replicas(model)
  return model, optimizer


def set_checkpoint_dir(test_replica_weights):
  """Set and create checkpoint directory

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
  return checkpoint_dir


def train(model, optimizer, train_loader, current_epoch, total_epochs, checkpoint_dir, writer, test_replica_weights):
  """Train model

  :param model: initialized model
  :param optimizer: initialized optimizer
  :param train_loader: loader for training data
  :param current_epoch: number of current epoch
  :param total_epochs: total number of epochs
  :param checkpoint_dir: save path for checkpoints
  :param writer: TensorBoardX Summary Writer
  :param test_replica_weights: bool, whether testing replica weights as part of integration tests
  """
  samples_seen = current_epoch * len(train_loader.dataset)
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    if torch.cuda.is_available():
      data, target = data.cuda(), target.cuda()
    data, target = Variable(data), Variable(target)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    samples_seen += len(data)
    if batch_idx % 10 == 0:
      print('Train Epoch: {}/{}\tLoss: {:.6f}'.format(current_epoch + 1, total_epochs, loss))
      # Write an image to TensorBoardX
      writer.add_image('images', data[0], samples_seen)

      # Write loss, accuracy and histogram of weights to Tensorboard
      writer.add_scalar('loss', loss, samples_seen)
      pred = output.data.max(1, keepdim=True)[1]
      acc = pred.eq(target.data.view_as(pred)).cpu().float().sum().item() / len(target)
      writer.add_scalar('acc', acc, samples_seen)

  state = {
    'epoch': current_epoch,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
  }
  if test_replica_weights:
    torch.save(state, os.path.join(checkpoint_dir, 'checkpoint'))
  else:
    eml.save(state, os.path.join(checkpoint_dir, 'checkpoint'))
  print('Model Saved to {}!\n'.format(checkpoint_dir))


def test(model, test_loader, samples_seen, writer):
  """Evaluate model on test set

  :param model: trained model
  :param test_loader: loader for test data
  :param samples_seen: number of training examples seen before running evaluation
  :param writer: TensorBoardX Summary Writer
  """
  model.eval()
  replica_test_loss = 0.
  replica_test_accuracy = 0.
  for data, target in test_loader:
    if torch.cuda.is_available():
      data, target = data.cuda(), target.cuda()
    data, target = Variable(data), Variable(target)
    output = model(data)
    # sum up batch loss
    replica_test_loss += F.nll_loss(output, target, size_average=False).item()
    # get the index of the max log-probability
    pred = output.data.max(1, keepdim=True)[1]
    replica_test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum().item()

  # Gather total test size, total accuracy, and total loss across all replicas
  total_test_size = eml.sync.replica_sum(np.float32(len(test_loader.dataset)))
  total_test_loss = eml.sync.replica_sum(np.float32(replica_test_loss)) / total_test_size
  total_test_accuracy = eml.sync.replica_sum(np.float32(replica_test_accuracy)) / total_test_size
  print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(total_test_loss, 100. * total_test_accuracy))
  writer.add_scalar('val_loss', total_test_loss, samples_seen)
  writer.add_scalar('val_acc', total_test_accuracy, samples_seen)


def main(args):
  # Write configuration from arguments to eml-cli
  eml.config.write_from_args(args)

  # Create Summary Writer for TensorBoardX.
  # log_dir needs to be set to eml.data.output_dir(). If training locally eml.data.output_dir() returns None.
  writer_dir = './logs' if eml.data.output_dir() is None else os.path.join(eml.data.output_dir(), 'logs')
  writer = SummaryWriter(log_dir=writer_dir)

  # Download data if necessary and create train and test data loaders
  train_loader, test_loader = create_data_loaders(args.data_dir, args.batch_size, args.run_on_subset)

  # Build model and optimizer
  model, optimizer = build_model()

  # Get checkpoint directory
  checkpoint_dir = set_checkpoint_dir(args.test_replica_weights)

  # Create a handler to automatically write a checkpoint when a job is preempted
  def save_handler(m, o, checkpoint_path):
    state = {
      'model_state': m.state_dict(),
      'optimizer_state': o.state_dict(),
    }
    eml.save(state, checkpoint_path)

  # Set the preempted checkpoint handler
  eml.preempted_handler(save_handler, model, optimizer, os.path.join(checkpoint_dir, 'preempted'))

  # If there is a predefined checkpoint, check that it exists and load it
  if args.restore_checkpoint_path:
    if os.path.isfile(args.restore_checkpoint_path):
      print('Loading model from checkpoint {}'.format(args.restore_checkpoint_path))
      if torch.cuda.is_available():
        checkpoint = torch.load(args.restore_checkpoint_path)
      else:
        checkpoint = torch.load(args.restore_checkpoint_path, map_location='cpu')
      model.load_state_dict(checkpoint['model_state'])
      optimizer.load_state_dict(checkpoint['optimizer_state'])
    else:
      raise IOError('No checkpoint found at %s' % args.restore_checkpoint_path)

  for epoch in range(args.epochs):
    # Train model
    eml.annotate(title='Train', comment='Start training', tags=[str(epoch)])
    train(model=model, optimizer=optimizer, train_loader=train_loader, current_epoch=epoch, total_epochs=args.epochs,
          checkpoint_dir=checkpoint_dir, writer=writer, test_replica_weights=args.test_replica_weights)
    # Validate against test set
    samples_seen = (1 + epoch) * len(train_loader.dataset)
    eml.annotate(title='Validation', comment='Start validation', tags=[str(epoch)])
    test(model=model, test_loader=test_loader, samples_seen=samples_seen, writer=writer)

  # Close TensorBoardX Summary Writer
  writer.close()

  # Run weight replica tests if flag is set
  if args.test_replica_weights:
    a = '/engine/outputs/0/checkpoint'
    b = '/engine/outputs/1/checkpoint'
    assert eml.compare_checkpoints(a, b), 'Weights do not match across replicas!'


if __name__ == '__main__':
  arguments = parser.parse_args()
  main(arguments)
