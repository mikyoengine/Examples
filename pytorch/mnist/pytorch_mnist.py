from __future__ import print_function
import os
import argparse

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
import engineml.torch as eml

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--restore-dir', type=str, default=None,
                    help='directory where saved checkpoint is stored')
parser.add_argument('--test-replica-weights', type=bool, default=False,
                    help='test that weights are identical across all GPU devices')


class Net(nn.Module):
  """
  A simple CNN in pytorch
  """
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)
  
  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x)


def build_model(learn_rate, momentum, use_cuda):
  """
  Build and initialize model and optimizer

  :param learn_rate: learn rate
  :param momentum: SGD momentum
  :param use_cuda: boolean if training on GPU

  :return: model, optimizer
  """
  model = Net()
  
  # Setup Model. init_devices pins your model to a single GPU while enabling NCCL communication
  eml.session.init_devices()
  
  if use_cuda:
    # Move model to GPU.
    model.cuda()
  
  # Scale learning rate by the number of GPUs.
  optimizer = optim.SGD(model.parameters(),
                        lr=eml.optimizer.scale_learning_rate(learn_rate),
                        momentum=momentum)
  # Wrap optimizer with distributed optimizer.
  optimizer = eml.optimizer.distribute(optimizer)
  
  # Synchronize all replica weights
  eml.session.sync_model_replicas(model)
  return model, optimizer


def set_checkpoint_dir(test_replica_weights):
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
  return checkpoint_dir


def download_datasets():
  """
  Download MNIST train and test sets

  :return: train_data, test_data
  """
  data_dir = 'data-%d' % eml.replica_id()
  train_dataset = \
    datasets.MNIST(data_dir, train=True, download=True,
                   transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307,), (0.3081,))
                   ]))
  test_dataset = \
    datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
    ]))
  return train_dataset, test_dataset


def create_data_loaders(train_batch_size, test_batch_size):
  """
  Create data loaders for train and test sets

  :return: train_loader, train_sampler, test_loader
  """
  train_dataset, test_dataset = download_datasets()
  
  # Use torch distributed data sampler with eml.num_replicas and eml.replica_id() to shard train data across replicas
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                  num_replicas=eml.num_replicas(),
                                                                  rank=eml.replica_id())
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=train_batch_size,
                                             num_workers=1,
                                             sampler=train_sampler)
  
  test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=test_batch_size,
                                            num_workers=1)
  return train_loader, train_sampler, test_loader


def train(model, optimizer, train_sampler, train_loader, start_epoch, epochs, use_cuda, checkpoint_dir, log_interval,
          writer):
  """
  Train model

  :param model: initialized model
  :param optimizer: initialized optimizer
  :param train_sampler: sampler to use to shard data across replicas
  :param train_loader: loader for training data
  :param start_epoch: if resuming training from checkpoint, which epoch to start from
  :param epochs: number of epoch to train
  :param use_cuda: boolean if training on GPU
  :param checkpoint_dir: save path for checkpoints
  :param log_interval: int for how often to print train loss
  :param writer: TensorBoardX Summary Writer
  """
  batch_num = 0
  for epoch in range(epochs):
    model.train()
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
      if use_cuda:
        data, target = data.cuda(), target.cuda()
      data, target = Variable(data), Variable(target)
      optimizer.zero_grad()
      output = model(data)
      loss = F.nll_loss(output, target)
      loss.backward()
      optimizer.step()
      batch_num += 1
      
      if batch_idx % log_interval == 0:
        print('Train Epoch: {}\tLoss: {:.6f}'.format((start_epoch + epoch), loss))
        
        # Write grid of 64 training images to TensorBoardX
        img_grid = torchvision.utils.make_grid(data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('Images', img_grid, batch_num)
        
        # Write loss and histogram of weights to Tensorboard
        writer.add_scalar('train_loss', loss, batch_num)
        for name, param in model.named_parameters():
          if param.requires_grad:
            writer.add_histogram(name, param.clone().cpu().data.numpy().reshape(-1), batch_num)
    
    state = {
      "epoch": epoch,
      "model_state": model.state_dict(),
      "optimizer_state": optimizer.state_dict(),
    }
    torch.save(state, os.path.join(checkpoint_dir, 'checkpoint.pt'))
    print("Model Saved at {}!\n".format(checkpoint_dir))
  print("Model finished training!")


def test(model, test_loader, use_cuda):
  """
  Evaluate model on test set

  :param model: trained model
  :param test_loader: loader for test data
  :param use_cuda: boolean if training on GPU
  """
  model.eval()
  test_loss = 0
  test_accuracy = 0.
  cnt = 0
  for data, target in test_loader:
    cnt += len(data)
    if use_cuda:
      data, target = data.cuda(), target.cuda()
    data, target = Variable(data), Variable(target)
    output = model(data)
    # sum up batch loss
    test_loss += F.nll_loss(output, target, size_average=False).item()
    # get the index of the max log-probability
    pred = output.data.max(1, keepdim=True)[1]
    test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()
  
  print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(test_loss / cnt, 100. * test_accuracy / cnt))


def main(args):
  # Create Summary Writer for TensorBoardX.
  # log_dir needs to be set to eml.data.output_dir(). If training locally eml.data.output_dir() returns None.
  writer_dir = eml.data.output_dir() or './logs'
  writer = SummaryWriter(log_dir=writer_dir)
  
  # Download data if necessary and create train and test data loaders
  train_loader, train_sampler, test_loader = create_data_loaders(args.batch_size, args.test_batch_size)
  
  # Build model and optimizer
  model, optimizer = build_model(args.lr, args.momentum, args.cuda)
  
  # Get checkpoint directory
  checkpoint_dir = set_checkpoint_dir(args.test_replica_weights)
  
  # Check to see if training from saved checkpoint and if so load model
  restore_dir = eml.data.input_dir() or args.restore_dir
  start_epoch = 0
  if restore_dir and os.path.isfile(os.path.join(restore_dir, 'checkpoint.pt')):
    print(
      "Loading model and optimizer from checkpoint '{}'".format(
        os.path.join(restore_dir, 'checkpoint.pt')
      )
    )
    checkpoint = torch.load(os.path.join(restore_dir, 'checkpoint.pt'))
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"]
  
  # Train model
  train(model, optimizer, train_sampler, train_loader, start_epoch, args.epochs, args.cuda, checkpoint_dir,
        args.log_interval, writer)
  # Validate against test set
  test(model, test_loader, args.cuda)
  
  # Close TensorBoardX Summary Writer
  writer.close()
  
  # Run weight replica tests if flag is set
  if args.test_replica_weights:
    a = '/engine/outputs/0/checkpoint.pt'
    b = '/engine/outputs/1/checkpoint.pt'
    assert eml.compare_checkpoints(a, b), "Weights don't match across replicas!"


if __name__ == "__main__":
  arguments = parser.parse_args()
  arguments.cuda = not arguments.no_cuda and torch.cuda.is_available()
  main(arguments)
