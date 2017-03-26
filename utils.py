from datetime import datetime
import numpy as np
from sklearn.utils import shuffle as group_shuffle

class Progress():
  """ Pretty print progress for neural net training """
  def __init__(self, batches, progress_bar=True, bar_length=30):
    self.progress_bar = progress_bar # boolean
    self.bar_length = bar_length
    self.t1 = datetime.now()
    self.train_start_time = self.t1
    self.batches = batches
    self.current_batch = 0
    self.epoch = 0

  def epoch_start(self):
    self.t1 = datetime.now()
    self.epoch += 1
    self.current_batch = 0 # reset batch

  def epoch_end(self):
    print()

  def print_train(self, loss):
    t2 = datetime.now()
    epoch_time = (t2 - self.t1).total_seconds()
    total_time = (t2 - self.train_start_time).total_seconds()/60
    print('epoch: {:2.0f} sec: {:>5.1f} | total min: {:>5.1f} | train loss: {:>3.4f} '.format(
        self.epoch, epoch_time, total_time, loss), end='')
    self.print_bar()

  def print_dec_eval(self, loss):
    print('| validation decoder loss: {:>3.4f} '.format(loss), end='')

  def print_class_eval(self, acc):
    print('| validation class acc: {:>3.4f} '.format(acc), end='')

  def print_bar(self):
    self.current_batch += 1
    end = '' if self.current_batch == self.batches else '\r'
    bars_full = int(self.current_batch/self.batches*self.bar_length)
    bars_empty = self.bar_length - bars_full
    print("| [{}{}] ".format(u"\u2586"*bars_full, '-'*bars_empty),end=end)

def make_batches(data, batch_size, num_batches,shuffle=True):
  """ Batches the passed data
  Args:
    data       : a list of numpy arrays
    batch_size : int
    shuffle    : should be true except when testing
  Returns:
    list of original numpy arrays but sliced
  """
  sk_seed = np.random.randint(0,10000)
  if shuffle: data = group_shuffle(*data, random_state=sk_seed)
  data_size = len(data[0])
  for batch_num in range(num_batches):
    start_index = batch_num * batch_size
    end_index = min((batch_num + 1) * batch_size, data_size)
    batch = []
    for d in data:
      batch.append(d[start_index:end_index])
    yield batch

