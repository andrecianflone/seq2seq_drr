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
    print('{:2.0f}: sec: {:>5.1f} | total min: {:>5.1f} | train loss: {:>3.4f} '.format(
        self.epoch, epoch_time, total_time, loss), end='')
    self.print_bar()

  def print_cust(self, msg):
    """ Print anything, append previous """
    print(msg, end='')

  def print_eval(self, msg, value):
    print('| {}: {:>3.4f} '.format(msg, value), end='')

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


class Metrics():
  """ Keeps score of metrics during training """
  def __init__(self):
    self.current_epoch = 0

    # Metrics updated at each epoch
    self._f1_micro = 0
    self._accuracy = 0
    self._f1 = 0

    # Updated when best
    self._f1_micro_best = 0
    self._accuracy_best = 0
    self._f1_best = 0
    self._f1_best_epoch = 0

  # F1 micro
  @property
  def f1_micro(self):
    """ Micro F1 score """
    return self._f1_micro

  @f1_micro.setter
  def f1_micro(self, value):
    self._f1_micro = value
    if self._f1_micro >= self._f1_micro_best:
      self._f1_micro_best = self._f1_micro

  @property
  def f1_micro_best(self):
    """ Best F1 score so far """
    return self._f1_micro_best

  # Accuracy
  @property
  def accuracy(self):
    """ Accuracy score """
    return self._accuracy

  @accuracy.setter
  def accuracy(self, value):
    self._accuracy = value
    if self._accuracy >= self._accuracy_best:
      self._accuracy_best = self._accuracy

  @property
  def accuracy_best(self):
    """ Best accuracy score so far """
    return self._accuracy_best

  # F1
  @property
  def f1(self):
    """ f1 score """
    return self._f1

  @f1.setter
  def f1(self, value):
    self._f1 = value
    self.current_epoch += 1
    if self._f1 >= self._f1_best:
      self._f1_best = self._f1
      self._f1_best_epoch = self.current_epoch

  @property
  def f1_best(self):
    """ Best f1 score so far """
    return self._f1_best

  @property
  def f1_best_epoch(self):
    """ Epoch which achieved best F1 """
    return self._f1_best_epoch

class Callback():
  """ Monitor training """
  def __init__(self, early_stop_epoch, metrics, prog_bar):
    """
    Args:
      early_stop_epoch : stop if not improved for these epochs
      metrics: a Metrics object with f1 property updated during training
    """
    self.metrics = metrics
    self.early_stop_epoch = early_stop_epoch
    self.stop_count = 0
    self.prog = prog_bar

  def early_stop(self):
    """ Check if f1 is decreasing """
    if self.metrics.f1 < self.metrics.f1_best:
      self.stop_count += 1
    else:
      self.stop_count = 0

    if self.stop_count >= self.early_stop_epoch:
      msg = "\nEarly stopping. Best f1 {} at epoch {}".format(\
            self.metrics.f1_best, self.metrics.f1_best_epoch)
      self.prog.print_cust(msg)
      return True
    else:
      return False
