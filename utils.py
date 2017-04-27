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

def make_batches_legacy(data, batch_size, num_batches,shuffle=True):
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
  def __init__(self, monitor):
    """
    Arg:
      monitor: best results based on this metric
    """
    self.monitor = monitor
    self.metric_best = 0
    self.metric_current = 0
    self._metric_dict = {}
    self.epoch_current = 0
    self.epoch_best = 0

  def update(self, name, value):
    """ Only save metric if best for monitored """
    if name == self.monitor:
      self._check_if_best(name, value)
    else:
      if self.epoch_current == self.epoch_best:
        self._metric_dict[name] = value

  def _check_if_best(self, name, value):
    self.epoch_current += 1
    self.metric_current = value
    if value >= self.metric_best:
      self._metric_dict[name] = value
      self.metric_best = value
      self.epoch_best = self.epoch_current
      self._metric_dict['epoch_best'] = self.epoch_best

  @property
  def metric_dict(self):
    """ Get the dictionary of metrics """
    return self._metric_dict

class Metrics_old():
  """ Keeps score of metrics during training """
  def __init__(self, monitor):
    """
    Arg:
      monitor: best results based on this metric
    """
    self.current_epoch = 0

    # Metrics updated at each epoch
    self._val_f1 = 0
    self._val_accuracy = 0
    self._test_f1 = 0
    self._blind_f1 = 0

    # Updated when best
    self._val_f1_best = 0
    self._val_accuracy_best = 0
    self._f1_best = 0
    self._f1_best_epoch = 0

  # TODO: multiple scores
  # F1 micro
  @property
  def val_f1(self):
    """ Micro F1 score """
    return self._val_f1

  @val_f1.setter
  def val_f1(self, value):
    self._val_f1 = value
    if self._val_f1 >= self._val_f1_best:
      self._val_f1_best = self._val_f1

  @property
  def val_f1_best(self):
    """ Best F1 score so far """
    return self._val_f1_best

  # val_accuracy
  @property
  def val_accuracy(self):
    """ Accuracy score """
    return self._val_accuracy

  @val_accuracy.setter
  def val_accuracy(self, value):
    self._val_accuracy = value
    if self._val_accuracy >= self._val_accuracy_best:
      self._val_accuracy_best = self._val_accuracy

  @property
  def val_accuracy_best(self):
    """ Best val_accuracy score so far """
    return self._val_accuracy_best

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

  # Test F1
  @property
  def test_f1(self):
    """ Test f1 score at best epoch """
    return self._test_f1

  @test_f1.setter
  def test_f1(self, value):
    if self._f1_best_epoch == self.current_epoch:
      self._test_f1 = value

  # Blind F1
  @property
  def blind_f1(self):
    """ Blind f1 score """
    return self._blind_f1

  @blind_f1.setter
  def blind_f1(self, value):
    if self._f1_best_epoch == self.current_epoch:
      self._blind_f1 = value

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
    if self.metrics.metric_current < self.metrics.metric_best:
      self.stop_count += 1
    else:
      self.stop_count = 0

    if self.stop_count >= self.early_stop_epoch:
      msg = "\nEarly stopping. Best f1 {} at epoch {}".format(\
            self.metrics.metric_best, self.metrics.epoch_best)
      self.prog.print_cust(msg)
      return True
    else:
      return False
