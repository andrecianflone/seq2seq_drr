import re
import numpy as np
import json
from collections import Counter
import codecs
from pprint import pprint
import random
import os.path
from conll_utils.scorer import f1_non_explicit

dtype='int32' # default numpy int dtype

# TODO : checkout tf.contrib preprocessing for tokenization

class Data():
  def __init__(self,
        max_arg_len, # max length of each argument
        maxlen, # maximum total length of input
        mapping_path='data/map_proper_conll.json',
        split_input=True, # boolean, split input into separate numpy arrays
        train_file = "data/train.json",
        val_file = "data/dev.json",
        decoder_targets=False, # second arg without bos
        bos_tag = None, # beginning of sequence tag
        eos_tag = None, # end of sequence tag
        vocab=None, # If none, will create the vocab
        inv_vocab=None): # If none, generates inverse vocab

    self.train_file   = train_file
    self.val_file     = val_file
    self.max_arg_len  = max_arg_len
    self.vocab        = None # set in method get_data
    self.inv_vocab    = inv_vocab # set in method get_data
    self.pad_val      = '<pad>'
    self.total_tokens = 0
    self.bos_tag      = bos_tag
    self.eos_tag      = eos_tag
    self.maxlen       = maxlen
    self.split_input  = split_input
    # Sense mapping dict, or list of dicts
    self.mapping_sense    = self.get_output_mapping(mapping_path)
    self.sense_to_one_hot = self.get_one_hot_dicts(self.mapping_sense)
    self.int_to_sense     = self.get_int_to_sense_dict(self.sense_to_one_hot)
    self.num_classes      = self.get_class_counts(self.mapping_sense)
    # array shape [samples, 1], or [samples,2] if split


  def get_data(self, w_sent_len=False):
    """ Return x/y for train/val """

    # TODO: because of all the list changes, should use collections.deque

    # Load data as lists of tokens
    x_train, y_train, seq_len_train, dec_targ_train= \
        self.load_from_file(self.train_file, self.max_arg_len)
    self.val_disc_list = []
    x_val, y_val, seq_len_val, dec_targ_val= \
        self.load_from_file(self.val_file, self.max_arg_len, self.val_disc_list)

    # Array with elements arg1 length, arg2 length
    self.seq_len_train = np.array(seq_len_train, dtype=dtype)
    self.seq_len_val = np.array(seq_len_val, dtype=dtype)

    # Map original sense (y value) to one hot output or multiple outputs (list)
    # These are already numpy arrays
    y_train = self.set_output_for_network(y_train)
    y_val = self.set_output_for_network(y_val)

    # Pad input according to split
    x_train = self.pad_input(x_train, self.seq_len_train, self.split_input)
    x_val   = self.pad_input(x_val, self.seq_len_val, self.split_input)
    dec_targ_train = self.pad_input(dec_targ_train, split=False)
    dec_targ_val = self.pad_input(dec_targ_val, split=False)

    # Create vocab
    if self.vocab == None:
      self.vocab, self.inv_vocab = self.create_vocab(x_train,x_val)
    self.total_tokens = len(self.vocab)

    # Integerize x
    x_train = self.integerize(x_train, self.vocab)
    x_val = self.integerize(x_val, self.vocab)
    dec_targ_train = self.integerize(dec_targ_train, self.vocab)
    dec_targ_val = self.integerize(dec_targ_val, self.vocab)

    # Make x into numpy arrays!
    x_train = np.array(x_train)
    x_val = np.array(x_val)
    dec_targ_train = np.array(dec_targ_train)
    dec_targ_val = np.array(dec_targ_val)

    # Split the input between arguments if so desired
    if self.split_input:
      x_train = self.split_x(x_train)
      x_val = self.split_x(x_val)

    return (x_train, y_train, dec_targ_train), (x_val, y_val, dec_targ_val)

  def get_seq_length(self):
    return self.seq_len_train , self.seq_len_val

  def set_output_for_network(self, y):
    """ Returns single list of y values, or multiple lists if multiple lists
    if the network has multiples outputs

    Return will depend on mapping_sense class param. If single dictionary,
    then returns single output. If list of multiple dictionaries, then returns
    list of outputs
    """
    if type(self.mapping_sense) is list:
      outputs = [] # list of list of y values for several network outputs
      for i, _ in enumerate(self.mapping_sense):
        labels = []
        for label in y:
          mapping = self.mapping_sense[i] # To map original label to new label
          one_hot_dict = self.sense_to_one_hot[i] # One hot of new label
          label = one_hot_dict[mapping[label]]
          labels.append(label)
        labels = np.array(labels) # swap to numpy array
        outputs.append(labels)
      return outputs
    else:
      outputs = [] # single list of y
      for label in y:
        label = self.sense_to_one_hot[self.mapping_sense[label]]
        outputs.append(label)
      outputs = np.array(outputs)
      return outputs

  def get_output_mapping(self, mapping_path):
    """ Returns single dict, or list of dicts of mapping"""
    if type(mapping_path) is list:
      maps = []
      for path in mapping_path:
        maps.append(self.dict_from_json(path))
      return maps
    else:
      return self.dict_from_json(mapping_path)

  def get_class_counts(self, mapping_sense):
    if type(mapping_sense) is list:
      classes = []
      for mappings in mapping_sense:
        classes.append(len(set(mappings.values())))
      return classes
    else:
      return len(set(mapping_sense.values()))

  def get_one_hot_dicts(self, mapping_sense):
    if type(mapping_sense) is list:
      one_hot = []
      for mapping in mapping_sense:
        senses = set(mapping.values())
        one_hot.append(self.one_hot_dict(senses))
      return one_hot
    else:
      return self.one_hot_dict(set(mapping_sense.values()))

  def get_int_to_sense_dict(self, sense_to_one_hot):
    """ Return dict mapping integer to label based on one hot dict"""
    int_to_sense = {}
    for k, v in sense_to_one_hot.items():
      key = np.argmax(v)
      int_to_sense[key] = k
    return int_to_sense

  def one_hot_dict(self,senses):
    """Return dictionary of one-hot encodings of list of items"""
    # Base vector, all zeros
    base = [0] * len(senses)
    embedding = {}
    for i, x in enumerate(senses):
      emb = base[:]
      emb[i] = 1
      embedding[x] = emb
    return embedding

  def dict_from_json(self, file_path):
    """ Load dictionary from a json file """
    with codecs.open(file_path, encoding='utf-8') as f:
      dictionary = json.load(f)
    return dictionary

  def split_x(self, x):
    x_1 = x[:,:self.max_arg_len]
    x_2 = x[:,self.max_arg_len:]
    x   = [x_1,x_2]
    return x

  def integerize(self, x, vocab):
    """ Swap all tokens for their integer values based on vocab """
    x_new = []
    for sample in x:
      tokens = [vocab[word] for word in sample]
      x_new.append(tokens)
    return x_new

  def pad_input(self, x, arg_len=None, split=False):
    x_new = []
    for i, sample in enumerate(x):
      # Pad end of individual arguments
      if split:
        arg1_len = arg_len[i][0]
        arg2_len = arg_len[i][1]
        pad1     = self.max_arg_len - arg1_len
        pad2     = self.max_arg_len - arg2_len
        sample   = sample[:arg1_len] + [self.pad_val] * pad1 +\
                   sample[arg1_len:] + [self.pad_val] * pad2
        x_new.append(sample)

      # Or pad only the end of the whole input
      else:
        pad = self.max_arg_len - len(sample)
        sample.extend([self.pad_val]*pad)
        x_new.append(sample)
    return x_new

  def load_from_file(self, path, max_arg_len, discourse_list=None):
    """ Parse the input
    Returns:
      x : list of tokenized discourse text
      y : list of labels
      arg_len : list of tuples (arg1_length, arg2_length)
      discourse_list : if !None, saves discourse info to this list
    """
    x = list(); y = list(); arg_len=list(); decoder_targets=list();
    with codecs.open(path, encoding='utf8') as pdfile:
      for line in pdfile:
        j = json.loads(line)
        if discourse_list is not None: discourse_list.append(j)
        arg1 = clean_str(j['Arg1']['RawText'])[:self.max_arg_len]
        arg2 = clean_str(j['Arg2']['RawText'])
        if self.bos_tag:
          arg2.insert(0, self.bos_tag)
        arg2 = arg2[:self.max_arg_len]
        dec_target = arg2[1:]
        dec_target.append(self.eos_tag)
        decoder_targets.append(dec_target)

        l1 = len(arg1)
        l2 = len(arg2)

        arg1.extend(arg2)
        # Return original sense, mapping done later
        label = j['Sense'][0]

        # Add sample to list of data
        x.append(arg1)
        y.append(label)
        arg_len.append((l1,l2))
    return x, y, arg_len, decoder_targets

  def add_tags(self, seq_list):
    """ Adds beginning and/or end of sequence tags if set """
    if self.bos_tag:
      seq_list.insert(0, self.bos_tag)
    if self.eos_tag:
      seq_list.append(self.eos_tag)
    return seq_list

  def create_vocab(self, train_data, val_data):
    """ Create a dictionary of words to int, and the reverse

    You'll want to save this, required for model restore
    """
    words = [word for sublist in train_data for word in sublist]
    words.extend([word for sublist in val_data for word in sublist])
    words.extend([self.eos_tag] * len(train_data)) # silly hack to add tag
    count = Counter(words) # word count
    # Vocab in descending order
    inv_vocab = [x[0] for x in count.most_common()]
    # Vocab with index position instead of word
    vocab = {x: i for i, x in enumerate(inv_vocab)}
    return vocab, inv_vocab

  def conll_f1_score(self, predictions):
    self.save_to_conll_format('tmp.json', predictions, self.val_disc_list)
    precision, recall, f1 = f1_non_explicit('tmp.json', 'data/dev.json')
    return f1

  def save_to_conll_format(self, path, predictions, discourse, append_file=False):
    """ Saves as json in conll format
    Args:
      path : path where to write the json
      predictions : sense prediction from the neural network, numpy array
      discourse : list of dicourse dictionary. Index of discourse in this list
            must match index of discourse in predictions array

    Writes Json file, where each line is:
    {   Arg1: {TokenList: [275, 276, 277], RawText: "raw text"},
        Arg2: {TokenList: [301, 302, 303], RawText: "raw text"},
        Connective: {TokenList: []},
        DocID: 'wsj_1000',
        Sense: ['Expansion.Conjunction'],
        Type: 'Implicit'
    }
    """

    if append_file == False:
      # Remove file if exists
      if os.path.isfile(path):
        os.remove(path)

    # Loop through results
    with codecs.open(path, mode='a', encoding='utf8') as pdtb:
      for i, disc in enumerate(discourse):
        sense_id = int(predictions[i])
        disc['Sense'] = [self.int_to_sense[sense_id]]
        json.dump(disc, pdtb) #indent to add to new line
        pdtb.write('\n')
    # print("\nSaved results as CoNLL json to here: ", path)

def clean_str(string):
  """
  Clean string, return tokenized list
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " \( ", string)
  string = re.sub(r"\)", " \) ", string)
  string = re.sub(r"\?", " \? ", string)
  string = re.sub(r"\s{2,}", " ", string)

  return string.strip().lower().split()

