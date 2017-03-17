import re
import numpy as np
import json
from collections import Counter
import codecs
from pprint import pprint
import random
import os.path

dtype='int32' # default numpy int dtype

# TODO : checkout tf.contrib preprocessing for tokenization

class Data():
  def __init__(self,
        max_arg_len, # max length of each argument
        maxlen, # maximum total length of input
        mapping_path='data/map_none.json',
        split_input=True, # boolean, split input into separate numpy arrays
        train_file = "data/train.json",
        val_file = "data/dev.json",
        vocab=None,
        inv_vocab=None):

    self.train_file   = train_file
    self.val_file     = val_file
    self.max_arg_len  = max_arg_len
    self.vocab        = None # set in method get_data
    self.inv_vocab    = inv_vocab # set in method get_data
    self.pad_val      = '<pad>'
    self.total_tokens = 0
    self.maxlen       = maxlen
    self.split_input  = split_input
    # Sense mapping dict, or list of dicts
    self.mapping_sense    = self.get_output_mapping(mapping_path)
    self.sense_to_one_hot = self.get_one_hot_dicts(self.mapping_sense)
    self.num_classes      = self.get_class_counts(self.mapping_sense)
    # array shape [samples, 1], or [samples,2] if split

  def get_embedding_matrix(self, model_path=None, save=False, load_saved=False):
    """ Get the embedding from word2vec, needed for embedding layer. Words
    not found in the word2vec dictionary are randomly initialized
    Args:
      model_path : path of the pretrained model
      save       : save the embedding to a json
      check_save : load from a json
    Returns:
      numpy array where row index equivalent to word id in self.vocab
    """
    embedding_file = "data/embedding.json"
    model_path = os.path.abspath(model_path)
    if model_path == None:
      assert load_saved == True
    if load_saved and os.path.isfile(embedding_file):
      return self._load_embedding_from_json(embedding_file, self.vocab)
    else:
      embedding = self._load_embedding_from_binary(model_path, self.vocab)
    if save:
      self._save_embedding(embedding_file, embedding, self.inv_vocab)
    return embedding

  def _save_embedding(self, embedding_file, emb_matrix, inv_vocab):
    """ Save embedding to json file
    Args:
      embedding_file : string, file location
      emb_matrix : array of [vocab_size x embedding size]
      inv_vocab : word list where index corresponds with emb_matrix row
    """
    word_to_emb = {}
    for i, v in enumerate(inv_vocab):
      word_to_emb[v] = emb_matrix[i].tolist()
    with open(embedding_file, 'w') as f:
      json.dump(word_to_emb, f)
    print('embedding json saved to ', embedding_file)

  def _load_embedding_from_json(self, embedding_file, vocab):
    """ Load embeddings from a json file"""
    word_to_emb = {}
    with open(embedding_file) as json_file:
      word_to_emb = json.load(json_file)
    emb_dim = len(list(word_to_emb.values())[0])
    return self._load_embedding(vocab,word_to_emb,emb_dim)

  def _load_embedding_from_binary(self, file_path, vocab):
    """ Load embeddings from a pretrained model, like word2vec
    Words not in vocab are randomly initialized
    """
    print("Loading embedding binary file, this could take a while")
    from gensim.models.keyedvectors import KeyedVectors
    model = KeyedVectors.load_word2vec_format(file_path, binary=True)
    # get length of a generic word as model dimension
    emb_dim = len(model['the'])
    return self._load_embedding(vocab,model,emb_dim)

  def _load_embedding(self, vocab, model, emb_dim):
    out_of_model = 0 # keep track how many random words
    emb_matrix = np.zeros((len(vocab), emb_dim), dtype=np.float32)
    for k, v in vocab.items():
      # Try to get word from the Word2Vec model
      try:
        emb_matrix[v] = model[k]
      except: # If not in Word2Vec, randomly initialize
        out_of_model += 1
        print("word '{}' was randomly initialized".format(k))
        emb_matrix[v] = np.random.rand(emb_dim)

    del model # Hint to Python to reduce memory
    print("words randomly initialized: {}".format(out_of_model))
    return emb_matrix

  def get_data(self, w_sent_len=False):
    """ Return x/y for train/val
    If
    """

    # Load data as lists of tokens
    x_train, y_train, seq_len_train= \
        self.load_from_file(self.train_file, self.max_arg_len)
    x_val, y_val, seq_len_val = \
        self.load_from_file(self.val_file, self.max_arg_len)

    # Array with elements arg1 length, arg2 length
    self.seq_len_train = np.array(seq_len_train, dtype=dtype)
    self.seq_len_val = np.array(seq_len_val, dtype=dtype)

    # Map original sense (y value) to one hot output or multiple outputs (list)
    # These are already numpy arrays
    y_train = self.set_output_for_network(y_train)
    y_val = self.set_output_for_network(y_val)

    # Pad input according to split
    x_train = self.pad_input(x_train, self.seq_len_train)
    x_val   = self.pad_input(x_val, self.seq_len_val)

    # Create vocab
    if self.vocab == None:
      self.vocab, self.inv_vocab = self.create_vocab(x_train,x_val)
    self.total_tokens = len(self.vocab)

    # Integerize x
    x_train = self.integerize(x_train, self.vocab, self.maxlen)
    x_val = self.integerize(x_val, self.vocab, self.maxlen)

    # Make x into numpy arrays!
    x_train = np.array(x_train)
    x_val = np.array(x_val)

    # Split the input between arguments if so desired
    if self.split_input:
      x_train = self.split_x(x_train)
      x_val = self.split_x(x_val)

    return (x_train, y_train), (x_val, y_val)

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

  def integerize(self, x, vocab, maxlen):
    """ Swap all tokens for their integer values based on vocab """
    x_new = []
    for sample in x:
      tokens = [vocab[word] for word in sample]
      x_new.append(tokens)
    return x_new

  def pad_input(self, x, arg_len):
    x_new = []
    for i, sample in enumerate(x):
      # Pad end of individual arguments
      if self.split_input:
        arg1_len = arg_len[i][0]
        arg2_len = arg_len[i][1]
        pad1     = self.max_arg_len - arg1_len
        pad2     = self.max_arg_len - arg2_len
        sample   = sample[:arg1_len] + [self.pad_val] * pad1 +\
                   sample[arg1_len:] + [self.pad_val] * pad2
        x_new.append(sample)

      # Or pad only the end of the whole input
      else:
        pad = maxlen - len(sample)
        sample = sample.extend([pad_val]*pad)
        x_new.append(sample)
    return x_new

  def load_from_file(self, path, max_arg_len):
    """ Parse the input
    Returns:
      x : list of tokenized discourse text
      y : list of labels
      arg_len : list of tuples (arg1_length, arg2_length)
    """
    x = list(); y = list(); arg_len=list();
    with codecs.open(path, encoding='utf8') as pdfile:
      for line in pdfile:
        j = json.loads(line)
        arg1 = clean_str(j['arg1'])[:self.max_arg_len]
        arg2 = clean_str(j['arg2'])[:self.max_arg_len]
        l1 = len(arg1)
        l2 = len(arg2)

        arg1.extend(arg2)
        # Return original sense, mapping done later
        label = j['sense']

        # Add sample to list of data
        x.append(arg1)
        y.append(label)
        arg_len.append((l1,l2))
    return x, y, arg_len

  def create_vocab(self, train_data, val_data):
    """ Create a dictionary of words to int, and the reverse

    You'll want to save this, required for model restore
    """
    words = [word for sublist in train_data for word in sublist]
    words.extend([word for sublist in val_data for word in sublist])
    count = Counter(words) # word count
    # Vocab in descending order
    inv_vocab = [x[0] for x in count.most_common()]
    # Vocab with index position instead of word
    vocab = {x: i for i, x in enumerate(inv_vocab)}
    return vocab, inv_vocab

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
