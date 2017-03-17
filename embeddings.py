"""
General tool to manage word embeddings.

Get embeddings from a binary file or json. Can save to json.
"""
import os.path
import json
import numpy as np

class Embeddings():
  def __init__(self, vocab, inverse_vocab):
    """
    Args:
      vocab         : dictionary of type {word_string : id_int}
      inverse_vocab : list of word_string where index corresponds to vocab id
    """
    self.vocab = vocab
    self.inv_vocab = inverse_vocab

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
