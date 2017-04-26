"""
Author: Andre Cianflone
"""
import random
import codecs
import os
import re
import json

def scan_folder(directory, output_file):
  """ Scan raw PDTB files and save as a single json """
  print("Scanning dir for pipe files: ", directory)
  for walk in _walklevel(directory, level=1): # for each subdir in root
    cur_dir = walk[0]
    if cur_dir == directory: continue # skip root
    for pipe_file in walk[2]: # for each file in subdir
      # For each discourse in pipe file
      try:
        for disc in _dict_from_pipe_file(cur_dir, pipe_file):
          # Append the discourse dictionary to json file
          _append_json_file(disc, output_file)
      except UnicodeDecodeError as e:
        print("Error in file: ", pipe_file)
        print(e)
        print("*****Process terminated*****")
        sys.exit()

def make_data_set(pdtb, mapping):
  """ From the master json, create datasets of positive/negative
  Note:
  - Breakdown according to Pitler et al, 2009
  - The training set is balanced 50/50, negative randomly sampled
  - Dev and test are balanced true/all_else
  - EntRel is merged into expansion
  - Only for Implicit and EntRel
  - NoRel, AltLex and Explicit ignored
  """
  train_range = range(2, 20+1)
  dev_range = range(0, 1+1)
  test_range = range(21, 22+1)
  mapping = _dict_from_json(mapping)
  relations = set(mapping.values())
  pdtb = _list_of_dict(pdtb)

  # Only these types
  types = ['Implicit', 'EntRel']
  train_data = _get_data(pdtb, types, mapping, train_range, relations, True)

def _get_data(pdtb, types, mapping, rng, relations, equal_negative=True):
  """ Returns a json of positive and negative """
  final_set = []
  for relation in relations:
    # Get positive set
    positive_set = _extract_disc(\
                              pdtb, relation, rng, types, mapping, 'positive')
    pos_ids = [disc['ID'] for disc in positive_set]

    # Get negative set
    negative_set = _extract_disc(\
                      pdtb, relation, rng, types, mapping, 'negative', pos_ids)

    # Balance the sets 50/50
    if equal_negative:
      positive_set, negative_set = _rebalance_sets(positive_set, negative_set)

    final_set.append(positive_set)
    final_set.append(negative_set)

def _rebalance_sets(positive_set, negative_set):
  """ The biggest set is reduced by random sampling """
  max_size = min(len(positive_set), len(negative_set))

  if len(positive_set) > max_size:
    random.shuffle(positive_set)
    positive_set = positive_set[0:max_size]

  if len(negative_set) > max_size:
    random.shuffle(negative_set)
    negative_set = negative_set[0:max_size]

  return positive_set, negative_set

def _extract_disc(pdtb, relation, sections, types, mapping, new_label,
    exclusion_set=None):
  """ Returns all true of relation, withing section range, or types
  Args:
    pdtb: full PDTB as list of dict
    relation: Top level relation, Example "Temporal"
    sections: a range of sections
    types: such as Implicit
    mapping: dictionary, map to this relation first
  """
  data_set = []
  for disc in pdtb:
    # Skip if not valid type
    tp = disc['Type']
    if tp not in types: continue

    # Skip if not section we want
    section = int(disc['Section'])
    if section not in sections: continue

    # Add the EntRel as sense
    if disc['Type'] == 'EntRel': disc['Sense'] = ['EntRel']

    # Skip if not relation we want
    rel = mapping[disc['Sense'][0]] # map the relation
    if rel not in relation: continue
    disc['Sense'] = rel

    # Check if not in the exclusion set
    if exclusion_set is not None:
      disc_id = disc['ID']
      if disc_id in exclusion_set:continue

    # Add the new label
    disc['Class'] = new_label

    data_set.append(disc)
  return data_set

########################################################
# HELPERS
########################################################

def _list_of_dict(file_path):
  dataset = []
  with codecs.open(file_path, encoding='utf-8') as f:
    for line in f:
      j = json.loads(line)
      dataset.append(j)
  return dataset

def _dict_from_json(file_path):
  """ Load dictionary from a json file """
  with codecs.open(file_path, encoding='utf-8') as f:
    dictionary = json.load(f)
  return dictionary

def _append_json_file(data_dict, file_path):
  """ Appends json file with data dictionary """

  with codecs.open(file_path, mode='a', encoding='utf8') as pdtb:
    json.dump(data_dict, pdtb) # append to end of json file
    pdtb.write('\n') # new line

def _dict_from_pipe_file(dirpath, filename):
  """ Yield dictionary for each discourse in file """

  # Get directory number
  dir_num = dirpath.split('/')[-1]

  # Full file path
  file_path = os.path.join(dirpath, filename)

  # Get the PDTB file number ID from filename
  doc_number = re.search('[0-9]+', filename).group(0)

  # Count discourse
  cnt = 0

  # Loop disc in file
  with codecs.open(file_path, mode='r', encoding='iso-8859-1') as pdtb:
    for line in pdtb:
      # Zero padded string count
      cnt_str = str(cnt).zfill(2)

      fields = line.split('|')
      data = {
          'Arg1': {
            'CharacterSpanList': fields[22],
            'RawText': fields[24],
            'Tokenized' : [[]]},
          'Arg2': {
            'CharacterSpanList': fields[32],
            'RawText': fields[34],
            'Tokenized' : [[]]},
          'Connective': {
            'CharacterSpanList': fields[3],
            'RawText': fields[5]},
          'DocID': filename,
          'Sense': _valid_list(fields[11], fields[12]),
          'Type': fields[0],
          'ID': doc_number + cnt_str,
          'Section': dir_num
          }
      yield data
      cnt += 1

def _walklevel(some_dir, level=1):
  """ Like os.walk but limit depth """
  some_dir = some_dir.rstrip(os.path.sep)
  assert os.path.isdir(some_dir)
  num_sep = some_dir.count(os.path.sep)
  for root, dirs, files in os.walk(some_dir):
    yield root, dirs, files
    num_sep_this = root.count(os.path.sep)
    if num_sep + level <= num_sep_this:
      del dirs[:]

def _valid_list(*args):
  """ Returns list of non empty strings """
  new_list = []
  # Unpack variable args
  for elem in args:
    if elem != '': new_list += [elem]
  return new_list

########################################################
# MAIN
########################################################
if __name__ == "__main__":
  # print('Converting PDTB pipe files into single JSON')
  # cur_dir = os.path.dirname(os.path.realpath(__file__))
  # output = 'new_relations.json'
  # scan_folder(cur_dir, output)
  # print('Done! Relations saved to: ', output)

  make_data_set('data/all_pdtb.json', 'data/map_pdtb_top.json')
