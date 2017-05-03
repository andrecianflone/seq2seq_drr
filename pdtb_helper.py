"""
Author: Andre Cianflone
"""
import random
import codecs
import os
import re
import json
import copy

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

def make_data_set(pdtb, mapping, rng, sampling="down", equal_negative=True):
  """ From the master json, create datasets of positive/negative
  Arg:
    sampling: if equal_negative is true, "down" will downsample largest,
      and "over" will oversample smallest category
  Note:
  - Breakdown according to Pitler et al, 2009
  - The training set is balanced 50/50, negative randomly sampled
  - Dev and test are balanced true/all_else
  - EntRel is merged into expansion
  - Only for Implicit and EntRel
  - NoRel, AltLex and Explicit ignored
  """

  mapping = _dict_from_json(mapping)
  relations = set(mapping.values())
  pdtb = _list_of_dict(pdtb)

  # Only these types
  types = ['Implicit', 'EntRel']
  final_set = []

  for relation in relations:
    # Get positive set
    positive_set = _extract_disc(\
                              pdtb, relation, rng, types, mapping, 'positive')
    pos_ids = [disc['ID'] for disc in positive_set]

    # Get negative set
    neg_relations = relations.copy()
    neg_relations.remove(relation)
    negative_set = _extract_disc(\
        pdtb, neg_relations, rng, types, mapping, 'negative', relation, pos_ids)

    # Maybe balance the sets 50/50 (if training set)
    if equal_negative and sampling == "over":
      positive_set, negative_set = _oversample_set(positive_set, negative_set)

    if equal_negative and sampling == "down":
      positive_set, negative_set = _downsample_set(positive_set, negative_set)

    final_set.extend(copy.deepcopy(positive_set))
    final_set.extend(copy.deepcopy(negative_set))

  return final_set

def _downsample_set(positive_set, negative_set):
  """ The largest set is reduced by random sampling """
  max_size = min(len(positive_set), len(negative_set))

  if len(positive_set) > max_size:
    random.shuffle(positive_set)
    positive_set = positive_set[0:max_size]

  if len(negative_set) > max_size:
    random.shuffle(negative_set)
    negative_set = negative_set[0:max_size]

  return positive_set, negative_set

def _oversample_set(positive_set, negative_set):
  """ The negative set is over sampled if smaller than positive """
  pos_size = len(positive_set)
  neg_size = len(negative_set)
  new_negative = negative_set[:]

  # Boost negative if smaller
  while neg_size < pos_size:
    missing = pos_size - neg_size
    max_add = min(missing, len(negative_set))
    random.shuffle(negative_set)
    negative = negative_set[0:max_add]
    new_negative.extend(negative)
    neg_size = len(new_negative)

  # Reduce negative if bigger
  if neg_size > pos_size:
    random.shuffle(negative_set)
    new_negative = negative_set[0:pos_size]

  return positive_set, new_negative

def _extract_disc(pdtb, relation, sections, types, mapping, new_label,
    new_relation=None, exclusion_set=None):
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

    # Skip if not the relation we want
    # Relation may be in a list or just a string (since remapped)
    if type(disc['Sense']) == list:
      if len(disc['Sense']) > 1: # for disc with two Senses
        rel1 = mapping[disc['Sense'][0]]
        rel2 = mapping[disc['Sense'][1]]
        if rel1 not in relation and rel2 not in relation: continue
        if rel1 in relation:
          rel = rel1
        if rel2 in relation:
          rel = rel2
      if len(disc['Sense']) == 1: # for disc with single Sense
        rel = mapping[disc['Sense'][0]]
        if rel not in relation: continue
    else: # if not a list
      rel = mapping[disc['Sense']]
      if rel not in relation: continue
    disc['Relation'] = rel # new key

    # Override relation for negative set
    if new_relation is not None:
      disc['Relation'] = new_relation

    # Check if not in the exclusion set
    if exclusion_set is not None:
      disc_id = disc['ID']
      if disc_id in exclusion_set:continue

    # Add the new label Key
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

def dict_to_json(disc_list, file_path):
  if os.path.isfile(file_path):
    os.remove(file_path)
  with codecs.open(file_path, mode='a', encoding='utf8') as pdtb:
    for disc in disc_list:
      json.dump(disc, pdtb) # append to end of json file
      pdtb.write('\n') # new line

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

  print('Getting training set')
  train_range = range(2, 20+1)
  train_data = make_data_set(\
            pdtb='data/all_pdtb.json',
            mapping='data/map_pdtb_top.json',
            rng=train_range,
            sampling="over",
            equal_negative=True)
  dict_to_json(train_data, 'data/one_v_all_no_entrel_train.json')

  print('Getting dev set')
  dev_range = range(0, 1+1)
  dev_data = make_data_set(\
            pdtb='data/all_pdtb.json',
            mapping='data/map_pdtb_top.json',
            rng=dev_range,
            sampling=None,
            equal_negative=False)
  dict_to_json(dev_data, 'data/one_v_all_no_entrel_dev.json')

  print('Getting test set')
  test_range = range(21, 22+1)
  test_data = make_data_set(\
            pdtb='data/all_pdtb.json',
            mapping='data/map_pdtb_top.json',
            rng=test_range,
            sampling=None,
            equal_negative=False)
  dict_to_json(test_data, 'data/one_v_all_no_entrel_test.json')
