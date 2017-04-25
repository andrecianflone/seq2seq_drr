"""
Author: Andre Cianflone
"""
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

def make_data_set(pdtb_json, mapping):
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

  # Only these types
  types = ['Implicit', 'EntRel']

def get_true(pdtb, relation, sections, types, mapping):
  """ Returns all true of relation, withing section range, or types
  Args:
    pdtb: full PDTB as json
    relation: Top level relation, Example "Temporal"
    sections: a range of sections
    types: such as Implicit
    mapping: dictionary, map to this relation first
  """

########################################################
# HELPERS
########################################################

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
  print('Converting PDTB pipe files into single JSON')
  cur_dir = os.path.dirname(os.path.realpath(__file__))
  output = 'new_relations.json'
  scan_folder(cur_dir, output)
  print('Done! Relations saved to: ', output)
