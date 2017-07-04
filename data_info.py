# Author: Andre Cianflone
# Print some useful info about the dataset
from collections import defaultdict
import json
import codecs
import matplotlib.pyplot as plt
from pprint import pprint

def file_info(filepath, explicit=None, sense_tag='Sense', only_positive=False):
  arg_len = defaultdict(int)
  count = 0
  senses = defaultdict(int)
  types = defaultdict(int)
  with open(filepath, encoding='utf8') as pdfile:
    for line in pdfile:
      count += 1
      # try:
      j = json.loads(line)
      # except Exception as e:
        # print('line ', count, ' invalid')
        # continue

      # Possible skip for positive filter
      if only_positive == True:
        if j['Class'] != "positive":
          continue

      # pprint(j)
      # input('...')
      # total_tokens = len(j['Arg1']['Tokenized'])
      # total_tokens += len(j['Arg2']['Tokenized'])
      total_tokens = len(j['Arg1']['TokenList'])
      total_tokens += len(j['Arg2']['TokenList'])
      if total_tokens > 120:
        total_tokens = 120
      arg_len[total_tokens] += 1

      typerel = str(j['Type'])
      types[typerel] += 1

      if sense_tag=='Sense':
        sense = str(j['Sense'][0])
      else:
        sense = str(j[sense_tag])
      if explicit == True:
        if typerel == 'Explicit':
          senses[sense] += 1
      elif explicit == False:
        if typerel != 'Explicit':
          senses[sense] += 1
      else:
        senses[sense] += 1

  # Add percents
  senses_perc = {}
  types_perc = {}
  for k, v in senses.items():
    senses_perc[k] = (v, "{:0.2f}".format(v/count))
  for k, v in types.items():
    types_perc[k] = (v, "{:0.2f}".format(v/count))

  return count, senses_perc, types_perc, arg_len

def plot_arg_len(arg_len):
  width = 1.0
  plt.bar(arg_len.keys(), arg_len.values(), width, color='b')
  plt.show()

def print_info(filepath, explicit=None, sense='Sense', plot=False, only_positive=False):
  print('='*79)
  print('Info for file: ', filepath)
  print('='*79)
  count, sense, types, arg_len = file_info(filepath, explicit, sense, only_positive)
  print('Lines in train: ', count)
  pprint(types)
  if explicit == False:
    print('*****IGNORING EXPLICIT****')
  pprint(sense)
  if plot:
    plot_arg_len(arg_len)

if __name__ == "__main__":
  # print_info('data/one_v_all_dev2.json', explicit=False, only_positive=False)
  # print_info('data/one_v_all_test.json', explicit=False, only_positive=False)
  # print_info('data/one_v_all_train.json', explicit=False, only_positive=False)
  print_info('data/train.json', True, sense='Sense')
  print_info('data/dev.json', True, sense='Sense')
  print_info('data/test.json', True, sense='Sense')
  print_info('data/blind.json', True, sense='Sense')


