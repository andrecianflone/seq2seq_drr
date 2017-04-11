
# Author: Andre Cianflone
# Format relations.json file to proper output.json for scorer.py
import json
import codecs
import os
import argparse

def convert(path_from, path_to):
  """ Converts from relations.json to proper output.json 
  Generally the files are the same except for the token lists which is only 
  the document level token indices.
  """

  # Delete path_to file if exists
  if os.path.isfile(path_to):
    os.remove(path_to)

  with open(path_from, encoding='utf8') as f_in:
    with open(path_to, mode='a', encoding='utf8') as f_out:
      for line in f_in:
        j = json.loads(line)
        j['Arg1']['TokenList'] = reduce_list(j['Arg1']['TokenList'])
        j['Arg2']['TokenList'] = reduce_list(j['Arg2']['TokenList'])
        j['Connective']['TokenList'] = reduce_list(j['Connective']['TokenList'])
        json.dump(j, f_out)
        f_out.write('\n')
  print('Conversion complete')

def reduce_list(old_list):
  """ Returns single list, from 3rd element in each list of list """
  new_list = []
  for ls in old_list:
    new_list.append(ls[2])
  return new_list

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Convert for scorer.py")
  parser.add_argument('path_from', help='original file')
  parser.add_argument('path_to', help='new file name')
  args = parser.parse_args()
  convert(args.path_from, args.path_to)
