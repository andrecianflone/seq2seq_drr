"""
python result_analysis.py trials/pdtb/cell_units one_v_all --metric val_f1
"""
from pprint import pprint
import json
import codecs
import argparse

def best(results, dataset, metric):
  """ Returns the best for each relation type based on metric """
  best = {}
  for r in results:
    # Check task
    if "dataset_name" not in r["params"]:
      continue
    if r["params"]["dataset_name"] != dataset:
      continue

    relation = r["params"]["relation"]

    # Skip if line does not have metric
    if metric not in r["metrics"]:
      continue
    score = float(r["metrics"][metric])
    if relation in best:
      if score > float(best[relation]["metrics"][metric]):
        best[relation] = r
    else:
      best[relation] = r
  return best

def graph(results, param):
  """ y as f1, x as varying parameter """
  pass

def load(filepath):
  """ Returns list of dictionaries from file """
  results = []
  with open(filepath, encoding='utf8') as pdfile:
    for line in pdfile:
      j = json.loads(line)
      results += [j]
  return results

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="analyze results")
  parser.add_argument('filepath', help='file path of results')
  parser.add_argument('dataset', help='"one_v_all" or "conll"')
  parser.add_argument('--metric', help='metric to check for best result',
      default='val_f1')
  args = parser.parse_args()
  results = load(args.filepath)
  print('-' * 80)
  print('Printing the best results based on metric: ', args.metric)
  print('-' * 80)
  pprint(best(results, args.dataset, args.metric))



