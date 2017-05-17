from pprint import pprint
import json
import codecs
import argparse

def best(results, metric):
  """ Returns the best for each relation type based on metric """
  best = {}
  for r in results:
    relation = r["params"]["relation"]
    f1 = float(r["metrics"][metric])
    if relation in best:
      if f1 > float(best[relation]["metrics"][metric]):
        best[relation] = r
    else:
      best[relation] = r
  return best

def graph(results, param):
  """ y as f1, x as varying parameter """
  pass

def load(filepath):
  results = []
  with open(filepath, encoding='utf8') as pdfile:
    for line in pdfile:
      j = json.loads(line)
      results += [j]
  return results

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="analyze results")
  parser.add_argument('filepath', help='file path of results')
  parser.add_argument('--metric', help='metric to check for best result')
  args = parser.parse_args()
  results = load(args.filepath)
  pprint(best(results, args.metric))



