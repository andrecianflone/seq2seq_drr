
import json
import codecs

def best(filepath):
  """ Returns the best f1 for each relation type """
  best = {}
  with open(filepath, encoding='utf8') as pdfile:
    for line in pdfile:
      d

def graph(results, param):
  """ y as f1, x as varying parameter """
  pass

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="analyze results")
  parser.add_argument('filepath', help='Hyperparam search over this param')
  args = parser.parse_args()
  best(args.filepath)
