import argparse

parser = argparse.ArgumentParser(description='Generates the necessary data and structures for a jupyter DeepNotebook')

parser.add_argument('data_loc', type=str)
parser.add_argument('-n', '--name')
parser.add_argument('-plt', '--precompute-plots', action='store_true')
parser.add_argument('-data', '--precompute-data', action='store_true')

if __name__ == "__main__":
