import os, sys
import random
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import zipfile
import pickle

from proto import perform_payment

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', default="ECL", type=str)
    args = parser.parse_args()
else:
     sys.exit()


base_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(base_dir, 
                                        "..", 
                                        "data"))
results_dir = os.path.abspath(os.path.join(base_dir, 
                                           "..",
                                           "results"))
os.makedirs(data_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
print('data_dir:', data_dir)
print('results_dir:', results_dir)


g = nx.read_gml(zipfile.ZipFile(os.path.join(data_dir,
                                    'snapshot.gml.geo.zip'), 'r')\
                                        .open('snapshot.gml.geo'))

print(g)

with open(os.path.join(data_dir, 
                            'txs.pkl'), 'rb') as f:
    
    txs = pickle.load(f)
    print(len(txs), 'transactions loaded')


random.seed(1313)
np.random.seed(1313)


def save_results(results):
    with open(os.path.join(results_dir, 
                                f'{args.proto}.pkl'), 'wb') as f:
        pickle.dump(results, f)  


results = []
for i, (u, v, amount) in enumerate(tqdm(txs[:100])): #[:100]
    r = perform_payment(g, u, v, amount, 
                               proto_type=args.proto,
                               max_count=5,
                               timeout=2)
    results.append(r)

    if i % 100 == 0:
        save_results(results)

save_results(results)

os._exit(0)
    