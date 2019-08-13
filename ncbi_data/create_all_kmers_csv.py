import gzip
import json
import os
import pandas as pd

kmers_file_name = 'all_kmers_file.txt.gz'
kmers_map_file_name = 'all_kmers_map.txt'

if os.name == 'nt':
    path = os.path.join('..', 'results_files')
else:
    path = os.path.join('.', 'results_files')

with gzip.open(os.path.join(path, kmers_file_name), 'rt') as f:
    all_kmers_dic = json.loads(f.read())

with open(os.path.join(path, kmers_map_file_name), 'r') as f:
    all_kmers_map = json.loads(f.read())

df = pd.DataFrame({key: pd.Series(val) for key, val in all_kmers_dic.items()})
df = df.T
df.to_csv(os.path.join(path, kmers_file_name.replace("txt.gz", "csv.gz")), compression="gzip")
