import gzip
import json
import os
import pandas as pd

# Config
dataset_file_name = 'all_kmers_file_SMALL.csv.gz'
amr_data_file_name = 'amr_data_summary.csv'
kmers_map_file_name = 'all_kmers_map_SMALL.txt'

antibiotic_for_test = 'amikacin'
ncbi_file_name_column = 'NCBI File Name'

# Config END
# *********************************************************************************************************************************

if os.name == 'nt':
    path = os.path.join('..', 'results_files')
else:
    path = os.path.join('.', 'results_files')

kmers_df = pd.read_csv(os.path.join(path, dataset_file_name), compression='gzip')
amr_df = pd.read_csv(os.path.join(path, amr_data_file_name))
with open(os.path.join(path, kmers_map_file_name), 'r') as f:
    all_kmers_map = json.loads(f.read())


# remove strains with no antibiotic data
ids_with_amr_data = [str(x) for x in list(amr_df['Index'])]
columns_to_leave = [x for x in list(kmers_df.columns) if x == 'Unnamed: 0' or x in ids_with_amr_data]
kmers_df = kmers_df[columns_to_leave]

# remove very common and very rare k-mers
rare_th = 1
common_th = kmers_df.shape[1] - 2
kmers_df = kmers_df[(kmers_df.astype(bool).sum(axis=1) > rare_th) & (kmers_df.astype(bool).sum(axis=1) < common_th)]

# replace columns names from index to 'NCBI File Name'
kmers_df = kmers_df.rename(columns=all_kmers_map)
kmers_df = kmers_df.set_index(['Unnamed: 0'])

# Transpose to have strains as rows and kmers as columns
kmers_df = kmers_df.T

# Get label of specific antibiotic
label_df = amr_df[[ncbi_file_name_column, antibiotic_for_test]]
label_df = label_df[label_df[antibiotic_for_test] != '-']
label_df = label_df.rename(columns={antibiotic_for_test: "label"})
label_df = label_df.set_index(['NCBI File Name'])

# Join (inner) between kmers_df and label_df
final_df = kmers_df.join(label_df, how="inner")
print("final_df have {} Strains with label and {} features".format(final_df.shape[0], final_df.shape[1]))

