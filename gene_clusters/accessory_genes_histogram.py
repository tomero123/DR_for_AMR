import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import os
import pandas as pd
from constants import Bacteria

# PARAMS
BACTERIA = Bacteria.PSEUDOMONAS_AUREGINOSA.value if len(sys.argv) <= 1 else sys.argv[1]
K = 10 if len(sys.argv) <= 2 else int(sys.argv[2])  # Choose K size
NUM_OF_PROCESSES = 1 if len(sys.argv) <= 3 else int(sys.argv[3])
# PARAMS END


prefix = '..' if os.name == 'nt' else '.'
summary_gene_files_path = os.path.join(prefix, "results_files", BACTERIA, "summary_gene_files")
strains_df = pd.read_csv(os.path.join(summary_gene_files_path, "ALL_STRAINS.csv"))
clusters_df = pd.read_csv(os.path.join(summary_gene_files_path, "CLUSTERS_DATA.csv"))

strains_columns = [x for x in clusters_df.columns if x.isdigit()]
accessory_df = clusters_df[clusters_df["accessory_cluster"]]
accessory_sum = accessory_df[strains_columns].sum()
accessory_sum.hist()

genes_sum = clusters_df[strains_columns].sum()
ratio = accessory_sum / genes_sum
