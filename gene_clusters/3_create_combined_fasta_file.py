import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import os
import pandas as pd
from tqdm import tqdm

from constants import Bacteria

# PARAMS
BACTERIA = Bacteria.PSEUDOMONAS_AUREGINOSA.value if len(sys.argv) <= 1 else sys.argv[1]

prefix = '..' if os.name == 'nt' else '.'
summary_gene_files_path = os.path.join(prefix, "results_files", BACTERIA, "summary_gene_files")
combined_files_path = os.path.join(prefix, "results_files", BACTERIA, "combined_genes_files")

seq_list = []

strains_df = pd.read_csv(os.path.join(summary_gene_files_path, "ALL_STRAINS.csv"))
for _, row in tqdm(strains_df.iterrows(), total=strains_df.shape[0]):
    file_name = row['file_name']
    strain_index = row['index']
    genes_df = pd.read_csv(os.path.join(combined_files_path, file_name))
    for seq_index, row in genes_df.iterrows():
        header = str(strain_index) + "|" + str(seq_index)
        protein = row["protein"]
        seq_list.append(f">{header}\n{protein}")

with open(os.path.join(summary_gene_files_path, "protein_combined.fasta"), "w") as file:
    file.write("\n".join(seq_list))
print("DONE!")
