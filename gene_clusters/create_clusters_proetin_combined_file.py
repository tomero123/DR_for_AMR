import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import os
import pandas as pd
from tqdm import tqdm

from constants import Bacteria

# PARAMS
BACTERIA = Bacteria.PSEUDOMONAS_AUREGINOSA.value if len(sys.argv) <= 1 else sys.argv[1]

if os.name == 'nt':
    cluster_output_file_name = "output70.txt.clstr"
else:
    cluster_output_file_name = "cd_hit_results.txt.clstr"


try:
    prefix = '..'
    summary_gene_files_path = os.path.join(prefix, "results_files", BACTERIA, "summary_gene_files")
    combined_files_path = os.path.join(prefix, "results_files", BACTERIA, "combined_genes_files")
except:
    prefix = '.'
    summary_gene_files_path = os.path.join(prefix, "results_files", BACTERIA, "summary_gene_files")
    combined_files_path = os.path.join(prefix, "results_files", BACTERIA, "combined_genes_files")

seq_list = []

strains_df = pd.read_csv(os.path.join(summary_gene_files_path, "ALL_STRAINS.csv"))
with open(os.path.join(summary_gene_files_path, cluster_output_file_name)) as cluster_file:
    for line in cluster_file:
        if line.startswith(">"):  # new cluster
            cluster_ind = line.split()[1]
        else:
            is_primary = True if "*" in line else False
            if is_primary:
                strain_ind = int(line.split(">")[1].split("|")[0])
                gene_ind = int(line.split(">")[1].split("|")[1].split(".")[0])
                file_name = strains_df[strains_df["index"] == strain_ind]['file_name'].values[0]
                genes_df = pd.read_csv(os.path.join(combined_files_path, file_name))
                protein_seq = genes_df[genes_df['Unnamed: 0'] == gene_ind]['protein'].values[0]
                header = f"cluster_{cluster_ind}"
                seq_list.append(f">{header}\n{protein_seq}")
                if int(cluster_ind) % 100 == 0:
                    print(f"Finished processing cluster: {cluster_ind}")

with open(os.path.join(summary_gene_files_path, "clusters_protein.fasta"), "w") as file:
    file.write("\n".join(seq_list))
print("DONE!")
