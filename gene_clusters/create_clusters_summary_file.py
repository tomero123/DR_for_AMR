import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import os
import pandas as pd
from constants import Bacteria

# PARAMS
BACTERIA = Bacteria.PSEUDOMONAS_AUREGINOSA.value if len(sys.argv) <= 1 else sys.argv[1]
# PARAMS END


prefix = '..' if os.name == 'nt' else '.'
summary_gene_files_path = os.path.join(prefix, "results_files", BACTERIA, "summary_gene_files")
combined_files_path = os.path.join(prefix, "results_files", BACTERIA, "combined_genes_files")


strains_df = pd.read_csv(os.path.join(summary_gene_files_path, "ALL_STRAINS.csv"))
clusters_df = pd.read_csv(os.path.join(summary_gene_files_path, "CLUSTERS_DATA.csv"))

cluster_output_file_name = "cd_hit_results.txt.clstr"

# Get accessory information
clusters_df = clusters_df[['cluster_ind', 'accessory_cluster']]

# Get primary gene name

locus_tag_list = []
gene_name_list = []

with open(os.path.join(summary_gene_files_path, cluster_output_file_name)) as cluster_file:
    for line in cluster_file:
        if line.startswith(">"):  # new cluster
            cluster_ind = line.split()[1]
            if int(cluster_ind) % 100 == 0:
                print(f"Started processing cluster_ind: {cluster_ind}")
        else:
            is_primary = True if "*" in line else False
            if is_primary:
                strain_ind = int(line.split(">")[1].split("|")[0])
                gene_ind = int(line.split(">")[1].split("|")[1].split(".")[0])
                file_name = strains_df[strains_df["index"] == strain_ind]['file_name'].values[0]
                genes_df = pd.read_csv(os.path.join(combined_files_path, file_name))
                locus_tag = genes_df[genes_df['Unnamed: 0'] == gene_ind]['locus_tag'].values[0]
                name_y = genes_df[genes_df['Unnamed: 0'] == gene_ind]['name_y'].values[0]

clusters_df['gene_name'] = gene_name_list
clusters_df['locus_tag'] = locus_tag_list

clusters_df.to_csv(os.path.join(summary_gene_files_path, "CLUSTERS_SUMMARY.csv"), index=False)
