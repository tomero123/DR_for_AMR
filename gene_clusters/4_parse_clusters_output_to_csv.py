import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import os
import pandas as pd
from collections import Counter
import json

from constants import Bacteria

# PARAMS
BACTERIA = Bacteria.PSEUDOMONAS_AUREGINOSA.value if len(sys.argv) <= 1 else sys.argv[1]

if os.name == 'nt':
    cluster_output_file_name = "output70.txt.clstr"
else:
    cluster_output_file_name = "cd_hit_results.txt.clstr"

prefix = '..' if os.name == 'nt' else '.'
summary_gene_files_path = os.path.join(prefix, "results_files", BACTERIA, "summary_gene_files")
strains_df = pd.read_csv(os.path.join(summary_gene_files_path, "ALL_STRAINS.csv"))
accessory_max_ratio_th = 0.8
accessory_min_count_th = 2

clusters_dict = {}
with open(os.path.join(summary_gene_files_path, cluster_output_file_name)) as cluster_file:
    for line in cluster_file:
        if line.startswith(">"):  # new cluster
            cluster_ind = line.split()[1]
            clusters_dict.update({cluster_ind: []})
        else:
            strain_ind = int(line.split(">")[1].split("|")[0])
            gene_ind = int(line.split(">")[1].split("|")[1].split(".")[0])
            clusters_dict[cluster_ind].append(strain_ind)

clusters_dict_count = {cluster: Counter(strain) for cluster, strain in clusters_dict.items()}
print("Finished creating clusters_dict_count")
clusters_df_strains = pd.DataFrame(clusters_dict_count).transpose()
print("Finished creating clusters_df_strains")
clusters_df_strains = clusters_df_strains.sort_index(axis=1)
num_of_strains = clusters_df_strains.shape[1]

clusters_df = clusters_df_strains.copy(deep=True)
clusters_df["n_zero_genes"] = clusters_df_strains.isnull().sum(axis=1)
clusters_df = clusters_df.fillna(0)
one_gene = (clusters_df_strains == 1).sum(axis=1)
clusters_df["n_one_gene"] = one_gene
multiple_genes = (clusters_df_strains > 1).sum(axis=1)
clusters_df["n_multiple_genes"] = multiple_genes

clusters_df["ratio_zero_genes"] = clusters_df["n_zero_genes"] / num_of_strains
clusters_df["ratio_one_gene"] = clusters_df["n_one_gene"] / num_of_strains
clusters_df["ratio_multiple_genes"] = clusters_df["n_multiple_genes"] / num_of_strains

clusters_df["accessory_cluster"] = (clusters_df["ratio_one_gene"] + clusters_df["ratio_multiple_genes"] < accessory_max_ratio_th) & \
                                   (clusters_df["n_one_gene"] + clusters_df["n_multiple_genes"] >= accessory_min_count_th)

# Write clusters_df to csv
clusters_df.index.set_names(['cluster_ind'], inplace=True)
clusters_df = clusters_df.reset_index()
clusters_df.to_csv(os.path.join(summary_gene_files_path, "CLUSTERS_DATA_COUNTS.csv"), index=False)

print("Finished Writing clusters_df to csv")

# calculate dicts for mapping of strains and their genes to clusters
accessory_clusters_ind = list(clusters_df["cluster_ind"][clusters_df["accessory_cluster"]])  # list of accessory clusters ind
strains_genes_dict = {}
strains_genes_dict_accessory = {}
with open(os.path.join(summary_gene_files_path, cluster_output_file_name)) as cluster_file:
    for line in cluster_file:
        if line.startswith(">"):  # new cluster
            cluster_ind = line.split()[1]
            clusters_dict.update({cluster_ind: []}) # new cluster
        else:
            strain_ind = int(line.split(">")[1].split("|")[0])
            gene_ind = int(line.split(">")[1].split("|")[1].split(".")[0])
            strains_genes_dict.setdefault(strain_ind, {})
            strains_genes_dict[strain_ind].setdefault(cluster_ind, [])
            strains_genes_dict[strain_ind][cluster_ind].append(gene_ind)
            # Add only genes of strains belonging to accessory clusters
            if cluster_ind in accessory_clusters_ind:
                strains_genes_dict_accessory.setdefault(strain_ind, {})
                strains_genes_dict_accessory[strain_ind].setdefault(cluster_ind, [])
                strains_genes_dict_accessory[strain_ind][cluster_ind].append(gene_ind)

print("Finished creating strains_genes_dict and strains_genes_dict_accessory")
# Write strains_genes_dict to json
with open(os.path.join(summary_gene_files_path, "STRAINS_GENES_DICT.json"), "w") as write_file:
    json.dump(strains_genes_dict, write_file)

# Write strains_genes_dict_accessory to json
with open(os.path.join(summary_gene_files_path, "STRAINS_GENES_DICT_ACCESSORY.json"), "w") as write_file:
    json.dump(strains_genes_dict_accessory, write_file)

print("DONE!")
