import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import os
import pandas as pd
from tqdm import tqdm
from collections import Counter

from constants import Bacteria

# PARAMS
BACTERIA = Bacteria.PSEUDOMONAS_AUREGINOSA.value if len(sys.argv) <= 1 else sys.argv[1]
cluster_output_file_name = "output70.txt.clstr"

prefix = '..' if os.name == 'nt' else '.'
summary_gene_files_path = os.path.join(prefix, "results_files", BACTERIA, "summary_gene_files")
strains_df = pd.read_csv(os.path.join(summary_gene_files_path, "ALL_STRAINS.csv"))
accessory_max_ratio_th = 0.8
accessory_min_count_th = 5

clusters_dict = {}
with open(os.path.join(summary_gene_files_path, cluster_output_file_name)) as cluster_file:
    for line in tqdm(cluster_file):
        if line.startswith(">"):  # new cluster
            cluster = line.split()[1]
            clusters_dict.update({cluster: []})
        else:
            strain = line.split(">")[1].split("|")[0]
            strain = int(strain)
            clusters_dict[cluster].append(strain)

clusters_dict_count = {cluster: Counter(strain) for cluster, strain in clusters_dict.items()}
clusters_df_strains = pd.DataFrame(clusters_dict_count).transpose()
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
# clusters_df = clusters_df.append(pd.Series(name="num_of_core_clusters"))
# clusters_df = clusters_df.append(pd.Series(name="single_copy_core_clusters"))

# for strain in clusters_df_strains:
#     strain_df = clusters_df.loc[:, [strain, "core_cluster"]]
#     core_df = strain_df[(strain_df[strain] > 0) & (strain_df["core_cluster"] > 0)]
#     clusters_df.loc["num_of_core_clusters", strain] = core_df[strain].count()
#
#     single_copy_df = strain_df[(strain_df[strain] == 1) & (strain_df["core_cluster"] > 0)]
#     clusters_df.loc["single_copy_core_clusters", strain] = single_copy_df[strain].count()

clusters_df.index.set_names(['cluster_ind'], inplace=True)
clusters_df.reset_index().to_csv(os.path.join(summary_gene_files_path, "CLUSTERS_DATA.csv"), index=False)
print("DONE!")
