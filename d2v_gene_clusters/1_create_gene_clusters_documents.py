import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import os
import pandas as pd
import pickle
from tqdm import tqdm

from constants import Bacteria

# PARAMS
BACTERIA = Bacteria.PSEUDOMONAS_AUREGINOSA.value if len(sys.argv) <= 1 else sys.argv[1]

prefix = '..' if os.name == 'nt' else '.'
summary_gene_files_path = os.path.join(prefix, "results_files", BACTERIA, "summary_gene_files")
output_gene_clusters_documents_path = os.path.join(prefix, "results_files", BACTERIA, "gene_clusters_documents")


if not os.path.exists(output_gene_clusters_documents_path):
    os.makedirs(output_gene_clusters_documents_path)

if os.name == 'nt':
    cluster_output_file_name = "output70.txt.clstr"
    all_strains_file = "ALL_STRAINS_SMALL.csv"
else:
    cluster_output_file_name = "cd_hit_results.txt.clstr"
    all_strains_file = "ALL_STRAINS.csv"

strains_genes_to_clusters_dict = {}
with open(os.path.join(summary_gene_files_path, cluster_output_file_name)) as cluster_file:
    for line in cluster_file:
        if line.startswith(">"):  # new cluster
            cluster_ind = int(line.split()[1])
        else:
            strain_ind = int(line.split(">")[1].split("|")[0])
            gene_ind = int(line.split(">")[1].split("|")[1].split(".")[0])
            strains_genes_to_clusters_dict.setdefault(strain_ind, {})
            strains_genes_to_clusters_dict[strain_ind][gene_ind] = cluster_ind

all_strains_df = pd.read_csv(os.path.join(summary_gene_files_path, all_strains_file))

for _, row in tqdm(all_strains_df.iterrows(), total=all_strains_df.shape[0]):
    strain_index = row['index']
    file = row['file']
    number_of_genes = row['number_of_genes']
    output_file_name = file + ".pickle"
    genes_to_clusters_dic = strains_genes_to_clusters_dict[strain_index]
    dic_len = len(genes_to_clusters_dic)
    if number_of_genes != dic_len:
        print(f"MISMATCH in genes len for strain_id: {strain_index}, strain name: {file}, number_of_genes: {number_of_genes}, genes_to_clusters_dic len: {dic_len}")
    clusters_list = []
    for gene_ind in range(dic_len):
        try:
            clusters_list.append(str(genes_to_clusters_dic[gene_ind]))
        except Exception as e:
            print(f"Couldn't find gene_ind: {gene_ind} for strain_id :{strain_index}")

    with open(os.path.join(output_gene_clusters_documents_path, output_file_name), 'wb') as f:
        pickle.dump(clusters_list, f, protocol=pickle.HIGHEST_PROTOCOL)

print("DONE!")
