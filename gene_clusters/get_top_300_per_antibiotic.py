import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import os
import pandas as pd
from constants import Bacteria

# PARAMS
BACTERIA = Bacteria.PSEUDOMONAS_AUREGINOSA.value if len(sys.argv) <= 1 else sys.argv[1]
# PARAMS END


results_folder_name = "2020_12_04_1808_GENE_CLUSTERS_FS_500"
top_x_features = 300

prefix = '..' if os.name == 'nt' else '.'
summary_gene_files_path = os.path.join(prefix, "results_files", BACTERIA, "summary_gene_files")
classic_ml_results_path = os.path.join(prefix, "results_files", BACTERIA, "classic_ml_results", results_folder_name)


clusters_df = pd.read_csv(os.path.join(summary_gene_files_path, "CLUSTERS_DATA.csv"))
clusters_df = clusters_df[['cluster_ind', 'accessory_cluster']]

antibiotic_list = ['imipenem', 'ceftazidime', 'meropenem', 'levofloxacin', 'amikacin']
# antibiotic_list = ['imipenem']

for antibiotic in antibiotic_list:
    importance_df = pd.read_csv(os.path.join(classic_ml_results_path, f"{antibiotic}_FS_IMPORTANCE_{results_folder_name}.csv"))
    importance_df = importance_df.rename(columns={'Unnamed: 0': 'cluster_ind'})
    importance_df = importance_df.iloc[:top_x_features]
    clusters_list = list(importance_df['cluster_ind'])
    importance_df = importance_df.merge(clusters_df, left_on='cluster_ind', right_on='cluster_ind', how='inner')
    importance_df.to_csv(os.path.join(summary_gene_files_path, f"{antibiotic}_IMPORTANCE.csv"), index=False)
    print(f"antibiotic: {antibiotic} have {importance_df['accessory_cluster'].sum()} accessory genes out of the top {importance_df.shape[0]} genes")

    seq_list = []
    with open(os.path.join(summary_gene_files_path, "clusters_protein.fasta")) as cluster_file:
        for line in cluster_file:
            if line.startswith(">"):  # new cluster
                header = line.replace("\n", "")
                cluster_ind = int(line.split("_")[1])
            else:
                protein_seq = line.replace("\n", "")
                if cluster_ind in clusters_list:
                    seq_list.append(f"{header}\n{protein_seq}")

    with open(os.path.join(summary_gene_files_path, f"{antibiotic}_top_300.fasta"), "w") as file:
        file.write("\n".join(seq_list))
    print("DONE!")
