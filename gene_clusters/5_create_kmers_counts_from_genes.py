import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

from pathos.multiprocessing import ProcessPool
import os
import pandas as pd
from constants import Bacteria

from gene_clusters.gene_clusters_utils import create_kmers_from_combined_csv

# PARAMS
BACTERIA = Bacteria.PSEUDOMONAS_AUREGINOSA.value if len(sys.argv) <= 1 else sys.argv[1]
K = 10 if len(sys.argv) <= 2 else int(sys.argv[2])  # Choose K size
NUM_OF_PROCESSES = 1 if len(sys.argv) <= 3 else int(sys.argv[3])
limit = 1
# PARAMS END


prefix = '..' if os.name == 'nt' else '.'
input_combined_genes_path = os.path.join(prefix, "results_files", BACTERIA, "combined_genes_files")
output_all_genes_kmers = os.path.join(prefix, "results_files", BACTERIA, "all_genes_kmers_files", f"K_{K}")
output_accessory_genes_kmers = os.path.join(prefix, "results_files", BACTERIA, "accessory_genes_kmers_files", f"K_{K}")
output_accessory_cds_from_genomic_files = os.path.join(prefix, "results_files", BACTERIA, "accessory_cds_from_genomic_files")
summary_gene_files_path = os.path.join(prefix, "results_files", BACTERIA, "summary_gene_files")
strains_df = pd.read_csv(os.path.join(summary_gene_files_path, "ALL_STRAINS.csv"))


if __name__ == '__main__':
    print(f"Start running on bacteria: {BACTERIA} with K={K}")
    input_list = []
    if not os.path.exists(output_all_genes_kmers):
        os.makedirs(output_all_genes_kmers)
    output_all_genes_list = os.listdir(output_all_genes_kmers)
    output_all_genes_list = [x for x in output_all_genes_list if ".txt.gz" in x]

    if not os.path.exists(output_accessory_genes_kmers):
        os.makedirs(output_accessory_genes_kmers)
    output_accessory_genes_list = os.listdir(output_accessory_genes_kmers)
    output_accessory_genes_list = [x for x in output_accessory_genes_list if ".txt.gz" in x]

    if not os.path.exists(output_accessory_cds_from_genomic_files):
        os.makedirs(output_accessory_cds_from_genomic_files)

    if limit:
        strains_df = strains_df[:limit]

    for _, row in strains_df.iterrows():
        file_name = row['file_name']
        strain_index = row['index']
        file = row['file']
        if file_name.replace(".csv.gz", ".txt.gz") not in output_all_genes_list or \
           file_name.replace(".csv.gz", ".txt.gz") not in output_accessory_genes_list:
            input_list.append([strain_index, file, K, input_combined_genes_path, output_all_genes_kmers, output_accessory_genes_kmers, output_accessory_cds_from_genomic_files, summary_gene_files_path])
    # input_list = input_list[0:10]
    print("Start processing {} files".format(len(input_list)))
    if NUM_OF_PROCESSES > 1:
        pool = ProcessPool(processes=NUM_OF_PROCESSES)
        pool.map(create_kmers_from_combined_csv, input_list)
    else:
        status_list = []
        for i in input_list:
            create_kmers_from_combined_csv(i)
    print("DONE!")
