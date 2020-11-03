import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import os
import pandas as pd
from tqdm import tqdm

from constants import Bacteria


# PARAMS
BACTERIA = Bacteria.PSEUDOMONAS_AUREGINOSA.value if len(sys.argv) <= 1 else sys.argv[1]
NUM_OF_PROCESSES = 1 if len(sys.argv) <= 2 else int(sys.argv[2])

prefix = '..' if os.name == 'nt' else '.'
input_folder_path = os.path.join(prefix, "results_files", BACTERIA, "combined_genes_files")
output_folder = os.path.join(prefix, "results_files", BACTERIA, "summary_gene_files")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

files_list = os.listdir(input_folder_path)
files_list = [x for x in files_list if ".csv.gz" in x]
print(f"Number of combined files: {len(files_list)}")

results_df = pd.DataFrame(columns=["file_name", "file", "number_of_genes"])

for file in tqdm(files_list):
    df = pd.read_csv(os.path.join(input_folder_path, file))
    results_df.loc[len(results_df)] = [file, file.replace(".csv.gz", ""), df.shape[0]]

results_df = results_df.reset_index()
results_df.to_csv(os.path.join(output_folder, "ALL_STRAINS.csv"), index=False)
print("DONE!")
