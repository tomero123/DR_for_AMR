import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import os
from pathos.multiprocessing import ProcessPool

from gene_clusters.gene_clusters_utils import create_merged_file
from constants import FileType, FILES_SUFFIX, Bacteria


# PARAMS
BACTERIA = Bacteria.PSEUDOMONAS_AUREGINOSA.value if len(sys.argv) <= 1 else sys.argv[1]
NUM_OF_PROCESSES = 1 if len(sys.argv) <= 2 else int(sys.argv[2])

prefix = '..' if os.name == 'nt' else '.'
input_folder_base = os.path.join(prefix, "results_files", BACTERIA)
output_folder = os.path.join(prefix, "results_files", BACTERIA, "combined_genes_files")
# PARAMS END

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

file_types = [FileType.FEATURE_TABLE.value, FileType.CDS_FROM_GENOMIC.value, FileType.PROTEIN.value]
file_names_dic = {}

for file_type in file_types:
    file_names_dic[file_type] = os.listdir(os.path.join(input_folder_base, f"{file_type}_files"))

file_names_combined = [x.replace(f"_{FileType.FEATURE_TABLE.value}{FILES_SUFFIX.get(FileType.FEATURE_TABLE.value)}", "") for x in file_names_dic[FileType.FEATURE_TABLE.value]]

# for file in file_names_combined:
#     create_merged_file(file, input_folder_base, output_folder)

output_files_list = os.listdir(output_folder)
output_files_list = [x for x in output_files_list if ".csv.gz" in x]

input_list = []
for ind, file_name in enumerate(file_names_combined):
    if file_name + ".csv.gz" not in output_files_list:
        input_list.append([ind, file_name, input_folder_base, output_folder])

print("Start processing {} files".format(len(input_list)))
if NUM_OF_PROCESSES > 1:
    pool = ProcessPool(processes=NUM_OF_PROCESSES)
    pool.map(create_merged_file, input_list)
else:
    status_list = []
    for i in input_list:
        create_merged_file(i)
print("DONE!")
