import sys
sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

from pathos.multiprocessing import ProcessPool
import gzip
from functools import partial
import os

from utils import create_kmers_file

# PARAMS
BACTERIA = "mycobacterium_tuberculosis" if len(sys.argv) < 2 else sys.argv[1]
NUM_OF_PROCESSES = 8
K = 10 if len(sys.argv) < 3 else int(sys.argv[2])  # Choose K size

prefix = '..' if os.name == 'nt' else '.'
input_folder = os.path.join(prefix, "results_files", BACTERIA, "genome_files")
output_folder = os.path.join(prefix, "results_files", BACTERIA, "kmers_files", f"K_{K}")
# PARAMS END

files_list = os.listdir(input_folder)
files_list = [x for x in files_list if ".fna.gz" in x]
_open = partial(gzip.open, mode='rt')


if __name__ == '__main__':
    print(f"Start running on bacteria: {BACTERIA} with K={K}")
    input_list = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_files_list = os.listdir(output_folder)
    output_files_list = [x for x in output_files_list if ".txt.gz" in x]
    for ind, file_name in enumerate(files_list):
        if file_name.replace(".fna", ".txt") not in output_files_list:
            input_list.append([ind, file_name, K, input_folder, output_folder])
    # input_list = input_list[0:10]
    print("Start processing {} files".format(len(input_list)))
    if NUM_OF_PROCESSES > 1:
        pool = ProcessPool(processes=NUM_OF_PROCESSES)
        pool.map(create_kmers_file, input_list)
    else:
        status_list = []
        for i in input_list:
            create_kmers_file(i)
    print("DONE!")
