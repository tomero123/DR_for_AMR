from pathos.multiprocessing import ProcessPool
import gzip
from functools import partial
import os
from ncbi_download_data.ncbi_utils import create_kmers_file

# PARAMS
NUM_OF_PROCESSES = 8
K = 10  # Choose K size
path = "C:/University 2nd degree/Thesis/Pseudomonas Aureginosa data/"
input_folder = "genome_files/"
output_folder = "kmers_files/"
# PARAMS END

files_list = os.listdir(path + input_folder)
files_list = [x for x in files_list if ".fna.gz" in x]
_open = partial(gzip.open, mode='rt')


if __name__ == '__main__':
    input_list = []
    for ind, file_name in enumerate(files_list):
        input_list.append([ind, file_name, K, path, input_folder, output_folder])
    # input_list = input_list[0:10]
    if NUM_OF_PROCESSES > 1:
        pool = ProcessPool(processes=NUM_OF_PROCESSES)
        pool.map(create_kmers_file, input_list)
    else:
        status_list = []
        for i in input_list:
            create_kmers_file(i)
