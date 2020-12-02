import sys
sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

from pathos.multiprocessing import ProcessPool
import gzip
from functools import partial
import os

from utils import create_kmers_file
from constants import Bacteria, FileType, FILES_SUFFIX

# PARAMS
BACTERIA = Bacteria.PSEUDOMONAS_AUREGINOSA.value if len(sys.argv) <= 1 else sys.argv[1]
FILE_TYPE = FileType.PROTEIN.value if len(sys.argv) <= 2 else sys.argv[2]
K = 10 if len(sys.argv) <= 3 else int(sys.argv[3])  # Choose K size
NUM_OF_PROCESSES = 1 if len(sys.argv) <= 4 else int(sys.argv[4])
# PARAMS END

prefix = '..' if os.name == 'nt' else '.'
input_folder = os.path.join(prefix, "results_files", BACTERIA, FILE_TYPE + "_files")
output_folder = os.path.join(prefix, "results_files", BACTERIA, FILE_TYPE + "_kmers_files", f"K_{K}")
file_suffix = FILES_SUFFIX.get(FILE_TYPE)


files_list = os.listdir(input_folder)
files_list = [x for x in files_list if file_suffix in x]
_open = partial(gzip.open, mode='rt')


if __name__ == '__main__':
    print(f"Start running on bacteria: {BACTERIA} with K={K}")
    input_list = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_files_list = os.listdir(output_folder)
    output_files_list = [x for x in output_files_list if ".txt.gz" in x]
    for ind, file_name in enumerate(files_list):
        if file_name.replace(file_suffix, ".txt.gz") not in output_files_list:
            input_list.append([ind, file_name, K, input_folder, output_folder, FILE_TYPE, file_suffix])
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
