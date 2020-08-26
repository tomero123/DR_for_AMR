import sys
sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import os
from pathos.multiprocessing import ProcessPool
import gzip
from functools import partial

from utils import create_genome_document
from enums import Bacteria, ProcessingMode

# PARAMS
BACTERIA = Bacteria.TEST.value if len(sys.argv) <= 1 else sys.argv[1]
PROCESSING_MODE = ProcessingMode.OVERLAPPING.value if len(sys.argv) <= 2 else sys.argv[2]  # can be "non_overlapping" or "overlapping"
K = 10 if len(sys.argv) <= 3 else int(sys.argv[3])  # Choose K size
NUM_OF_PROCESSES = 1 if len(sys.argv) <= 4 else int(sys.argv[4])
SHIFT_SIZE = 1 if len(sys.argv) <= 5 else int(sys.argv[5]) # relevant only for PROCESSING_MODE "overlapping"

prefix = '..' if os.name == 'nt' else '.'
input_folder = os.path.join(prefix, "results_files", BACTERIA, "genome_files")
if PROCESSING_MODE == "overlapping":
    output_folder = os.path.join(prefix, "results_files", BACTERIA, "genome_documents", f"overlapping_{SHIFT_SIZE}", f"K_{K}")
elif PROCESSING_MODE == "non_overlapping":
    output_folder = os.path.join(prefix, "results_files", BACTERIA, "genome_documents", "non_overlapping", f"K_{K}")
files_list = os.listdir(input_folder)
files_list = [x for x in files_list if ".fna.gz" in x]
print(f"Total files in folder: {len(files_list)}")
_open = partial(gzip.open, mode='rt')

if __name__ == '__main__':
    print(f"Start running on bacteria: {BACTERIA} with K={K}")
    input_list = []
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_files_list = os.listdir(output_folder)
    output_files_list = [x for x in output_files_list if ".pkl" in x]
    for ind, file_name in enumerate(files_list):
        if file_name.replace(".fna.gz", ".pkl") not in output_files_list:
            input_list.append([ind, file_name, K, PROCESSING_MODE, SHIFT_SIZE, input_folder, output_folder])
    # input_list = input_list[0:10]
    print("Start processing {} files".format(len(input_list)))
    if NUM_OF_PROCESSES > 1:
        pool = ProcessPool(processes=NUM_OF_PROCESSES)
        pool.map(create_genome_document, input_list)
    else:
        status_list = []
        for i in input_list:
            create_genome_document(i)
    print("DONE!")
