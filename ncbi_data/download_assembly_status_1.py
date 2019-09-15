import sys
sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")

import os
import pandas as pd
from pathos.multiprocessing import ProcessPool
from ncbi_data.ncbi_utils import open_ftp_file

# PARAMS

BACTERIA = "mycobacterium_tuberculosis"
NUM_OF_PROCESSES = 1
limit = None  # if None - take all files found else limit
input_file_name = "genomes_proks166.csv"
output_file_name = "{}_data.csv".format(BACTERIA)

########################################################################

NCBI_FTP_SITE = "ftp.ncbi.nlm.nih.gov"
FTP_FILE_NAME = "assembly_status.txt"

PREFIX = '..' if os.name == 'nt' else '.'
PATH = os.path.join(PREFIX, "data_files", BACTERIA)
CSV_INPUT_FILE_PATH = os.path.join(PATH, input_file_name)
CSV_OUTPUT_FILE_PATH = os.path.join(PATH, output_file_name)

# PARAMS END

if __name__ == '__main__':
    df = pd.read_csv(CSV_INPUT_FILE_PATH)
    n_rows = df.shape[0]
    input_list = []
    for ind in range(n_rows):
        folder_ind = df["RefSeq FTP"][ind].find("/genomes")
        ftp_sub_folder = df["RefSeq FTP"][ind][folder_ind:]
        strain_name = df["Strain"][ind]
        input_list.append([ind, ftp_sub_folder, strain_name, FTP_FILE_NAME, NCBI_FTP_SITE])
    if limit is not None:
        input_list = input_list[:limit]
    if NUM_OF_PROCESSES > 1:
        pool = ProcessPool(processes=NUM_OF_PROCESSES)
        status_list = pool.map(open_ftp_file, input_list)
    else:
        status_list = []
        for i in input_list:
            status_list.append(open_ftp_file(i))
    # print(status_list)
    status_df = pd.DataFrame({"strain_validation": [x[0] for x in status_list], "status": [x[1].replace("status=", "") for x in status_list]})
    df = pd.concat([df, status_df], axis=1)
    df.to_csv(CSV_OUTPUT_FILE_PATH, index=False)
    print("DONE!")
