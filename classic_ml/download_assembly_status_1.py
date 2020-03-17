import sys
sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")

import os
import pandas as pd
from pathos.multiprocessing import ProcessPool

from classic_ml.utils import open_ftp_file

# PARAMS

BACTERIA = "mycobacterium_tuberculosis" if len(sys.argv) < 2 else sys.argv[1]
NUM_OF_PROCESSES = 1
limit = None  # if None - take all files found else limit
input_file_name = "mycobacterium_tuberculosis_data.csv"
output_file_name = "{}_data.csv".format(BACTERIA)

########################################################################

NCBI_FTP_SITE = "ftp.ncbi.nlm.nih.gov"
FTP_FILE_NAME = "assembly_status.txt"

PREFIX = '..' if os.name == 'nt' else '.'
PATH = os.path.join(PREFIX, "data_files", BACTERIA)
CSV_INPUT_FILE_PATH = os.path.join(PATH, input_file_name)
# CSV_OUTPUT_FILE_PATH = os.path.join(PATH, output_file_name)

# PARAMS END

if __name__ == '__main__':
    df = pd.read_csv(CSV_INPUT_FILE_PATH)
    if 'status' not in df:
        df['strain_validation'] = "NA"
        df['status'] = "NA"
    n_rows = df.shape[0]
    print(f"Loaded csv file with {n_rows} rows!")
    input_list = []
    for ind in range(n_rows):
        folder_ind = df["RefSeq FTP"][ind].find("/genomes")
        ftp_sub_folder = df["RefSeq FTP"][ind][folder_ind:]
        strain_name = df["Strain"][ind]
        status = df["status"][ind]
        input_list.append([ind, ftp_sub_folder, strain_name, FTP_FILE_NAME, NCBI_FTP_SITE, status])
    if limit is not None:
        input_list = input_list[:limit]
    if NUM_OF_PROCESSES > 1:
        pool = ProcessPool(processes=NUM_OF_PROCESSES)
        status_list = pool.map(open_ftp_file, input_list)
    else:
        status_list = []
        for ind, ftp_file in enumerate(input_list):
            cur_status_list = open_ftp_file(ftp_file)
            cur_strain_validation = cur_status_list[0]
            cur_status = cur_status_list[1].replace("status=", "")
            df.at[ind, 'strain_validation'] = cur_strain_validation
            df.at[ind, 'status'] = cur_status
            if ind % 10 == 0:
                df.to_csv(CSV_INPUT_FILE_PATH, index=False)
                print(f"Saved DF after {ind} iterations")
    # print(status_list)
    df.to_csv(CSV_INPUT_FILE_PATH, index=False)
    print("DONE!")
