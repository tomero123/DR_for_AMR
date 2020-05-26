import sys
sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import pandas as pd
from pathos.multiprocessing import ProcessPool
import os

from utils import download_ftp_file

# PARAMS
BACTERIA = "genome_mix" if len(sys.argv) < 2 else sys.argv[1]
NUM_OF_PROCESSES = 10 if len(sys.argv) < 3 else int(sys.argv[2])
limit = None  # if None - take all files found else limit

prefix = '..' if os.name == 'nt' else '.'
DEST_PATH = os.path.join(prefix, "results_files", BACTERIA, "genome_files")
CSV_FILE_PATH = os.path.join(prefix, "data_files", BACTERIA, "{}_data.csv".format(BACTERIA))
NCBI_FTP_SITE = "ftp.ncbi.nlm.nih.gov"
# PARAMS END


if __name__ == '__main__':
    df = pd.read_csv(CSV_FILE_PATH)
    # ignore all rows where status is not 'latest'
    df = df[df['status'] == 'latest'].reset_index()
    n_rows = df.shape[0]
    input_list = []
    if not os.path.exists(DEST_PATH):
        os.makedirs(DEST_PATH)
    files_list = os.listdir(DEST_PATH)
    files_list = [x for x in files_list if ".fna.gz" in x]
    for ind in range(n_rows):
        try:
            folder_ind = df["RefSeq FTP"][ind].find("/genomes")
        except Exception as e:
            continue
        ftp_sub_folder = df["RefSeq FTP"][ind][folder_ind:]
        strain_name = df["Strain"][ind]
        ftp_file_name = ftp_sub_folder.split("/")[-1] + "_genomic.fna.gz"
        if ftp_file_name not in files_list:
            input_list.append([ind, ftp_sub_folder, strain_name, ftp_file_name, NCBI_FTP_SITE, DEST_PATH])
    if limit is not None:
        input_list = input_list[:limit]
    print("Start downloading {} files".format(len(input_list)))
    if NUM_OF_PROCESSES > 1:
        pool = ProcessPool(processes=NUM_OF_PROCESSES)
        pool.map(download_ftp_file, input_list)
    else:
        status_list = []
        for i in input_list:
            download_ftp_file(i)
    print("DONE!")