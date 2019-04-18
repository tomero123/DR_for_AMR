import sys
sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")

import pandas as pd
from pathos.multiprocessing import ProcessPool
import os

from ncbi_data.ncbi_utils import download_ftp_file

# PARAMS
NUM_OF_PROCESSES = 8
DEST_PATH = "../results_files/genome_files/"
CSV_FILE_PATH = "../data_files/pa_data.csv"
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
    for ind in range(n_rows):
        folder_ind = df["RefSeq FTP"][ind].find("/genomes")
        ftp_sub_folder = df["RefSeq FTP"][ind][folder_ind:]
        strain_name = df["Strain"][ind]
        ftp_file_name = ftp_sub_folder.split("/")[-1] + "_genomic.fna.gz"
        input_list.append([ind, ftp_sub_folder, strain_name, ftp_file_name, NCBI_FTP_SITE, DEST_PATH])
    # input_list = input_list[0:10]
    if NUM_OF_PROCESSES > 1:
        pool = ProcessPool(processes=NUM_OF_PROCESSES)
        pool.map(download_ftp_file, input_list)
    else:
        status_list = []
        for i in input_list:
            download_ftp_file(i)
