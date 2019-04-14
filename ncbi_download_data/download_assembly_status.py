import pandas as pd
from pathos.multiprocessing import ProcessPool
from ncbi_download_data.ftp_download_utils import open_ftp_file

# PARAMS
NUM_OF_PROCESSES = 8
DEST_PATH = "C:/University 2nd degree/Thesis/Pseudomonas Aureginosa data/"
CSV_FILE_PATH = "../data_files/genomes_proks.csv"
NCBI_FTP_SITE = "ftp.ncbi.nlm.nih.gov"
FTP_FILE_NAME = "assembly_status.txt"
# PARAMS END


if __name__ == '__main__':
    df = pd.read_csv(CSV_FILE_PATH)
    n_rows = df.shape[0]
    input_list = []
    for ind in range(n_rows):
        folder_ind = df["RefSeq FTP"][ind].find("/genomes")
        ftp_sub_folder = df["RefSeq FTP"][ind][folder_ind:]
        strain_name = df["Strain"][ind]
        input_list.append([ind, ftp_sub_folder, strain_name, FTP_FILE_NAME, NCBI_FTP_SITE])
    # input_list = input_list[0:20]
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
    df.to_csv("../data_files/pa_data.csv", index=False)
