import pandas as pd
from ftplib import FTP
from pathos.multiprocessing import ProcessPool

# PARAMS
NUM_OF_PROCESSES = 8
DEST_PATH = "C:/University 2nd degree/Thesis/Pseudomonas Aureginosa data/"
CSV_FILE_PATH = "./data_files/genomes_proks.csv"
NCBI_FTP_SITE = "ftp.ncbi.nlm.nih.gov"
ASSEMBLY_STATUS_FILE = "assembly_status.txt"
# PARAMS END


def download_assembly_status(input_list):
    """
    Download one assembly_status.txt file
    :param ref_seq_ftp: path of specific ftp folder (coming after 'ftp.ncbi.nlm.nih.gov')
    :param strain_name: name of specific strain
    """
    try:
        ind = input_list[0]
        ref_seq_ftp = input_list[1]
        strain_name = input_list[2]
        file_name = input_list[3]
        if ref_seq_ftp == '-':
            print(f"SKIP! no RefSeq: {strain_name}, index: {ind}")
            return
        ftp = FTP(NCBI_FTP_SITE)
        ftp.login()
        ftp.cwd(ref_seq_ftp)
        # replace "/" with "$" so it will be possible to save the file
        output_file_path = DEST_PATH + strain_name.replace("/", "$") + ".txt"
        with open(output_file_path, 'wb') as f:
            ftp.retrbinary('RETR ' + file_name, f.write)
        print(f"Downloaded assembly_status for: {strain_name}, index: {ind}")
    except Exception as e:
        print(f"ERROR at downloading assembly_status for: {strain_name}, index: {ind}, message: {e}")


if __name__ == '__main__':
    df = pd.read_csv(CSV_FILE_PATH)
    n_rows = df.shape[0]
    input_list = []
    for ind in range(n_rows):
        folder_ind = df["RefSeq FTP"][ind].find("/genomes")
        ref_seq_ftp = df["RefSeq FTP"][ind][folder_ind:]
        strain_name = df["Strain"][ind]
        input_list.append([ind, ref_seq_ftp, strain_name, ASSEMBLY_STATUS_FILE])
    # input_list = [input_list[158]]
    if NUM_OF_PROCESSES > 1:
        pool = ProcessPool(processes=NUM_OF_PROCESSES)
        pool.map(download_assembly_status, input_list)
    else:
        for i in input_list:
            download_assembly_status(i)
