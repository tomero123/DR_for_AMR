from ftplib import FTP
from io import StringIO
from Bio import SeqIO
import json
import gzip
from functools import partial

_open = partial(gzip.open, mode='rt')


def open_ftp_file(input_list):
    """
    Download one assembly_status.txt file
    :param input_list - list including the following fields:
    1) ind: index of the current item
    2) ref_seq_ftp: path of specific ftp folder (coming after 'ftp.ncbi.nlm.nih.gov')
    3) strain_name: name of specific strain
    4) ftp_file_name: name of the ftp file to download/open
    5) ftp_file_site: site name to open the ftp connection with
    :return: list where first element is strain_name and second is the status string from assembly_status.txt
    """
    try:
        ind = input_list[0]
        ftp_sub_folder = input_list[1]
        strain_name = input_list[2]
        ftp_file_name = input_list[3]
        ftp_site = input_list[4]
        if ftp_sub_folder == '-':
            print(f"SKIP! ftp_sub_folder is: {ftp_sub_folder} for strain: {strain_name}, index: {ind}")
            return "None"
        ftp = FTP(ftp_site)
        ftp.login()
        ftp.cwd(ftp_sub_folder)
        line_reader = StringIO()
        ftp.retrlines('RETR ' + ftp_file_name, line_reader.write)
        str_to_return = line_reader.getvalue()
        line_reader.close()
        print(f"Opened file for: {strain_name}, index: {ind}")
        return [strain_name, str_to_return]
    except Exception as e:
        print(f"ERROR at open_ftp_file for: {strain_name}, index: {ind}, message: {e}")
        return "None"


def download_ftp_file(input_list):
    """
    Download one assembly_status.txt file
    :param input_list - list including the following fields:
    1) ind: index of the current item
    2) ref_seq_ftp: path of specific ftp folder (coming after 'ftp.ncbi.nlm.nih.gov')
    3) strain_name: name of specific strain
    4) ftp_file_name: name of the ftp file to download/open
    5) ftp_file_site: site name to open the ftp connection with
    6) dest_path: destenation of the output files folder. if None then return the str retreived and don't save the file
    """
    try:
        ind = input_list[0]
        ftp_sub_folder = input_list[1]
        strain_name = input_list[2]
        ftp_file_name = input_list[3]
        ftp_site = input_list[4]
        dest_path = input_list[5]
        if ftp_sub_folder == '-':
            print(f"SKIP! ftp_sub_folder is: {ftp_sub_folder} for strain: {strain_name}, index: {ind}")
            return
        ftp = FTP(ftp_site)
        ftp.login()
        ftp.cwd(ftp_sub_folder)
        # replace "/" with "$" so it will be possible to save the file
        if dest_path is not None:
            output_file_path = dest_path + ftp_file_name
            with open(output_file_path, 'wb') as f:
                ftp.retrbinary('RETR ' + ftp_file_name, f.write)
            print(f"Downloaded file for: {strain_name}, index: {ind}")
        else:
            line_reader = StringIO()
            ftp.retrlines('RETR ' + ftp_file_name, line_reader.write)
            str_to_return = line_reader.getvalue()
            line_reader.close()
            print(f"Opened file for: {strain_name}, index: {ind}")
            return [strain_name, str_to_return]
    except Exception as e:
        print(f"ERROR at download_ftp_file for: {strain_name}, index: {ind}, message: {e}")


def create_kmers_file(input_list):
    """
    get one fasta file and creating a kmers mapping file
    :param input_list - list including the following fields:
    1) ind: index of the current item
    2) ref_seq_ftp: path of specific ftp folder (coming after 'ftp.ncbi.nlm.nih.gov')
    3) strain_name: name of specific strain
    4) ftp_file_name: name of the ftp file to download/open
    5) ftp_file_site: site name to open the ftp connection with
    6) dest_path: destenation of the output files folder. if None then return the str retreived and don't save the file
    """
    try:
        ind = input_list[0]
        file_name = input_list[1]
        K = input_list[2]
        input_folder = input_list[3]
        output_folder = input_list[4]
        print(f"Started processing: {file_name}")
        kmers_dic = {}
        fasta_sequences = SeqIO.parse(_open(input_folder + file_name), 'fasta')
        for fasta in fasta_sequences:
            name, sequence = fasta.id, str(fasta.seq)
            for start_ind in range(len(sequence) - K + 1):
                key = sequence[start_ind:start_ind + K]
                if key in kmers_dic:
                    kmers_dic[key] += 1
                else:
                    kmers_dic[key] = 1
        with gzip.open(output_folder + file_name.replace(".fna.gz", ".txt.gz"), 'wt') as outfile:
            json.dump(kmers_dic, outfile)
    except Exception as e:
        print(f"ERROR at create_kmers_file for: {file_name}, index: {ind}, message: {e}")
