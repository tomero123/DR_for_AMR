from ftplib import FTP
from io import StringIO
from Bio import SeqIO
import json
import gzip
from functools import partial
import os
import pandas as pd
import pickle
import datetime

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
        status = input_list[5]
        if not pd.isna(status) and status != "NA":
            str_to_return = status
            print(f"Took status from csv: {strain_name}, index: {ind}")
        else:
            if ftp_sub_folder == '-':
                print(f"SKIP! ftp_sub_folder is: {ftp_sub_folder} for strain: {strain_name}, index: {ind}")
                return "None"
            ftp = FTP(ftp_site, timeout=10)
            ftp.login()
            ftp.cwd(ftp_sub_folder)
            line_reader = StringIO()
            ftp.retrlines('RETR ' + ftp_file_name, line_reader.write)
            str_to_return = line_reader.getvalue()
            line_reader.close()
            print(f"Opened ftp for: {strain_name}, index: {ind}")
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
            output_file_path = os.path.join(dest_path, ftp_file_name)
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
        fasta_sequences = SeqIO.parse(_open(os.path.join(input_folder, file_name)), 'fasta')
        for fasta in fasta_sequences:
            name, sequence = fasta.id, str(fasta.seq)
            for start_ind in range(len(sequence) - K + 1):
                key = sequence[start_ind:start_ind + K]
                if key in kmers_dic:
                    kmers_dic[key] += 1
                else:
                    kmers_dic[key] = 1
        with gzip.open(os.path.join(output_folder, file_name.replace(".fna.gz", ".txt.gz")), 'wt') as outfile:
            json.dump(kmers_dic, outfile)
    except Exception as e:
        print(f"ERROR at create_kmers_file for: {file_name}, index: {ind}, message: {e}")


def create_genome_document(input_list):
    """
    get one fasta file and creating a document, which is python list with words,
    based on the parameters K,
    :param input_list - list including the following fields:
    1) ind: index of the current item
    2) file_name: path of specific ftp folder (coming after 'ftp.ncbi.nlm.nih.gov')
    3) K: kmer size
    4) PROCESSING_MODE: can be "non_overlapping" or "overlapping"
    5) SHIFT_SIZE: size of jumps between words, relevant only for PROCESSING_MODE "overlapping"
    6) dest_path: destenation of the output files folder. if None then return the str retreived and don't save the file
    """
    try:
        ind = input_list[0]
        file_name = input_list[1]
        K = input_list[2]
        PROCESSING_MODE = input_list[3]
        SHIFT_SIZE = input_list[4]
        input_folder = input_list[5]
        output_folder = input_list[6]
        print(f"Started processing: {file_name}")
        if PROCESSING_MODE == "overlapping":
            fasta_sequences = SeqIO.parse(_open(os.path.join(input_folder, file_name)), 'fasta')
            document_list = []
            for fasta in fasta_sequences:
                name, sequence = fasta.id, str(fasta.seq)
                for start_ind in range(0, len(sequence) - K + 1, SHIFT_SIZE):
                    key = sequence[start_ind:start_ind + K]
                    document_list.append(key)
            with open(os.path.join(output_folder, file_name.replace(".fna.gz", ".pkl")), 'wb') as outfile:
                pickle.dump(document_list, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        elif PROCESSING_MODE == "non_overlapping":
            for k_ind in range(K):
                fasta_sequences = SeqIO.parse(_open(os.path.join(input_folder, file_name)), 'fasta')
                document_list = []
                for fasta in fasta_sequences:
                    name, sequence = fasta.id, str(fasta.seq)
                    for start_ind in range(k_ind, len(sequence) - K + 1)[::K]:
                        key = sequence[start_ind:start_ind + K]
                        document_list.append(key)
                with open(os.path.join(output_folder, file_name.replace(".fna.gz", f"_ind_{k_ind+1}.pkl")), 'wb') as outfile:
                    pickle.dump(document_list, outfile, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise Exception(f"PROCESSING_MODE: {PROCESSING_MODE} is invalid!")
    except Exception as e:
        print(f"ERROR at create_genome_document for: {file_name}, index: {ind}, message: {e}")


def get_file_name(prefix, ext):
    time = datetime.datetime.now()
    year = str(time.year) if time.year > 9 else '0' + str(time.year)
    month = str(time.month) if time.month > 9 else '0' + str(time.month)
    day = str(time.day) if time.day > 9 else '0' + str(time.day)
    hour = str(time.hour) if time.hour > 9 else '0' + str(time.hour)
    minute = str(time.minute) if time.minute > 9 else '0' + str(time.minute)
    if prefix is not None and ext is not None:
        file_name = "{}_{}_{}_{}_{}{}.{}".format(prefix, year, month, day, hour, minute, ext)
    elif prefix is None and ext is not None:
        file_name = "{}_{}_{}_{}{}.{}".format(year, month, day, hour, minute, ext)
    elif prefix is not None and ext is None:
        file_name = "{}_{}_{}_{}_{}{}".format(prefix, year, month, day, hour, minute)
    else:
        file_name = "{}_{}_{}_{}{}".format(year, month, day, hour, minute)
    return file_name


def get_time_as_str():
    time = datetime.datetime.now()
    year = str(time.year) if time.year > 9 else '0' + str(time.year)
    month = str(time.month) if time.month > 9 else '0' + str(time.month)
    day = str(time.day) if time.day > 9 else '0' + str(time.day)
    hour = str(time.hour) if time.hour > 9 else '0' + str(time.hour)
    minute = str(time.minute) if time.minute > 9 else '0' + str(time.minute)
    file_name = "{}_{}_{}_{}{}".format(year, month, day, hour, minute)
    return file_name


def get_train_and_test_groups(n_folds):
    train_groups_list = []
    test_groups_list = list(range(1, n_folds + 1))
    for i in test_groups_list:
        train_groups_list.append([x for x in test_groups_list if i != x])
    return train_groups_list, test_groups_list
