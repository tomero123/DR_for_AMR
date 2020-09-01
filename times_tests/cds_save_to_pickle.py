import gzip
import pickle
import sys

from enums import ProcessingMode

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import os
import random

from Bio import SeqIO
from gensim.models import doc2vec
from functools import partial


def _get_document_from_fasta(sequence, processing_mode, k, shift_size):
    '''
    Function takes a fasta sequence and return a list of documents.
    The number of documents to be returned is deterimend by processing_mode.
    if processing_mode = NON_OVERLAPPING: will return K documents in the list.
    if processing_mode = OVERLAPPING: will return 1 document in the list.
    :param sequence: fasta sequence (str)
    '''
    documents_list = []
    if processing_mode == ProcessingMode.NON_OVERLAPPING.value:
        for k_ind in range(k):
            cur_doc = []
            for start_ind in range(k_ind, len(sequence) - k + 1)[::k]:
                key = sequence[start_ind:start_ind + k]
                cur_doc.append(key)
            documents_list.append(cur_doc)
    elif processing_mode == ProcessingMode.OVERLAPPING.value:
        cur_doc = []
        for start_ind in range(0, len(sequence) - k + 1, shift_size):
            key = sequence[start_ind:start_ind + k]
            cur_doc.append(key)
        documents_list.append(cur_doc)
    else:
        raise Exception(f"PROCESSING_MODE: {processing_mode} is invalid!")
    return documents_list





prefix = '..' if os.name == 'nt' else '.'
input_folder = os.path.join(prefix, "results_files", "test", "cds_genome_files")
model_save_name = "example.model"
files_list = os.listdir(input_folder)
files_list = [x for x in files_list if ".fna.gz" in x]
document_id = 0
_open = partial(gzip.open, mode='rt')
PROCESSING_MODE = ProcessingMode.OVERLAPPING.value
K = 10
SHIFT_SIZE = 1

all_genes_list = []
for file_ind, file_name in enumerate(files_list):
    try:
        fasta_sequences = SeqIO.parse(_open(os.path.join(input_folder, file_name)), 'fasta')
        seq_id = 0
        for fasta in fasta_sequences:
            x = random.random()
            # if x <= 0.5:
            #     continue
            seq_id += 1
            name, sequence = fasta.id, str(fasta.seq)
            documents_list = _get_document_from_fasta(sequence, PROCESSING_MODE, K, SHIFT_SIZE)
            all_genes_list.append(documents_list[0])

        if file_ind % 1 == 0:
            print(f"Finished processing file #{file_ind}, file_name:{file_name.replace('.fna.gz', '')}, number of genes: {seq_id} document_id: {document_id}")
    except Exception as e:
        print(f"****ERROR IN PARSING file: {file_name}, seq_id: {seq_id},")
        print(f"name: {name}  sequence: {sequence}")
        print(f"Error message: {e}")
    document_id += 1

with open(os.path.join(input_folder, "all_files.pkl"), 'wb') as outfile:
    pickle.dump(all_genes_list, outfile, protocol=pickle.HIGHEST_PROTOCOL)