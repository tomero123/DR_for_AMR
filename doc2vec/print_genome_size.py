import sys

from Bio import SeqIO

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import multiprocessing
import datetime
import json
import os
import pandas as pd
import numpy as np
import xgboost
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn import metrics
import time

from doc2vec.Doc2VecTrainer import Doc2VecLoader
from utils import get_file_name, _open
from constants import Bacteria, ProcessingMode


BACTERIA = Bacteria.PSEUDOMONAS_AUREGINOSA.value
K = 10
PROCESSING_MODE = ProcessingMode.NON_OVERLAPPING.value
SHIFT_SIZE = 1
prefix = '..' if os.name == 'nt' else '.'


input_folder = os.path.join(prefix, "results_files", BACTERIA, "genome_files")
files_list = os.listdir(input_folder)

for file_name in files_list:
    fasta_sequences = SeqIO.parse(_open(os.path.join(input_folder, file_name)), 'fasta')
    document_list = []
    for fasta_ind, fasta in enumerate(fasta_sequences):
        name, sequence = fasta.id, str(fasta.seq)
        print(f"File name: {file_name} ; fasta_ind: {fasta_ind} ; seq len : {len(sequence)}")

