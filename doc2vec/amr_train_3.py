import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import multiprocessing
import os
import time
import pandas as pd
import datetime

from doc2vec.Doc2VecTrainer import Doc2VecLoader


def get_label_df(amr_file_path, files_list, antibiotic):
    amr_df = pd.read_csv(amr_file_path)
    file_name_col = 'NCBI File Name'
    strain_col = 'Strain'
    label_df = amr_df[[file_name_col, strain_col, antibiotic]]
    # Remove antibiotics without resistance data
    label_df = label_df[label_df[antibiotic] != '-']
    # Remove antibiotics with label 'I'
    label_df = label_df[label_df[antibiotic] != 'I']
    # Remove strains which are not in "files list"
    ind_to_save = label_df[file_name_col].apply(lambda x: True if x in [x.replace(".pkl", ".txt.gz") for x in files_list] else False)
    label_df = label_df[ind_to_save]
    return label_df


if __name__ == '__main__':
    # PARAMS
    BACTERIA = "pseudomonas_aureginosa" if len(sys.argv) < 2 else sys.argv[1]
    K = 3 if len(sys.argv) < 3 else int(sys.argv[2])  # Choose K size
    MODEL_NAME = "d2v_2020_04_15_1051.model" if len(sys.argv) < 4 else int(sys.argv[3])  # Model Name
    PROCESSING_MODE = "overlapping"  # can be "non_overlapping" or "overlapping"
    SHIFT_SIZE = 1  # relevant only for PROCESSING_MODE "overlapping"
    workers = multiprocessing.cpu_count()
    amr_data_file_name = "amr_data_summary.csv"
    antibiotic = "amikacin"
    # PARAMS END

    now = time.time()
    now_date = datetime.datetime.now()
    print(f"Started running on: {now_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Started xgboost training for bacteria: {BACTERIA} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE}")
    prefix = '..' if os.name == 'nt' else '.'
    input_folder = os.path.join(prefix, "results_files", BACTERIA, "genome_documents", f"{PROCESSING_MODE}_{SHIFT_SIZE}", f"K_{K}")
    models_folder = os.path.join(prefix, "results_files", BACTERIA, "models", f"{PROCESSING_MODE}_{SHIFT_SIZE}", f"K_{K}")
    amr_file_path = os.path.join(prefix, 'results_files', BACTERIA, amr_data_file_name)
    files_list = os.listdir(input_folder)
    files_list = [x for x in files_list if ".pkl" in x]

    doc2vec_loader = Doc2VecLoader(input_folder, files_list, os.path.join(models_folder, MODEL_NAME))
    label_df = get_label_df(amr_file_path, files_list, antibiotic)
    em_df = doc2vec_loader.run()
    # print(doc2vec_model.most_similar(['GTT'])[0:2])
    print(f"label_df shape: {label_df.shape}")
    print(f"em_df shape: {em_df.shape}")
    x = 1