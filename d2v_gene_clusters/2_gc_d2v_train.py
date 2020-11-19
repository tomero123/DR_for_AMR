import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import multiprocessing
import os
import time
import datetime
import pandas as pd

from utils import get_time_as_str
from constants import Bacteria, ProcessingMode, RawDataType
from MyLogger import Logger
from d2v_gene_clusters.Doc2VecGeneClusters import Doc2VecGeneClusters

if __name__ == '__main__':
    # PARAMS
    BACTERIA = Bacteria.PSEUDOMONAS_AUREGINOSA.value if len(sys.argv) <= 1 else sys.argv[1]
    VECTOR_SIZE = 300 if len(sys.argv) <= 2 else int(sys.argv[2])
    WINDOW_SIZE = 5 if len(sys.argv) <= 3 else int(sys.argv[3])
    workers = multiprocessing.cpu_count()
    USE_ONLY_LABELED_STRAINS = False  # if True take only strains that have AMR label
    # PARAMS END

    conf_str = f"_GENE_CLUSTERS"
    folder_time = get_time_as_str()
    model_folder_name = folder_time + conf_str

    now = time.time()
    now_date = datetime.datetime.now()

    prefix = '..' if os.name == 'nt' else '.'
    gene_clusters_documents_files_folder = "gene_clusters_documents"
    input_folder = os.path.join(prefix, "results_files", BACTERIA, gene_clusters_documents_files_folder)
    models_folder = os.path.join(prefix, "results_files", BACTERIA, "cds_models", model_folder_name)
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    log_path = os.path.join(models_folder, f"log_{model_folder_name}.txt")
    sys.stdout = Logger(log_path)
    print(f"Started running on: {now_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Started GeneClusters dov2vec training for bacteria: {BACTERIA} num of workers: {workers} model_folder_name: {model_folder_name}")
    files_list = os.listdir(input_folder)
    files_list = [x for x in files_list if ".txt" in x]

    if USE_ONLY_LABELED_STRAINS:
        original_files_list_len = len(files_list)
        amr_df = pd.read_csv(os.path.join(prefix, "results_files", BACTERIA, "amr_labels.csv"))
        labeled_files_list = list(amr_df["NCBI File Name"])
        files_list = [x for x in files_list if x.replace(".txt", "") in labeled_files_list]
        new_files_list_len = len(files_list)
        print(f"Using only labeled files ; Original number of strains: {original_files_list_len} New number of strains: {new_files_list_len}")

    conf_dict = {
        "bacteria": BACTERIA,
        "workers": workers,
        "training_strains_number": len(files_list)
    }

    trainer = Doc2VecGeneClusters(input_folder, models_folder, files_list, folder_time, VECTOR_SIZE, WINDOW_SIZE, workers, conf_dict)
    trainer.run()
    print(f"Finished training for bacteria: {BACTERIA} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE} model_folder_name: {model_folder_name} in {round((time.time() - now) / 3600, 4)} hours")
    now_date = datetime.datetime.now()
    print(f"Finished running on: {now_date.strftime('%Y-%m-%d %H:%M:%S')}")
