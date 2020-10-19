import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import multiprocessing
import os
import time
import datetime

from doc2vec_cds.Doc2VecCDS import Doc2VecCDS
from utils import get_time_as_str
from constants import Bacteria, ProcessingMode
from MyLogger import Logger

if __name__ == '__main__':
    # PARAMS
    BACTERIA = Bacteria.PSEUDOMONAS_AUREGINOSA.value if len(sys.argv) <= 1 else sys.argv[1]
    PROCESSING_MODE = ProcessingMode.NON_OVERLAPPING.value if len(sys.argv) <= 2 else sys.argv[2]  # can be "non_overlapping" or "overlapping"
    VECTOR_SIZE = 300 if len(sys.argv) <= 3 else int(sys.argv[3])
    WINDOW_SIZE = 5 if len(sys.argv) <= 4 else int(sys.argv[4])
    K = 10 if len(sys.argv) <= 5 else int(sys.argv[5])  # Choose K size
    SHIFT_SIZE = 1 if len(sys.argv) <= 6 else int(sys.argv[6])  # relevant only for PROCESSING_MODE "overlapping"
    workers = multiprocessing.cpu_count()
    NUMBER_OF_TRAINING_STRAINS = 500  # if None take ALL strains
    # PARAMS END

    conf_str = f"_PM_{PROCESSING_MODE}_K_{K}_SS_{SHIFT_SIZE}"
    folder_time = get_time_as_str()
    model_folder_name = folder_time + conf_str

    now = time.time()
    now_date = datetime.datetime.now()

    prefix = '..' if os.name == 'nt' else '.'
    input_folder = os.path.join(prefix, "results_files", BACTERIA, "cds_genome_files")
    models_folder = os.path.join(prefix, "results_files", BACTERIA, "cds_models", model_folder_name)
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    log_path = os.path.join(models_folder, f"log_{model_folder_name}.txt")
    sys.stdout = Logger(log_path)
    print(f"Started running on: {now_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Started dov2vec training for bacteria: {BACTERIA} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE} num of workers: {workers} model_folder_name: {model_folder_name}")
    files_list = os.listdir(input_folder)
    files_list = [x for x in files_list if ".fna.gz" in x]

    if NUMBER_OF_TRAINING_STRAINS:
        files_list = files_list[:NUMBER_OF_TRAINING_STRAINS]

    conf_dict = {
        "bacteria": BACTERIA,
        "processing_mode": PROCESSING_MODE,
        "k": K,
        "shift_size": SHIFT_SIZE,
        "workers": workers,
        "training_strains_number": len(files_list)
    }

    trainer = Doc2VecCDS(input_folder, models_folder, files_list, folder_time, PROCESSING_MODE, K, SHIFT_SIZE, VECTOR_SIZE, WINDOW_SIZE, workers, conf_dict)
    trainer.run()
    print(f"Finished training for bacteria: {BACTERIA} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE} model_folder_name: {model_folder_name} in {round((time.time() - now) / 3600, 4)} hours")
    now_date = datetime.datetime.now()
    print(f"Finished running on: {now_date.strftime('%Y-%m-%d %H:%M:%S')}")
