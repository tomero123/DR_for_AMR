import sys
sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import multiprocessing
import os
import time
import datetime

from doc2vec.Doc2VecTrainer import Doc2VecTrainer
from utils import get_file_name
from enums import Bacteria


if __name__ == '__main__':
    # PARAMS
    BACTERIA = Bacteria.GENOME_MIX.value if len(sys.argv) <= 1 else sys.argv[1]
    PROCESSING_MODE = "overlapping" if len(sys.argv) <= 2 else sys.argv[2]  # can be "non_overlapping" or "overlapping"
    K = 3 if len(sys.argv) <= 3 else int(sys.argv[3])  # Choose K size
    NUM_OF_PROCESSES = 10 if len(sys.argv) <= 4 else int(sys.argv[4])
    SHIFT_SIZE = 1  # relevant only for PROCESSING_MODE "overlapping"
    workers = multiprocessing.cpu_count()
    # workers = 1
    # PARAMS END

    now = time.time()
    now_date = datetime.datetime.now()
    print(f"Started running on: {now_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Started dov2vec training for bacteria: {BACTERIA} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE} num of workers: {workers}")
    prefix = '..' if os.name == 'nt' else '.'
    if PROCESSING_MODE == "overlapping":
        input_folder = os.path.join(prefix, "results_files", BACTERIA, "genome_documents", f"overlapping_{SHIFT_SIZE}", f"K_{K}")
        models_folder = os.path.join(prefix, "results_files", BACTERIA, "models", f"overlapping_{SHIFT_SIZE}", f"K_{K}")
    elif PROCESSING_MODE == "non_overlapping":
        input_folder = os.path.join(prefix, "results_files", BACTERIA, "genome_documents", "non_overlapping", f"K_{K}")
        models_folder = os.path.join(prefix, "results_files", BACTERIA, "models", "non_overlapping", f"K_{K}")
    else:
        raise Exception(f"PROCESSING_MODE: {PROCESSING_MODE} is invalid!")
    files_list = os.listdir(input_folder)
    files_list = [x for x in files_list if ".pkl" in x]
    #

    model_save_name = get_file_name("", "model")
    trainer = Doc2VecTrainer(input_folder, models_folder, files_list, model_save_name, PROCESSING_MODE, workers)
    trainer.run()
    print(f"Finished training for bacteria: {BACTERIA} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE} in {round((time.time() - now) / 3600, 4)} hours")
    now_date = datetime.datetime.now()
    print(f"Finished running on: {now_date.strftime('%Y-%m-%d %H:%M:%S')}")
