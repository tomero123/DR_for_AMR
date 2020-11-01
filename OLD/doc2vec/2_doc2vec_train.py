import sys
sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import multiprocessing
import os
import time
import datetime

from OLD.doc2vec.Doc2VecTrainer import Doc2VecTrainer
from utils import get_file_name
from constants import Bacteria, ProcessingMode

if __name__ == '__main__':
    # PARAMS
    BACTERIA = Bacteria.GENOME_MIX_NEW.value if len(sys.argv) <= 1 else sys.argv[1]
    PROCESSING_MODE = ProcessingMode.NON_OVERLAPPING.value if len(sys.argv) <= 2 else sys.argv[2]  # can be "non_overlapping" or "overlapping"
    VECTOR_SIZE = 1024 if len(sys.argv) <= 3 else int(sys.argv[3])
    WINDOW_SIZE = 5 if len(sys.argv) <= 4 else int(sys.argv[4])
    K = 3 if len(sys.argv) <= 5 else int(sys.argv[5])  # Choose K size
    SHIFT_SIZE = 1  # relevant only for PROCESSING_MODE "overlapping"
    workers = multiprocessing.cpu_count()
    # PARAMS END

    model_save_name = get_file_name("", "model")

    now = time.time()
    now_date = datetime.datetime.now()
    print(f"Started running on: {now_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Started dov2vec training for bacteria: {BACTERIA} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE} num of workers: {workers} model_name: {model_save_name}")
    prefix = '..' if os.name == 'nt' else '.'
    if PROCESSING_MODE == ProcessingMode.OVERLAPPING.value:
        input_folder = os.path.join(prefix, "results_files", BACTERIA, "genome_documents", f"overlapping_{SHIFT_SIZE}", f"K_{K}")
        models_folder = os.path.join(prefix, "results_files", BACTERIA, "models", f"overlapping_{SHIFT_SIZE}", f"K_{K}")
    elif PROCESSING_MODE == ProcessingMode.NON_OVERLAPPING.value:
        input_folder = os.path.join(prefix, "results_files", BACTERIA, "genome_documents", "non_overlapping", f"K_{K}")
        models_folder = os.path.join(prefix, "results_files", BACTERIA, "models", "non_overlapping", f"K_{K}")
    else:
        raise Exception(f"PROCESSING_MODE: {PROCESSING_MODE} is invalid!")
    files_list = os.listdir(input_folder)
    files_list = [x for x in files_list if ".pkl" in x]
    #

    trainer = Doc2VecTrainer(input_folder, models_folder, files_list, model_save_name, PROCESSING_MODE, VECTOR_SIZE, WINDOW_SIZE, workers)
    trainer.run()
    print(f"Finished training for bacteria: {BACTERIA} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE} model_name: {model_save_name} in {round((time.time() - now) / 3600, 4)} hours")
    now_date = datetime.datetime.now()
    print(f"Finished running on: {now_date.strftime('%Y-%m-%d %H:%M:%S')}")
