import sys


sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import multiprocessing
import os
import time

from doc2vec.Doc2VecTrainer import Doc2VecTrainer
from utils import get_file_name


if __name__ == '__main__':
    # PARAMS
    BACTERIA = "pseudomonas_aureginosa" if len(sys.argv) < 2 else sys.argv[1]
    NUM_OF_PROCESSES = 1
    K = 3 if len(sys.argv) < 3 else int(sys.argv[2])  # Choose K size
    PROCESSING_MODE = "overlapping"  # can be "non_overlapping" or "overlapping"
    SHIFT_SIZE = 1  # relevant only for PROCESSING_MODE "overlapping"
    workers = multiprocessing.cpu_count()
    # workers = 1
    # PARAMS END

    now = time.time()
    print(f"Started training for bacteria: {BACTERIA} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE} num of workers: {workers}")
    prefix = '..' if os.name == 'nt' else '.'
    input_folder = os.path.join(prefix, "results_files", BACTERIA, "genome_documents", f"{PROCESSING_MODE}_{SHIFT_SIZE}", f"K_{K}")
    models_folder = os.path.join(prefix, "results_files", BACTERIA, "models", f"{PROCESSING_MODE}_{SHIFT_SIZE}", f"K_{K}")
    files_list = os.listdir(input_folder)
    files_list = [x for x in files_list if ".pkl" in x]
    #

    model_save_name = get_file_name("", "model")
    trainer = Doc2VecTrainer(input_folder, models_folder, files_list, model_save_name)
    trainer.run()
    print(f"Finished training for bacteria: {BACTERIA} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE} in {round((time.time() - now) / 60, 4)} minutes")
