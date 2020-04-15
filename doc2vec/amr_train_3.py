import sys

from doc2vec.Doc2VecTrainer import Doc2VecLoader

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

from doc2vec.doc2vec_train_2 import Doc2VecTrainer
import gc
import multiprocessing
import os
import pickle
import time
from gensim.corpora import dictionary
from gensim.models.deprecated.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

if __name__ == '__main__':
    # PARAMS
    BACTERIA = "pseudomonas_aureginosa" if len(sys.argv) < 2 else sys.argv[1]
    K = 3 if len(sys.argv) < 3 else int(sys.argv[2])  # Choose K size
    MODEL_NAME = "d2v_2020_04_15_1011.model" if len(sys.argv) < 4 else int(sys.argv[3])  # Model Name
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

    doc2vec_loader = Doc2VecLoader(os.path.join(models_folder, MODEL_NAME))
    doc2vec_model = doc2vec_loader.run()
    print(doc2vec_model.most_similar(['hey'])[0:2])
    x = 1