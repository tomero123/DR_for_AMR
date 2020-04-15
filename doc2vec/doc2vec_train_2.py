import sys
sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import gc
import multiprocessing
import os
import pickle
import time
import gensim
from gensim.models import doc2vec
# from gensim.models.doc2vec import TaggedDocument

from utils import get_file_name


class GenomeDocs(object):
    def __init__(self, input_folder, files_list):
        self.input_folder = input_folder
        self.files_list = files_list

    def __iter__(self):
        for ind, file_name in enumerate(self.files_list):
            with open(os.path.join(self.input_folder, file_name), 'rb') as f:
                cur_doc = pickle.load(f)
                print(f"ind: {ind}, doc len: {len(cur_doc)}")
                # yield dictionary.doc2bow(cur_doc)
                yield doc2vec.TaggedDocument(cur_doc, [file_name])


class Doc2VecTrainer(object):
    def __init__(self, input_folder, models_folder, files_list, model_save_name, workers=1, load_existing=False):
        self.input_folder = input_folder
        self.models_folder = models_folder
        self.files_list = files_list
        self.model_save_name = model_save_name
        self.workers = workers
        self.load_existing = load_existing

    def run(self):
        print('app started')

        gc.collect()
        if self.load_existing:
            print('loading an exiting model')
            # model = Doc2Vec.load(PATH_TO_EXISTING_MODEL)
        else:
            print("Started training!")
            print(f"doc2vec FAST_VERSION: {doc2vec.FAST_VERSION}")
            corpus_data = GenomeDocs(self.input_folder, self.files_list)

            model = doc2vec.Doc2Vec(size=128, window=10, min_count=3, sample=1e-4, negative=5, workers=self.workers, dm=1)
            print('building vocabulary...')
            model.build_vocab(corpus_data)

            model.train(corpus_data, total_examples=model.corpus_count, epochs=20)

            if not os.path.exists(self.models_folder):
                os.makedirs(self.models_folder)

            model.save(os.path.join(self.models_folder,"d2v" + model_save_name))
            model.save_word2vec_format(os.path.join(self.models_folder, "w2v" + model_save_name))

        print('total docs learned %s' % (len(model.docvecs)))


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
    trainer = Doc2VecTrainer(input_folder, models_folder, files_list[:5], model_save_name)
    trainer.run()
    print(f"Finished training for bacteria: {BACTERIA} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE} in {round((time.time() - now) / 60, 4)} minutes")
