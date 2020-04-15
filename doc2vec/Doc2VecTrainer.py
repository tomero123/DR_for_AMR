import sys
sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import gc
import os
import pickle
import pandas as pd
import numpy as np
from gensim.models import doc2vec


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
    def __init__(self, input_folder, models_folder, files_list, model_save_name, workers=1):
        self.input_folder = input_folder
        self.models_folder = models_folder
        self.files_list = files_list
        self.model_save_name = model_save_name
        self.workers = workers

    def run(self):
        gc.collect()
        print(f"Number of documents: {len(self.files_list)}")
        print(f"doc2vec FAST_VERSION: {doc2vec.FAST_VERSION}")
        corpus_data = GenomeDocs(self.input_folder, self.files_list)

        model = doc2vec.Doc2Vec(size=128, window=10, min_count=3, sample=1e-4, negative=5, workers=self.workers, dm=1)
        print('building vocabulary...')
        model.build_vocab(corpus_data)

        model.train(corpus_data, total_examples=model.corpus_count, epochs=20)

        if not os.path.exists(self.models_folder):
            os.makedirs(self.models_folder)

        model.save(os.path.join(self.models_folder, "d2v" + self.model_save_name))
        model.save_word2vec_format(os.path.join(self.models_folder, "w2v" + self.model_save_name))

        print('total docs learned %s' % (len(model.docvecs)))


class Doc2VecLoader(object):
    def __init__(self, input_folder, files_list, load_existing_path=None):
        self.input_folder = input_folder
        self.files_list = files_list
        self.model = doc2vec.Doc2Vec.load(load_existing_path)

    def run(self):
        gc.collect()
        print('Loading an exiting model')
        print(f"Number of documents: {len(self.files_list)}")
        print(f"doc2vec FAST_VERSION: {doc2vec.FAST_VERSION}")
        vector_size = None
        for ind, file_name in enumerate(self.files_list):
            with open(os.path.join(self.input_folder, file_name), 'rb') as f:
                cur_doc = pickle.load(f)
                cur_vec = self.model.infer_vector(cur_doc)
                if vector_size is None:
                    vector_size = cur_vec.shape[0]
                    results_array = cur_vec
                else:
                    results_array = np.vstack((results_array, cur_vec))
        columns_names = [f"f_{x+1}" for x in range(vector_size)]
        em_df = pd.DataFrame(results_array, columns=columns_names)
        em_df.insert(loc=0, column="file_name", value=self.files_list)
        return em_df
