import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import gc
import os
import pickle
import pandas as pd
import numpy as np
from gensim.models import doc2vec

from enums import ProcessingMode


class GenomeDocs(object):
    def __init__(self, input_folder, files_list, document_id_dic):
        self.input_folder = input_folder
        self.files_list = files_list
        self.document_id_dic = document_id_dic

    def __iter__(self):
        for ind, file_name in enumerate(self.files_list):
            with open(os.path.join(self.input_folder, file_name), 'rb') as f:
                cur_doc = pickle.load(f)
                print(f"ind: {ind}, doc len: {len(cur_doc)}")
                document_id = self.document_id_dic[file_name]
                yield doc2vec.TaggedDocument(cur_doc, [document_id])


class Doc2VecTrainer(object):
    def __init__(self, input_folder, models_folder, files_list, model_save_name, processing_mode, workers=1):
        self.input_folder = input_folder
        self.models_folder = models_folder
        self.files_list = files_list
        self.model_save_name = model_save_name
        self.workers = workers
        self.processing_mode = processing_mode
        self.document_id_dic = self.get_document_id_dic()

    def run(self):
        gc.collect()
        print(f"Number of documents: {len(self.files_list)}")
        print(f"doc2vec FAST_VERSION: {doc2vec.FAST_VERSION}")
        corpus_data = GenomeDocs(self.input_folder, self.files_list, self.document_id_dic)

        model = doc2vec.Doc2Vec(size=1024, window=10, min_count=3, sample=1e-4, negative=5, workers=self.workers, dm=1)
        print('building vocabulary...')
        model.build_vocab(corpus_data)

        model.train(corpus_data, total_examples=model.corpus_count, epochs=20)

        if not os.path.exists(self.models_folder):
            os.makedirs(self.models_folder)

        model.save(os.path.join(self.models_folder, "d2v" + self.model_save_name))
        model.save_word2vec_format(os.path.join(self.models_folder, "w2v" + self.model_save_name))

        print('total docs learned %s' % (len(model.docvecs)))

    def get_document_id_dic(self):
        document_id_dic = {}
        for file_name in self.files_list:
            document_id = file_name.replace(".pkl", "")
            if self.processing_mode == "non_overlapping":
                document_id = document_id[:document_id.rfind("ind") - 1]
            document_id_dic[file_name] = document_id
        return document_id_dic


class Doc2VecLoader(object):
    def __init__(self, input_folder, files_list, k, processing_mode, load_existing_path=None):
        self.input_folder = input_folder
        self.files_list = files_list
        self.k = k
        self.processing_mode = processing_mode
        self.model = doc2vec.Doc2Vec.load(load_existing_path)

    def run(self):
        gc.collect()
        print('Loading an exiting model')
        print(f"Number of documents: {len(self.files_list)}")
        print(f"doc2vec FAST_VERSION: {doc2vec.FAST_VERSION}")
        vector_size = None
        all_results = []
        file_names = [x.replace(".pkl", ".txt.gz") for x in self.files_list]
        for ind, file_name in enumerate(self.files_list):
            with open(os.path.join(self.input_folder, file_name), 'rb') as f:
                cur_doc = pickle.load(f)
                cur_vec = self.model.infer_vector(cur_doc)
                if vector_size is None:
                    vector_size = cur_vec.shape[0]
                all_results.append(cur_vec)
        columns_names = [f"f_{x + 1}" for x in range(vector_size)]
        em_df = pd.DataFrame(all_results, columns=columns_names)
        em_df.insert(loc=0, column="file_name", value=file_names)
        return em_df
