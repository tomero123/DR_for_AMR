import datetime
import gc
import os
import pickle
import pandas as pd
import time
import multiprocessing

from gensim.models import doc2vec


class GenomeDocs(object):
    def __init__(self, input_folder, files_list):
        self.input_folder = input_folder
        self.files_list = files_list

    def __iter__(self):
        for document_id, file_name in enumerate(self.files_list):
            with open(os.path.join(self.input_folder, file_name), 'rb') as f:
                cur_doc = pickle.load(f)
                print(f"ind: {document_id}, doc len: {len(cur_doc)}")
                yield doc2vec.TaggedDocument(cur_doc, [document_id])


class Doc2VecTrainer(object):
    def __init__(self, input_folder, models_folder, files_list, model_save_name, vector_size, window_size,workers):
        self.input_folder = input_folder
        self.models_folder = models_folder
        self.files_list = files_list
        self.model_save_name = model_save_name
        self.workers = workers
        self.vector_size = vector_size
        self.window_size = window_size

    def run(self):
        gc.collect()
        print(f"Number of documents: {len(self.files_list)}")
        print(f"doc2vec FAST_VERSION: {doc2vec.FAST_VERSION}")
        corpus_data = GenomeDocs(self.input_folder, self.files_list)

        # params
        vector_size = self.vector_size
        dm = 1
        min_count = 100
        sample = 1e-4
        negative = 5
        window = self.window_size
        epochs = 20

        model = doc2vec.Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, sample=sample, negative=negative, workers=self.workers, dm=dm)
        print(f"model params:\nvector_size: {vector_size}\nwindow: {window}\ndm: {dm}\nmin_count: {min_count}\n"
              f"sample: {sample}\nnegative: {negative}\nworkers: {self.workers}")
        print('building vocabulary...')
        model.build_vocab(corpus_data)

        model.train(corpus_data, total_examples=model.corpus_count, epochs=epochs)

        if not os.path.exists(self.models_folder):
            os.makedirs(self.models_folder)

        model.save(os.path.join(self.models_folder, "d2v" + self.model_save_name))
        model.save_word2vec_format(os.path.join(self.models_folder, "w2v" + self.model_save_name))

        print('total docs learned %s' % (len(model.docvecs)))


class Doc2VecLoader(object):
    def __init__(self, input_folder, files_list, k, processing_mode, load_existing_path=None):
        self.input_folder = input_folder
        self.files_list = files_list
        self.k = k
        self.processing_mode = processing_mode
        self.model = doc2vec.Doc2Vec.load(load_existing_path)

    def run(self):
        now = time.time()
        gc.collect()
        print(f"Loading doc2vec model. Number of documents: {len(self.files_list)}. doc2vec FAST_VERSION: {doc2vec.FAST_VERSION}")
        vector_size = None
        all_results = []
        file_names = [x.replace(".pkl", ".txt.gz") for x in self.files_list]
        for ind, file_name in enumerate(self.files_list):
            with open(os.path.join(self.input_folder, file_name), 'rb') as f:
                cur_doc = pickle.load(f)
                # cur_vec = self.model.infer_vector(cur_doc)
                cur_vec = self.model.infer_vector(cur_doc,  alpha=0.1, min_alpha=0.0001, steps=100)
                if vector_size is None:
                    vector_size = cur_vec.shape[0]
                all_results.append(cur_vec)
        columns_names = [f"f_{x + 1}" for x in range(vector_size)]
        em_df = pd.DataFrame(all_results, columns=columns_names)
        em_df.insert(loc=0, column="file_name", value=file_names)
        print(f"Finished creating embeddings in {round((time.time() - now) / 60, 4)} minutes")
        return em_df


if __name__ == '__main__':
    input_folder = r"C:\tomer_thesis\results_files\test\genome_files"
    model_save_name = "example.model"
    files_list = os.listdir(input_folder)
    files_list = [x for x in files_list if ".pkl" in x]
    VECTOR_SIZE = 300
    WINDOW_SIZE = 5
    workers = multiprocessing.cpu_count()
    #

    trainer = Doc2VecTrainer(input_folder, input_folder, files_list, model_save_name, VECTOR_SIZE, WINDOW_SIZE, workers)
    trainer.run()
    now_date = datetime.datetime.now()
    print(f"Finished running on: {now_date.strftime('%Y-%m-%d %H:%M:%S')}")
