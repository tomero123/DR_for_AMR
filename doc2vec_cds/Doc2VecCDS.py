import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import gc
import os
import pickle
import pandas as pd
import gzip
from functools import partial
from enums import ProcessingMode
from Bio import SeqIO
from gensim.models import doc2vec
from gensim.models.callbacks import CallbackAny2Vec

_open = partial(gzip.open, mode='rt')


class GenomeDocsCDS(object):
    def __init__(self, input_folder, files_list, processing_mode, k, shift_size):
        self.input_folder = input_folder
        self.files_list = files_list
        self.processing_mode = processing_mode
        self.k = k
        self.shift_size = shift_size

    def __iter__(self):
        document_id = 0
        for file_ind, file_name in enumerate(self.files_list):
            # print(f"ind: {ind}, doc len: {len(cur_doc)}")
            fasta_sequences = SeqIO.parse(_open(os.path.join(self.input_folder, file_name)), 'fasta')
            seq_id = 0
            for fasta in fasta_sequences:
                seq_id += 1
                name, sequence = fasta.id, str(fasta.seq)
                documents_list = self._get_document_from_fasta(sequence)
                for doc_ind, doc in enumerate(documents_list):
                    yield doc2vec.TaggedDocument(doc, [document_id])
            if file_ind % 1 == 0:
                print(f"Finished processing file #{file_ind}, file_name:{file_name.replace('.fna.gz', '')}, number of genes: {seq_id} document_id: {document_id}")
            document_id += 1

    def _get_document_from_fasta(self, sequence: str):
        '''
        Function takes a fasta sequence and return a list of documents.
        The number of documents to be returned is deterimend by processing_mode.
        if processing_mode = NON_OVERLAPPING: will return K documents in the list.
        if processing_mode = OVERLAPPING: will return 1 document in the list.
        :param sequence: fasta sequence (str)
        '''
        documents_list = []
        if self.processing_mode == ProcessingMode.NON_OVERLAPPING.value:
            for k_ind in range(self.k):
                cur_doc = []
                for start_ind in range(k_ind, len(sequence) - self.k + 1)[::self.k]:
                    key = sequence[start_ind:start_ind + self.k]
                    cur_doc.append(key)
                documents_list.append(cur_doc)
        elif self.processing_mode == ProcessingMode.OVERLAPPING.value:
            cur_doc = []
            for start_ind in range(0, len(sequence) - self.k + 1, self.shift_size):
                key = sequence[start_ind:start_ind + self.k]
                cur_doc.append(key)
            documents_list.append(cur_doc)
        else:
            raise Exception(f"PROCESSING_MODE: {self.processing_mode} is invalid!")
        return documents_list


class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1


class Doc2VecCDS(object):
    def __init__(self, input_folder, models_folder, files_list, model_save_name, processing_mode, k, shift_size, vector_size, window_size, workers):
        self.input_folder = input_folder
        self.models_folder = models_folder
        self.files_list = files_list
        self.model_save_name = model_save_name
        self.workers = workers
        self.processing_mode = processing_mode
        self.k = k
        self.shift_size = shift_size
        self.vector_size = vector_size
        self.window_size = window_size
        # self.document_id_dic = self.get_document_id_dic()

    def run(self):
        gc.collect()
        print(f"Number of documents: {len(self.files_list)}")
        print(f"doc2vec FAST_VERSION: {doc2vec.FAST_VERSION}")
        corpus_data = GenomeDocsCDS(self.input_folder, self.files_list, self.processing_mode, self.k, self.shift_size)

        # PARAMS
        vector_size = self.vector_size
        window = self.window_size
        dm = 1
        min_count = 50
        sample = 1e-4
        negative = 5
        epochs = 10
        # PARAMS END

        model = doc2vec.Doc2Vec(vector_size=vector_size, window=window, min_count=min_count,
                                sample=sample, negative=negative, workers=self.workers, dm=dm,
                                # compute_loss=True, callbacks=[callback()]
                                )
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


class Doc2VecCDSLoader(object):
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
