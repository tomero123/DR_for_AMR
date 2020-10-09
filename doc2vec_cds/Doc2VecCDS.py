import sys

from doc2vec_cds.EpochSaver import EpochSaver

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import gc
import os
import json
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
            try:
                fasta_sequences = SeqIO.parse(_open(os.path.join(self.input_folder, file_name)), 'fasta')
                seq_id = 0
                for fasta in fasta_sequences:
                    seq_id += 1
                    name, sequence = fasta.id, str(fasta.seq)
                    documents_list = self._get_document_from_fasta(sequence, self.processing_mode, self.k, self.shift_size)
                    for doc_ind, doc in enumerate(documents_list):
                        yield doc2vec.TaggedDocument(doc, [document_id])
                if file_ind % 1 == 0:
                    print(f"Finished processing file #{file_ind}, file_name:{file_name}, number of genes: {seq_id} document_id: {document_id}")
            except Exception as e:
                print(f"****ERROR IN PARSING file: {file_name}, seq_id: {seq_id},")
                print(f"name: {name}  sequence: {sequence}")
                print(f"Error message: {e}")
            document_id += 1

    @staticmethod
    def _get_document_from_fasta(sequence: str, processing_mode, k, shift_size):
        '''
        Function takes a fasta sequence and return a list of documents.
        The number of documents to be returned is deterimend by processing_mode.
        if processing_mode = NON_OVERLAPPING: will return K documents in the list.
        if processing_mode = OVERLAPPING: will return 1 document in the list.
        :param sequence: fasta sequence (str)
        '''
        documents_list = []
        if processing_mode == ProcessingMode.NON_OVERLAPPING.value:
            for k_ind in range(k):
                cur_doc = []
                for start_ind in range(k_ind, len(sequence) - k + 1)[::k]:
                    key = sequence[start_ind:start_ind + k]
                    cur_doc.append(key)
                documents_list.append(cur_doc)
        elif processing_mode == ProcessingMode.OVERLAPPING.value:
            cur_doc = []
            for start_ind in range(0, len(sequence) - k + 1, shift_size):
                key = sequence[start_ind:start_ind + k]
                cur_doc.append(key)
            documents_list.append(cur_doc)
        else:
            raise Exception(f"PROCESSING_MODE: {processing_mode} is invalid!")
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
    def __init__(self, input_folder, models_folder, files_list, folder_time, processing_mode, k, shift_size, vector_size, window_size, workers, conf_dict):
        self.input_folder = input_folder
        self.models_folder = models_folder
        self.files_list = files_list
        self.folder_time = folder_time
        self.workers = workers
        self.processing_mode = processing_mode
        self.k = k
        self.shift_size = shift_size
        self.vector_size = vector_size
        self.window_size = window_size
        self.conf_dict = conf_dict
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
        min_count = 20
        sample = 1e-4
        negative = 5
        epochs = 20
        # PARAMS END

        self.conf_dict["vector_size"] = vector_size
        self.conf_dict["window"] = window
        self.conf_dict["dm"] = dm
        self.conf_dict["min_count"] = min_count
        self.conf_dict["sample"] = sample
        self.conf_dict["negative"] = negative
        self.conf_dict["epochs"] = epochs

        with open(os.path.join(self.models_folder, "model_conf.json"), "w") as write_file:
            json.dump(self.conf_dict, write_file)

        model = doc2vec.Doc2Vec(vector_size=vector_size, window=window, min_count=min_count,
                                sample=sample, negative=negative, workers=self.workers, dm=dm,
                                callbacks=[EpochSaver(os.path.join(self.models_folder, "checkpoints"))],
                                # compute_loss=True, callbacks=[callback()]
                                )
        print(f"model params:\nvector_size: {vector_size}\nwindow: {window}\ndm: {dm}\nmin_count: {min_count}\n"
              f"sample: {sample}\nnegative: {negative}\nworkers: {self.workers}\nepochs: {epochs}")

        print('building vocabulary...')
        model.build_vocab(corpus_data)
        model.train(corpus_data, total_examples=model.corpus_count, epochs=epochs)

        model.save(os.path.join(self.models_folder, "d2v.model"))
        # model.save_word2vec_format(os.path.join(self.models_folder, "w2v.model"))

        print('total docs learned %s' % (len(model.docvecs)))
        print(f"Saved model to {self.models_folder}")


class Doc2VecCDSLoader(object):
    def __init__(self, input_folder, labeled_files_dic, k, processing_mode, shift_size, models_folder):
        self.input_folder = input_folder
        self.labeled_files_dic = labeled_files_dic
        self.k = k
        self.processing_mode = processing_mode
        self.shift_size = shift_size
        self.model = doc2vec.Doc2Vec.load(os.path.join(models_folder, "d2v.model"))
        self.model.delete_temporary_training_data(keep_doctags_vectors=False, keep_inference=True)

    def run(self):
        gc.collect()
        print('Loading an exiting model')
        print(f"Number of documents: {len(self.labeled_files_dic)}")
        print(f"doc2vec FAST_VERSION: {doc2vec.FAST_VERSION}")
        vector_size = None
        embeddings_results = []
        metadata_results = []
        metadata_results_full = []
        file_ind = 0
        for file_name, file_id in self.labeled_files_dic.items():
            try:
                file_ind += 1
                fasta_sequences = SeqIO.parse(_open(os.path.join(self.input_folder, file_name + "_cds_from_genomic.fna.gz")), 'fasta')
                seq_id = 0
                for fasta in fasta_sequences:
                    seq_id += 1
                    seq_name, sequence = fasta.id, str(fasta.seq)
                    documents_list = GenomeDocsCDS._get_document_from_fasta(sequence, self.processing_mode, self.k, self.shift_size)
                    for doc_ind, cur_doc in enumerate(documents_list):
                        cur_vec = self.model.infer_vector(cur_doc)
                        if vector_size is None:
                            vector_size = cur_vec.shape[0]
                        embeddings_results.append(cur_vec)
                        metadata_results.append([file_id, seq_id, doc_ind])
                        metadata_results_full.append([file_id, file_name, seq_id, seq_name, doc_ind])
                if file_id % 1 == 0:
                    print(f"Finished processing file#{file_ind} file_id: {file_id}, file_name: {file_name}, number of genes: {seq_id}")
            except Exception as e:
                print(f"****ERROR IN PARSING file: {file_name}, seq_id: {seq_id},")
                print(f"name: {seq_name}  sequence: {sequence}")
                print(f"Error message: {e}")

        columns_names = [f"f_{x + 1}" for x in range(vector_size)]
        em_df = pd.DataFrame(embeddings_results, columns=columns_names)
        metadata_df = pd.DataFrame(metadata_results, columns=["file_id", "seq_id", "doc_ind"])
        metadata_df_full = pd.DataFrame(metadata_results_full, columns=["file_ind", "file_name", "seq_id", "seq_name", "doc_ind"])
        final_df = pd.concat([metadata_df, em_df], axis=1)
        return final_df, metadata_df_full
