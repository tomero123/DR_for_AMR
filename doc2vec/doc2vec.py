import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")

import os
import pickle
from gensim.models import Doc2Vec
from gensim.models.deprecated.doc2vec import TaggedDocument

# PARAMS
BACTERIA = "pseudomonas_aureginosa" if len(sys.argv) < 2 else sys.argv[1]
NUM_OF_PROCESSES = 1
K = 3 if len(sys.argv) < 3 else int(sys.argv[2])  # Choose K size
PROCESSING_MODE = "overlapping"  # can be "non_overlapping" or "overlapping"
SHIFT_SIZE = 1  # relevant only for PROCESSING_MODE "overlapping"

prefix = '..' if os.name == 'nt' else '.'
input_folder = os.path.join(prefix, "results_files", BACTERIA, "genome_documents", f"{PROCESSING_MODE}_{SHIFT_SIZE}", f"K_{K}")
files_list = os.listdir(input_folder)
files_list = [x for x in files_list if ".pkl" in x]

if __name__ == '__main__':
    print(f"Start running on bacteria: {BACTERIA} with K={K}")
    all_documents = []
    for ind, file_name in enumerate(files_list):
        with open(os.path.join(input_folder, file_name), 'rb') as f:
            cur_doc = pickle.load(f)
            all_documents.append(cur_doc)


documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(all_documents)]
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
print("DONE!!!")