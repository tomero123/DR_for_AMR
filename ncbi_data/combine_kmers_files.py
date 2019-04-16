import gzip
import os
import json

# PARAMS
path = "C:/University 2nd degree/Thesis/Pseudomonas Aureginosa data/"
input_folder = "kmers_files/"
# PARAMS END

files_list = os.listdir(path + input_folder)
files_list = [x for x in files_list if ".txt.gz" in x]
n_of_files = len(files_list)

if __name__ == '__main__':
    all_kmers_dic = {}
    for ind, file_name in enumerate(files_list):
        print(f"Started processing: {file_name}")
        with gzip.open(path + input_folder + file_name, "rt") as f:
            kmers_dic = json.loads(f.read())
            for kmer in kmers_dic.keys():
                if kmer in all_kmers_dic:
                    all_kmers_dic[kmer][ind] = kmers_dic[kmer]
                else:
                    all_kmers_dic[kmer] = [0] * n_of_files
                    all_kmers_dic[kmer][ind] = kmers_dic[kmer]
    with gzip.open(path + "all_kmers_file.txt.gz", 'wt') as outfile:
        json.dump(all_kmers_dic, outfile)
