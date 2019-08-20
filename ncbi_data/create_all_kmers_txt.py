import gzip
import os
import json

# PARAMS
input_folder = "../results_files/kmers_files/"
all_kmers_file_txt_name = "all_kmers_file.txt.gz"
all_kmers_map_file_name = "all_kmers_map.txt"
limit = None  # if None - take all files from kmers_files else limit
# PARAMS END

files_list = os.listdir(input_folder)
files_list = [x for x in files_list if ".txt.gz" in x]
n_of_files = len(files_list)

if __name__ == '__main__':
    all_kmers_dic = {}
    mapping_dic = {}
    for ind, file_name in enumerate(files_list):
        mapping_dic[ind] = file_name
        print(f"Started processing: {file_name}")
        with gzip.open(input_folder + file_name, "rt") as f:
            kmers_dic = json.loads(f.read())
            files_names_list = list(kmers_dic.keys()) if limit is None else list(kmers_dic.keys())[:limit]
            for kmer in files_names_list:
                if kmer in all_kmers_dic:
                    all_kmers_dic[kmer][ind] = kmers_dic[kmer]
                else:
                    all_kmers_dic[kmer] = [0] * n_of_files
                    all_kmers_dic[kmer][ind] = kmers_dic[kmer]
    with gzip.open("../results_files/" + all_kmers_file_txt_name, 'wt') as outfile:
        json.dump(all_kmers_dic, outfile)
    with open("../results_files/" + all_kmers_map_file_name, 'w') as outfile2:
        json.dump(mapping_dic, outfile2)
    print("DONE!")
