import gzip
import os
import json
import pandas as pd

# PARAMS
from tqdm import tqdm

prefix = '..' if os.name == 'nt' else '.'
results_files_path = os.path.join(prefix, 'results_files')
input_folder = os.path.join(results_files_path, 'kmers_files')
amr_data_file_path = os.path.join(results_files_path, 'amr_data_summary.csv')

all_kmers_file_csv_name = "all_kmers_file.csv.gz"
all_kmers_map_file_name = "all_kmers_map.txt"

limit = None  # if None - take all files from kmers_files else limit
# PARAMS END

files_list = os.listdir(input_folder)
files_list = [x for x in files_list if ".txt.gz" in x]

if __name__ == '__main__':
    print("Total files in folder: {}".format(len(files_list)))
    amr_df = pd.read_csv(amr_data_file_path)
    files_with_amr_data = list(amr_df['NCBI File Name'])
    all_kmers_dic = {}
    mapping_dic = {}
    if limit is not None:
        files_list = files_list[:limit]
        print("Total files after using limit: {}".format(len(files_list)))
    # Keep only strains with amr data
    files_list = [x for x in files_list if x in files_with_amr_data]
    n_of_files = len(files_list)
    print("Total files with AMR data: {}".format(n_of_files))
    for ind, file_name in enumerate(tqdm(files_list)):
        mapping_dic[ind] = file_name
        # print(f"Started processing: {file_name}")
        with gzip.open(os.path.join(input_folder, file_name), "rt") as f:
            kmers_dic = json.loads(f.read())
            for kmer in kmers_dic.keys():
                if kmer in all_kmers_dic:
                    all_kmers_dic[kmer][ind] = kmers_dic[kmer]
                else:
                    all_kmers_dic[kmer] = [0] * n_of_files
                    all_kmers_dic[kmer][ind] = kmers_dic[kmer]

    # remove very rare and/or very common k-mers
    print("Number of unique kmers: {}".format(len(all_kmers_dic)))
    rare_th = 1
    # common_th = n_of_files - 2
    for k in list(all_kmers_dic.keys()):
        num_of_non_zero = [x for x in all_kmers_dic[k] if x > 0]
        if len(num_of_non_zero) <= rare_th:
            del all_kmers_dic[k]
    print("Number of unique kmers after removal of rare and/or common kmers: {}".format(len(all_kmers_dic)))
    df = pd.DataFrame({key: pd.Series(val) for key, val in all_kmers_dic.items()})
    print("Finished creating dataframe")
    df = df.T
    df.to_csv(os.path.join(results_files_path, all_kmers_file_csv_name), compression="gzip")
    print("Finished saving dataframe!")

    with open(os.path.join(results_files_path, all_kmers_map_file_name), 'w') as outfile2:
        json.dump(mapping_dic, outfile2)
    print("DONE!")
