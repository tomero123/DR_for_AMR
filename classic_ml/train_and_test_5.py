import sys
sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import os
import pandas as pd
import xgboost
import time
import datetime
import json

from classic_ml.classic_ml_utils import get_final_df, train_test_and_write_results, get_kmers_df, \
    get_current_results_folder, get_label_df, train_test_and_write_results_cv
from MyLogger import Logger
from enums import Bacteria, ANTIBIOTIC_DIC, TestMethod

# *********************************************************************************************************************************
# Config
BACTERIA = Bacteria.PSEUDOMONAS_AUREGINOSA.value if len(sys.argv) <= 1 else sys.argv[1]
K = 10 if len(sys.argv) <= 2 else int(sys.argv[2])  # Choose K size
TEST_METHOD = TestMethod.CV.value if len(sys.argv) <= 3 else sys.argv[3]  # can be either "train_test" or "cv"

remove_intermediate = True

# Model params
random_seed = 1
rare_th = 5  # remove kmer if it appears in number of strains which is less or equal than rare_th
common_th_subtract = None  # remove kmer if it appears in number of strains which is more or equal than number_of_strains - common_th
features_selection_n = 300  # number of features to leave after feature selection
max_depth = 4
n_estimators = 300
subsample = 0.8
max_features = 0.8  # like max_features in sklearn
learning_rate = 0.1
n_jobs = 10
model = xgboost.XGBClassifier(random_state=random_seed)
model_params = {
    "max_depth": max_depth,
    "n_estimators": n_estimators,
    "subsample": subsample,
    "max_features": max_features,  # like max_features in sklearn
    "learning_rate": learning_rate,
    "n_jobs": n_jobs
}
if os.name == 'nt':
    model = xgboost.XGBClassifier(random_state=random_seed)
    model_params = {
        "n_estimators": 2,
        "max_features": 0.8,  # like max_features in sklearn
        "learning_rate": 0.5,
    }
    antibiotic_list = ['levofloxacin', 'ceftazidime']
else:
    antibiotic_list = ANTIBIOTIC_DIC.get(BACTERIA)


params_dict = {
    "bacteria": BACTERIA,
    "test_method": TEST_METHOD,
    "K": K,
    "model": str(model.__class__)
}
params_dict.update(model.get_params())

print(f"STARTED running at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n params: {params_dict}")
# *********************************************************************************************************************************
# Constant PARAMS
if os.name == 'nt':
    dataset_file_name = 'all_kmers_file_SMALL_50.csv.gz'
    kmers_map_file_name = 'all_kmers_map_SMALL_50.txt'
else:
    dataset_file_name = f'all_kmers_file_K_{K}.csv.gz'
    kmers_map_file_name = f'all_kmers_map_K_{K}.txt'


amr_data_file_name = 'amr_labels.csv'

prefix = '..' if os.name == 'nt' else '.'
path = os.path.join(prefix, 'results_files', BACTERIA)

# Config END
# *********************************************************************************************************************************

now_global = time.time()
kmers_df, kmers_original_count, kmers_final_count = get_kmers_df(path, dataset_file_name, kmers_map_file_name, rare_th, common_th_subtract)
all_results_dic = {"antibiotic": [], "accuracy": [], "f1_score": [], "auc": [], "recall": [], "precision": []}

amr_df = pd.read_csv(os.path.join(path, amr_data_file_name))
results_file_folder = get_current_results_folder(features_selection_n)
results_path = os.path.join(path, "classic_ml_results", results_file_folder)
if not os.path.exists(results_path):
    os.makedirs(results_path)
log_path = os.path.join(results_path, f"log_{results_file_folder}.txt")
sys.stdout = Logger(log_path)

print(f"Started bacteria: {BACTERIA} with antibiotics: {str(antibiotic_list)}")
for antibiotic in antibiotic_list:
    print(f"Started running get_final_df for bacteria: {BACTERIA}, antibiotic: {antibiotic}")
    now = time.time()
    label_df = get_label_df(amr_df, antibiotic)
    final_df = get_final_df(antibiotic, kmers_df, label_df)
    print(f"Finished running get_final_df for bacteria: {BACTERIA}, antibiotic: {antibiotic} in {round((time.time() - now) / 60, 4)} minutes")
    results_file_name = f"{antibiotic}_RESULTS_{results_file_folder}.xlsx"
    results_file_path = os.path.join(results_path, results_file_name)
    if TEST_METHOD == TestMethod.TRAIN_TEST.value:
        train_test_and_write_results(final_df, amr_df, results_file_path, model, model_params, antibiotic, kmers_original_count, kmers_final_count, features_selection_n, all_results_dic)
    elif TEST_METHOD == TestMethod.CV.value:
        train_test_and_write_results_cv(final_df, amr_df, results_file_path, model, model_params, antibiotic, kmers_original_count, kmers_final_count, features_selection_n, all_results_dic, random_seed)
    else:
        raise Exception("Invalid test_method")
print(all_results_dic)
all_results_df = pd.DataFrame(all_results_dic)
writer = pd.ExcelWriter(os.path.join(results_path, f"ALL_RESULTS_{results_file_folder}.xlsx"), engine='xlsxwriter')
all_results_df.iloc[::-1].to_excel(writer, sheet_name="Sheet1", index=False)
workbook = writer.book
workbook.close()


params_dict.update(all_results_dic)

with open(os.path.join(results_path, "params.json"), "w") as write_file:
    json.dump(params_dict, write_file)


print(f"DONE! Finished running at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} bacteria: {BACTERIA}, antibiotics: {str(antibiotic_list)} in {round((time.time() - now_global) / 60, 4)} minutes")
