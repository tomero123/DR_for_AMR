import sys
import traceback

from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import compute_sample_weight

from doc2vec_cds.Doc2VecCDS import Doc2VecCDSLoader

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import multiprocessing
import datetime
import json
import os
import pandas as pd
import numpy as np
import xgboost
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn import metrics
import time

from doc2vec.Doc2VecTrainer import Doc2VecLoader
from utils import get_file_name
from enums import Bacteria, ProcessingMode


def get_label_dic(amr_file_path, files_list, antibiotic, processing_mode, k):
    amr_df = pd.read_csv(amr_file_path)
    file_name_col = 'NCBI File Name'
    strain_col = 'Strain'
    label_df = amr_df[[file_name_col, strain_col, antibiotic]]
    # Remove antibiotics without resistance data
    label_df = label_df[label_df[antibiotic] != '-']
    # Remove antibiotics with label 'I'
    label_df = label_df[label_df[antibiotic] != 'I']
    # Remove strains which are not in "files list"
    ind_to_save = label_df[file_name_col].apply(lambda x: True if x in [x.replace(".fna.gz", ".txt.gz") for x in files_list] else False)
    new_label_df = label_df[ind_to_save]
    new_label_df.rename(columns={"NCBI File Name": "file_name", antibiotic: "label"}, inplace=True)
    label_dic = {}
    for index, row in new_label_df.iterrows():
        label_dic[row['file_name']] = [row['label'], row['Strain']]
    return new_label_df, label_dic


def write_data_to_excel(results_df, results_file_path, classes, model_parmas, all_results_dic):
    try:
        writer = pd.ExcelWriter(results_file_path, engine='xlsxwriter')
        name = 'Sheet1'
        col_ind = 0
        row_ind = 0
        results_df.to_excel(writer, sheet_name=name, startcol=col_ind, startrow=row_ind, index=False)
        col_ind += results_df.shape[1] + 1
        y_true = [1 if x == "R" else 0 for x in list(results_df['Label'])]
        y_pred = [1 if x == "R" else 0 for x in list(results_df['Prediction'])]
        y_pred_score = list(results_df['Resistance score'])
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        confusion_matrix_df = pd.DataFrame(confusion_matrix, columns=[x + "_Prediction" for x in classes], index=[x + "_Actual" for x in classes])
        confusion_matrix_df.to_excel(writer, sheet_name=name, startcol=col_ind, startrow=row_ind, index=True)
        row_ind += confusion_matrix_df.shape[0] + 2
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1_score = metrics.f1_score(y_true, y_pred)
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred_score)
        auc = metrics.auc(fpr, tpr)
        all_results_dic["antibiotic"].append(antibiotic)
        all_results_dic["accuracy"].append(accuracy)
        all_results_dic["f1_score"].append(f1_score)
        all_results_dic["auc"].append(auc)
        evaluation_list = [["accuracy", accuracy], ["f1_score", f1_score], ["auc", auc], ["model_parmas", model_parmas]]
        evaluation_df = pd.DataFrame(evaluation_list, columns=["metric", "value"])
        evaluation_df.to_excel(writer, sheet_name=name, startcol=col_ind, startrow=row_ind, index=False)
        workbook = writer.book
        worksheet = writer.sheets[name]
        # percent_format = workbook.add_format({'num_format': '0.00%'})
        worksheet.set_column('A:Z', 15)
        workbook.close()
        return accuracy, f1_score, auc
    except Exception as e:
        print(f"ERROR at write_data_to_excel, message: {e}")
        traceback.print_exc()


def train_test_and_write_results_cv(final_df, results_file_path, model, k_folds, num_of_processes, random_seed,
                                    antibiotic, all_results_dic, processing_mode, k):
    try:
        X = final_df.drop(['file_ind', 'file_name', 'seq_id', 'seq_name', 'doc_ind', 'strain', 'label'], axis=1).copy()
        y = final_df[['label']].copy()

        files_names = list(final_df['file_name'])
        strains_list = list(final_df['strain'])

        # Create weight according to the ratio of each class
        sample_weight = compute_sample_weight(class_weight='balanced', y=y['label'])

        cv = StratifiedKFold(k_folds, random_state=random_seed, shuffle=True)
        print("Started running Cross Validation for {} folds with {} processes".format(k_folds, num_of_processes))
        now = time.time()
        classes = np.unique(y.values.ravel())
        susceptible_ind = list(classes).index("S")
        resistance_ind = list(classes).index("R")
        temp_scores = cross_val_predict(model, X, y.values.ravel(), cv=cv,
                                        method='predict_proba',
                                        n_jobs=num_of_processes)
        predictions = []
        for p in temp_scores:
            if p[susceptible_ind] > p[resistance_ind]:
                predictions.append("S")
            else:
                predictions.append("R")

        results_df = pd.DataFrame({
            'Strain': strains_list,
            'File name': files_names,
            'Label': y.values.ravel(),
            'Susceptible score': [x[susceptible_ind] for x in temp_scores],
            'Resistance score': [x[resistance_ind] for x in temp_scores],
            'Prediction': predictions
        })

        f = {'Strain': 'first', 'Label': 'first', 'Resistance score': 'mean'}
        results_df_agg = results_df.groupby('File name', as_index=False).agg(f)

        model_parmas = json.dumps(model.get_params())
        accuracy, f1_score, auc = write_data_to_excel(results_df, results_file_path, classes, model_parmas, all_results_dic)
        return accuracy, f1_score, auc
        # print(f"Finished running train_test_and_write_results_cv for antibiotic: {antibiotic} in {round((time.time() - now) / 60, 4)} minutes")
    except Exception as e:
        print(f"ERROR at train_test_and_write_results_cv, message: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    # PARAMS
    MODEL_BACTERIA = Bacteria.GENOME_MIX_NEW.value if len(sys.argv) <= 1 else sys.argv[1]
    K = 10 if len(sys.argv) <= 2 else int(sys.argv[2])  # Choose K size
    random_seed = 1
    num_of_processes = 10
    k_folds = 10
    SHIFT_SIZE = 2  # relevant only for PROCESSING_MODE "overlapping"
    workers = multiprocessing.cpu_count()
    amr_data_file_name = "amr_data_summary.csv"
    # BACTERIA list
    BACTERIA_LIST = [Bacteria.MYCOBACTERIUM_TUBERCULOSIS.value,
                     # Bacteria.PSEUDOMONAS_AUREGINOSA.value
                     ]
    # Define list of model_names and processing method
    D2V_MODEL_PROCESSING_MODE_LIST = [
        ["d2v_2020_09_01_1231.model", ProcessingMode.OVERLAPPING.value],
    ]
    # antibiotic dic
    ANTIBIOTIC_DIC = {
        Bacteria.PSEUDOMONAS_AUREGINOSA.value: ['amikacin', 'levofloxacin', 'meropenem', 'ceftazidime', 'imipenem'],
        Bacteria.MYCOBACTERIUM_TUBERCULOSIS.value: ['isoniazid', 'ethambutol', 'rifampin', 'streptomycin', 'pyrazinamide']
    }
    # model = xgboost.XGBClassifier(random_state=random_seed)
    # model_params = {'max_depth': 4, 'n_estimators': 300, 'max_features': 0.8, 'subsample': 0.8, 'learning_rate': 0.1}
    model = KNeighborsClassifier(n_neighbors=5)
    # PARAMS END
    prefix = '..' if os.name == 'nt' else '.'
    for conf in D2V_MODEL_PROCESSING_MODE_LIST:
        D2V_MODEL_NAME = conf[0]
        PROCESSING_MODE = conf[1]
        for BACTERIA in BACTERIA_LIST:
            antibiotic_list = ANTIBIOTIC_DIC.get(BACTERIA)
            current_date_folder = get_file_name(D2V_MODEL_NAME.replace(".model", ""), None)
            if PROCESSING_MODE == ProcessingMode.OVERLAPPING.value:
                models_folder = os.path.join(prefix, "results_files", MODEL_BACTERIA, "cds_models", f"overlapping_{SHIFT_SIZE}", f"K_{K}")
                results_file_folder = os.path.join(prefix, "results_files", BACTERIA, "cds_embeddings_classification_results", f"overlapping_{SHIFT_SIZE}", f"K_{K}", current_date_folder)
            elif PROCESSING_MODE == ProcessingMode.NON_OVERLAPPING.value:
                models_folder = os.path.join(prefix, "results_files", MODEL_BACTERIA, "cds_models", "non_overlapping", f"K_{K}")
                results_file_folder = os.path.join(prefix, "results_files", BACTERIA, "cds_embeddings_classification_results", "non_overlapping", f"K_{K}", current_date_folder)
            input_folder = os.path.join(prefix, "results_files", BACTERIA, "cds_genome_files")
            amr_file_path = os.path.join(prefix, 'results_files', BACTERIA, amr_data_file_name)
            now_total = time.time()
            now_date = datetime.datetime.now()
            print(f"Started running on: {now_date.strftime('%Y-%m-%d %H:%M:%S')}  D2V_MODEL_NAME: {D2V_MODEL_NAME}  PROCESSING_MODE: {PROCESSING_MODE}  BACTERIA: {BACTERIA}")
            all_results_dic = {"antibiotic": [], "accuracy": [], "f1_score": [], "auc": []}
            for antibiotic in antibiotic_list:
                t1 = time.time()
                print(f"Started xgboost training for bacteria: {BACTERIA} antibiotic: {antibiotic} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE}")
                if not os.path.exists(results_file_folder):
                    os.makedirs(results_file_folder)
                results_file_name = D2V_MODEL_NAME.replace("d2v", antibiotic).replace(".model", ".xlsx")
                results_file_path = os.path.join(results_file_folder, results_file_name)
                files_list = os.listdir(input_folder)
                files_list = [x for x in files_list if ".fna.gz" in x]
                # get AMR data df
                label_df, label_dic = get_label_dic(amr_file_path, files_list, antibiotic, PROCESSING_MODE, K)
                t2 = time.time()
                print(f"Finished running get_label_dic in {round((t2-t1) / 60, 4)} minutes")
                # get only the files with label for the specific antibiotic
                files_list = [x for x in files_list if x.replace(".fna.gz", ".txt.gz") in list(label_dic.keys())]
                now = time.time()
                doc2vec_loader = Doc2VecCDSLoader(input_folder, label_dic, K, PROCESSING_MODE, SHIFT_SIZE, os.path.join(models_folder, D2V_MODEL_NAME))
                final_df = doc2vec_loader.run()
                t3 = time.time()
                print(f"Finished extracting embeddings in {round((t3 - t2) / 60, 4)} minutes")
                accuracy, f1_score, auc = train_test_and_write_results_cv(final_df, results_file_path, model, k_folds, num_of_processes, random_seed, antibiotic, all_results_dic, PROCESSING_MODE, K)
                t4 = time.time()
                print(f"Finished training xgboost for bacteria: {BACTERIA} antibiotic: {antibiotic} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE} in {round((t4-t3) / 60, 4)} minutes  accuracy: {accuracy}  f1_score: {f1_score}   auc: {auc}")
            all_results_df = pd.DataFrame(all_results_dic)
            writer = pd.ExcelWriter(os.path.join(results_file_folder, f"ALL_RESULTS_{current_date_folder}.xlsx"), engine='xlsxwriter')
            all_results_df.to_excel(writer, sheet_name="Sheet1", index=False)
            workbook = writer.book
            workbook.close()
            now_date = datetime.datetime.now()
            print(f"Finished running on: {now_date.strftime('%Y-%m-%d %H:%M:%S')} after {round((time.time() - now_total) / 3600, 4)} hours; D2V_MODEL_NAME: {D2V_MODEL_NAME}  PROCESSING_MODE: {PROCESSING_MODE}  BACTERIA: {BACTERIA}")
    print("DONE!")
