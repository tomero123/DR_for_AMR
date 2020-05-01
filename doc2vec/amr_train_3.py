import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import multiprocessing
import datetime

from doc2vec.Doc2VecTrainer import Doc2VecLoader

import json
import os
import pandas as pd
import numpy as np
import xgboost
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn import metrics
import time


def get_label_df(amr_file_path, files_list, antibiotic):
    amr_df = pd.read_csv(amr_file_path)
    file_name_col = 'NCBI File Name'
    strain_col = 'Strain'
    label_df = amr_df[[file_name_col, strain_col, antibiotic]]
    # Remove antibiotics without resistance data
    label_df = label_df[label_df[antibiotic] != '-']
    # Remove antibiotics with label 'I'
    label_df = label_df[label_df[antibiotic] != 'I']
    # Remove strains which are not in "files list"
    ind_to_save = label_df[file_name_col].apply(lambda x: True if x in [x.replace(".pkl", ".txt.gz") for x in files_list] else False)
    label_df = label_df[ind_to_save]
    label_df.rename(columns={"NCBI File Name": "file_name", antibiotic: "label"}, inplace=True)
    return label_df


def write_data_to_excel(results_df, results_file_path, classes, model_parmas):
    try:
        writer = pd.ExcelWriter(results_file_path, engine='xlsxwriter')
        name = 'Sheet1'
        col_ind = 0
        row_ind = 0
        results_df.to_excel(writer, sheet_name=name, startcol=col_ind, startrow=row_ind, index=False)
        col_ind += results_df.shape[1] + 1
        y_true = list(results_df['Label'])
        y_pred = list(results_df['Prediction'])
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=classes)
        confusion_matrix_df = pd.DataFrame(confusion_matrix, columns=[x + "_Prediction" for x in classes], index=[x + "_Actual" for x in classes])
        confusion_matrix_df.to_excel(writer, sheet_name=name, startcol=col_ind, startrow=row_ind, index=True)
        row_ind += confusion_matrix_df.shape[0] + 2
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1_score = metrics.f1_score(y_true, y_pred, labels=classes, pos_label="R")
        evaluation_list = [["accuracy", accuracy], ["f1_score", f1_score], ["model_parmas", model_parmas]]
        evaluation_df = pd.DataFrame(evaluation_list, columns=["metric", "value"])
        evaluation_df.to_excel(writer, sheet_name=name, startcol=col_ind, startrow=row_ind, index=False)
        workbook = writer.book
        worksheet = writer.sheets[name]
        # percent_format = workbook.add_format({'num_format': '0.00%'})
        worksheet.set_column('A:Z', 15)
        workbook.close()
        # write_roc_curve(y_pred, y_true, results_file_path)
        print('Finished creating results file!')
    except Exception as e:
        print("Error in write_roc_curve.error message: {}".format(e))


def train_test_and_write_results_cv(final_df, results_file_path, model, model_params, k_folds, num_of_processes, random_seed, antibiotic):
    try:
        X = final_df.drop(['label', 'file_name', 'Strain'], axis=1).copy()
        y = final_df[['label']].copy()

        files_names = list(final_df['file_name'])
        strains_list = list(final_df['Strain'])

        # Create weight according to the ratio of each class
        resistance_weight = (y['label'] == "S").sum() / (y['label'] == "R").sum() \
            if (y['label'] == "S").sum() / (y['label'] == "R").sum() > 0 else 1
        sample_weight = np.array([resistance_weight if i == "R" else 1 for i in y['label']])
        print("Resistance_weight for antibiotic: {} is: {}".format(antibiotic, resistance_weight))

        model.set_params(**model_params)
        cv = StratifiedKFold(k_folds, random_state=random_seed, shuffle=True)
        print("Started running Cross Validation for {} folds with {} processes".format(k_folds, num_of_processes))
        now = time.time()
        classes = np.unique(y.values.ravel())
        susceptible_ind = list(classes).index("S")
        resistance_ind = list(classes).index("R")
        temp_scores = cross_val_predict(model, X, y.values.ravel(), cv=cv,
                                        fit_params={'sample_weight': sample_weight}, method='predict_proba',
                                        n_jobs=num_of_processes)
        predictions = []
        for p in temp_scores:
            if p[susceptible_ind] > p[resistance_ind]:
                predictions.append("S")
            else:
                predictions.append("R")
        results_df = pd.DataFrame({
            'Strain': strains_list, 'File name': files_names, 'Label': y.values.ravel(),
            'Susceptible score': [x[susceptible_ind] for x in temp_scores],
            'Resistance score': [x[resistance_ind] for x in temp_scores],
            'Prediction': predictions
        })
        model_parmas = json.dumps(model.get_params())
        write_data_to_excel(results_df, results_file_path, classes, model_parmas)
        print(f"Finished running train_test_and_write_results_cv for antibiotic: {antibiotic} in {round((time.time() - now) / 60, 4)} minutes")
    except Exception as e:
        print(f"ERROR at train_test_and_write_results_cv, message: {e}")


if __name__ == '__main__':
    # PARAMS
    BACTERIA = "pseudomonas_aureginosa" if len(sys.argv) < 2 else sys.argv[1]
    K = 3 if len(sys.argv) < 3 else int(sys.argv[2])  # Choose K size
    random_seed = 1
    num_of_processes = 10
    k_folds = 10
    if os.name == 'nt':
        D2V_MODEL_NAME = "d2v_2020_04_18_1235.model"  # Model Name
    else:
        D2V_MODEL_NAME = "d2v_2020_04_18_1240.model" if len(sys.argv) < 4 else int(sys.argv[3])  # Model Name
    PROCESSING_MODE = "overlapping"  # can be "non_overlapping" or "overlapping"
    SHIFT_SIZE = 1  # relevant only for PROCESSING_MODE "overlapping"
    workers = multiprocessing.cpu_count()
    amr_data_file_name = "amr_data_summary.csv"
    antibiotic_list = ['amikacin', 'levofloxacin', 'meropenem', 'ceftazidime', 'imipenem']
    model = xgboost.XGBClassifier(random_state=random_seed)
    model_params = {'max_depth': 4, 'n_estimators': 300, 'max_features': 0.8, 'subsample': 0.8, 'learning_rate': 0.1}
    # PARAMS END
    prefix = '..' if os.name == 'nt' else '.'
    input_folder = os.path.join(prefix, "results_files", BACTERIA, "genome_documents", f"{PROCESSING_MODE}_{SHIFT_SIZE}", f"K_{K}")
    models_folder = os.path.join(prefix, "results_files", BACTERIA, "models", f"{PROCESSING_MODE}_{SHIFT_SIZE}", f"K_{K}")
    amr_file_path = os.path.join(prefix, 'results_files', BACTERIA, amr_data_file_name)
    results_file_folder = models_folder.replace("models", "embeddings_classification_results")

    now_total = time.time()
    now_date = datetime.datetime.now()
    print(f"Started running on: {now_date.strftime('%Y-%m-%d %H:%M:%S')}")
    for antibiotic in antibiotic_list:
        now = time.time()
        print(f"Started xgboost training for bacteria: {BACTERIA} antibiotic: {antibiotic} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE}")
        if not os.path.exists(results_file_folder):
            os.makedirs(results_file_folder)
        results_file_name = D2V_MODEL_NAME.replace("d2v", antibiotic).replace(".model", ".xlsx")
        results_file_path = os.path.join(results_file_folder, results_file_name)
        files_list = os.listdir(input_folder)
        files_list = [x for x in files_list if ".pkl" in x]
        # get AMR data df
        label_df = get_label_df(amr_file_path, files_list, antibiotic)
        # get only the files with label for the specific antibiotic
        files_list = [x for x in files_list if x.replace(".pkl", ".txt.gz") in list(label_df['file_name'])]
        doc2vec_loader = Doc2VecLoader(input_folder, files_list, os.path.join(models_folder, D2V_MODEL_NAME))
        em_df = doc2vec_loader.run()
        final_df = em_df.join(label_df.set_index('file_name'), on='file_name')
        train_test_and_write_results_cv(final_df, results_file_path, model, model_params, k_folds, num_of_processes, random_seed, antibiotic)
        print(f"label_df shape: {label_df.shape}")
        print(f"em_df shape: {em_df.shape}")
        print(f"Finished training xgboost for bacteria: {BACTERIA} antibiotic: {antibiotic} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE} in {round((time.time() - now) / 60, 4)} minutes")
    now_date = datetime.datetime.now()
    print(f"Finished running on: {now_date.strftime('%Y-%m-%d %H:%M:%S')} after {round((time.time() - now_total) / 3600, 4)} hours")
