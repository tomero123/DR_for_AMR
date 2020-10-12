import sys

from utils import get_time_as_str

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import json
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import xgboost
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn import metrics
import matplotlib.pyplot as plt
import traceback
import time


def get_kmers_df(path, dataset_file_name, kmers_map_file_name, rare_th, common_th_subtract):
    try:
        now = time.time()
        kmers_df = pd.read_csv(os.path.join(path, dataset_file_name), compression='gzip')
        kmers_original_count = kmers_df.shape[0]
        print("kmers_df shape: {}".format(kmers_df.shape))
        # # remove too rare and too common kmers
        if rare_th:
            non_zero_strains_count = kmers_df.astype(bool).sum(axis=1)
            kmers_df = kmers_df[non_zero_strains_count > rare_th]
            print("rare_th: {} ; kmers_df shape after rare kmers removal: {}".format(rare_th, kmers_df.shape))
        if common_th_subtract:
            non_zero_strains_count = kmers_df.astype(bool).sum(axis=1)
            kmers_df = kmers_df[non_zero_strains_count < kmers_df.shape[1] - common_th_subtract]
            print("common kmers value: {} ; kmers_df shape after common kmers removal: {}".format(kmers_df.shape[1] - common_th_subtract, kmers_df.shape))
        kmers_final_count = kmers_df.shape[0]
        with open(os.path.join(path, kmers_map_file_name), 'r') as f:
            all_kmers_map = json.loads(f.read())
        kmers_df = kmers_df.rename(columns=all_kmers_map)
        kmers_df = kmers_df.set_index(['Unnamed: 0'])
        # Transpose to have strains as rows and kmers as columns
        kmers_df = kmers_df.T
        kmers_df.index = kmers_df.index.str.replace("_genomic.txt.gz", "")
        print("kmers_df shape: {}".format(kmers_df.shape))
        print("Finished running get_kmers_df in {} minutes".format(round((time.time() - now)/60), 4))
        return kmers_df, kmers_original_count, kmers_final_count
    except Exception as e:
        print(f"ERROR at get_kmers_df, message: {e}")
        traceback.print_exc()


def get_label_df(amr_df, antibiotic):
    file_name_col = 'NCBI File Name'
    file_id_col = 'file_id'
    strain_col = 'Strain'
    label_df = amr_df[[file_id_col, file_name_col, strain_col, antibiotic]]
    # Remove antibiotics without resistance data
    label_df = label_df[label_df[antibiotic] != '-']
    # Remove antibiotics with label 'I'
    label_df = label_df[label_df[antibiotic] != 'I']
    # # Remove strains which are not in "files list"
    # ind_to_save = label_df[file_name_col].apply(lambda x: True if x in [x.replace("_cds_from_genomic.fna.gz", "") for x in files_list] else False)
    # new_label_df = label_df[ind_to_save]
    label_df.rename(columns={"NCBI File Name": "file_name", antibiotic: "label"}, inplace=True)
    return label_df


def get_final_df(antibiotic, kmers_df, label_df):
    try:
        print(f"antibiotic: {antibiotic}, label_df shape: {label_df.shape}")
        if os.name == 'nt':
            # LOCAL ONLY!!!!$#!@!#@#$@!#@!
            final_df = kmers_df.join(label_df, how="left")
            final_df["label"] = ["S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R"]
            final_df["file_id"] = list(range(5,55))
        else:
            # Join (inner) between kmers_df and label_df
            final_df = kmers_df.merge(label_df, how="inner", right_on="file_name", left_index=True)
        print("final_df for antibiotic: {} have {} Strains with label and {} features".format(antibiotic, final_df.shape[0], final_df.shape[1] - 2))
        return final_df
    except Exception as e:
        print(f"ERROR at get_final_df, message: {e}")
        traceback.print_exc()


def train_test_and_write_results(final_df, amr_df, results_file_path, model, model_params, antibiotic, kmers_original_count, kmers_final_count, features_selection_n, all_results_dic):
    try:
        non_features_columns = ['file_id', 'file_name', 'Strain', 'label']
        train_file_id_list = list(amr_df[amr_df[f"{antibiotic}_is_train"] == 1]["file_id"])
        test_file_id_list = list(amr_df[amr_df[f"{antibiotic}_is_train"] == 0]["file_id"])
        final_df['label'].replace('R', 1, inplace=True)
        final_df['label'].replace('S', 0, inplace=True)
        final_df_train = final_df[final_df["file_id"].isin(train_file_id_list)]
        final_df_test = final_df[final_df["file_id"].isin(test_file_id_list)]
        X_train = final_df_train.drop(non_features_columns, axis=1).copy()
        y_train = final_df_train[['label']].copy()
        X_test = final_df_test.drop(non_features_columns, axis=1).copy()
        y_test = final_df_test[['label']].copy()
        print(f"X_train size: {X_train.shape}  y_train size: {y_train.shape}  X_test size: {X_test.shape}  y_test size: {y_test.shape}")

        # Create weight according to the ratio of each class
        resistance_weight = (y_train['label'] == 0).sum() / (y_train['label'] == 1).sum() \
            if (y_train['label'] == 0).sum() / (y_train['label'] == 1).sum() > 0 else 1
        sample_weight = np.array([resistance_weight if i == 1 else 1 for i in y_train['label']])
        print("Resistance_weight for antibiotic: {} is: {}".format(antibiotic, resistance_weight))

        # Features Selection
        if features_selection_n:
            print(f"Started Feature selection model fit antibiotic: {antibiotic}")
            model.set_params(**model_params)
            now = time.time()
            model.fit(X_train, y_train.values.ravel(), sample_weight=sample_weight)
            # Write csv of data after FS
            d = model.feature_importances_
            most_important_index = sorted(range(len(d)), key=lambda i: d[i], reverse=True)[:features_selection_n]
            temp_df = X_train.iloc[:, most_important_index]
            temp_df["label"] = y_train.values.ravel()
            temp_df.to_csv(results_file_path.replace("RESULTS", "FS_DATA").replace("xlsx", "csv"), index=False)
            importance_list = []
            for ind in most_important_index:
                importance_list.append([X_train.columns[ind], d[ind]])
            importance_df = pd.DataFrame(importance_list, columns=["kmer", "score"])
            importance_df.to_csv(results_file_path.replace("RESULTS", "FS_IMPORTANCE").replace("xlsx", "csv"), index=False)
            print(f"Finished running Feature selection model fit for antibiotic: {antibiotic} in {round((time.time() - now) / 60, 4)} minutes ; X_train.shape: {X_train.shape}")
            print(f"Started Feature selection SelectFromModel for antibiotic: {antibiotic}")
            now = time.time()
            selection = SelectFromModel(model, threshold=-np.inf, prefit=True, max_features=features_selection_n)
            X_train = selection.transform(X_train)
            X_test = selection.transform(X_test)
            print(f"Finished running Feature selection SelectFromModel for antibiotic: {antibiotic} in {round((time.time() - now) / 60, 4)} minutes ; X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}")

        model.set_params(**model_params)
        model.fit(X_train, y_train.values.ravel(), sample_weight=sample_weight)

        # Save model
        model_file_path = results_file_path.replace(".xlsx", "_MODEL.p")
        with open(model_file_path, 'wb') as f:
            pickle.dump(model, f)

        temp_scores = model.predict_proba(X_test)
        true_results = y_test.values.ravel()

        predictions = []
        for p in temp_scores:
            if p[0] > p[1]:
                predictions.append(0)
            else:
                predictions.append(1)
        results_df = pd.DataFrame({
            'file_id': list(final_df_test['file_id']),
            'Strain': list(final_df_test['Strain']),
            'File name': list(final_df_test['file_name']),
            'Label': true_results,
            'Resistance score': [x[1] for x in temp_scores],
            'Prediction': predictions
        })
        model_parmas = json.dumps(model.get_params())
        write_data_to_excel(antibiotic, results_df, results_file_path, model_parmas, kmers_original_count, kmers_final_count, all_results_dic)
        print(f"Finished running train_test_and_write_results_cv for antibiotic: {antibiotic} in {round((time.time() - now) / 60, 4)} minutes")
    except Exception as e:
        print(f"ERROR at train_test_and_write_results_cv, message: {e}")
        traceback.print_exc()


def train_test_and_write_results_cv(final_df, amr_df, results_file_path, model, model_params, antibiotic, kmers_original_count, kmers_final_count, features_selection_n, all_results_dic, random_seed):
    try:
        k_folds = 10
        num_of_processes = 10
        non_features_columns = ['file_id', 'file_name', 'Strain', 'label']

        final_df['label'].replace('R', 1, inplace=True)
        final_df['label'].replace('S', 0, inplace=True)
        X = final_df.drop(non_features_columns, axis=1).copy()
        y = final_df[['label']].copy()

        print(f"X size: {X.shape}  y size: {y.shape}")

        # Create weight according to the ratio of each class
        resistance_weight = (y['label'] == 0).sum() / (y['label'] == 1).sum() \
            if (y['label'] == 0).sum() / (y['label'] == 1).sum() > 0 else 1
        sample_weight = np.array([resistance_weight if i == 1 else 1 for i in y['label']])
        print("Resistance_weight for antibiotic: {} is: {}".format(antibiotic, resistance_weight))

        # Features Selection
        if features_selection_n:
            print(f"Started running Feature selection for antibiotic: {antibiotic}")
            model.set_params(**model_params)
            now = time.time()
            model.fit(X, y.values.ravel(),
                      # sample_weight=sample_weight
                      )
            # Write csv of data after FS
            d = model.feature_importances_
            most_important_index = sorted(range(len(d)), key=lambda i: d[i], reverse=True)[:features_selection_n]
            temp_df = X.iloc[:, most_important_index]
            temp_df["label"] = y.values.ravel()
            temp_df.to_csv(results_file_path.replace("RESULTS", "FS_DATA").replace("xlsx", "csv"), index=False)
            importance_list = []
            for ind in most_important_index:
                importance_list.append([X.columns[ind], d[ind]])
            importance_df = pd.DataFrame(importance_list, columns=["kmer", "score"])
            importance_df.to_csv(results_file_path.replace("RESULTS", "FS_IMPORTANCE").replace("xlsx", "csv"), index=False)
            selection = SelectFromModel(model, threshold=-np.inf, prefit=True, max_features=features_selection_n)
            X = selection.transform(X)
            print(f"Finished running Feature selection for antibiotic: {antibiotic} in {round((time.time() - now) / 60, 4)} minutes ; X.shape: {X.shape}")

        model.set_params(**model_params)
        cv = StratifiedKFold(k_folds, random_state=random_seed, shuffle=True)
        print("Started running Cross Validation for {} folds with {} processes".format(k_folds, num_of_processes))
        now = time.time()
        temp_scores = cross_val_predict(model, X, y.values.ravel(), cv=cv,
                                        fit_params={'sample_weight': sample_weight}, method='predict_proba',
                                        n_jobs=num_of_processes)

        true_results = y.values.ravel()

        predictions = []
        for p in temp_scores:
            if p[0] > p[1]:
                predictions.append(0)
            else:
                predictions.append(1)
        results_df = pd.DataFrame({
            'file_id': list(final_df['file_id']),
            'Strain': list(final_df['Strain']),
            'File name': list(final_df['file_name']),
            'Label': true_results,
            'Resistance score': [x[1] for x in temp_scores],
            'Prediction': predictions
        })
        model_parmas = json.dumps(model.get_params())
        write_data_to_excel(antibiotic, results_df, results_file_path, model_parmas, kmers_original_count,
                            kmers_final_count, all_results_dic)
        print(
            f"Finished running train_test_and_write_results_cv for antibiotic: {antibiotic} in {round((time.time() - now) / 60, 4)} minutes")
    except Exception as e:
        print(f"ERROR at train_test_and_write_results_cv, message: {e}")


def write_data_to_excel(antibiotic, results_df, results_file_path, model_parmas, kmers_original_count, kmers_final_count, all_results_dic):
    try:
        writer = pd.ExcelWriter(results_file_path, engine='xlsxwriter')
        name = 'Sheet1'
        col_ind = 0
        row_ind = 0
        results_df.to_excel(writer, sheet_name=name, startcol=col_ind, startrow=row_ind, index=False)
        col_ind += results_df.shape[1] + 1
        y_true = list(results_df['Label'])
        y_pred = list(results_df['Prediction'])
        y_pred_score = list(results_df['Resistance score'])
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_score)
        confusion_matrix_df = pd.DataFrame(confusion_matrix, columns=[x + "_Prediction" for x in ["S", "R"]], index=[x + "_Actual" for x in ["S", "R"]])
        confusion_matrix_df.to_excel(writer, sheet_name=name, startcol=col_ind, startrow=row_ind, index=True)
        row_ind += confusion_matrix_df.shape[0] + 2
        accuracy = metrics.accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)
        f1_score = metrics.f1_score(y_true, y_pred)
        auc = metrics.auc(fpr, tpr)
        all_results_dic["antibiotic"].append(antibiotic)
        all_results_dic["accuracy"].append(accuracy)
        all_results_dic["precision"].append(precision)
        all_results_dic["recall"].append(recall)
        all_results_dic["f1_score"].append(f1_score)
        all_results_dic["auc"].append(auc)
        evaluation_list = [["accuracy", accuracy], ["precision", precision], ["recall", recall], ["f1_score", f1_score],
                           ["auc", auc], ["model_parmas", model_parmas], ["kmers_original_count", kmers_original_count],
                           ["kmers_final_count", kmers_final_count]]
        evaluation_df = pd.DataFrame(evaluation_list, columns=["metric", "value"])
        evaluation_df.to_excel(writer, sheet_name=name, startcol=col_ind, startrow=row_ind, index=False)
        workbook = writer.book
        worksheet = writer.sheets[name]
        # percent_format = workbook.add_format({'num_format': '0.00%'})
        worksheet.set_column('A:Z', 15)
        workbook.close()
        print(f"Finished creating results for antibiotic: {antibiotic} ; accuracy: {accuracy}  f1_score: {f1_score}  auc: {auc} precision: {precision} recall: {recall}")
    except Exception as e:
        print("Error in write_roc_curve.error message: {}".format(e))


def write_roc_curve(y_pred, y_true, results_file_path):
    try:
        labels = [int(i == 1) for i in y_true]
        predictions = [int(i == 1) for i in y_pred]
        fpr, tpr, _ = metrics.roc_curve(labels, predictions)
        auc = round(metrics.roc_auc_score(labels, predictions), 3)
        plt.figure(figsize=(10, 10))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.plot(fpr, tpr, label="auc=" + str(auc))
        plt.savefig(results_file_path.replace(".xlsx", ".png"),  bbox_inches="tight")
    except Exception as e:
        print("Error in write_roc_curve.error message: {}".format(e))


def get_current_results_folder(results_folder_name, features_selection_n, test_method):
    current_results_folder = get_time_as_str()
    if results_folder_name is not None:
        current_results_folder += f"_{results_folder_name}"
    else:
        if features_selection_n:
            current_results_folder += f"_FS_{features_selection_n}"
        current_results_folder += f"_{test_method}"
    return current_results_folder
