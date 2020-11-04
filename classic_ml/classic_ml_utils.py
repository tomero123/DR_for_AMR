import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import multiprocessing
# import pathos.multiprocessing as multiprocessing
import json
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
import traceback
import time
import datetime

from constants import PREDEFINED_FEATURES_LIST, TIME_STR
from utils import get_time_as_str

pd.options.mode.chained_assignment = None  # default='warn'


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
        # Remove kmers with have "N" in them
        temp_count = kmers_df.shape[0]
        kmers_df = kmers_df[~kmers_df['Unnamed: 0'].str.contains("N")]
        kmers_final_count = kmers_df.shape[0]
        print(f"Removed {temp_count - kmers_final_count} kmers that include 'N'")
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
        # final_df = kmers_df.merge(label_df, how="inner", right_on="file_name", left_index=True)
        print("final_df for antibiotic: {} have {} Strains with label and {} features".format(antibiotic, final_df.shape[0], final_df.shape[1] - 2))
        return final_df
    except Exception as e:
        print(f"ERROR at get_final_df, message: {e}")
        traceback.print_exc()


def train_test_and_write_results(final_df, amr_df, results_file_path, model, antibiotic, kmers_original_count, kmers_final_count, features_selection_n, all_results_dic, bacteria, use_predefined_features_list):
    try:
        non_features_columns = ['file_id', 'file_name', 'Strain', 'label']
        train_file_id_list = list(amr_df[amr_df[f"{antibiotic}_group"].isin([1, 2, 3, 4])]["file_id"])
        test_file_id_list = list(amr_df[amr_df[f"{antibiotic}_group"] == 5]["file_id"])
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

        if use_predefined_features_list:
            predefined_features_list = PREDEFINED_FEATURES_LIST.get(bacteria, {}).get(antibiotic, [])
            if predefined_features_list:
                print(f"Using predefined_features_list in size: {len(predefined_features_list)} for bacteria: {bacteria} and antibiotic: {antibiotic}")
                X_train = X_train[predefined_features_list]
                X_test = X_test[predefined_features_list]
            else:
                raise Exception(f"Couldn't find valid predefined_features_list for bacteria: {bacteria} and antibiotic: {antibiotic}")

        # Features Selection
        elif features_selection_n:
            print(f"Started Feature selection model fit antibiotic: {antibiotic}")
            now = time.time()
            model.fit(X_train, y_train.values.ravel(), sample_weight=sample_weight)
            # Write csv of data after FS
            d = model.feature_importances_
            most_important_index = sorted(range(len(d)), key=lambda i: d[i], reverse=True)[:features_selection_n]
            temp_df = X_train.iloc[:, most_important_index]
            temp_df["label"] = y_train.values.ravel()
            temp_df["file_id"] = list(final_df_train.loc[:, 'file_id'])
            temp_df["Strain"] = list(final_df_train.loc[:, 'Strain'])
            temp_df["file_name"] = list(final_df_train.loc[:, 'file_name'])
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

        model.fit(X_train, y_train.values.ravel(), sample_weight=sample_weight)

        # Save model
        model_file_path = results_file_path.replace(".xlsx", "_MODEL.p")
        with open(model_file_path, 'wb') as f:
            pickle.dump(model, f)

        temp_scores = model.predict_proba(X_test)
        true_results = y_test.values.ravel()

        predictions = [1 if p[1] >= p[0] else 0 for p in temp_scores]
        resistance_score = [x[1] for x in temp_scores]

        results_dic = {
            'Fold': "NA",
            'file_id': list(final_df_test['file_id']),
            'Strain': list(final_df_test['Strain']),
            'File name': list(final_df_test['file_name']),
            "true_results": true_results,
            "resistance_score": resistance_score,
            "predictions": predictions,
        }

        results_list = [results_dic]

        model_parmas = json.dumps(model.get_params())
        write_data_to_excel(antibiotic, results_list, results_file_path, model_parmas, kmers_original_count, kmers_final_count, all_results_dic)
        print(f"Finished running train_test_and_write_results_cv for antibiotic: {antibiotic} in {round((time.time() - now) / 60, 4)} minutes")
    except Exception as e:
        print(f"ERROR at train_test_and_write_results, message: {e}")
        traceback.print_exc()


def train_test_and_write_results_cv(final_df_file_path, amr_df, results_file_path, model, antibiotic, kmers_original_count, kmers_final_count, features_selection_n, all_results_dic, use_multiprocess):
    try:
        now = time.time()
        n_folds = amr_df[f"{antibiotic}_group"].max()
        train_groups_list, test_groups_list = get_train_and_test_groups(n_folds)

        inputs_list = []

        print(f"{datetime.datetime.now().strftime(TIME_STR)} STARTED creating folds data. antibiotic: {antibiotic}")
        for train_group_list, test_group in zip(train_groups_list, test_groups_list):
            train_file_id_list = list(amr_df[amr_df[f"{antibiotic}_group"].isin(train_group_list)]["file_id"])
            test_file_id_list = list(amr_df[amr_df[f"{antibiotic}_group"] == test_group]["file_id"])
            inputs_list.append([test_group, final_df_file_path, train_file_id_list, test_file_id_list, results_file_path, model, antibiotic, features_selection_n])

        print(f"{datetime.datetime.now().strftime(TIME_STR)} FINISHED creating folds data. antibiotic: {antibiotic}")
        print(f"{datetime.datetime.now().strftime(TIME_STR)} STARTED training models. antibiotic: {antibiotic}")
        if use_multiprocess:
            with multiprocessing.Pool(processes=n_folds) as pool:
                results_list = pool.starmap(train_test_one_fold, inputs_list)
        else:
            results_list = []
            for i in inputs_list:
                results_list.append(train_test_one_fold(*i))

        model_parmas = json.dumps(model.get_params())
        write_data_to_excel(antibiotic, results_list, results_file_path, model_parmas, kmers_original_count,
                            kmers_final_count, all_results_dic)
        print(f"***{datetime.datetime.now().strftime(TIME_STR)} FINISHED training models for antibiotic: {antibiotic} in {round((time.time() - now) / 60, 4)} minutes***")
    except Exception as e:
        print(f"ERROR at train_test_and_write_results_cv, message: {e}")
        traceback.print_exc()


def write_data_to_excel(antibiotic, results_list, results_file_path, model_parmas, kmers_original_count, kmers_final_count, all_results_dic):
    try:
        writer = pd.ExcelWriter(results_file_path, engine='xlsxwriter')
        name = 'Sheet1'
        col_ind = 0
        row_ind = 0
        results_df = pd.DataFrame(columns=['file_id', 'file_name', 'strain', 'label', 'resistance_score', 'prediction', 'fold'])
        evaluation_df = pd.DataFrame(columns=['fold', 'auc', 'accuracy', 'f1_score', 'precision', 'recall'])
        for fold_dic in results_list:
            fold = fold_dic['Fold']
            file_id_list = fold_dic["file_id"]
            file_name_list = fold_dic["File name"]
            strain_list = fold_dic["Strain"]
            fold_list = [fold] * len(file_id_list)
            y_true = fold_dic['true_results']
            y_pred = fold_dic['predictions']
            y_pred_score = fold_dic['resistance_score']
            results_df = results_df.append(pd.DataFrame({
                'file_id': file_id_list,
                'file_name': file_name_list,
                'strain': strain_list,
                'label': y_true,
                'resistance_score': y_pred_score,
                'prediction': y_pred,
                'fold': fold_list
            }))
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_score)
            accuracy = metrics.accuracy_score(y_true, y_pred)
            precision = metrics.precision_score(y_true, y_pred)
            recall = metrics.recall_score(y_true, y_pred)
            f1_score = metrics.f1_score(y_true, y_pred)
            auc = metrics.auc(fpr, tpr)
            all_results_dic["antibiotic"].append(antibiotic)
            all_results_dic["fold"].append(fold)
            all_results_dic["accuracy"].append(accuracy)
            all_results_dic["precision"].append(precision)
            all_results_dic["recall"].append(recall)
            all_results_dic["f1_score"].append(f1_score)
            all_results_dic["auc"].append(auc)
            evaluation_df.loc[len(evaluation_df.index)] = [fold, auc, accuracy, f1_score, precision, recall]

        # Add mean and std to evaluation_df
        evaluation_mean = list(evaluation_df.mean())
        evaluation_std = list(evaluation_df.std())
        evaluation_mean[0] = "mean"
        evaluation_std[0] = "std"
        evaluation_df.loc[len(evaluation_df.index)] = evaluation_mean
        evaluation_df.loc[len(evaluation_df.index)] = evaluation_std

        results_df.to_excel(writer, sheet_name=name, startcol=col_ind, startrow=row_ind, index=False)
        col_ind += results_df.shape[1] + 1
        confusion_matrix = metrics.confusion_matrix(list(results_df['label']), list(results_df['prediction']))
        confusion_matrix_df = pd.DataFrame(confusion_matrix, columns=[x + "_Prediction" for x in ["S", "R"]], index=[x + "_Actual" for x in ["S", "R"]])
        confusion_matrix_df.to_excel(writer, sheet_name=name, startcol=col_ind, startrow=row_ind, index=True)
        row_ind += confusion_matrix_df.shape[0] + 2
        evaluation_df.to_excel(writer, sheet_name=name, startcol=col_ind, startrow=row_ind, index=False)
        workbook = writer.book
        worksheet = writer.sheets[name]
        # percent_format = workbook.add_format({'num_format': '0.00%'})
        worksheet.set_column('A:Z', 15)
        workbook.close()
        print(f"Finished creating results for antibiotic: {antibiotic} ; auc: {evaluation_mean[1]} accuracy: {evaluation_mean[2]} f1_score: {evaluation_mean[3]} precision: {evaluation_mean[4]} recall: {evaluation_mean[5]}")
    except Exception as e:
        print("Error in write_data_to_excel.error message: {}".format(e))
        traceback.print_exc()


def get_current_results_folder(results_folder_name, features_selection_n, test_method):
    current_results_folder = get_time_as_str()
    if results_folder_name is not None:
        current_results_folder += f"_{results_folder_name}"
    else:
        if features_selection_n:
            current_results_folder += f"_FS_{features_selection_n}"
        current_results_folder += f"_{test_method}"
    return current_results_folder


def convert_results_df_to_new_format(all_results_agg, metrics_order):
    antibiotic_list = sorted(all_results_agg["antibiotic"].unique())
    mean_std_list = []
    mean_list = []
    new_results_columns = []
    for col in metrics_order:
        for antibiotic in antibiotic_list:
            mean = round(float(all_results_agg[col][(all_results_agg["antibiotic"] == antibiotic) & (all_results_agg["category"] == "mean")]), 3)
            std = round(float(all_results_agg[col][(all_results_agg["antibiotic"] == antibiotic) & (all_results_agg["category"] == "std")]), 3)
            mean_list.append(mean)
            mean_std_list.append(f"{mean}+-{std}")
            new_results_columns.append(f"{col}_{antibiotic}")
    return pd.DataFrame(data=[mean_list, mean_std_list], columns=new_results_columns)


def get_agg_results_df(all_results_df, metrics_order):
    mean_df = all_results_df.groupby("antibiotic").mean()
    mean_df["category"] = "mean"
    std_df = all_results_df.groupby("antibiotic").std()
    std_df["category"] = "std"
    all_results_agg = mean_df.append(std_df)
    all_results_agg = all_results_agg.loc[:, ["category"] + metrics_order].sort_values(by=["category", "antibiotic"])
    return all_results_agg.reset_index()


def get_all_resulst_df(all_results_dic, metrics_order):
    all_results_df = pd.DataFrame(all_results_dic)
    all_results_df = all_results_df.loc[:, ["antibiotic", "fold"] + metrics_order].sort_values(by=["antibiotic", "fold"])
    return all_results_df


def get_train_and_test_groups(n_folds):
    train_groups_list = []
    test_groups_list = list(range(1, n_folds + 1))
    for i in test_groups_list:
        train_groups_list.append([x for x in test_groups_list if i != x])
    return train_groups_list, test_groups_list


def train_test_one_fold(test_group, final_df_file_path, train_file_id_list, test_file_id_list, results_file_path, model, antibiotic, features_selection_n):
    final_df = pd.read_csv(final_df_file_path, compression='gzip')
    final_df['label'].replace('R', 1, inplace=True)
    final_df['label'].replace('S', 0, inplace=True)
    final_df_train = final_df[final_df["file_id"].isin(train_file_id_list)]
    final_df_test = final_df[final_df["file_id"].isin(test_file_id_list)]
    non_features_columns = ['file_id', 'file_name', 'Strain', 'label']
    X_train = final_df_train.drop(non_features_columns, axis=1).copy()
    y_train = final_df_train[['label']].copy()
    X_test = final_df_test.drop(non_features_columns, axis=1).copy()
    y_test = final_df_test[['label']].copy()
    print(f"{datetime.datetime.now().strftime(TIME_STR)} FOLD#{test_group} X_train size: {X_train.shape}  y_train size: {y_train.shape}  X_test size: {X_test.shape}  y_test size: {y_test.shape}")

    # Create weight according to the ratio of each class
    resistance_weight = (y_train['label'] == 0).sum() / (y_train['label'] == 1).sum() \
        if (y_train['label'] == 0).sum() / (y_train['label'] == 1).sum() > 0 else 1
    sample_weight = np.array([resistance_weight if i == 1 else 1 for i in y_train['label']])
    print(f"{datetime.datetime.now().strftime(TIME_STR)} FOLD#{test_group} Resistance_weight for antibiotic: {antibiotic} is: {resistance_weight}")

    # Features Selection
    if features_selection_n:
        print(f"{datetime.datetime.now().strftime(TIME_STR)} FOLD#{test_group} Started running Feature selection for antibiotic: {antibiotic}")
        now = time.time()
        model.fit(X_train, y_train.values.ravel(), sample_weight=sample_weight)
        # Write csv of data after FS
        d = model.feature_importances_
        most_important_index = sorted(range(len(d)), key=lambda i: d[i], reverse=True)[:features_selection_n]
        temp_df = X_train.iloc[:, most_important_index]
        temp_df["label"] = y_train.values.ravel()
        temp_df["file_id"] = list(final_df_train.loc[:, 'file_id'])
        temp_df["Strain"] = list(final_df_train.loc[:, 'Strain'])
        temp_df["file_name"] = list(final_df_train.loc[:, 'file_name'])
        temp_df.to_csv(results_file_path.replace("RESULTS", "FS_DATA").replace("xlsx", "csv"), index=False)
        importance_list = []
        for ind in most_important_index:
            importance_list.append([X_train.columns[ind], d[ind]])
        importance_df = pd.DataFrame(importance_list, columns=["kmer", "score"])
        importance_df.to_csv(results_file_path.replace("RESULTS", "FS_IMPORTANCE").replace("xlsx", "csv"), index=False)
        selection = SelectFromModel(model, threshold=-np.inf, prefit=True, max_features=features_selection_n)
        X_train = selection.transform(X_train)
        X_test = selection.transform(X_test)
        print(f"{datetime.datetime.now().strftime(TIME_STR)} FOLD#{test_group} Finished running Feature selection for antibiotic: {antibiotic} in {round((time.time() - now) / 60, 4)} minutes ; X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}")

    # model.fit(X_train, y_train.values.ravel(), sample_weight=sample_weight)

    # eval_set = [(X_test, y_test)]
    model.fit(X_train, y_train.values.ravel(), sample_weight=sample_weight,
              # eval_metric="error", eval_set=eval_set, # verbose=True,
              # early_stopping_rounds=20
              )

    temp_scores = model.predict_proba(X_test)
    true_results = y_test.values.ravel()
    resistance_score = [x[1] for x in temp_scores]
    predictions = [1 if p[1] >= p[0] else 0 for p in temp_scores]

    results_dic = {
        'Fold': test_group,
        'file_id': list(final_df_test['file_id']),
        'Strain': list(final_df_test['Strain']),
        'File name': list(final_df_test['file_name']),
        "true_results": true_results,
        "resistance_score": resistance_score,
        "predictions": predictions,
    }

    return results_dic
