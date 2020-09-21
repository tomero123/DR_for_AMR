import sys

from doc2vec_cds.FaissKNeighbors import FaissKNeighbors

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import os
import traceback
import json
import pandas as pd
import numpy as np
from sklearn import metrics


def get_label_df(amr_df, files_list, antibiotic):
    file_name_col = 'NCBI File Name'
    file_id_col = 'file_id'
    strain_col = 'Strain'
    label_df = amr_df[[file_id_col, file_name_col, strain_col, antibiotic]]
    # Remove antibiotics without resistance data
    label_df = label_df[label_df[antibiotic] != '-']
    # Remove antibiotics with label 'I'
    label_df = label_df[label_df[antibiotic] != 'I']
    # Remove strains which are not in "files list"
    ind_to_save = label_df[file_name_col].apply(lambda x: True if x in [x.replace("_cds_from_genomic.fna.gz", "") for x in files_list] else False)
    new_label_df = label_df[ind_to_save]
    new_label_df.rename(columns={"NCBI File Name": "file_name", antibiotic: "label"}, inplace=True)
    return new_label_df


def write_data_to_excel(writer, antibiotic, agg_method, results_df, classes, model_parmas, all_results_dic):
    try:
        col_ind = 0
        row_ind = 0
        results_df.to_excel(writer, sheet_name=agg_method, startcol=col_ind, startrow=row_ind, index=False)
        col_ind += results_df.shape[1] + 1
        y_true = [1 if x == "R" else 0 for x in list(results_df['Label'])]
        y_pred = [1 if x == "R" else 0 for x in list(results_df['Prediction'])]
        y_pred_score = list(results_df['Resistance score'])
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        confusion_matrix_df = pd.DataFrame(confusion_matrix, columns=[x + "_Prediction" for x in classes], index=[x + "_Actual" for x in classes])
        confusion_matrix_df.to_excel(writer, sheet_name=agg_method, startcol=col_ind, startrow=row_ind, index=True)
        row_ind += confusion_matrix_df.shape[0] + 2
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1_score = metrics.f1_score(y_true, y_pred)
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred_score)
        auc = metrics.auc(fpr, tpr)
        all_results_dic["antibiotic"].append(antibiotic)
        all_results_dic["agg_method"].append(agg_method)
        all_results_dic["accuracy"].append(accuracy)
        all_results_dic["f1_score"].append(f1_score)
        all_results_dic["auc"].append(auc)
        evaluation_list = [["accuracy", accuracy], ["f1_score", f1_score], ["auc", auc], ["model_parmas", model_parmas]]
        evaluation_df = pd.DataFrame(evaluation_list, columns=["metric", "value"])
        evaluation_df.to_excel(writer, sheet_name=agg_method, startcol=col_ind, startrow=row_ind, index=False)
        worksheet = writer.sheets[agg_method]
        # percent_format = workbook.add_format({'num_format': '0.00%'})
        worksheet.set_column('A:Z', 15)
        return accuracy, f1_score, auc
    except Exception as e:
        print(f"ERROR at write_data_to_excel, message: {e}")
        traceback.print_exc()


def train_test_and_write_results_cv(final_df, antibiotic, results_file_path, model, all_results_dic, amr_df, use_faiss_knn):
    try:
        non_features_columns = ['file_id', 'seq_id', 'doc_ind', 'label']
        train_file_id_list = list(amr_df[amr_df[f"{antibiotic}_is_train"] == 1]["file_id"])
        test_file_id_list = list(amr_df[amr_df[f"{antibiotic}_is_train"] == 0]["file_id"])
        final_df_train = final_df[final_df["file_id"].isin(train_file_id_list)]
        final_df_test = final_df[final_df["file_id"].isin(test_file_id_list)]
        X_train = final_df_train.drop(non_features_columns, axis=1).copy()
        y_train = final_df_train[['label']].copy()
        X_test = final_df_test.drop(non_features_columns, axis=1).copy()
        y_test = final_df_test[['label']].copy()
        print(f"X_train size: {X_train.shape}  y_train size: {y_train.shape}  X_test size: {X_test.shape}  y_test size: {y_test.shape}")

        test_files_ids = list(final_df_test['file_id'])
        # strains_list = list(final_df['Strain'])

        # Create weight according to the ratio of each class
#         sample_weight = compute_sample_weight(class_weight='balanced', y=y['label'])
        classes = np.unique(y_train.values.ravel())
        susceptible_ind = list(classes).index("S")
        resistance_ind = list(classes).index("R")
        # cv = StratifiedKFold(k_folds, random_state=random_seed, shuffle=True)
        # temp_scores = cross_val_predict(model, X, y.values.ravel(), cv=cv,
        #                                 method='predict_proba',
        #                                 n_jobs=num_of_processes)
        # true_results = y.values.ravel()
        if use_faiss_knn and os.name != 'nt':
            model_faiss = FaissKNeighbors(7)
            model_faiss.fit(X_train, y_train.values.ravel())
            temp_scores = model_faiss.predict_proba(X_test)
        else:
            model.fit(X_train, y_train.values.ravel())
            temp_scores = model.predict_proba(X_test)
        true_results = y_test.values.ravel()

        test_agg_list = ["mean_highest$100", "mean_highest$300", "mean_highest$600", "mean_all"]
        results_dic = {}
        writer = pd.ExcelWriter(results_file_path, engine='xlsxwriter')

        results_df = pd.DataFrame({
            'file_id': test_files_ids,
            'Label': true_results,
            'Resistance score': [x[resistance_ind] for x in temp_scores],
        })

        for agg_method in test_agg_list:
            results_df_agg = get_results_agg_df(agg_method, results_df, amr_df)
            model_parmas = json.dumps(model.get_params())
            accuracy, f1_score, auc = write_data_to_excel(writer, antibiotic, agg_method, results_df_agg, classes, model_parmas, all_results_dic)
            print(f"antibiotic: {antibiotic}  aggregation method: {agg_method} accuracy: {accuracy}  f1_score: {f1_score}  auc: {auc}")
            results_dic[agg_method] = [accuracy, f1_score, auc]

        writer.save()
        return results_dic
    except Exception as e:
        print(f"ERROR at train_test_and_write_results_cv, message: {e}")
        traceback.print_exc()


def get_results_agg_df(agg_method, results_df, amr_df):
    try:
        # test_agg_list = ["mean_all", "mean_highest$100", "mean_highest$300", "mean_highest$600"]
        f = {'Label': 'first', 'Resistance score': 'mean'}
        if agg_method == "mean_all":
            results_df_agg = results_df.groupby('file_id', as_index=False).agg(f)
        elif "mean_highest" in agg_method:
            top_x = int(agg_method.split("$")[1])
            results_df_agg = results_df.sort_values('Resistance score', ascending=False).groupby('file_id', as_index=False).head(top_x).groupby('file_id', as_index=False).agg(f)
        else:
            raise Exception(f"agg_method: {agg_method} is invalid!")
        results_df_agg['Prediction'] = np.where(results_df_agg['Resistance score'] > 0.5, 'R', 'S')
        final_df = results_df_agg.merge(amr_df[['file_id', 'NCBI File Name']], on='file_id', how='inner')
        return final_df
    except Exception as e:
        print(f"ERROR at get_results_agg_df, message: {e}")
        traceback.print_exc()
