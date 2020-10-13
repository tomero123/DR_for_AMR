import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import os
import traceback
import json
import pickle
import pandas as pd
import numpy as np
import xgboost
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from doc2vec_cds.FaissKNeighbors import FaissKNeighbors
from utils import get_time_as_str


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


def write_data_to_excel_scores_agg(writer, antibiotic, agg_method, results_df, model_parmas, all_results_dic):
    try:
        col_ind = 0
        row_ind = 0
        results_df.sort_values(by="Resistance score", ascending=False).to_excel(writer, sheet_name=agg_method, startcol=col_ind, startrow=row_ind, index=False)
        col_ind += results_df.shape[1] + 1
        y_true = list(results_df['Label'])
        y_pred_score = list(results_df['Resistance score'])
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_score)
        num_pos_class = len([x for x in y_true if x == 1])
        num_neg_class = len([x for x in y_true if x == 0])
        max_accuracy, resistance_threshold = get_metric_and_best_threshold_from_roc_curve(tpr, fpr, thresholds, num_pos_class, num_neg_class)
        print(f"max_accuracy: {max_accuracy}, best_threshold: {resistance_threshold}")
        y_pred = [1 if x > resistance_threshold else 0 for x in y_pred_score]
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        confusion_matrix_df = pd.DataFrame(confusion_matrix, columns=[x + "_Prediction" for x in ["S", "R"]], index=[x + "_Actual" for x in ["S", "R"]])
        confusion_matrix_df.to_excel(writer, sheet_name=agg_method, startcol=col_ind, startrow=row_ind, index=True)
        row_ind += confusion_matrix_df.shape[0] + 2
        accuracy = metrics.accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)
        f1_score = metrics.f1_score(y_true, y_pred)
        auc = metrics.auc(fpr, tpr)
        all_results_dic["antibiotic"].append(antibiotic)
        all_results_dic["agg_method"].append(agg_method)
        all_results_dic["accuracy"].append(accuracy)
        all_results_dic["precision"].append(precision)
        all_results_dic["recall"].append(recall)
        all_results_dic["f1_score"].append(f1_score)
        all_results_dic["auc"].append(auc)
        evaluation_list = [["accuracy", accuracy], ["precision", precision], ["recall", recall], ["f1_score", f1_score],
                           ["auc", auc], ["model_parmas", model_parmas], ["resistance_threshold", resistance_threshold]]
        evaluation_df = pd.DataFrame(evaluation_list, columns=["metric", "value"])
        evaluation_df.to_excel(writer, sheet_name=agg_method, startcol=col_ind, startrow=row_ind, index=False)
        worksheet = writer.sheets[agg_method]
        # percent_format = workbook.add_format({'num_format': '0.00%'})
        worksheet.set_column('A:Z', 15)
        return accuracy, f1_score, auc, precision, recall
    except Exception as e:
        print(f"ERROR at write_data_to_excel, message: {e}")
        traceback.print_exc()


def write_data_to_excel_embeddings_agg(antibiotic, results_df, results_file_path, model_parmas, all_results_dic):
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
        all_results_dic["agg_method"].append("NA")
        all_results_dic["accuracy"].append(accuracy)
        all_results_dic["precision"].append(precision)
        all_results_dic["recall"].append(recall)
        all_results_dic["f1_score"].append(f1_score)
        all_results_dic["auc"].append(auc)
        evaluation_list = [["accuracy", accuracy], ["precision", precision], ["recall", recall], ["f1_score", f1_score],
                           ["auc", auc], ["model_parmas", model_parmas]]
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


def train_test_scores_aggregation(final_df, antibiotic, results_file_path, all_results_dic, amr_df, model_classifier, knn_k_size, use_faiss_knn):
    try:
        non_features_columns = ['file_id', 'seq_id', 'doc_ind', 'label']
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

        test_files_ids = list(final_df_test['file_id'])

        if model_classifier == "knn":
            if use_faiss_knn and os.name != 'nt':
                print(f"Using FaissKNeighbors with K: {knn_k_size}")
                model = FaissKNeighbors(knn_k_size)
            else:
                model = KNeighborsClassifier(n_neighbors=knn_k_size)
        elif model_classifier == "xgboost":
            model = xgboost.XGBClassifier(max_depth=4, n_estimators=300, subsample=0.8, max_features=0.8, learning_rate=0.1, n_jobs=10)
        else:
            raise Exception(f"model_classifier: {model_classifier} is invalid!")

        # Create weight according to the ratio of each class
        resistance_weight = (y_train['label'] == 0).sum() / (y_train['label'] == 1).sum() \
            if (y_train['label'] == 0).sum() / (y_train['label'] == 1).sum() > 0 else 1
        sample_weight = np.array([resistance_weight if i == 1 else 1 for i in y_train['label']])
        print("Resistance_weight for antibiotic: {} is: {}".format(antibiotic, resistance_weight))

        if model_classifier == "xgboost":
            model.fit(X_train, y_train.values.ravel(), sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train.values.ravel())

        # Save model
        if not use_faiss_knn:
            model_file_path = results_file_path.replace(".xlsx", "_MODEL.p")
            with open(model_file_path, 'wb') as f:
                pickle.dump(model, f)

        temp_scores = model.predict_proba(X_test)
        true_results = y_test.values.ravel()

        test_agg_list = ["mean_highest$100", "mean_highest$300", "mean_highest$600", "mean_all"]
        writer = pd.ExcelWriter(results_file_path, engine='xlsxwriter')

        results_df = pd.DataFrame({
            'file_id': test_files_ids,
            'Label': true_results,
            'Resistance score': [x[1] for x in temp_scores],
        })

        for agg_method in test_agg_list:
            results_df_agg = get_results_agg_df(agg_method, results_df, amr_df)
            model_parmas = json.dumps(model.get_params())
            accuracy, f1_score, auc, precision, recall = write_data_to_excel_scores_agg(writer, antibiotic, agg_method, results_df_agg, model_parmas, all_results_dic)
            print(f"antibiotic: {antibiotic}  aggregation method: {agg_method} accuracy: {accuracy}  f1_score: {f1_score}  auc: {auc}, precision: {precision}, recall: {recall}")

        writer.save()
    except Exception as e:
        print(f"ERROR at train_test_scores_aggregation, message: {e}")
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
        # results_df_agg['Prediction'] = np.where(results_df_agg['Resistance score'] > 0.5, 'R', 'S')
        final_df = results_df_agg.merge(amr_df[['file_id', 'NCBI File Name']], on='file_id', how='inner')
        return final_df
    except Exception as e:
        print(f"ERROR at get_results_agg_df, message: {e}")
        traceback.print_exc()


def train_test_embeddings_aggregation(final_df, antibiotic, results_file_path, all_results_dic, amr_df, model_classifier, knn_k_size, use_faiss_knn):
    try:
        non_features_columns = ['file_id', 'label']  # 'seq_id', 'doc_ind'
        # aggregate final_df by file_id and average all embeddings
        agg_final_df = final_df.groupby('file_id')[[x for x in final_df.columns if x.startswith("f_")]].mean()
        agg_final_df['label'] = final_df.groupby('file_id')['label'].max()
        agg_final_df.reset_index(level=0, inplace=True)
        train_file_id_list = list(amr_df[amr_df[f"{antibiotic}_is_train"] == 1]["file_id"])
        test_file_id_list = list(amr_df[amr_df[f"{antibiotic}_is_train"] == 0]["file_id"])
        agg_final_df['label'].replace('R', 1, inplace=True)
        agg_final_df['label'].replace('S', 0, inplace=True)
        final_df_train = agg_final_df[agg_final_df["file_id"].isin(train_file_id_list)]
        final_df_test = agg_final_df[agg_final_df["file_id"].isin(test_file_id_list)]
        X_train = final_df_train.drop(non_features_columns, axis=1).copy()
        y_train = final_df_train[['label']].copy()
        X_test = final_df_test.drop(non_features_columns, axis=1).copy()
        y_test = final_df_test[['label']].copy()
        print(f"X_train size: {X_train.shape}  y_train size: {y_train.shape}  X_test size: {X_test.shape}  y_test size: {y_test.shape}")

        if model_classifier == "knn":
            if use_faiss_knn and os.name != 'nt':
                print(f"Using FaissKNeighbors with K: {knn_k_size}")
                model = FaissKNeighbors(knn_k_size)
            else:
                model = KNeighborsClassifier(n_neighbors=knn_k_size)
        elif model_classifier == "xgboost":
            model = xgboost.XGBClassifier(max_depth=4, n_estimators=300, subsample=0.8, max_features=0.8, learning_rate=0.1, n_jobs=10)
        else:
            raise Exception(f"model_classifier: {model_classifier} is invalid!")

        # Create weight according to the ratio of each class
        resistance_weight = (y_train['label'] == 0).sum() / (y_train['label'] == 1).sum() \
            if (y_train['label'] == 0).sum() / (y_train['label'] == 1).sum() > 0 else 1
        sample_weight = np.array([resistance_weight if i == 1 else 1 for i in y_train['label']])
        print("Resistance_weight for antibiotic: {} is: {}".format(antibiotic, resistance_weight))

        if model_classifier == "xgboost":
            model.fit(X_train, y_train.values.ravel(), sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train.values.ravel())

        # Save model
        if not use_faiss_knn:
            model_file_path = results_file_path.replace(".xlsx", "_MODEL.p")
            with open(model_file_path, 'wb') as f:
                pickle.dump(model, f)

        temp_scores = model.predict_proba(X_test)
        true_results = y_test.values.ravel()

        predictions = [1 if p[1] >= p[0] else 0 for p in temp_scores]

        results_df = pd.DataFrame({
            'file_id': list(final_df_test['file_id']),
            'Strain': list(final_df_test['Strain']),
            'File name': list(final_df_test['file_name']),
            'Label': true_results,
            'Resistance score': [x[1] for x in temp_scores],
            'Prediction': predictions
        })
        model_parmas = json.dumps(model.get_params())
        write_data_to_excel_embeddings_agg(antibiotic, results_df, results_file_path, model_parmas, all_results_dic)

    except Exception as e:
        print(f"ERROR at train_test_scores_aggregation, message: {e}")
        traceback.print_exc()


def get_metric_and_best_threshold_from_roc_curve(tpr, fpr, thresholds, num_pos_class, num_neg_class):
    tp = tpr * num_pos_class
    tn = (1 - fpr) * num_neg_class
    acc = (tp + tn) / (num_pos_class + num_neg_class)

    best_threshold = thresholds[np.argmax(acc)]
    max_accuracy = np.amax(acc)
    return max_accuracy, best_threshold


def get_current_results_folder(model_classifier, knn_k_size):
    current_results_folder = get_time_as_str()
    if model_classifier == "knn":
        current_results_folder += f"_knn_{knn_k_size}"
    elif model_classifier == "xgboost":
        current_results_folder += "_xgboost"
    return current_results_folder
