import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import time
import datetime
import traceback
import pandas as pd
import numpy as np
from sklearn import metrics
from pathos import multiprocessing


from utils import get_time_as_str, get_train_and_test_groups
from constants import TIME_STR, AggregationMethod, ProcessingMode


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


def get_results_agg_df(agg_method, results_df, amr_df):
    try:
        # test_agg_list = ["mean_all", "mean_highest$100", "mean_highest$300", "mean_highest$600"]
        f = {'label': 'first', 'strain': 'first', 'resistance_score': 'mean'}
        if agg_method == "mean_all":
            results_df_agg = results_df.groupby('file_id', as_index=False).agg(f)
        elif "mean_highest" in agg_method:
            top_x = int(agg_method.split("$")[1])
            results_df_agg = results_df.sort_values('resistance_score', ascending=False).groupby('file_id', as_index=False).head(top_x).groupby('file_id', as_index=False).agg(f)
        else:
            raise Exception(f"agg_method: {agg_method} is invalid!")

        # get prediction
        y_true = results_df_agg['label']
        y_pred_score = list(results_df_agg['resistance_score'])
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_score)
        num_pos_class = len([x for x in y_true if x == 1])
        num_neg_class = len([x for x in y_true if x == 0])
        if agg_method != "mean_all":
            max_accuracy, resistance_threshold = get_metric_and_best_threshold_from_roc_curve(tpr, fpr, thresholds, num_pos_class, num_neg_class)
            print(f"max_accuracy: {max_accuracy}, best_threshold: {resistance_threshold}")
        else:
            resistance_threshold = 0.5
            print("Using resistance_threshold = 0.5 for mean_all aggregation")

        prediction = [1 if x > resistance_threshold else 0 for x in y_pred_score]
        results_df_agg['prediction'] = prediction

        final_df = results_df_agg.merge(amr_df[['file_id', 'NCBI File Name']], on='file_id', how='inner')
        return final_df
    except Exception as e:
        print(f"ERROR at get_results_agg_df, message: {e}")
        traceback.print_exc()


def train_cv_from_cds_embeddings(final_df, amr_df, results_file_path, model, antibiotic, all_results_dic, use_multiprocess, MODEL_CLASSIFIER, NON_OVERLAPPING_USE_SEQ_AGGREGATION, EMBEDDINGS_AGGREGATION_METHOD, AGGREGATION_METHOD, PROCESSING_MODE):
    try:
        now = time.time()
        n_folds = amr_df[f"{antibiotic}_group"].max()
        train_groups_list, test_groups_list = get_train_and_test_groups(n_folds)

        inputs_list = []

        # print(f"{datetime.datetime.now().strftime(TIME_STR)} STARTED creating folds data. antibiotic: {antibiotic}")
        for train_group_list, test_group in zip(train_groups_list, test_groups_list):
            train_file_id_list = list(amr_df[amr_df[f"{antibiotic}_group"].isin(train_group_list)]["file_id"])
            test_file_id_list = list(amr_df[amr_df[f"{antibiotic}_group"] == test_group]["file_id"])
            inputs_list.append([test_group, train_file_id_list, test_file_id_list, final_df, antibiotic, amr_df, model, NON_OVERLAPPING_USE_SEQ_AGGREGATION, EMBEDDINGS_AGGREGATION_METHOD, PROCESSING_MODE])

        # print(f"{datetime.datetime.now().strftime(TIME_STR)} FINISHED creating folds data. antibiotic: {antibiotic}")
        print(f"{datetime.datetime.now().strftime(TIME_STR)} STARTED training models. antibiotic: {antibiotic}")
        if use_multiprocess:
            print("Using multiprocessing")
            with multiprocessing.Pool(processes=n_folds) as pool:
                if AGGREGATION_METHOD == AggregationMethod.SCORES.value:
                    results_list = pool.starmap(scores_agg_one_fold, inputs_list)
                elif AGGREGATION_METHOD == AggregationMethod.EMBEDDINGS.value:
                    results_list = pool.starmap(embeddings_agg_one_fold, inputs_list)
        else:
            print("Using 1 process")
            results_list = []
            for i in inputs_list:
                if AGGREGATION_METHOD == AggregationMethod.SCORES.value:
                    results_list.append(scores_agg_one_fold(*i))
                elif AGGREGATION_METHOD == AggregationMethod.EMBEDDINGS.value:
                    results_list.append(embeddings_agg_one_fold(*i))

        cds_write_data_to_excel(antibiotic, results_list, results_file_path, all_results_dic)
        print(f"***{datetime.datetime.now().strftime(TIME_STR)} FINISHED training models for antibiotic: {antibiotic} in {round((time.time() - now) / 60, 4)} minutes***")
    except Exception as e:
        print(f"ERROR at train_test_and_write_results_cv, message: {e}")
        traceback.print_exc()


def scores_agg_one_fold(test_group, train_file_id_list, test_file_id_list, final_df, antibiotic, amr_df, model, NON_OVERLAPPING_USE_SEQ_AGGREGATION, embeddings_aggregation_method, PROCESSING_MODE):
    now = time.time()
    non_features_columns = ['file_id', 'NCBI File Name', 'Strain', 'label', 'seq_id', 'doc_ind']
    final_df['label'].replace('R', 1, inplace=True)
    final_df['label'].replace('S', 0, inplace=True)
    final_df = final_df.merge(amr_df[['file_id', 'NCBI File Name', 'Strain']], on='file_id', how='inner')
    # Use mean aggregation for all sequences when method = non_overlapping
    if PROCESSING_MODE == ProcessingMode.NON_OVERLAPPING.value and NON_OVERLAPPING_USE_SEQ_AGGREGATION:
        agg_final_df = final_df.groupby(['file_id', 'seq_id'])[[x for x in final_df.columns if x.startswith("f_")]].mean()
        agg_final_df['label'] = final_df.groupby(['file_id', 'seq_id'])['label'].max()
        agg_final_df.reset_index(inplace=True)
        final_df_train = agg_final_df[agg_final_df["file_id"].isin(train_file_id_list)]
        final_df_test = agg_final_df[agg_final_df["file_id"].isin(test_file_id_list)]
    else:
        final_df_train = final_df[final_df["file_id"].isin(train_file_id_list)]
        final_df_test = final_df[final_df["file_id"].isin(test_file_id_list)]
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

    if model.__class__.__name__ == "XGBClassifier":
        eval_set = [(X_test, y_test)]
        model.fit(X_train, y_train.values.ravel(), sample_weight=sample_weight,
                  eval_metric="auc", eval_set=eval_set, verbose=True,
                  early_stopping_rounds=15
                  )
    else:
        model.fit(X_train, y_train.values.ravel())

    test_agg_list = ["mean_highest$100", "mean_highest$300", "mean_highest$600", "mean_all"]

    temp_scores = model.predict_proba(X_test)
    true_results = y_test.values.ravel()

    resistance_score = [x[1] for x in temp_scores]
    test_files_ids = list(final_df_test['file_id'])
    strain = list(final_df_test['Strain'])

    results_list = []

    results_df = pd.DataFrame({
        'file_id': test_files_ids,
        'label': true_results,
        'resistance_score': resistance_score,
        'strain': strain
    })

    for agg_method in test_agg_list:
        results_df_agg = get_results_agg_df(agg_method, results_df, amr_df)

        results_dic = {
            'agg_method': agg_method,
            'Fold': test_group,
            'file_id': list(results_df_agg['file_id']),
            'Strain': list(results_df_agg['strain']),
            'File name': list(results_df_agg['NCBI File Name']),
            "true_results": results_df_agg['label'],
            "resistance_score": results_df_agg['resistance_score'],
            "predictions": results_df_agg['prediction'],
        }

        results_list.append(results_dic)

    print(f"{datetime.datetime.now().strftime(TIME_STR)} FOLD#{test_group} Finished training classifier in {round((time.time() - now) / 60, 4)} minutes")
    return results_list


def embeddings_agg_one_fold(test_group, train_file_id_list, test_file_id_list, final_df, antibiotic, amr_df, model, NON_OVERLAPPING_USE_SEQ_AGGREGATION, embeddings_aggregation_method, PROCESSING_MODE):
    now = time.time()
    if embeddings_aggregation_method == "mean":
        agg_final_df = final_df.groupby('file_id')[[x for x in final_df.columns if x.startswith("f_")]].mean()
    elif embeddings_aggregation_method == "max":
        agg_final_df = final_df.groupby('file_id')[[x for x in final_df.columns if x.startswith("f_")]].max()
    else:
        raise Exception(f"embeddings_aggregation_method: {embeddings_aggregation_method} is invalid!")

    agg_final_df['label'] = final_df.groupby('file_id')['label'].max()
    agg_final_df.reset_index(level=0, inplace=True)

    non_features_columns = ['file_id', 'NCBI File Name', 'Strain', 'label']
    agg_final_df['label'].replace('R', 1, inplace=True)
    agg_final_df['label'].replace('S', 0, inplace=True)
    agg_final_df = agg_final_df.merge(amr_df[['file_id', 'NCBI File Name', 'Strain']], on='file_id', how='inner')
    final_df_train = agg_final_df[agg_final_df["file_id"].isin(train_file_id_list)]
    final_df_test = agg_final_df[agg_final_df["file_id"].isin(test_file_id_list)]
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

    if model.__class__.__name__ == "XGBClassifier":
        eval_set = [(X_test, y_test)]
        model.fit(X_train, y_train.values.ravel(), sample_weight=sample_weight,
                  eval_metric="auc", eval_set=eval_set, verbose=True,
                  early_stopping_rounds=15
                  )
    else:
        model.fit(X_train, y_train.values.ravel())

    temp_scores = model.predict_proba(X_test)
    true_results = y_test.values.ravel()
    resistance_score = [x[1] for x in temp_scores]
    predictions = [1 if p[1] >= p[0] else 0 for p in temp_scores]

    results_list = [{
        'agg_method': "NA",
        'Fold': test_group,
        'file_id': list(final_df_test['file_id']),
        'Strain': list(final_df_test['Strain']),
        'File name': list(final_df_test['NCBI File Name']),
        "true_results": true_results,
        "resistance_score": resistance_score,
        "predictions": predictions,
    }]

    print(f"{datetime.datetime.now().strftime(TIME_STR)} FOLD#{test_group} Finished training classifier in {round((time.time() - now) / 60, 4)} minutes")
    return results_list


def cds_write_data_to_excel(antibiotic, results_list, results_file_path, all_results_dic):
    try:
        writer = pd.ExcelWriter(results_file_path, engine='xlsxwriter')
        workbook = writer.book
        agg_method_n = len(results_list[0])
        for agg_method_ind in range(agg_method_n):
            col_ind = 0
            row_ind = 0
            results_df = pd.DataFrame(columns=['file_id', 'file_name', 'strain', 'label', 'resistance_score', 'prediction', 'fold'])
            evaluation_df = pd.DataFrame(columns=['fold', 'auc', 'accuracy', 'f1_score', 'precision', 'recall'])
            for result in results_list:
                fold_dic = result[agg_method_ind]
                agg_method = fold_dic["agg_method"]
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
                all_results_dic["agg_method"].append(agg_method)
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

            results_df.to_excel(writer, sheet_name=agg_method, startcol=col_ind, startrow=row_ind, index=False)
            col_ind += results_df.shape[1] + 1
            confusion_matrix = metrics.confusion_matrix(list(results_df['label']), list(results_df['prediction']))
            confusion_matrix_df = pd.DataFrame(confusion_matrix, columns=[x + "_Prediction" for x in ["S", "R"]], index=[x + "_Actual" for x in ["S", "R"]])
            confusion_matrix_df.to_excel(writer, sheet_name=agg_method, startcol=col_ind, startrow=row_ind, index=True)
            row_ind += confusion_matrix_df.shape[0] + 2
            evaluation_df.to_excel(writer, sheet_name=agg_method, startcol=col_ind, startrow=row_ind, index=False)
            worksheet = writer.sheets[agg_method]
            # percent_format = workbook.add_format({'num_format': '0.00%'})
            worksheet.set_column('A:Z', 15)
        workbook.close()
        print(f"Finished creating results for antibiotic: {antibiotic} ; auc: {evaluation_mean[1]} accuracy: {evaluation_mean[2]} f1_score: {evaluation_mean[3]} precision: {evaluation_mean[4]} recall: {evaluation_mean[5]}")
    except Exception as e:
        print("Error in cds_write_data_to_excel.error message: {}".format(e))
        traceback.print_exc()


def cds_get_all_resulst_df(all_results_dic, metrics_order):
    all_results_df = pd.DataFrame(all_results_dic)
    all_results_df = all_results_df.loc[:, ["antibiotic", "agg_method", "fold"] + metrics_order].sort_values(by=["antibiotic", "agg_method", "fold"])
    return all_results_df


def cds_get_agg_results_df(all_results_df, metrics_order):
    mean_df = all_results_df.groupby(["antibiotic", "agg_method"]).mean()
    mean_df["category"] = "mean"
    std_df = all_results_df.groupby(["antibiotic", "agg_method"]).std()
    std_df["category"] = "std"
    all_results_agg = mean_df.append(std_df)
    all_results_agg = all_results_agg.loc[:, ["category"] + metrics_order].sort_values(by=["category", "antibiotic", "agg_method"])
    return all_results_agg.reset_index()


def get_metric_and_best_threshold_from_roc_curve(tpr, fpr, thresholds, num_pos_class, num_neg_class):
    tp = tpr * num_pos_class
    tn = (1 - fpr) * num_neg_class
    acc = (tp + tn) / (num_pos_class + num_neg_class)

    best_threshold = thresholds[np.argmax(acc)]
    max_accuracy = np.amax(acc)
    return max_accuracy, best_threshold


def get_current_results_folder(results_folder_name, model_classifier, knn_k_size):
    current_results_folder = get_time_as_str()
    if results_folder_name is not None:
        current_results_folder += f"_{results_folder_name}"
    else:
        if model_classifier == "knn":
            current_results_folder += f"_knn_{knn_k_size}"
        elif model_classifier == "xgboost":
            current_results_folder += "_xgboost"
    return current_results_folder


def cds_convert_results_df_to_new_format(all_results_agg, metrics_order):
    antibiotic_list = sorted(all_results_agg["antibiotic"].unique())
    agg_method_list = sorted(all_results_agg["agg_method"].unique())
    final_list = []
    for agg_method in agg_method_list:
        mean_std_list = [agg_method]
        mean_list = [agg_method]
        new_results_columns = ["agg_method"]
        for col in metrics_order:
            for antibiotic in antibiotic_list:
                mean = round(float(all_results_agg[col][(all_results_agg["antibiotic"] == antibiotic) & (all_results_agg["category"] == "mean") & (all_results_agg["agg_method"] == agg_method)]), 3)
                std = round(float(all_results_agg[col][(all_results_agg["antibiotic"] == antibiotic) & (all_results_agg["category"] == "std") & (all_results_agg["agg_method"] == agg_method)]), 3)
                mean_list.append(mean)
                mean_std_list.append(f"{mean}+-{std}")
                new_results_columns.append(f"{col}_{antibiotic}")
        final_list.append(mean_list)
        final_list.append(mean_std_list)
    return pd.DataFrame(data=final_list, columns=new_results_columns)
