import multiprocessing
import time
import datetime
import traceback
import numpy as np

from doc2vec_cds.cds_utils import cds_write_data_to_excel
from utils import get_train_and_test_groups
from constants import TIME_STR


def train_cv_from_gene_clusters_embeddings(final_df, amr_df, results_file_path, model, antibiotic, all_results_dic, use_multiprocess, MODEL_CLASSIFIER):
    try:
        now = time.time()
        n_folds = amr_df[f"{antibiotic}_group"].max()
        train_groups_list, test_groups_list = get_train_and_test_groups(n_folds)

        inputs_list = []

        for train_group_list, test_group in zip(train_groups_list, test_groups_list):
            train_file_id_list = list(amr_df[amr_df[f"{antibiotic}_group"].isin(train_group_list)]["file_id"])
            test_file_id_list = list(amr_df[amr_df[f"{antibiotic}_group"] == test_group]["file_id"])
            inputs_list.append([test_group, train_file_id_list, test_file_id_list, final_df, antibiotic, amr_df, model])

        print(f"{datetime.datetime.now().strftime(TIME_STR)} STARTED training models. antibiotic: {antibiotic}")
        if use_multiprocess:
            print("Using multiprocessing")
            with multiprocessing.Pool(processes=n_folds) as pool:
                results_list = pool.starmap(gene_clusters_train_one_fold, inputs_list)
        else:
            print("Using 1 process")
            results_list = []
            for i in inputs_list:
                results_list.append(gene_clusters_train_one_fold(*i))

        cds_write_data_to_excel(antibiotic, results_list, results_file_path, all_results_dic)
        print(f"***{datetime.datetime.now().strftime(TIME_STR)} FINISHED training models for antibiotic: {antibiotic} in {round((time.time() - now) / 60, 4)} minutes***")
    except Exception as e:
        print(f"ERROR at train_test_and_write_results_cv, message: {e}")
        traceback.print_exc()


def gene_clusters_train_one_fold(test_group, train_file_id_list, test_file_id_list, final_df, antibiotic, amr_df, model):
    try:
        now = time.time()
        non_features_columns = ['file_id', 'NCBI File Name', 'Strain', 'label']
        final_df['label'].replace('R', 1, inplace=True)
        final_df['label'].replace('S', 0, inplace=True)

        final_df = final_df.merge(amr_df[['file_id', 'NCBI File Name', 'Strain']], on='file_id', how='inner')
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
    except Exception as e:
        print(f"ERROR at embeddings_agg_one_fold, message: {e}")
        traceback.print_exc()
