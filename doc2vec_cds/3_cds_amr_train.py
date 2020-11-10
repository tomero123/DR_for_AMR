import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import multiprocessing
import datetime
import os
import pandas as pd
import time
import json
import xgboost
from sklearn.neighbors import KNeighborsClassifier
from doc2vec_cds.FaissKNeighbors import FaissKNeighbors

from doc2vec_cds.cds_utils import get_label_df, get_current_results_folder, \
    cds_convert_results_df_to_new_format, train_cv_from_cds_embeddings, cds_get_all_resulst_df, cds_get_agg_results_df
from doc2vec_cds.Doc2VecCDS import Doc2VecCDSLoader
from constants import Bacteria, ANTIBIOTIC_DIC, EMBEDDING_DF_FILE_NAME, METADATA_DF_FILE_NAME, AggregationMethod, \
    ClassifierType, RawDataType
from MyLogger import Logger

if __name__ == '__main__':
    # PARAMS
    MODEL_BACTERIA = Bacteria.PSEUDOMONAS_AUREGINOSA.value if len(sys.argv) <= 1 else sys.argv[1]
    MODEL_CLASSIFIER = ClassifierType.XGBOOST.value if len(sys.argv) <= 2 else sys.argv[2]  # can be "knn" or "xgboost"
    AGGREGATION_METHOD = AggregationMethod.SCORES.value if len(sys.argv) <= 3 else sys.argv[3]  # can be "scores" or "embeddings"
    RESULTS_FOLDER_NAME = None if len(sys.argv) <= 4 else sys.argv[4]
    NON_OVERLAPPING_USE_SEQ_AGGREGATION = False  # relevant only if non_overlapping and AGGREGATION_METHOD = "scores"
    if len(sys.argv) > 5 and sys.argv[5] == "true":
        NON_OVERLAPPING_USE_SEQ_AGGREGATION = True

    EMBEDDINGS_AGGREGATION_METHOD = "mean"  # can be "max" or "mean"
    LOAD_EMBEDDING_DF = True  # if True then load embedding_df if it exists otherwise calculate. If False - always calculate
    USE_MULTIPROCESS = False

    # XGBoost params - relevant only if MODEL_CLASSIFIER = ClassifierType.XGBOOST.value
    MAX_DEPTH = 4
    N_ESTIMATORS = 300
    SUBSAMPLE = 0.8
    COLSAMPLE_BYTREE = 0.8
    LEARNING_RATE = 0.1
    n_jobs = 64
    random_seed = 1
    # knn params - relevant only if MODEL_CLASSIFIER = ClassifierType.KNN.value
    use_faiss_knn = True
    knn_k_size = 5

    workers = multiprocessing.cpu_count()
    amr_data_file_name = "amr_labels.csv"
    prefix = '.'
    # BACTERIA list
    BACTERIA_LIST = [
        # Bacteria.TEST.value,
        Bacteria.PSEUDOMONAS_AUREGINOSA.value,
        # Bacteria.MYCOBACTERIUM_TUBERCULOSIS.value,
    ]
    # Define list of model_names and processing method
    D2V_MODELS_LIST = [
        # "2020_11_07_2316_PM_overlapping_K_10_SS_1",
        "2020_11_07_2316_PM_non_overlapping_K_10_SS_1"
    ]
    # PARAMS END
    # IF RUNNING LOCAL (WINDOWS)
    if os.name == 'nt':
        D2V_MODELS_LIST = ["2020_11_09_1818_PM_overlapping_K_10_SS_1"]
        prefix = '..'

    # if "SSH_CONNECTION" in os.environ:
    #     prefix = '..'

    # Choose classifier
    if MODEL_CLASSIFIER == ClassifierType.KNN.value:
        if use_faiss_knn and os.name != 'nt':
            print(f"Using FaissKNeighbors with K: {knn_k_size}")
            model = FaissKNeighbors(knn_k_size)
        else:
            model = KNeighborsClassifier(n_neighbors=knn_k_size)
    elif MODEL_CLASSIFIER == ClassifierType.XGBOOST.value:
        model_params = {
            "max_depth": MAX_DEPTH,
            "n_estimators": N_ESTIMATORS,
            "subsample": SUBSAMPLE,
            "colsample_bytree": COLSAMPLE_BYTREE,  # like max_features in sklearn
            "learning_rate": LEARNING_RATE,
            "n_jobs": n_jobs,
            "random_state": random_seed
        }
        model = xgboost.XGBClassifier(**model_params)
    else:
        raise Exception(f"model_classifier: {MODEL_CLASSIFIER} is invalid!")

    for d2v_model_folder_name in D2V_MODELS_LIST:
        models_folder = os.path.join(prefix, "results_files", MODEL_BACTERIA, "cds_models", d2v_model_folder_name)
        with open(os.path.join(models_folder, "model_conf.json"), "r") as read_file:
            model_conf = json.load(read_file)
        PROCESSING_MODE = model_conf["processing_mode"]
        K = model_conf["k"]
        SHIFT_SIZE = model_conf["shift_size"]
        RAW_DATA_TYPE = model_conf["raw_data_type"]

        for BACTERIA in BACTERIA_LIST:
            current_results_folder = get_current_results_folder(RESULTS_FOLDER_NAME, MODEL_CLASSIFIER, knn_k_size)
            embedding_df_folder = os.path.join(prefix, "results_files", BACTERIA, "cds_embeddings_df", d2v_model_folder_name)
            results_file_folder = os.path.join(prefix, "results_files", BACTERIA, "cds_embeddings_classification_results", d2v_model_folder_name, current_results_folder)
            if not os.path.exists(results_file_folder):
                os.makedirs(results_file_folder)
            log_path = os.path.join(results_file_folder, f"log_{current_results_folder}.txt")
            sys.stdout = Logger(log_path)
            all_results_dic = {"antibiotic": [], "agg_method": [], "fold": [], "accuracy": [], "f1_score": [], "auc": [], "recall": [], "precision": []}

            params_dict = {
                "bacteria": BACTERIA,
                "model_classifier": MODEL_CLASSIFIER,
                "aggregation_method": AGGREGATION_METHOD,
                "d2v_model": d2v_model_folder_name,
                "load_embedding_df": LOAD_EMBEDDING_DF,
            }
            if MODEL_CLASSIFIER == ClassifierType.KNN.value:
                params_dict["knn_k_size"] = knn_k_size
                params_dict["use_faiss_knn"] = use_faiss_knn

            elif MODEL_CLASSIFIER == ClassifierType.XGBOOST.value:
                params_dict["max_depth"] = MAX_DEPTH
                params_dict["n_estimators"] = N_ESTIMATORS
                params_dict["subsample"] = SUBSAMPLE
                params_dict["max_features"] = COLSAMPLE_BYTREE
                params_dict["learning_rate"] = LEARNING_RATE
                params_dict["n_jobs"] = n_jobs

            for key, val in model_conf.items():
                params_dict[f"d2v_{key}"] = val

            antibiotic_list = ANTIBIOTIC_DIC.get(BACTERIA)
            cds_genome_files_folder = "accessory_cds_from_genomic_files" if RAW_DATA_TYPE == RawDataType.ACCESSORY_GENES.value else "cds_from_genomic_files"
            genome_files_input_folder = os.path.join(prefix, "results_files", BACTERIA, cds_genome_files_folder)
            amr_file_path = os.path.join(prefix, 'results_files', BACTERIA, amr_data_file_name)
            now_total = time.time()
            now_date = datetime.datetime.now()
            print(f"Started running on: {now_date.strftime('%Y-%m-%d %H:%M:%S')}  D2V_MODEL_NAME: {d2v_model_folder_name}  PROCESSING_MODE: {PROCESSING_MODE}  BACTERIA: {BACTERIA}  MODEL_CLASSIFIER: {MODEL_CLASSIFIER}")
            print(f"results_file_folder path: {results_file_folder}")
            # Get embeddings df
            files_list = os.listdir(genome_files_input_folder)
            files_list = [x for x in files_list if ".fna.gz" in x]
            print(f"len files_list: {len(files_list)}")
            # get AMR data df
            amr_df = pd.read_csv(amr_file_path)
            full_labeled_files_dic = dict(zip(amr_df["NCBI File Name"], amr_df["file_id"]))
            labeled_files_dic = {}
            for x in files_list:
                genome_file_name = x.replace("_cds_from_genomic.fna.gz", "")
                if genome_file_name in full_labeled_files_dic:
                    labeled_files_dic[genome_file_name] = full_labeled_files_dic[genome_file_name]
            print(f"len labeled_files_list: {len(labeled_files_dic)}")
            if not os.path.exists(embedding_df_folder):
                os.makedirs(embedding_df_folder)
            if LOAD_EMBEDDING_DF and os.path.exists(os.path.join(embedding_df_folder, EMBEDDING_DF_FILE_NAME)):
                print(f"Loading embedding_df from disk. path: {embedding_df_folder}")
                embedding_df = pd.read_hdf(os.path.join(embedding_df_folder, EMBEDDING_DF_FILE_NAME))
                metadata_df_full = pd.read_csv(os.path.join(embedding_df_folder, METADATA_DF_FILE_NAME))
            else:
                # get only the files with label for the specific antibiotic
                t1 = time.time()
                doc2vec_loader = Doc2VecCDSLoader(genome_files_input_folder, labeled_files_dic, K, PROCESSING_MODE, SHIFT_SIZE, models_folder)
                embedding_df, metadata_df_full = doc2vec_loader.run()
                t2 = time.time()
                print(f"Finished extracting embeddings in {round((t2 - t1) / 60, 4)} minutes")
                print(f"Started saving embeddings_df to hdf")
                embedding_df.to_hdf(os.path.join(embedding_df_folder, EMBEDDING_DF_FILE_NAME), key='stage', mode='w')
                metadata_df_full.to_csv(os.path.join(embedding_df_folder, METADATA_DF_FILE_NAME), index=False)
                t3 = time.time()
                print(f"Finished saving embeddings_df to hdf in {round((t3 - t2) / 60, 4)} minutes")

            for antibiotic in antibiotic_list:
                t1 = time.time()
                results_file_name = f"{antibiotic}_{current_results_folder}.xlsx"
                results_file_path = os.path.join(results_file_folder, results_file_name)
                label_df = get_label_df(amr_df, files_list, antibiotic)
                final_df = embedding_df.merge(label_df[['file_id', 'label']], on='file_id', how='inner')
                t2 = time.time()
                print(f"Finished creating final_df for bacteria: {BACTERIA} antibiotic: {antibiotic} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE} in {round((t2-t1) / 60, 4)} minutes")
                print(f"Started classifier training for bacteria: {BACTERIA} antibiotic: {antibiotic} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE}  MODEL_CLASSIFIER: {MODEL_CLASSIFIER}")
                train_cv_from_cds_embeddings(final_df, amr_df, results_file_path, model, antibiotic, all_results_dic,
                                             USE_MULTIPROCESS, MODEL_CLASSIFIER, NON_OVERLAPPING_USE_SEQ_AGGREGATION,
                                             EMBEDDINGS_AGGREGATION_METHOD, AGGREGATION_METHOD)

                t3 = time.time()

                print(f"Finished training classifier for bacteria: {BACTERIA} antibiotic: {antibiotic} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE} in {round((t3-t2) / 60, 4)} minutes")

            # Write ALL_RESULTS
            metrics_order = ["auc", "accuracy", "f1_score", "recall", "precision"]
            all_results_df = cds_get_all_resulst_df(all_results_dic, metrics_order)
            all_results_agg = cds_get_agg_results_df(all_results_df, metrics_order)
            all_results_df_new_format = cds_convert_results_df_to_new_format(all_results_agg, metrics_order)
            writer = pd.ExcelWriter(os.path.join(results_file_folder, f"ALL_RESULTS_{current_results_folder}.xlsx"), engine='xlsxwriter')
            all_results_df_new_format.to_excel(writer, sheet_name="Agg_Results_New_Format", index=False)
            all_results_agg.to_excel(writer, sheet_name="Agg_Results", index=False)
            all_results_df.to_excel(writer, sheet_name="All_Folds", index=False)
            writer.save()

            params_dict.update(all_results_dic)

            with open(os.path.join(results_file_folder, "params.json"), "w") as write_file:
                json.dump(params_dict, write_file)

            now_date = datetime.datetime.now()
            print(f"Finished running on: {now_date.strftime('%Y-%m-%d %H:%M:%S')} after {round((time.time() - now_total) / 3600, 4)} hours; D2V_MODEL_NAME: {d2v_model_folder_name}  PROCESSING_MODE: {PROCESSING_MODE}  BACTERIA: {BACTERIA}  MODEL_CLASSIFIER: {MODEL_CLASSIFIER}")
    print("DONE!")
