import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import multiprocessing
import datetime
import os
import pandas as pd
import time
import json

from doc2vec_cds.cds_utils import get_label_df, train_test_and_write_results_cv
from doc2vec_cds.Doc2VecCDS import Doc2VecCDSLoader
from enums import Bacteria, ANTIBIOTIC_DIC, EMBEDDING_DF_FILE_NAME, METADATA_DF_FILE_NAME
from utils import get_time_as_str

if __name__ == '__main__':
    # PARAMS
    MODEL_BACTERIA = Bacteria.PSEUDOMONAS_AUREGINOSA.value if len(sys.argv) <= 1 else sys.argv[1]
    MODEL_CLASSIFIER = "knn" if len(sys.argv) <= 2 else sys.argv[2]  # can be "knn" or "xgboost"
    KNN_K_SIZE = 5 if len(sys.argv) <= 3 else sys.argv[3]
    LOAD_EMBEDDING_DF = True
    USE_FAISS_KNN = True
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
        "2020_09_01_1931_PM_overlapping_K_10_SS_2",
    ]

    # PARAMS END
    # IF RUNNING LOCAL (WINDOWS)
    if os.name == 'nt':
        D2V_MODELS_LIST = ["2020_09_21_1227_PM_overlapping_K_10_SS_2"]
        prefix = '..'
    # if "SSH_CONNECTION" in os.environ:
    #     prefix = '..'
    for d2v_model_folder_name in D2V_MODELS_LIST:
        models_folder = os.path.join(prefix, "results_files", MODEL_BACTERIA, "cds_models", d2v_model_folder_name)
        with open(os.path.join(models_folder, "model_conf.json"), "r") as read_file:
            model_conf = json.load(read_file)
        PROCESSING_MODE = model_conf["processing_mode"]
        K = model_conf["k"]
        SHIFT_SIZE = model_conf["shift_size"]
        for BACTERIA in BACTERIA_LIST:
            current_date_folder = get_time_as_str()
            embedding_df_folder = os.path.join(prefix, "results_files", BACTERIA, "cds_embeddings_df", d2v_model_folder_name)
            results_file_folder = os.path.join(prefix, "results_files", BACTERIA, "cds_embeddings_classification_results", d2v_model_folder_name, current_date_folder)
            all_results_dic = {"antibiotic": [], "agg_method": [], "accuracy": [], "f1_score": [], "auc": [], "recall": [], "precision": []}
            antibiotic_list = ANTIBIOTIC_DIC.get(BACTERIA)
            input_folder = os.path.join(prefix, "results_files", BACTERIA, "cds_genome_files")
            amr_file_path = os.path.join(prefix, 'results_files', BACTERIA, amr_data_file_name)
            now_total = time.time()
            now_date = datetime.datetime.now()
            print(f"Started running on: {now_date.strftime('%Y-%m-%d %H:%M:%S')}  D2V_MODEL_NAME: {d2v_model_folder_name}  PROCESSING_MODE: {PROCESSING_MODE}  BACTERIA: {BACTERIA}")
            print(f"results_file_folder path: {results_file_folder}")
            # Get embeddings df
            files_list = os.listdir(input_folder)
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
                doc2vec_loader = Doc2VecCDSLoader(input_folder, labeled_files_dic, K, PROCESSING_MODE, SHIFT_SIZE, models_folder)
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
                if not os.path.exists(results_file_folder):
                    os.makedirs(results_file_folder)
                results_file_name = f"{antibiotic}_{current_date_folder}.xlsx"
                results_file_path = os.path.join(results_file_folder, results_file_name)
                label_df = get_label_df(amr_df, files_list, antibiotic)
                final_df = embedding_df.merge(label_df[['file_id', 'label']], on='file_id', how='inner')
                t2 = time.time()
                print(f"Finished creating final_df for bacteria: {BACTERIA} antibiotic: {antibiotic} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE} in {round((t2-t1) / 60, 4)} minutes")
                print(f"Started classifier training for bacteria: {BACTERIA} antibiotic: {antibiotic} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE}")
                results_dic = train_test_and_write_results_cv(final_df, antibiotic, results_file_path, all_results_dic, amr_df, MODEL_CLASSIFIER, KNN_K_SIZE, USE_FAISS_KNN)
                t3 = time.time()

                print(f"Finished training classifier for bacteria: {BACTERIA} antibiotic: {antibiotic} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE} in {round((t3-t2) / 60, 4)} minutes")
            all_results_df = pd.DataFrame(all_results_dic)
            writer = pd.ExcelWriter(os.path.join(results_file_folder, f"ALL_RESULTS_{current_date_folder}.xlsx"), engine='xlsxwriter')
            all_results_df.to_excel(writer, sheet_name="Sheet1", index=False)
            # workbook = writer.book
            # workbook.close()
            writer.save()
            now_date = datetime.datetime.now()
            print(f"Finished running on: {now_date.strftime('%Y-%m-%d %H:%M:%S')} after {round((time.time() - now_total) / 3600, 4)} hours; D2V_MODEL_NAME: {d2v_model_folder_name}  PROCESSING_MODE: {PROCESSING_MODE}  BACTERIA: {BACTERIA}")
    print("DONE!")
