import sys

sys.path.append("/home/local/BGU-USERS/tomeror/tomer_thesis")
sys.path.append("/home/tomeror/tomer_thesis")

import multiprocessing
import datetime
import os
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier

from doc2vec_cds.cds_utils import get_label_df, train_test_and_write_results_cv
from doc2vec_cds.Doc2VecCDS import Doc2VecCDSLoader
from enums import Bacteria, ProcessingMode, ANTIBIOTIC_DIC, EMBEDDING_DF_FILE_NAME, METADATA_DF_FILE_NAME
from utils import get_file_name


if __name__ == '__main__':
    # PARAMS
    MODEL_BACTERIA = Bacteria.PSEUDOMONAS_AUREGINOSA.value if len(sys.argv) <= 1 else sys.argv[1]
    K = 10 if len(sys.argv) <= 2 else int(sys.argv[2])  # Choose K size
    SHIFT_SIZE = 2  # relevant only for PROCESSING_MODE "overlapping"
    LOAD_EMBEDDING_DF = True
    KNN_K_SIZE = 7
    workers = multiprocessing.cpu_count()
    amr_data_file_name = "amr_labels.csv"
    prefix = '.'
    # BACTERIA list
    BACTERIA_LIST = [
        Bacteria.PSEUDOMONAS_AUREGINOSA.value,
        # Bacteria.MYCOBACTERIUM_TUBERCULOSIS.value,
    ]
    # Define list of model_names and processing method
    D2V_MODEL_PROCESSING_MODE_LIST = [
        ["d2v_2020_09_01_1931.model", ProcessingMode.OVERLAPPING.value],
    ]
    # model = xgboost.XGBClassifier(random_state=random_seed)
    # model_params = {'max_depth': 4, 'n_estimators': 300, 'max_features': 0.8, 'subsample': 0.8, 'learning_rate': 0.1}
    model = KNeighborsClassifier(n_neighbors=KNN_K_SIZE)
    # PARAMS END
    # IF RUNNING LOCAL (WINDOWS)
    if os.name == 'nt':
        D2V_MODEL_PROCESSING_MODE_LIST = [["d2v_2020_08_11_2132.model", ProcessingMode.OVERLAPPING.value]]
        prefix = '..'
    for conf in D2V_MODEL_PROCESSING_MODE_LIST:
        D2V_MODEL_NAME = conf[0]
        PROCESSING_MODE = conf[1]
        for BACTERIA in BACTERIA_LIST:
            all_results_dic = {"antibiotic": [], "agg_method": [], "accuracy": [], "f1_score": [], "auc": []}
            antibiotic_list = ANTIBIOTIC_DIC.get(BACTERIA)
            current_date_folder = get_file_name(D2V_MODEL_NAME.replace(".model", ""), None)
            if PROCESSING_MODE == ProcessingMode.OVERLAPPING.value:
                models_folder = os.path.join(prefix, "results_files", MODEL_BACTERIA, "cds_models", f"overlapping_{SHIFT_SIZE}", f"K_{K}")
                results_file_folder = os.path.join(prefix, "results_files", BACTERIA, "cds_embeddings_classification_results", f"overlapping_{SHIFT_SIZE}", f"K_{K}", current_date_folder)
                embedding_df_folder = os.path.join(prefix, "results_files", BACTERIA, "cds_embeddings_df", f"overlapping_{SHIFT_SIZE}", f"K_{K}")
            elif PROCESSING_MODE == ProcessingMode.NON_OVERLAPPING.value:
                models_folder = os.path.join(prefix, "results_files", MODEL_BACTERIA, "cds_models", "non_overlapping", f"K_{K}")
                results_file_folder = os.path.join(prefix, "results_files", BACTERIA, "cds_embeddings_classification_results", "non_overlapping", f"K_{K}", current_date_folder)
                embedding_df_folder = os.path.join(prefix, "results_files", BACTERIA, "cds_embeddings_df", "non_overlapping", f"K_{K}")
            else:
                raise Exception(f"PROCESSING_MODE: {PROCESSING_MODE} is invalid!")
            input_folder = os.path.join(prefix, "results_files", BACTERIA, "cds_genome_files")
            amr_file_path = os.path.join(prefix, 'results_files', BACTERIA, amr_data_file_name)
            now_total = time.time()
            now_date = datetime.datetime.now()
            print(f"Started running on: {now_date.strftime('%Y-%m-%d %H:%M:%S')}  D2V_MODEL_NAME: {D2V_MODEL_NAME}  PROCESSING_MODE: {PROCESSING_MODE}  BACTERIA: {BACTERIA}")
            print(f"results_file_folder path: {results_file_folder}")
            # Get embeddings df
            files_list = os.listdir(input_folder)
            files_list = [x for x in files_list if ".fna.gz" in x]
            print(f"len files_list: {len(files_list)}")
            # get AMR data df
            amr_df = pd.read_csv(amr_file_path)
            full_labeled_files_dic = dict(zip(amr_df["NCBI File Name"], amr_df["file_id"]))
            labeled_files_dic = dict((k, full_labeled_files_dic[k]) for k in [x.replace("_cds_from_genomic.fna.gz", "") for x in files_list])
            print(f"len labeled_files_list: {len(labeled_files_dic)}")
            if not os.path.exists(embedding_df_folder):
                os.makedirs(embedding_df_folder)
            if LOAD_EMBEDDING_DF and os.path.exists(os.path.join(embedding_df_folder, EMBEDDING_DF_FILE_NAME)):
                embedding_df = pd.read_hdf(os.path.join(embedding_df_folder, EMBEDDING_DF_FILE_NAME))
                metadata_df_full = pd.read_csv(os.path.join(embedding_df_folder, METADATA_DF_FILE_NAME))
            else:
                # get only the files with label for the specific antibiotic
                t1 = time.time()
                doc2vec_loader = Doc2VecCDSLoader(input_folder, labeled_files_dic, K, PROCESSING_MODE, SHIFT_SIZE,
                                                  os.path.join(models_folder, D2V_MODEL_NAME))
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
                results_file_name = D2V_MODEL_NAME.replace("d2v", antibiotic).replace(".model", ".xlsx")
                results_file_path = os.path.join(results_file_folder, results_file_name)
                label_df = get_label_df(amr_df, files_list, antibiotic)
                final_df = embedding_df.merge(label_df[['file_id', 'label']], on='file_id', how='inner')
                t2 = time.time()
                print(f"Finished creating final_df for bacteria: {BACTERIA} antibiotic: {antibiotic} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE} in {round((t2-t1) / 60, 4)} minutes")
                print(f"Started classifier training for bacteria: {BACTERIA} antibiotic: {antibiotic} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE}")
                results_dic = train_test_and_write_results_cv(final_df, antibiotic, results_file_path, model, all_results_dic, amr_df)
                t3 = time.time()
                print(f"Finished training classifier for bacteria: {BACTERIA} antibiotic: {antibiotic} processing mode: {PROCESSING_MODE} shift size: {SHIFT_SIZE} in {round((t3-t2) / 60, 4)} minutes")
            all_results_df = pd.DataFrame(all_results_dic)
            writer = pd.ExcelWriter(os.path.join(results_file_folder, f"ALL_RESULTS_{current_date_folder}.xlsx"), engine='xlsxwriter')
            all_results_df.to_excel(writer, sheet_name="Sheet1", index=False)
            # workbook = writer.book
            # workbook.close()
            writer.save()
            now_date = datetime.datetime.now()
            print(f"Finished running on: {now_date.strftime('%Y-%m-%d %H:%M:%S')} after {round((time.time() - now_total) / 3600, 4)} hours; D2V_MODEL_NAME: {D2V_MODEL_NAME}  PROCESSING_MODE: {PROCESSING_MODE}  BACTERIA: {BACTERIA}")
    print("DONE!")
