import json
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn import metrics
import matplotlib.pyplot as plt
import traceback
import time


def get_kmers_df(path, dataset_file_name, kmers_map_file_name):
    try:
        now = time.time()
        kmers_df = pd.read_csv(os.path.join(path, dataset_file_name), compression='gzip')
        print("kmers_df shape: {}".format(kmers_df.shape))
        with open(os.path.join(path, kmers_map_file_name), 'r') as f:
            all_kmers_map = json.loads(f.read())
        kmers_df = kmers_df.rename(columns=all_kmers_map)
        kmers_df = kmers_df.set_index(['Unnamed: 0'])
        # Transpose to have strains as rows and kmers as columns
        kmers_df = kmers_df.T
        print("kmers_df shape: {}".format(kmers_df.shape))
        print("Finished running get_kmers_df in {} minutes".format(round((time.time() - now)/60), 4))
        return kmers_df
    except Exception as e:
        print(f"ERROR at get_kmers_df, message: {e}")
        traceback.print_exc()


def get_final_df(path, kmers_df, amr_data_file_name, antibiotic, ncbi_file_name_column, strain_column, remove_intermediate):
    try:
        now = time.time()
        amr_df = pd.read_csv(os.path.join(path, amr_data_file_name))
        # Get label of specific antibiotic
        label_df = amr_df[[ncbi_file_name_column, strain_column, antibiotic]]
        # Remove antibiotics without resistance data
        label_df = label_df[label_df[antibiotic] != '-']
        # Remove antibiotics with label 'I'
        if remove_intermediate:
            label_df = label_df[label_df[antibiotic] != 'I']
        label_df = label_df.rename(columns={antibiotic: "label"})
        label_df = label_df.set_index(['NCBI File Name'])
        print("label_df shape: {}".format(label_df.shape))
        if os.name == 'nt':
            # REMOVE!$#!@!#@#$@!#@!
            final_df = kmers_df.join(label_df, how="left")
            final_df["label"] = ["S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "S", "R", "R", "R", "R", "R", "R", "R", "R", "R", "R"]
            final_df = final_df.append(final_df)
            final_df = final_df.append(final_df)
        else:
            # Join (inner) between kmers_df and label_df
            final_df = kmers_df.join(label_df, how="inner")
        print("final_df for antibiotic: {} have {} Strains with label and {} features".format(antibiotic, final_df.shape[0], final_df.shape[1] - 2))
        print("Finished running get_final_df for antibiotic: {} in {} minutes".format(antibiotic, round((time.time() - now)/60), 4))
        return final_df
    except Exception as e:
        print(f"ERROR at get_final_df, message: {e}")
        traceback.print_exc()


def train_test_and_write_results_cv(final_df, results_file_path, model, model_params, k_folds, num_of_processes, random_seed, strain_column, antibiotic):
    try:
        now = time.time()
        X = final_df.drop(['label', strain_column], axis=1).copy()
        Y = final_df[['label']].copy()

        # Create weight according to the ratio of each class
        resistance_weight = (Y['label'] == "S").sum() / (Y['label'] == "R").sum() \
            if (Y['label'] == "S").sum() / (Y['label'] == "R").sum() > 0 else 1
        sample_weight = np.array([resistance_weight if i == "R" else 1 for i in Y['label']])
        print("Resistance_weight for antibiotic: {} is: {}".format(antibiotic, resistance_weight))

        model.set_params(**model_params)
        cv = StratifiedKFold(k_folds, random_state=random_seed, shuffle=True)
        print("Started running Cross Validation for {} folds with {} processes".format(k_folds, num_of_processes))
        classes = np.unique(Y.values.ravel())
        susceptible_ind = list(classes).index("S")
        resistance_ind = list(classes).index("R")
        temp_scores = cross_val_predict(model, X, Y.values.ravel(), cv=cv,
                                        fit_params={'sample_weight': sample_weight}, method='predict_proba',
                                        n_jobs=num_of_processes)
        predictions = []
        for p in temp_scores:
            if p[susceptible_ind] > p[resistance_ind]:
                predictions.append("S")
            else:
                predictions.append("R")
        strains_list = list(final_df[strain_column])
        files_names = list(X.index)
        results_df = pd.DataFrame({
            'Strain': strains_list, 'File name': files_names, 'Label': Y.values.ravel(),
            'Susceptible score': [x[susceptible_ind] for x in temp_scores],
            'Resistance score': [x[resistance_ind] for x in temp_scores],
            'Prediction': predictions
        })
        write_data_to_excel(results_df, results_file_path, classes)
        print("Finished running train_test_and_write_results_cv for antibiotic: {} in {} minutes".format(antibiotic, round((time.time() - now) / 60), 4))
    except Exception as e:
        print(f"ERROR at train_test_and_write_results_cv, message: {e}")


def write_data_to_excel(results_df, results_file_path, classes):
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
        confusion_matrix_df = pd.DataFrame(confusion_matrix, columns=[x + "_Actual" for x in classes], index=[x + "_Pred" for x in classes])
        confusion_matrix_df.to_excel(writer, sheet_name=name, startcol=col_ind, startrow=row_ind, index=True)
        row_ind += confusion_matrix_df.shape[0] + 2
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1_score = metrics.f1_score(y_true, y_pred, labels=classes, pos_label="R")
        evaluation_list = [["accuracy", accuracy], ["f1_score", f1_score]]
        evaluation_df = pd.DataFrame(evaluation_list, columns=["metric", "value"])
        evaluation_df.to_excel(writer, sheet_name=name, startcol=col_ind, startrow=row_ind, index=False)
        workbook = writer.book
        worksheet = writer.sheets[name]
        # percent_format = workbook.add_format({'num_format': '0.00%'})
        worksheet.set_column('A:Z', 15)
        workbook.close()
        print('Finished creating results file!')
    except Exception as e:
        print("Error in write_roc_curve.error message: {}".format(e))


def write_roc_curve(raw_score_list, labels, path, results_file_name, txt):
    try:
        results_path = path + "Plot_Results\\"
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        fpr, tpr, _ = metrics.roc_curve(labels, raw_score_list)
        auc = round(metrics.roc_auc_score(labels, raw_score_list), 3)
        plt.figure(figsize=(10, 10))
        plt.legend(loc=4)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.annotate(txt, ha='right', va='bottom', xy=(1, 0), fontsize=10)
        plt.plot(fpr, tpr, label="auc=" + str(auc))
        plt.savefig(results_path + results_file_name.replace(".xlsx", ".png"))
    except Exception as e:
        print("Error in write_roc_curve.error message: {}".format(e))


# *********************************************************************************************************************************
# Config

# antibiotic_list = ['amikacin', 'meropenem', 'Meropenem', 'Levofloxacin', 'ceftazidime']
antibiotic_list = ['Levofloxacin', 'ceftazidime']
remove_intermediate = True

# Model params
random_seed = 1
num_of_processes = 10  # relevant only if test_mode = "cv"
k_folds = 10  # relevant only if test_mode = "cv"
model = GradientBoostingClassifier(random_state=random_seed)
if os.name == 'nt':
    model_params = {'n_estimators': 5}

else:
    # model_params = {}
    model_params = {'max_depth': 4, 'n_estimators': 300, 'max_features': 0.8, 'subsample': 0.8}

# *********************************************************************************************************************************
# Constant PARAMS
if os.name == 'nt':
    dataset_file_name = 'all_kmers_file_SMALL_50.csv.gz'
    kmers_map_file_name = 'all_kmers_map.txt'
else:
    dataset_file_name = 'all_kmers_file.csv.gz'
    kmers_map_file_name = 'all_kmers_map.txt'
amr_data_file_name = 'amr_data_summary.csv'

ncbi_file_name_column = 'NCBI File Name'
strain_column = 'Strain'

prefix = '..' if os.name == 'nt' else '.'
path = os.path.join(prefix, 'results_files')

# Config END
# *********************************************************************************************************************************
kmers_df = get_kmers_df(path, dataset_file_name, kmers_map_file_name)
for antibiotic in antibiotic_list:
    results_path = os.path.join(path, "CV_Results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    results_file_name = "{}_RESULTS.xlsx".format(antibiotic)
    final_df = get_final_df(path, kmers_df, amr_data_file_name, antibiotic, ncbi_file_name_column, strain_column, remove_intermediate)
    train_test_and_write_results_cv(final_df, os.path.join(results_path, results_file_name), model, model_params, k_folds, num_of_processes, random_seed, strain_column, antibiotic)
