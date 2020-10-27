import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

from constants import Bacteria, ANTIBIOTIC_DIC


BACTERIA = Bacteria.PSEUDOMONAS_AUREGINOSA.value
PATH = f"C:\\tomer_thesis\\results_files\\{BACTERIA}\\"
K_FOLDS = 5

df = pd.read_csv(PATH + "amr_labels.csv")
anti = reversed(ANTIBIOTIC_DIC[BACTERIA])

for a in anti:
    groups_dic = {}
    group_id = 0
    small = df[df[a].isin(["R", "S"])]
    ind = small.index
    y = small[a]
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True)
    for _, test_index in skf.split(ind, y):
        group_id += 1
        for i in test_index:
            groups_dic[ind[i]] = group_id
    l = []
    for i in range(len(df)):
        l.append(groups_dic.get(i, -1))
    df[f"{a}_group"] = l

df.to_csv(PATH + "amr_labels_FINAL.csv", index=False)
