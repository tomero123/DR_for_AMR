import pandas as pd
from sklearn.model_selection import train_test_split

from constants import Bacteria, ANTIBIOTIC_DIC


BACTERIA = Bacteria.PSEUDOMONAS_AUREGINOSA.value
PATH = f"C:\\tomer_thesis\\results_files\\{BACTERIA}\\"


df = pd.read_csv(PATH + "amr_labels.csv")
anti = ANTIBIOTIC_DIC[BACTERIA]
for a in anti:
    small = df[df[a].isin(["R", "S"])]
    ind = small.index
    y = small[a]
    X_train, X_test, y_train, y_test = train_test_split(ind, y, test_size=0.25)
    l = []
    for i in range(len(df)):
        if i in X_train:
            l.append(1)
        elif i in X_test:
            l.append(0)
        else:
            l.append(-1)
    df[f"{a}_is_train"] = l

df.to_csv(PATH + "amr_labels_FINAL.csv", index=False)
