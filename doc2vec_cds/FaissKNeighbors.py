import numpy as np
import faiss


class FaissKNeighbors:
    def __init__(self, k, resistance_ind):
        self.index = None
        self.y = None
        self.k = k
        self.resistance_ind = resistance_ind

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(np.ascontiguousarray(X.astype(np.float32)))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions

    def predict_proba(self, X):
        distances, indices = self.index.search(np.ascontiguousarray(X.astype(np.float32)), k=self.k)
        votes = self.y[indices]
        s_score = (votes == "S").sum(axis=1) / len(votes[0])
        r_score = (votes == "R").sum(axis=1) / len(votes[0])
        if self.resistance_ind == 1:
            return np.vstack((s_score, r_score)).T
        elif self.resistance_ind == 0:
            return np.vstack((r_score, s_score)).T
        else:
            raise Exception(f"resistance_ind should be either 0 or 1 and it actual: {self.resistance_ind}")

    def get_params(self):
        return {
            "k": self.k,
            "resistance_ind": self.resistance_ind
        }
