import numpy as np
import os
if os.name != 'nt':
    import faiss


class FaissKNeighbors:
    def __init__(self, k):
        self.index = None
        self.y = None
        self.k = k

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
        s_score = (votes == 1).sum(axis=1) / len(votes[0])
        r_score = (votes == 0).sum(axis=1) / len(votes[0])
        return np.vstack((s_score, r_score)).T

    def get_params(self):
        return {
            "k": self.k,
        }
