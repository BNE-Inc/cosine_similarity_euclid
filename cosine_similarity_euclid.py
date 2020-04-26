import numpy as np
from sklearn.preprocessing import Normalizer


class fast_consine_similarity:
    def __init__(self, X, tree='ball_tree'):
        if type(X) == np.ndarray:
            self.normalizer = Normalizer()
            if tree == 'kd_tree':
                from sklearn.neighbors import KDTree
                self.tree = KDTree(self.normalizer.fit_transform(X), leaf_size=int(2*(X.shape[0]//2)))
            else:
                from sklearn.neighbors import BallTree
                self.tree = BallTree(self.normalizer.fit_transform(X), leaf_size=int(2*(X.shape[0]//2)))
        else:
            print("X must be numpy.ndarray.")


    def query(self, X, k=1, normalize=True):
        if normalize:
            _, idx = self.tree.query(self.normalizer.transform(X), dualtree=True, k=k)
            return idx
        else:
            _, idx = self.tree.query(X, dualtree=True, k=k)
            return idx
