import numpy as np
from numpy.linalg import norm

from benchopt import BaseObjective


class Objective(BaseObjective):
    name = "Sparse Support Recovery"

    is_convex = False
    min_benchopt_version = "1.5.0"

    def set_data(self, X, y, w_true):
        self.X, self.y, self.w_true = X, y, w_true
        self.n_features = self.X.shape[1]

    def get_one_result(self):
        return {'beta': np.zeros(self.n_features)}

    def evaluate_result(self, beta):
        beta = beta.astype(np.float64)  # avoid float32 numerical errors
        support_beta = beta != 0
        K_beta = max(1, support_beta.sum())

        # compute residuals
        diff = self.y - self.X @ beta

        # compute primal objective and duality gap
        p_obj = .5 * diff.dot(diff)
        res = dict(value=p_obj, support_size=(beta != 0).sum())

        if self.w_true is not None:
            support_true = self.w_true != 0
            K_true = max(1, support_true.sum())

            res['d_w'] = norm(self.w_true - beta) / norm(self.w_true)
            res['acc'] = (support_beta == support_true).mean()
            res['precision'] = (support_beta * support_true).sum() / K_beta
            res['recall'] = (support_beta * support_true).sum() / K_true

        return res

    def get_objective(self):
        return dict(X=self.X, y=self.y)
