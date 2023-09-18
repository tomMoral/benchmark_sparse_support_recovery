import warnings

from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from celer import Lasso
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    name = 'Celer'
    sampling_strategy = 'iteration'

    install_cmd = 'conda'
    requirements = ['pip:celer']
    references = [
        'M. Massias, A. Gramfort and J. Salmon, ICML, '
        '"Celer: a Fast Solver for the Lasso with Dual Extrapolation", '
        'vol. 80, pp. 3321-3330 (2018)'
    ]

    parameters = {
        'reg': [0.1, 0.01, 0.001]
    }

    def set_objective(self, X, y, ):
        self.X, self.y = X, y
        self.lmbd = self.reg * self._get_lambda_max()

        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        n_samples = self.X.shape[0]
        self.lasso = Lasso(
            alpha=self.lmbd / n_samples, max_iter=1, max_epochs=100000,
            tol=1e-12, prune=True, fit_intercept=False,
            warm_start=False, positive=False, verbose=False,
        )

    def run(self, n_iter):
        if n_iter == 0:
            self.coef = np.zeros([self.X.shape[1]])
        else:
            self.lasso.max_iter = n_iter
            self.lasso.fit(self.X, self.y)

            coef = self.lasso.coef_.flatten()
            self.coef = coef

    @staticmethod
    def get_next(previous):
        "Linear growth for n_iter."
        return previous + 1

    def get_result(self):
        return {'beta': self.coef}

    def _get_lambda_max(self):
        return abs(self.X.T.dot(self.y)).max()
