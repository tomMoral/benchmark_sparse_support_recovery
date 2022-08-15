import numpy as np

from benchopt import BaseDataset
from benchopt.datasets.simulated import make_correlated_data


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples, n_features, density': [
            (100, 20, 0.2),
            (100, 100, 0.1),
            (100, 300, 0.03)
        ],
        'rho': [0, 0.6],
        'random_state': [42]
    }

    def get_data(self):
        rng = np.random.RandomState(self.random_state)

        X, y, w_true = make_correlated_data(
            self.n_samples, self.n_features, rho=self.rho,
            density=self.density, random_state=rng
        )

        data = dict(X=X, y=y, w_true=w_true)

        return data
