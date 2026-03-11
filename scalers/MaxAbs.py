import numpy as np

from .BaseScaler import BaseScaler


class MaxAbs(BaseScaler):
    """Scale each feature by its maximum absolute value."""

    def __init__(self, stat=None):
        super().__init__()
        if stat is not None:
            self.max_abs = []
            for key in stat.keys():
                self.max_abs.append(stat[key]["max_abs"])
            self.max_abs = np.array(self.max_abs)

    def fit(self, data):
        self.max_abs = np.max(np.abs(data), axis=0)

    def transform(self, data):
        return self.divide_no_nan(data, self.max_abs)

    def inverse_transform(self, data):
        return data * self.max_abs
