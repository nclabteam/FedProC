import numpy as np


class BaseScaler:
    def __init__(self, *args, **kwargs):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        pass

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data

    @staticmethod
    def divide_no_nan(a, b):
        """
        a/b where the resulted NaN or Inf are replaced by 0.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            result = a / b
            result[np.isnan(result)] = 0.0
            result[np.isinf(result)] = 0.0
        return result
