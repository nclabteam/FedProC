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

    def divide_no_nan(self, a, b):
        """
        a/b where the resulted NaN or Inf are replaced by 0.
        """
        result = a / b
        result[result != result] = 0.0
        result[result == np.inf] = 0.0
        return result
