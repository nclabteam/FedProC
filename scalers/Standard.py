import numpy as np

from .BaseScaler import BaseScaler


class Standard(BaseScaler):
    def __init__(self, stat=None):
        super().__init__()
        if stat is not None:
            self.mean = []
            self.std = []
            for key in stat.keys():
                self.mean.append(stat[key]["mean"])
                self.std.append(stat[key]["std"])
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)

    def fit(self, data):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)

    def transform(self, data):
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        return self.divide_no_nan((np.asarray(data, dtype=np.float32) - mean), std)

    def inverse_transform(self, data):
        return (np.asarray(data, dtype=np.float32) * np.asarray(self.std, dtype=np.float32)) + np.asarray(self.mean, dtype=np.float32)
