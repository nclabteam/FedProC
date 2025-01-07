import numpy as np

from .BaseScaler import BaseScaler


class StandardScaler(BaseScaler):
    def __init__(self, stat=None):
        super(StandardScaler, self).__init__()
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
        return self.divide_no_nan((data - self.mean), self.std)

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
