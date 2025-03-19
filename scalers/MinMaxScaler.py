import numpy as np

from .BaseScaler import BaseScaler


class MinMaxScaler(BaseScaler):
    def __init__(self, stat=None):
        super().__init__()
        if stat is not None:
            self.min = []
            self.max = []
            for key in stat.keys():
                self.min.append(stat[key]["min"])
                self.max.append(stat[key]["max"])
            self.min = np.array(self.min)
            self.max = np.array(self.max)

    def fit(self, data):
        self.min = data.min(axis=0)
        self.max = data.max(axis=0)

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min
