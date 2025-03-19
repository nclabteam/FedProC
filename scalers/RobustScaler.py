import numpy as np

from .BaseScaler import BaseScaler


class RobustScaler(BaseScaler):
    def __init__(self, stat=None):
        super().__init__()
        if stat is not None:
            self.q1s = []
            self.q3s = []
            for key in stat.keys():
                self.q1s.append(stat[key]["q1"])
                self.q3s.append(stat[key]["q3"])
            self.q1s = np.array(self.q1s)
            self.q3s = np.array(self.q3s)

    def fit(self, data):
        self.q1s = np.percentile(data, 25, axis=0)
        self.q3s = np.percentile(data, 75, axis=0)

    def transform(self, data):
        return (data - self.q1s) / (self.q3s - self.q1s)

    def inverse_transform(self, data):
        return data * (self.q3s - self.q1s) + self.q1s
