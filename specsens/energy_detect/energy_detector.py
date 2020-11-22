import numpy as np

class EnergyDetector():
    def __init__(self):
        pass

    @classmethod
    def get(self, x):
        assert x.ndim == 1, 'x must have exactly 1 dimension'
        return np.sum(np.abs(x) ** 2.)
