import numpy as np
from scipy import stats

from specsens import util


class WhiteGaussianNoise:
    def __init__(self, f_sample=1000.0, n=None, t_sec=None, seed=None):
        assert f_sample > 0., 'f_sample must be greater than 0'
        if n is not None:
            assert n > 0, 'n must be greater than 0'
            self.f_sample = f_sample
            self.num_samples = int(n)
            self.t_sec = self.num_samples / self.f_sample
        elif t_sec is not None:
            assert t_sec > 0., 't_sec must be greater than 0'
            self.f_sample = f_sample
            self.t_sec = t_sec
            self.num_samples = int(self.t_sec * self.f_sample)
        else:
            assert False, 'either n or t_sec needed'
        self.rng = np.random.default_rng(seed)  # random number generator

    def signal(self, power=1., dB=True):
        if dB:
            x = util.dB_to_factor_power(power)  # set power level using dB
        else:
            x = power  # set power level using factor
        stat = stats.multivariate_normal(mean=[0., 0.],
                                         cov=[[.5 * x, 0.], [0., .5 * x]])
        x = stat.rvs(size=self.num_samples, random_state=self.rng).view(
            np.complex128).reshape(self.num_samples)
        return np.asarray(x)
