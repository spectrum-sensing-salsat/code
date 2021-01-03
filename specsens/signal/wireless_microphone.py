import numpy as np

from specsens import util


class WirelessMicrophone:
    def __init__(self, f_sample, num_samples=None, t_sec=None, seed=None):
        assert f_sample > 0., 'f_sample must be greater than 0'
        if num_samples is not None:
            assert num_samples > 0., 'num_samples must be greater than 0'
            self.f_sample = f_sample
            self.num_samples = int(num_samples)
            self.t_sec = self.num_samples / self.f_sample
        elif t_sec is not None:
            assert t_sec > 0., 't_sec must be greater than 0'
            self.f_sample = f_sample
            self.t_sec = t_sec
            self.num_samples = int(self.t_sec * self.f_sample)
        else:
            assert False, 'either num_samples or t_sec needed'
        self.rng = np.random.default_rng(seed)  # random number generator

    def signal(self, f_center, f_deviation, f_modulation, power=1., dB=True):
        assert f_deviation > 0., 'f_deviation must be greater than 0'
        assert f_modulation > 0., 'f_modulation must be greater than 0'
        t = np.arange(self.num_samples) / self.f_sample  # time vector
        x = np.exp(1.j *
                   (2. * np.pi * f_center * t + f_deviation / f_modulation *
                    np.sin(2. * np.pi * f_modulation * t) +
                   2. * np.pi * self.rng.random()))  # random phase
        x -= np.mean(x)  # remove bias
        x /= np.std(x)  # normalize
        if dB:
            x *= util.dB_to_factor(power)  # set power level using dB
        else:
            x *= power  # set power level using factor
        return np.asarray(x)

    def silent(self, f_center, power=1., dB=True):
        return self.signal(f_center=f_center,
                           f_deviation=5000.,
                           f_modulation=32000.,
                           power=power,
                           dB=dB)

    def soft(self, f_center, power=1., dB=True):
        '''Best option for general performance tests.'''
        return self.signal(f_center=f_center,
                           f_deviation=15000.,
                           f_modulation=3900.,
                           power=power,
                           dB=dB)

    def loud(self, f_center, power=1., dB=True):
        return self.signal(f_center=f_center,
                           f_deviation=32600.,
                           f_modulation=13400.,
                           power=power,
                           dB=dB)

    def very_silent(self, f_center, power=1., dB=True):
        return self.signal(f_center=f_center,
                           f_deviation=50.,
                           f_modulation=32000.,
                           power=power,
                           dB=dB)
