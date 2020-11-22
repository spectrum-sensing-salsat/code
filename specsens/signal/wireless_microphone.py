import numpy as np

class WirelessMicrophone:
    def __init__(self, f_sample, num_samples=None, t_sec=None):
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

    def get_signal(self, f_center, f_deviation, f_modulation, dB=0.):
        assert f_deviation > 0., 'f_deviation must be greater than 0'
        assert f_modulation > 0., 'f_modulation must be greater than 0'
        t = np.arange(self.num_samples) / self.f_sample
        x = np.exp(1.j *
                   (2. * np.pi * f_center * t + f_deviation / f_modulation *
                    np.sin(2. * np.pi * f_modulation * t)))
        x -= np.mean(x) # remove bias
        x /= np.std(x) # normalize
        x *= 10.**(dB / 20.) # set power level
        return x

    def get_silent(self, f_center, dB=0.):
        return self.get_signal(f_center, 5000, 32000, dB)

    def get_soft(self, f_center, dB=0.):
        return self.get_signal(f_center, 3900, 15000, dB)

    def get_loud(self, f_center, dB=0.):
        return self.get_signal(f_center, 13400, 32600, dB)
