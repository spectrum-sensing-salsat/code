import numpy as np
from scipy import stats

from specsens import WirelessMicrophone


class WidebandSignal():
    def __init__(self, num_samples=None, t_sec=None, num_bands=1, num_steps=1):
        '''WidebandSignal is currently only supporting 100kHz wide bands.
        By supplying a number of bands (num_bands), one indirectly determines the signals bandwith.'''

        assert num_bands > 0, 'num_bands must be greater than 0'
        self.num_bands = num_bands
        self.band_width = 1e5
        self.f_sample = self.band_width * num_bands

        if num_samples is not None:
            assert num_samples > 0., 'num_samples must be greater than 0'
            self.num_samples = int(num_samples)
            self.t_sec = self.num_samples / self.f_sample
        elif t_sec is not None:
            assert t_sec > 0., 't_sec must be greater than 0'
            self.t_sec = t_sec
            self.num_samples = int(self.t_sec * self.f_sample)
        else:
            assert False, 'either num_samples or t_sec needed'

        assert num_steps > 0, 'num_steps must be greater than 0'
        self.num_steps = int(num_steps)
        self.t_sec_per_step = self.t_sec / self.num_steps
        self.num_samples_per_step = self.num_samples // self.num_steps

        print(
            'Created WidebandSignal with %.1f MHz total bandwidth and %d samples per step'
            % (self.f_sample / 1e6, self.num_samples_per_step))

    def signal(self, mat, dB=True):
        assert (
            self.num_steps,
            self.num_bands) == np.shape(mat), 'Matrix dimensions do not match'
        mat = mat.astype(np.float32)
        wm = WirelessMicrophone(f_sample=self.f_sample,
                                num_samples=self.num_samples)
        sig = np.zeros([self.num_samples], dtype=np.complex128)
        for i in range(self.num_bands):
            band_sig = np.zeros([self.num_samples], dtype=np.complex128)
            f_center = (i + .5) * self.band_width - self.f_sample / 2.
            power = self.smooth_power(
                np.repeat(mat[:, i], self.num_samples // self.num_steps))
            sig += wm.soft(f_center, power, dB=dB)
        return sig, self.f_sample

    def smooth_power(self, power):
        sec_len = int(self.num_samples_per_step *
                      0.2)  # make the logistic 20% of the step length
        for i in range(1, self.num_steps):
            sec = power[i * self.num_samples_per_step -
                        sec_len // 2:i * self.num_samples_per_step +
                        sec_len // 2]
            left = np.mean(sec[0:len(sec) // 2])
            right = np.mean(sec[len(sec) // 2:-1])
            diff = right - left
            x = np.linspace(0, len(sec) - 1, len(sec))
            log = stats.logistic.cdf(x,
                                     len(sec) / 2 - .5,
                                     len(sec) * 1. / 15.) * diff + left
            # plt.figure(figsize=(8, 6))
            # plt.plot(x, sec, 'kx')
            # plt.plot(x, log, 'rx')
            # plt.show()
            power[i * self.num_samples_per_step -
                  sec_len // 2:i * self.num_samples_per_step +
                  sec_len // 2] = log
        return power
