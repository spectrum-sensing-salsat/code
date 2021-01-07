import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
from scipy import stats

from specsens import util_wide
from specsens import WirelessMicrophone

class WidebandSignal():
    def __init__(self,
                 num_samples=None,
                 t_sec=None,
                 num_bands=1,
                 num_steps=1,
                 seed=None,
                 band_width=1e5):
        '''
        WidebandSignal is currently only supporting 100kHz wide bands.
        By supplying a number of bands (num_bands),
        you indirectly determine the signals bandwith.
        '''

        assert num_bands > 0, 'num_bands must be greater than 0'
        self.num_bands = num_bands
        self.band_width = band_width
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
        self.seed = seed

        print(
            'Created WidebandSignal with %.1f MHz total bandwidth and %d samples per step.'
            % (self.f_sample / 1e6, self.num_samples_per_step))

    def signal(self, mat, dB=True):
        '''
        Return a wideband signal from a signal power matrix.
        '''

        assert (
            self.num_bands,
            self.num_steps) == np.shape(mat), 'Matrix dimensions do not match'

        # create 'WirelessMicrophone' used to generate all signal samples
        wm = WirelessMicrophone(f_sample=self.f_sample,
                                num_samples=self.num_samples,
                                seed=self.seed)

        # transpose and convert to float32 format
        mat = mat.astype(np.float32).T

        # create emtpy signal
        sig = np.zeros([self.num_samples], dtype=np.complex128)

        # loop over all bands
        for i in range(self.num_bands):

            # determine center frequency for current band
            f_center = (i + .5) * self.band_width - self.f_sample / 2.

            # get smoothed power
            power = self.smooth_power(
                np.repeat(mat[:, i], self.num_samples // self.num_steps))

            # add signal from current band to signal
            sig += wm.soft(f_center, power, dB=dB)

        return sig, self.f_sample

    def smooth_power(self, power):
        '''
        Helps smooth the transition from one to another power level,
        which reduces artifacts. This done by fitting a logistic curve to the
        changing power level.
        '''

        sec_len = int(self.num_samples_per_step *
                      0.2)  # make the logistic 20% of the step length long

        # loop over all steps and smooth the transition
        for i in range(1, self.num_steps):

            # cut out a section of length 'sec_len'
            sec = power[i * self.num_samples_per_step -
                        sec_len // 2:i * self.num_samples_per_step +
                        sec_len // 2]

            # determine parameters for logistic curve
            left = np.mean(sec[0:len(sec) // 2])
            right = np.mean(sec[len(sec) // 2:-1])
            diff = right - left
            x = np.linspace(0, len(sec) - 1, len(sec))
            log = stats.logistic.cdf(x,
                                     len(sec) / 2 - .5,
                                     len(sec) * 1. / 15.) * diff + left

            # replace 'hard' step by logistic function
            power[i * self.num_samples_per_step -
                  sec_len // 2:i * self.num_samples_per_step +
                  sec_len // 2] = log

        return power

    @classmethod
    def plot(self, mat):
        '''Plot the power matrix.'''
        util_wide.plot_matrix(mat, xlabel='Time [Steps]', plabel='Power [dB]')
