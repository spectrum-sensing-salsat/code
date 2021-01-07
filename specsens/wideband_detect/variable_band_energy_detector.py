import numpy as np
from scipy import signal
from scipy import fft

from specsens import util
from specsens import edge_detector
from specsens import chi2_stats


class VariableBandEnergyDetector():
    def __init__(
            self,
            fft_len=1024,  # length of fft
            freqs=None,  # frequency band, x axis, used by the power spectrum
            noise_power=0.,  # in dB
            pfa=0.1,  # probability of false alarm
            smooth=10.,  # number of ffts used for smoothing
            scale=5,  # number of multi-scale wavelet products
            min_height=0.2,  # min height required
            min_freq=3e4,  # min frequency spacing required
            energy_compensation=0.5):  # compensate for shrinking bands

        self.fft_len = fft_len
        self.freqs = freqs
        self.noise_power = noise_power
        self.pfa = pfa
        self.smooth = smooth
        self.scale = scale
        self.min_height = min_height
        self.min_freq = min_freq
        self.energy_compensation = energy_compensation
        # list used for power spectrum smoothing
        self.ps_list = np.zeros(shape=(self.fft_len, 0))

    def ps_smooth(self, ps):
        '''Smooth spectrum using Bartlett's method.'''
        # if list is full, remove 'oldest' entry
        if np.size(self.ps_list, 1) >= self.smooth:
            self.ps_list = np.delete(self.ps_list, 0, axis=1)

        # add new power spectrum to list
        self.ps_list = np.concatenate((self.ps_list, np.array([ps]).T), axis=1)

        # return smooth spectrum
        return np.mean(self.ps_list, axis=1)

    def threshold(self, band):
        '''Returns true if energy in band is above threshold.'''
        thr = chi2_stats.thr(
            noise_power=self.noise_power,
            pfa=self.pfa,
            n=band[3],  # width of band
            dB=True)
        thr /= self.energy_compensation
        return band[0] > thr

    def energies(self, peak, ps):
        '''
        Converts peaks (edges) to dictionaries with actual bands.
        Keys are the center indices of each band.
        The values contain a tuple with:
        [0] - energy in band
        [1] - lower edge index of band
        [2] - upper edge index of band
        [3] - width of band in index
        '''
        assert len(ps) == self.fft_len, 'ps length does not match fft_len'
        energ = dict()

        # no peaks? retun empty dict
        if len(peak) == 0:
            return energ

        # handle outer edge and first edge in spectrum
        energ[self.cen(0, peak[0])] = (self.en(ps, 0, peak[0]), 0, peak[0],
                                       np.abs(peak[0] - 0))

        # handle edges in the spectrum
        for i, _ in enumerate(peak[:-1]):
            energ[self.cen(peak[i],
                           peak[i + 1])] = (self.en(ps, peak[i], peak[i + 1]),
                                            peak[i], peak[i + 1],
                                            np.abs(peak[i + 1] - peak[i]))

        # handle outer edge and last edge in spectrum
        energ[self.cen(peak[-1], self.fft_len -
                       1)] = (self.en(ps, peak[-1], self.fft_len - 1),
                              peak[-1], self.fft_len - 1,
                              np.abs(peak[-1] - self.fft_len - 1))
        return energ

    def en(self, ps, start, stop):
        '''Return energy in a band.'''
        return np.sum(ps[start:stop])

    def cen(self, start, stop):
        '''Return center index for a band.'''
        return int(start + np.abs(stop - start) / 2.)

    def detect(self, ps):
        '''Variable band detection on power spectrum.'''

        # smooth the power spectrum
        ps_smooth = self.ps_smooth(ps)

        # convert to dB for edge detector
        ps_dB = util.dB_power(ps_smooth)

        # edge detection
        prod, peak, peakf = edge_detector(ps_dB,
                                          self.freqs,
                                          scale=self.scale,
                                          min_height=self.min_height,
                                          min_freq=self.min_freq)
        # compute energies for subbands
        dics = self.energies(peak, ps_smooth)

        # ḱeep entry if above threshold
        dics = {k: v for k, v in dics.items() if self.threshold(v)}

        return ps_smooth, peak, peakf, dics
