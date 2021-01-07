import numpy as np
from scipy import signal
from scipy import fft
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')

from specsens import util
from specsens import Stft


def spectrum_plot_1d(sig,
                     f_sample,
                     window='flattop',
                     nfft=1024,
                     dB=True,
                     type='our',
                     title=None,
                     linewidth=0.75):

    assert len(sig) == nfft, 'signal length does not match nfft'

    if type == 'matplotlib':
        assert False, 'matplotlib is not yet supported'

    elif type == 'scipy':
        assert False, 'scipy is not yet supported'

    elif type == 'our':
        plt.figure(figsize=(8, 6))
        if title is not None:
            plt.title('PSD of %s' % title)
        sft = Stft(n=nfft, window=window)
        f, x = sft.stft(sig, f_sample, normalized=True, dB=dB)
        plt.plot(f, x, linewidth=linewidth, aa=True)

    else:
        assert False, 'unknown type'

    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power [dB]' if dB else 'Power [linear]')
    plt.grid(linewidth=0.5)
    plt.show()
