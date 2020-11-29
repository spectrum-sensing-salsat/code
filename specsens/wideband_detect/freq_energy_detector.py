import numpy as np
from scipy import signal
from scipy import fft

from specsens import util
from specsens import Stft

class FreqEnergyDetector():

    def __init__(self, stft: Stft = None):
        print("FreqEnergyDetector")
