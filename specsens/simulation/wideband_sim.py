import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
import multiprocessing as mp
import tqdm
from functools import partial

from specsens import util
from specsens import util_sim
from specsens import WirelessMicrophone
from specsens import WhiteGaussianNoise
from specsens import WidebandEnergyDetector
from specsens import Stft
from specsens import chi2_stats


def generation(f_sample, length_sec, itrs, noise_power, signal_power,
               noise_uncert, threshold, window, fft_len, num_bands, f_center,
               band_to_detect, seeds):

    # create new signal objects
    wm = WirelessMicrophone(f_sample=f_sample, t_sec=length_sec, seed=seeds[0])
    wgn = WhiteGaussianNoise(f_sample=f_sample,
                             t_sec=length_sec,
                             seed=seeds[1])

    # local rng
    rng = np.random.default_rng(seeds[2])

    # calculate noise power with uncertainty
    gen_noise_power = rng.normal(loc=noise_power, scale=noise_uncert)

    # store results in 'result' array energies in 'energy' array
    result = np.array([])
    energy = np.array([])

    # 'inner' interations loop
    for _ in range(itrs):

        # generate signal
        sig = wm.soft(f_center=f_center, power=signal_power, dB=True)

        # generate noise
        noise = wgn.signal(power=gen_noise_power, dB=True)

        # randomly decide whether signal should be present
        sig_present = rng.choice([True, False])
        if sig_present:
            both = sig + noise
        else:
            both = noise

        # create a Short Time Fourier Transform object
        sft = Stft(n=fft_len, window=window)

        # use the stft to transform the signal into the frequency domain
        f, psd = sft.stft(both, f_sample, normalized=False, dB=False)

        # create a Wideband Energy Detector object
        fed = WidebandEnergyDetector(num_bands=num_bands,
                                     f_sample=f_sample,
                                     fft_len=fft_len,
                                     freqs=f)

        # compute energy for all bands
        bands = fed.detect(psd)

        # energy detector
        eng = bands[band_to_detect]

        # threshold
        sig_detected = eng > threshold

        # log detection outcome
        if sig_present and sig_detected:
            result = np.append(result, 1)
        elif sig_present and not sig_detected:
            result = np.append(result, 2)
        elif not sig_present and sig_detected:
            result = np.append(result, 3)
        else:
            result = np.append(result, 4)

        # log energy
        energy = np.append(energy, eng)

    # calculate statistics and store in arrays
    pfa_tmp = np.sum(result == 3) / (np.sum(result == 3) + np.sum(result == 4))
    pd_tmp = np.sum(result == 1) / (np.sum(result == 1) + np.sum(result == 2))

    return pfa_tmp, pd_tmp, energy, result


def wideband_sim(
    gens=50,  # generations, number of environments
    itrs=300,  # iterations, number of tests in each environment
    f_sample=1e6,  # in Hz
    signal_power=0.,  # in dB
    f_center=-1e5,  # signal center frequency
    noise_power=0.,  # in dB
    length_sec=None,  # length of each section in seconds
    num_samples=None,  # number of samples
    theo_pfa=0.1,  # probability of false alarm
    threshold=None,  # threshold used for detection
    noise_uncert=0.0,  # standard deviation of the noise normal distribution
    seed=None,  # random seed used for rng
    num_procs=None,  # number of processes to run in parallel
    window='box',  # window used with fft
    fft_len=1024,  # samples used for fft
    num_bands=1,  # total number of bands
    band_to_detect=0):  # band to 'search' for signal in

    # set number of processes used
    if num_procs is None:
        num_procs = mp.cpu_count()
    assert num_procs > 0, 'num_procs must be greater than 0'
    assert num_procs <= gens, 'num_procs must be less or equal to gens'

    # check and calculate length (in seconds and number of samples)
    if num_samples is not None:
        assert num_samples > 0., 'num_samples must be greater than 0'
        length_sec = num_samples / f_sample
    elif length_sec is not None:
        assert length_sec > 0., 'length_sec must be greater than 0'
        length_sec = length_sec
        num_samples = int(f_sample * length_sec)
    else:
        assert False, 'either num_samples or length_sec needed'

    # calculate threshold
    if threshold is None:
        threshold = chi2_stats.thr(noise_power=noise_power,
                                   pfa=theo_pfa,
                                   n=fft_len // num_bands,
                                   dB=True)

    print('---- Simulation parameters ----')
    print('Generations:    %d' % (gens))
    print('Iterations:     %d' % (itrs))
    print('Total iters:    %d' % (gens * itrs))
    print('Signal power:   %.2f dB' % (signal_power))
    print('Sig cent. freq: %.1f Hz' % (f_center))
    print('Noise power:    %.2f dB' % (noise_power))
    print('Noise uncert:   %.2f dB' % (noise_uncert))
    print('SNR:            %.2f dB' % (signal_power - noise_power))
    print('Signal length:  %.6f s' % (length_sec))
    print('Signal samples: %d' % (num_samples))
    print('FFT length:     %d' % (fft_len))
    print('Num. of bands:  %d' % (num_bands))
    print('Band to detect: %d' % (band_to_detect))

    # calculate pd (only needed for prints)
    theo_pd = chi2_stats.pd(noise_power,
                            signal_power,
                            threshold,
                            n=fft_len // num_bands,
                            dB=True,
                            num_bands=num_bands)

    print('---- Simulation stats theory ----')
    print('Prob false alarm: %.4f' % (theo_pfa))
    print('Prob detection:   %.4f' % (theo_pd))
    print('Threshold:        %.4f' % (threshold))

    print('---- Running simulation ----')
    print('Using %d processes on %d cores' % (num_procs, mp.cpu_count()))

    pfas = list()  # probability of false alarm list
    pds = list()  # probability of detection list
    current_time = None  # time variable used for 'runtime_stats'

    # generate child seeds for wm and wgn
    seed_seq = np.random.SeedSequence(seed)
    seeds = list(zip(seed_seq.spawn(gens), seed_seq.spawn(gens),
                     seed_seq.spawn(gens)))

    # prepare parallel execution
    p = mp.Pool(processes=num_procs)
    f = partial(generation, f_sample, length_sec, itrs, noise_power,
                signal_power, noise_uncert, threshold, window, fft_len,
                num_bands, f_center, band_to_detect)

    # run simulation while showing progress bar
    res = list(tqdm.tqdm(p.imap(f, seeds), total=gens))

    # cleanup parallel execution
    p.close()
    p.join()

    # 'unwrap' res tuples
    pfas = [r[0] for r in res]
    pds = [r[1] for r in res]
    energies = np.ravel([r[2] for r in res])
    results = np.ravel([r[3] for r in res])

    # compute stats from lists
    pfa = np.sum(pfas) / gens
    pd = np.sum(pds) / gens

    # compute energy distributions
    engs_both = energies[np.where(results <= 2)[0]]
    engs_noise = energies[np.where(results > 2)[0]]

    print('---- Simulation stats ----')
    print('Prob false alarm theory: %.4f' % (theo_pfa))
    print('Prob false alarm sim:    %.4f' % (pfa))
    print('Prob detection theory:   %.4f' % (theo_pd))
    print('Prob detection sim:      %.4f' % (pd))

    # print the convergence diagrams
    util_sim.print_convergence(gens, pfas, pds, theo_pfa, theo_pd)

    # print energy distributions
    util_sim.print_distribution(engs_both, engs_noise, fft_len // num_bands,
                                signal_power, noise_power, threshold,
                                num_bands)

    return pfa, pd
