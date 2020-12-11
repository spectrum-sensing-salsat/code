import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial

from specsens import util
from specsens import util_sim
from specsens import WirelessMicrophone
from specsens import WhiteGaussianNoise
from specsens import EnergyDetector
from specsens import chi2_stats


def sim_narrow_parallel(gens=100,
               itrs=100,
               f_sample=1e6,
               signal_strength=0.,
               noise_strength=0.,
               length_sec=None,
               num_samples=None,
               theo_pfa=0.1,
               threshold=None,
               noise_un=0.):

    # Check and calculate length, in seconds and number of samples
    if num_samples is not None:
        assert num_samples > 0., 'num_samples must be greater than 0'
        length_sec = num_samples / f_sample
    elif length_sec is not None:
        assert length_sec > 0., 'length_sec must be greater than 0'
        length_sec = length_sec
        num_samples = int(f_sample * length_sec)
    else:
        assert False, 'either num_samples or length_sec needed'

    # Calculate threshold
    if threshold is None:
        threshold = chi2_stats.get_thr(noise_power=noise_strength,
                                          pfa=theo_pfa,
                                          n=num_samples,
                                          dB=True)

    print('---- Simulation parameter ----')
    print('Generations:    %d' % (gens))
    print('Iterations:     %d' % (itrs))
    print('Total Iters:    %d' % (gens * itrs))
    print('Signal power:   %.2f dB' % (signal_strength))
    print('Noise power:    %.2f dB' % (noise_strength))
    print('Noise uncert.:  %.2f dB' % (noise_un))
    print('SNR:            %.2f dB' % (signal_strength - noise_strength))
    print('Signal length:  %.6f sec' % (length_sec))
    print('Signal samples: %d' % (num_samples))

    theo_pd = chi2_stats.get_pd(noise_strength,
                                   signal_strength,
                                   threshold,
                                   num_samples,
                                   dB=True)
    print('---- Simulation stats theory ----')
    print('Prob false alarm %.4f' % (theo_pfa))
    print('Prob detection   %.4f' % (theo_pd))
    print('Threshold        %.4f' % (threshold))

    print('---- Running simulation ----')
    print('using %d threads on %d cores' % (mp.cpu_count(), mp.cpu_count()))

    pfas = list()  # Probability of false alarm list
    pds = list()  # Probability of detection list
    current_time = None

    # Calculate noise for each generation
    noise_strength = np.random.normal(noise_strength, noise_un, gens)

    # Outer generations loop
    for i in range(gens):

        # Prepare parallel execution
        p = mp.Pool(processes=mp.cpu_count())
        f = partial(iteration, f_sample, length_sec, signal_strength, noise_strength[i], threshold)

        # Run itertations in parallel and store results in result array
        result = p.map(f, np.arange(randi, randi+itrs))

        # Cleanup parallel execution
        p.close()
        p.join()

        # result = np.array([])
        # for j in range(itrs):
        #     result = np.append(result, iteration(f_sample, length_sec, signal_strength, noise_strength[i], threshold, 0))

        # Convert to numpy array
        result = np.asarray(result)

        # Calculate statistics and store in arrays
        pfa_tmp = np.sum(result == 3) / (np.sum(result == 3) + np.sum(result == 4))
        pd_tmp = np.sum(result == 1) / (np.sum(result == 1) + np.sum(result == 2))
        pfas.append(pfa_tmp)
        pds.append(pd_tmp)

        # Print simulation progress
        rem, percent, current_time = util_sim.runtime_stats(current_time, gens, i)
        print('%6.2fs left at %5.2f%%' % (rem, percent))

    # Compute stats from lists
    pfa = np.sum(pfas) / gens
    pd = np.sum(pds) / gens

    print('---- Simulation stats ----')
    print('Prob false alarm theory %.4f' % (theo_pfa))
    print('Prob false alarm sim    %.4f' % (pfa))
    print('Prob detection theory   %.4f' % (theo_pd))
    print('Prob detection sim      %.4f' % (pd))

    util_sim.print_convergence(gens, pfas, pds)
    return pfa, pd


def iteration(f_sample, length_sec, signal_strength, noise_strength, threshold, j):

    np.random.seed(seed=j)

    wm = WirelessMicrophone(f_sample=f_sample, t_sec=length_sec)
    wgn = WhiteGaussianNoise(f_sample=f_sample, t_sec=length_sec)

    # Generate signal, center frequency does not matter with single band ED
    sig = wm.get_soft(f_center=1e5, power=signal_strength, dB=True)

    # Generate noise
    noise = wgn.get_signal(power=noise_strength,
                           dB=True)

    # Randomly decide whether signal should be present
    sig_present = bool(np.random.randint(2))
    if sig_present:
        both = sig + noise
    else:
        both = noise

    # Classic (single band) energy detector
    eng = EnergyDetector.get(both)

    # Threshold
    sig_detected = eng > threshold

    # Log signal and detection outcome
    if sig_present and sig_detected:
        return 1
    elif sig_present and not sig_detected:
        return 2
    elif not sig_present and sig_detected:
        return 3
    else:
        return 4
