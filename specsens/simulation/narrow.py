import numpy as np
import matplotlib.pyplot as plt

from specsens import util
from specsens import util_sim
from specsens import WirelessMicrophone
from specsens import WhiteGaussianNoise
from specsens import EnergyDetector
from specsens import chi2_stats


def sim_narrow(gens=100,
               itrs=100,
               f_sample=1e6,
               signal_power=0.,
               noise_power=0.,
               length_sec=None,
               num_samples=None,
               theo_pfa=0.1,
               threshold=None,
               noise_uncert=0.,
               seed=None):

    # check and calculate length, in seconds and number of samples
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
                                   n=num_samples,
                                   dB=True)

    print('---- Simulation parameter ----')
    print('Generations:    %d' % (gens))
    print('Iterations:     %d' % (itrs))
    print('Total iters:    %d' % (gens * itrs))
    print('Signal power:   %.2f dB' % (signal_power))
    print('Noise power:    %.2f dB' % (noise_power))
    print('Noise uncert:   %.2f dB' % (noise_uncert))
    print('SNR:            %.2f dB' % (signal_power - noise_power))
    print('Signal length:  %.6f sec' % (length_sec))
    print('Signal samples: %d' % (num_samples))

    # calculate pd (only needed for prints)
    theo_pd = chi2_stats.pd(noise_power,
                            signal_power,
                            threshold,
                            num_samples,
                            dB=True)

    print('---- Simulation stats theory ----')
    print('Prob false alarm %.4f' % (theo_pfa))
    print('Prob detection   %.4f' % (theo_pd))
    print('Threshold        %.4f' % (threshold))

    print('---- Running simulation ----')

    pfas = list()  # probability of false alarm list
    pds = list()  # probability of detection list
    current_time = None  # time variable used for 'runtime_stats'

    # generate child seeds for wm and wgn
    ss = np.random.SeedSequence(seed)
    seeds = list(zip(ss.spawn(gens), ss.spawn(gens)))

    # outer 'generations' loop
    for i in range(gens):

        # create new signal objects
        wm = WirelessMicrophone(f_sample=f_sample, t_sec=length_sec, seed=seeds[i][0])
        wgn = WhiteGaussianNoise(f_sample=f_sample, t_sec=length_sec, seed=seeds[i][1])

        # calculate noise power with uncertainty
        gen_noise_power = np.random.normal(noise_power, noise_uncert)

        # run itertations and store results in 'result' array
        result = np.array([])

        # inner 'interations' loop
        for j in range(itrs):

            # generate signal (center frequency does not matter with simple ED)
            sig = wm.soft(f_center=1e5, power=signal_power, dB=True)

            # generate noise
            noise = wgn.signal(power=gen_noise_power, dB=True)

            # randomly decide whether signal should be present
            sig_present = bool(np.random.randint(2))
            if sig_present:
                both = sig + noise
            else:
                both = noise

            # classic (single band) energy detector
            eng = EnergyDetector.get(both)

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

        # convert to numpy array
        result = np.asarray(result)

        # calculate statistics and store in arrays
        pfa_tmp = np.sum(result == 3) / (np.sum(result == 3) +
                                         np.sum(result == 4))
        pd_tmp = np.sum(result == 1) / (np.sum(result == 1) +
                                        np.sum(result == 2))
        pfas.append(pfa_tmp)
        pds.append(pd_tmp)

        # Print simulation progress
        rem, percent, current_time = util_sim.runtime_stats(
            current_time, gens, i)
        print('%6.2fs left at %5.2f%%' % (rem, percent))

    # compute stats from lists
    pfa = np.sum(pfas) / gens
    pd = np.sum(pds) / gens

    print('---- Simulation stats ----')
    print('Prob false alarm theory %.4f' % (theo_pfa))
    print('Prob false alarm sim    %.4f' % (pfa))
    print('Prob detection theory   %.4f' % (theo_pd))
    print('Prob detection sim      %.4f' % (pd))

    # print the convergence diagrams
    util_sim.print_convergence(gens, pfas, pds)

    return pfa, pd
