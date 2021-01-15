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
from specsens import Stft
from specsens import WidebandEnergyDetector
from specsens import noise_est as noise_esti


def generation(f_sample, length_sec, itrs, noise_power, signal_power,
               noise_uncert, window, fft_len, f_center, num_bands,
               band_noise_est, cov_size, seeds):

    # create new signal objects
    wm = WirelessMicrophone(f_sample=f_sample, t_sec=length_sec, seed=seeds[0])
    wgn = WhiteGaussianNoise(f_sample=f_sample,
                             t_sec=length_sec,
                             seed=seeds[1])

    # local rng
    rng = np.random.default_rng(seeds[2])

    # calculate noise power with uncertainty
    gen_noise_power = rng.normal(loc=noise_power, scale=noise_uncert)

    # list of noise estimation errors
    errs_time = np.array([])
    errs_band = np.array([])
    errs_eig_avg = np.array([])
    errs_eig_hist = np.array([])
    errs_eig_kde = np.array([])
    errs_eig_mle = np.array([])

    # 'inner' interations loop
    for _ in range(itrs):

        # generate signal
        sig = wm.soft(f_center=f_center, power=signal_power, dB=True)

        # generate noise
        noise = wgn.signal(power=gen_noise_power, dB=True)

        # create acutal signal
        both = sig + noise

        # time noise estimation
        est_time = util.dB_power(np.mean(np.abs(noise)**2.))

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
        # in band noise estimation
        est_band = util.dB_power(bands[band_noise_est] / (fft_len / num_bands))

        # eigenvalue noise estimation
        est_eig_avg, est_eig_hist, est_eig_kde, est_eig_mle = noise_esti.estimate(
            both, int(f_sample * length_sec), l=cov_size)

        #         print(est_time)
        #         print(est_band)
        #         print(est_eig_avg)
        #         print(est_eig_hist)
        #         print(est_eig_kde)
        #         print(est_eig_mle)

        # calculate errors
        err_time = util.dB_rel_err(gen_noise_power, est_time)
        err_band = util.dB_rel_err(gen_noise_power, est_band)
        err_eig_avg = util.dB_rel_err(gen_noise_power, est_eig_avg)
        err_eig_hist = util.dB_rel_err(gen_noise_power, est_eig_hist)
        err_eig_kde = util.dB_rel_err(gen_noise_power, est_eig_kde)
        err_eig_mle = util.dB_rel_err(gen_noise_power, est_eig_mle)

        # append to errors list
        errs_time = np.append(errs_time, err_time)
        errs_band = np.append(errs_band, err_band)
        errs_eig_avg = np.append(errs_eig_avg, err_eig_avg)
        errs_eig_hist = np.append(errs_eig_hist, err_eig_hist)
        errs_eig_kde = np.append(errs_eig_kde, err_eig_kde)
        errs_eig_mle = np.append(errs_eig_mle, err_eig_mle)

    # calculate average error and return
    return np.mean(errs_time), np.mean(errs_band), np.mean(
        errs_eig_avg), np.mean(errs_eig_hist), np.mean(errs_eig_kde), np.mean(
            errs_eig_mle)


def estimation_comparison_sim(
        gens=50,  # generations, number of environments
        itrs=300,  # iterations, number of tests in each environment
        f_sample=1e6,  # in Hz
        signal_power=0.,  # in dB
        f_center=-1e5,  # signal center frequency
        noise_power=0.,  # in dB
        length_sec=None,  # length of each section in seconds
        num_samples=None,  # number of samples
        noise_uncert=0.0,  # standard deviation of the noise normal distribution
        seed=None,  # random seed used for rng
        num_procs=None,  # number of processes to run in parallel
        window='box',  # window used with fft
        fft_len=1024,  # samples used for fft
        num_bands=1,  # total number of bands
        band_noise_est=None,  # band to use for noise estimation
        cov_size=50):

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
    print('Band noise est: %d' % (band_noise_est))

    print('---- Running simulation ----')
    print('Using %d processes on %d cores' % (num_procs, mp.cpu_count()))

    # generate child seeds for wm and wgn
    seed_seq = np.random.SeedSequence(seed)
    seeds = list(
        zip(seed_seq.spawn(gens), seed_seq.spawn(gens), seed_seq.spawn(gens)))

    # prepare parallel execution
    p = mp.Pool(processes=num_procs)
    f = partial(generation, f_sample, length_sec, itrs, noise_power,
                signal_power, noise_uncert, window, fft_len, f_center,
                num_bands, band_noise_est, cov_size)

    # run simulation while showing progress bar
    res = list(tqdm.tqdm(p.imap(f, seeds), total=gens))

    # cleanup parallel execution
    p.close()
    p.join()

    # 'unwrap' res tuples
    errs_time = [r[0] for r in res]
    errs_band = [r[1] for r in res]
    errs_eig_avg = [r[2] for r in res]
    errs_eig_hist = [r[3] for r in res]
    errs_eig_kde = [r[4] for r in res]
    errs_eig_mle = [r[5] for r in res]

    err_time = np.mean(errs_time)
    err_band = np.mean(errs_band)
    err_eig_avg = np.mean(errs_eig_avg)
    err_eig_hist = np.mean(errs_eig_hist)
    err_eig_kde = np.mean(errs_eig_kde)
    err_eig_mle = np.mean(errs_eig_mle)

    print('---- Simulation stats ----')
    print('Err time:             %.4f dB' % (err_time))
    print('Err band:             %.4f dB' % (err_band))
    print('Err eigenval average: %.4f dB' % (err_eig_avg))
    print('Err eigenval hist:    %.4f dB' % (err_eig_hist))
    print('Err eigenval kde:     %.4f dB' % (err_eig_kde))
    print('Err eigenval mle:     %.4f dB' % (err_eig_mle))

    return err_time, err_band, err_eig_avg, err_eig_hist, err_eig_kde, err_eig_mle
