# Spectrum Sensing for the SALSAT Nanosatellite

This repository is part of my bachelor thesis: *Spectrum Sensing for the SALSAT Nanosatellite*. It provides algorithms and simulations for blind wideband spectrum sensing with special emphasis on space applications. More background information on the SALSAT project can be found [here](https://www.raumfahrttechnik.tu-berlin.de/menue/forschung/aktuelle_projekte/salsat/parameter/en/). The code is a constant *work in progress* and ever-evolving. So stay with me if something is not working or behaving "weird". You can always contact me (details at the bottom).

This repository is split up into three main directories:

* `specsens/` the python package that contains all relevant code and algorithms.
* `notebooks/` contains usage examples for all parts of the `specsens` package.
* `graphs/` contains the code used to create the graphs in my thesis.

## Usage

The code is written in `python 3.6` using the packages listed in `requirements.txt`. Consider using a [virtual environment](https://docs.python.org/3/tutorial/venv.html). **Ensure that you have matplotlib** `3.1.x` installed. Other versions might give weird artifacts. I am still looking into the problem at this point in time.

### Step by step installation guide

Starting with a *clean* Ubuntu 18.04 system (should also work with other Linux distributions) you will need to follow these steps:

* Clone the repository using `git`, run `git clone git@git.tu-berlin.de:salsat/spectrum-sensing.git` (you will need to have `ssh` with GitLab setup for this to work)
* Navigate into the repository, run `cd spectrum-sensing`
* Ensure you have Python 3, run `python3 --version`
* Install Python Virtual Environments, run `sudo apt install python3-venv`
* Install `pip3`, run `sudo apt install python3-pip`
* Create a new virtual environment, run `python3 -m venv specsens-env`
* To activate the virtual environment, run `source specsens-env/bin/activate`
* (To deactivate the virtual environment, run `deactivate`)
* Use the `requirements.txt` to install required python packages, run `pip3 install -r requirements.txt`
* Finally, start Jupyter Notebook, run `jupyter notebook`

If you already have some of the parts installed, just skip the corresponding steps. Feel free to open an issue or contact me if you run into any problems.

## Specsens Package

In order to organize and structure my work, I created the `specsens` package. It contains most of the algorithms and simulations that I developed. It behaves like a regular python package (one can just `import specsens`). It consists of the following directories:

* `eigenvalue_detect/` Eigenvalue detection algorithms based on signal covariance matrices.
* `energy_detect/` Energy detector and functions to calculate performance statistics (prob. of false alarm, prob. of detection, etc.).
* `noise_estimation/` Noise estimation methods and related utilities. (**Currently still part of eigenvalue_detect, work in progress.**)
* `plot/` Plotting functions to visualize signals and detection results.
* `signal/` Signal and noise generators used mainly for simulations, as well as some utilities that simplify working with signals.
* `simulation/` Simulations to evaluate performance statistics.
* `wideband_detect/` Wideband detection algorithms (wideband energy detector, wavelet edge detector, etc.).

When using the `specsens` package locally you need to add the following lines before you import the package. This tells the python interpreter where to find the package:

```
import sys
sys.path.insert(0, '..')
```

Replace `..` with the path to the `specsens` package, when moving things around.

## Notebooks

The notebooks provide a visual presentation of the `specsens` package and its functionalities. They allow you to get an overview of what the algorithms are doing in order to better understand them. Feel free to play around with the code and observe the behavior, especially in the simulations.

* `01_signal_and_noise.ipynb` basic overview over complex signal and noise generation + time-domain visualization.
* `02_simple_energy_detector.ipynb` simple narrowband time-domain energy detector.
* `03_energy_detector_statistics.ipynb` analytical performance statistics overview and comparison (prob. of false alarm, prob. of detection, etc.) ( comparison between Chi-square based statistics and central limit theorem based statistics).
* `04_energy_detector_simulation.ipynb` simulation using simple energy detector and comparison between analytical and numerical statistics.
* `05_short_time_fourier_transform.ipynb` short-time Fourier transform + frequency domain visualization.
* `06_wideband_signal.ipynb` wideband signal generation using signal matrix.
* `06a_doppler_signal.ipynb` wideband doppler signal generation that aims to realistically reproduce doppler shifts.
* `07_wideband_detect.ipynb` short-time Fourier transform-based wideband energy detection.
* `07a_wideband_detect_simulation.ipynb` simulation for wideband energy detection.
* `07b_wideband_detect_doppler.ipynb` wideband detection of doppler signal.
* `08_edge_detect.ipynb` spectrum edge detection using wavelet transforms.
* `09_variable_band_detect.ipynb` variable band wideband energy detection using edge detection.
* `09a_variable_band_detect_doppler.ipynb` variable band wideband energy detection of doppler signal.
* `10_noise_estimation_simulation.ipynb` energy detection using noise estimation.
* `11_eigenvalue_detector.ipynb` eigenvalue detector based on covariance matrix.
* `12_eigenvalue_detector_simulation.ipynb` simulation of eigenvalue detector.
* `13_eigenvalue_detector_filter.ipynb` eigenvalue detector with bandpass filter.
* `14_eigenvalue_detector_whitening.ipynb` eigenvalue detector with bandpass filter using noise whitening.
* `15_eigenvalue_simulation_whitening.ipynb` simulation of eigenvalue detector with bandpass filter and noise whitening.
* `16_eigenvalue_noise_estimation.ipynb` noise power estimation using covariance matrix eigenvalues.
* `17_eigenvalue_noise_estimation_simulation.ipynb` simulation of wideband energy detection using eigenvalue noise power estimation.
* `18_noise_estimation_comparison.ipynb` comparison simulation of noise estimation techniques.

## TODO

### General

* [x] Upload first version to gitlab
* [x] Notebook documentation in readme
* [ ] Upload papers and create basic paper index that helps to *connect* paper and python implementation
* [ ] Documentation for every function using [sphinx](https://docs.python-guide.org/writing/documentation/)
* [ ] Upload specsens to PyPi

### Functional

* [x] Time space plotting functions. (`time_plot.py`)
* [ ] Implement sigmoid detection output
* [ ] Cleanup variable band spectrum sensing
* [x] Add pfa and pd eigenvalue stats
* [x] Create eigenvalue simulation with filter and whitening matrix
* [ ] Combine eigenvalue detection and noise estimation based energy detection, use energy detection as heuristic and binary search pattern to find *noise only* band to use for noise estimation
* [x] Use small eigenvalues for noise estimation

## Author

Fabian Peddinghaus [peddinghaus@tu-berlin.de](mailto:peddinghaus@tu-berlin.de)
