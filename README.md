# Spectrum Sensing for the SALSAT Nanosatellite

This repository is part of my bachelor thesis: *Spectrum Sensing for the SALSAT Nanosatellite*. It provides algorithms and simulations for blind wideband spectrum sensing with special emphasis on space applications. More background information on the SALSAT project can be found [here](https://www.raumfahrttechnik.tu-berlin.de/menue/forschung/aktuelle_projekte/salsat/parameter/en/).

This repository is split up into two main directories:
* `specsens/` the python package that contains all relevant code and algorithms.
* `notebooks/` contains usage examples for all parts of the `specsens` package.

## Usage

The code is written in `python 3.6` using the packages listed in [`requirements.txt`](./requirements.txt). Consider using a [virtual environment](https://docs.python.org/3/tutorial/venv.html). **Ensure that you have matplotlib `3.1.x` installed. Other versions might give weird artifacts. I am still looking into the problem at this point in time.**

### Step by step installation guide
Starting with a *clean* Ubuntu 18.04 system (should also work with other Linux distributions) you will need to follow these steps:
- Clone the repository using `git`, run `git clone git@git.tu-berlin.de:salsat/spectrum-sensing.git`
- Navigate into the repository, run `cd spectrum-sensing`
- Ensure you have Python 3, run `python3 --version`
- Install Python Virtual Environments, run `sudo apt install python3-venv`
- Install `pip3`, run `sudo apt install python3-pip`
- Create a new virtual environment, run `python3 -m venv specsens-env`
- To activate the virtual environment, run `source specsens-env/bin/activate`
- (To deactivate the virtual environment, run `deactivate`)
- Use the `requirements.txt` to install required python packages, run `pip3 install -r requirements.txt`
- Finally, start Jupyter Notebook, run `jupyter notebook`

If you already have some of parts installed, just skip the corresponding steps. Feel free to open an issue or contact me if you run into any problems.

## Specsens Package
In order to organize and structure my work I created the `specsens` package. It contains most of the algorithms and simulations that I developed. It behaves like a regular python package (one can just `import specsens`) and consists of the following directories:
* `signal/` signal and noise generators used mainly for simulations, as well as some utilities that simplify working with signals.
* `plot/` plotting functions to visualize signals.
* `energy_detect/` energy detector and functions to calculate performance statistics (prob. of false alarm, prob. of detection, etc.).
* `wideband_detect/` wideband detection algorithms (wideband energy detector, spectrum edge detector, etc.).
* `eigenvalue_detect/` eigenvalue detection algorithms based on signal covariance matrices
* `simulation/` simulations to evaluate performance statistics.

When using the `specsens` package locally you need to add the following lines before you import the package. This tells the python interpreter where to find the package:
```python
import sys
sys.path.insert(0, '..')
```
Replace `..` with the path to the `specsens` package, when moving things around.


## Notebooks
The notebooks provide a visual presentation of the `specsens` package and its functionalities. They allow you to get an overview of what the algorithms are doing in order to better understand them. Feel free to play around with the code and observe the behavior, especially in the simulations.
- `01_signal_and_noise.ipynb` basic overview over complex signal and noise generation + time domain visualization.
- `02_simple_energy_detector.ipynb` simple narrow band time domain energy detector.
- `03_energy_detector_statistics.ipynb` analytical performance statistics overview and comparison (prob. of false alarm, prob. of detection, etc.) ( comparison between Chi-square based statistics and central limit theorem based statistics).
- `04_energy_detector_simulation.ipynb` simulation using simple energy detector and comparison between analytical and numerical statistics.
- `05_short_time_fourier_transform.ipynb` short time Fourier transform + frequency domain visualization.
- `06_wideband_signal.ipynb` wideband signal generation using signal matrix.
- `06a_doppler_signal.ipynb` wideband doppler signal generation that aims to realistically reproduce doppler shifts.
- `07_wideband_detect.ipynb` short time fourier transform based wide band energy detection.
- `07a_wideband_detect_simulation.ipynb` simulation for wide band energy detection.
- `07b_wideband_detect_doppler.ipynb` wideband detection of doppler singal.
- `08_edge_detect.ipynb` spectrum edge detection using wavelet transforms.
- `09_variable_band_detect.ipynb` variable band wideband energy detection using edge detection.
- `09a_variable_band_detect_doppler.ipynb` variable band wideband energy detection of doppler signal.
- `10_noise_estimation_simulation.ipynb` energy detection using noise estimation.
- `11_eigenvalue_detector.ipynb` eigenvalue detector based on covariance matrix.
- `12_eigenvalue_detector_simulation.ipynb` simulation of eigenvalue detector.
- `13_eigenvalue_detector_filter.ipynb` eigenvalue detector with bandpass filter.
- `14_eigenvalue_detector_whitening.ipynb` eigenvalue detector with bandpass filter using noise prewhitening.
- `15_eigenvalue_whitening_simualtion.ipynb` simulation of eigenvalue detector using bandpass filter and noise prewhitening.

## TODO
### General
- [x] Upload first version to gitlab
- [] Notebook documentation in readme
- [] Upload papers and create basic paper index that helps to *connect* paper and python implementation
- [] Documentation for every function using sphinx(not sure if I will do that tho) (https://docs.python-guide.org/writing/documentation/)
- [] Upload specsens to PyPi


### Functional
- [] Time space plotting functions. (`time_plot.py`)
- [] Implement sigmoid detection output
- [] Add pfa and pd eigenvalue stats
- [] Create eigenvalue simulation with filter and prewhitening matrix
- [] Combine eigenvalue detection and noise estimation based energy detection, use energy detection as heuristic and binary search pattern to find *noise only* band to use for noise estimation

## Author
Fabian Peddinghaus <peddinghaus@tu-berlin.de>
