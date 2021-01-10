# Bachelor Thesis: Spectrum Sensing for the SALSAT Nanosatellite

This repository contains code for the Bachelor thesis *Spectrum Sensing for the SALSAT Nanosatellite*. It aims to provide algorithms and simulations for *blind wideband spectrum sensing* with special emphasis on space applications. More background information on the SALSAT project can be found [here](https://www.raumfahrttechnik.tu-berlin.de/menue/forschung/aktuelle_projekte/salsat/parameter/en/).

This repository is split up into two main directories:
* `specsens/` the python package that contains all relevant code and algorithms.
* `notebooks/` contains usage examples for all parts of the `specsens` package.

Not relevant for *normal* use:
* `graphs/` code that generates graphics for the thesis (LaTex).
* `archive/` old code no longer relevant.

The code is written in `python 3.6` using the packages listed in [`requirements.txt`](./requirements.txt). **Ensure that you have matplotlib `3.1.x` installed. Other versions might give weird artifacts. I am still looking into the problem at this point in time.**

To run the notebooks you will need to install [Jupyter Notebook](https://jupyter.org/install.html).

## Specsens Package
In order to organize and structure my work I created the `specsens` package. It contains most of the algorithms and simulations that I developed. It behaves like a regular python package (`import specsens`) and consists of the following directories:
* `signal/` signal and noise generators used mainly for simulations, as well as some utilities that simplify working with signals.
* `plot/` plotting functions to visualize signals.
* `energy_detect/` energy detector and functions to calculate performance statistics (prob. of false alarm, prob. of detection, etc.).
* `wideband_detect/` wideband detection algorithms (wideband energy detector, spectrum edge detector, etc.).
* `eigenvalue_detect/` eigenvalue detection algorithms based on signal covariance matrices
* `simulation/` simulations to evaluate performance statistics.

## Notebooks
**Coming soon**

## TODO
#### General
- [] Notebook documentation in readme
- [] Upload papers and create basic paper index that helps to *connect* paper and python implementation
- [] Documentation for every function using sphinx(not sure if I will do that tho) (https://docs.python-guide.org/writing/documentation/)


#### Functional
- [] Time space plotting functions. (`time_plot.py`)
- [] Implement sigmoid detection output
- [] Add pfa and pd eigenvalue stats
- [] Create eigenvalue simulation with filter and prewhitening matrix
- [] Combine eigenvalue detection and noise estimation based energy detection, use energy detection as heuristic and binary search pattern to find *noise only* band to use for noise estimation

## Author
Fabian Peddinghaus <peddinghaus@tu-berlin.de>
