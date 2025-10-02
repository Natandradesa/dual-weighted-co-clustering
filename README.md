# A Dual-Weighted Gaussian Kernel-Based Method for Fuzzy Co-Clustering

This repository contains the official implementation of the article [A Dual-Weighted Gaussian Kernel-Based Method for Fuzzy Co-Clustering](https://ieeexplore.ieee.org/abstract/document/11152270) by José Nataniel A. de Sá, Marcelo R. P. Ferreira, and Francisco de A. T. de Carvalho.

## Organization

The code is organized as follows:

* The folder **algoritms** contains the implementation of the algoritms used in the experiments. The implementation od the proposed method is in the file double_subspace_coclustering.py.

* The folder **configurations** contains the hyperparameter setting of the algorithms.

* The folder **datasets** containd the real datasets used in the experiments.

* The folder **results_real** contains the results of the experiments with real datasets.

* The folder **results_synthetic** contains the results of the Monte Carlo simulation.

* The file aplications_real.ipynb cotains the code for the application on real datasets.

* The file data_generation.py contains the code to denerated the synthetic datasets used in the Monte Carlo simulation.

* The file metrics.py contains the implementation of some evaluation metrics. 

* The file practical_example.ipynb contains a demo of the proposed algorithm DWGKFDK.

* The file results_real_analysis.ipynb contains the analisysof the results obtained in the application to real datasets.

* The file simulation.ipynb contains the code of the Monte Carlo simulation.

## Requirements

The codes in this repository requires numpy, scipy, scikit-learn, matplotlib, pandas, seaborn and jupyter. The Environment can be installed using the command: pip install -r requirements.txt

## Citation 

If you use this code in your research, please cite the following paper:

J. N. A. de Sá, M. R. P. Ferreira and F. de A. T. de Carvalho, "A Dual-Weighted Gaussian Kernel-Based Method for Fuzzy Co-Clustering," 2025 IEEE International Conference on Fuzzy Systems (FUZZ), Reims, France, 2025, pp. 1-6, doi: 10.1109/FUZZ62266.2025.11152270

