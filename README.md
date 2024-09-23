# $\tau$-statistics paper

## Scope of this repository

This repository is aimed at reproducing the figures in Guerrini et al (in prep.).

## Installation of the dependencies

Some of the scripts require to install the package `shear_psf_leakage` (https://github.com/martinkilbinger/shear_psf_leakage).

It can be installed with `pip` using the following command:

```bash
pip install shear_psf_leakage
```

## Semi-analytical covariance matrix for the $\tau$-statistics

The notebook `cov_comparison.ipynb` allows to reproduce the figures in Section 4. The required data in available in the folder `data` including the $\rho$- and $\tau$-statistics of the catalog. The notebook also performs the inference of the parameters $\alpha$, $\beta$ and $\eta$ in the error model introduced in Eq.(8).

## Redefinition of the $\tau$-statistics

The notebook `shear_scalar_correlator.ipynb` performs the $\rho$- and $\tau$-statistics analysis with the redefined $\tau$-statistic with the size residuals considered as a scalar field. It reproduced the figures in Section 5.

## Gaussianity test

In Appendix C, a test of gaussianity is performed to check if the Gaussian assumption for the likelihood is correct. This analysis is performed in the notebook `tau_non_gaussian.ipynb`.

## Computation time

The scripts `computation_time.py` and `emcee_vs_ls.py` were used to evaluate the computation time of the jackknife resampling approach against the semi-analytical covariance and the MCMC sampling against the least-square approach to sample the parameters of the error model.

## Authors

Sacha Guerrini <sacha.guerrini@cea.fr>

Martin Kilbinger <martin.kilbinger@cea.fr>

Hubert Leterme

Axel Guinot

Fabian Hervas Peters

Jingwei Wang
