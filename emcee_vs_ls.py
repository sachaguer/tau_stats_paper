"""
Script computation_time.py

Allows to get the computation time  of the different methods on SP_v1.3_LFmask_8k
"""
import numpy as np
import time
import yaml

from shear_psf_leakage.rho_tau_stat import RhoStat, TauStat, PSFErrorFit

if __name__ == "__main__":

    data_directory = './data/'

    n_samples = 10_000

    #TreeCorr config
    theta_min = 0.1
    theta_max = 250
    nbins = 20
    sep_units = 'arcmin'
    coord_units = 'degrees'

    # ## Set up
    TreeCorrConfig_xi = {
        'ra_units': coord_units,
        'dec_units': coord_units,
        'min_sep': theta_min,
        'max_sep': theta_max,
        'sep_units': sep_units,
        'nbins': nbins,
    }

    #Create objects to sample
    rho_stat_handler = RhoStat(output='.', treecorr_config=TreeCorrConfig_xi, verbose=True)
    tau_stat_handler = TauStat(catalogs=rho_stat_handler.catalogs, output='.', treecorr_config=TreeCorrConfig_xi, verbose=True)

    psf_fitter = PSFErrorFit(rho_stat_handler, tau_stat_handler, data_directory)

    path_rho = 'rho_stats_SP_v1.3_LFmask_8k.fits'
    path_tau = 'tau_stats_SP_v1.3_LFmask_8k.fits'
    path_cov_rho = 'cov_rho_SP_v1.3_LFmask_8k.npy'
    path_cov_tau = 'cov_tau_SP_v1.3_LFmask_8k.npy'

    #Load data and covariance
    psf_fitter.load_rho_stat(path_rho)
    psf_fitter.load_tau_stat(path_tau)

    psf_fitter.load_covariance(path_cov_rho, cov_type='rho')
    psf_fitter.load_covariance(path_cov_tau, cov_type='tau')

    #Get samples from least-squares
    start_time = time.time()

    _ = psf_fitter.get_least_squares_params_samples(npatch=150, apply_debias=True)

    end_time = time.time()

    print(f"Time to get least-squares samples: {(end_time - start_time)/60:.3f} min")

    #Get samples from emcee
    nwalkers = 124

    start_time = time.time()

    _ = psf_fitter.run_chain(
        nwalkers=nwalkers,
        nsamples=n_samples,
        npatch=150,
        apply_debias=True,
        savefig=None
    )

    end_time = time.time()

    print(f"Time to get emcee samples: {(end_time - start_time)/60:.3f} min")





    
