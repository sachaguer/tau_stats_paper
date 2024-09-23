"""
Script computation_time.py

Allows to get the computation time  of the different methods on SP_v1.3_LFmask_8k
"""
import numpy as np
import time
import yaml

from utils import get_theory_cov, get_jackknife_cov

if __name__ == "__main__":

    # Load config file
    path_config = '/home/guerrini/sp_validation/notebooks/cosmo_val/cat_config.yaml'

    # Base directory for data, on candide
    data_base_dir = '/n17data/mkilbing/astro/data/'


    #Version
    version = "SP_v1.3_LFmask_8k"

    all_keys = ['nz']
    all_keys.append(version)

    with open(path_config, 'r') as file:
        cat = yaml.load(file.read(), Loader=yaml.FullLoader)

    # Set full paths
    for ver in all_keys:
        for key in cat[ver]:
            if "path" in cat[ver][key]:
                cat[ver][key]["path"] = f"{data_base_dir}/{cat[ver]['subdir']}/{cat[ver][key]['path']}"

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

    start = time.time()

    # Get theory covariance
    get_theory_cov(cat, version, TreeCorrConfig_xi, outdir='.', nbin_ang=100, nbin_rad=200)

    end = time.time()
    print(f"Time to get theory covariance: {(end - start)/60:.3f} min")

    start = time.time()

    # Get jackknife covariance
    get_jackknife_cov(cat, version, TreeCorrConfig_xi, outdir='.', ncov=100)

    end = time.time()
    print(f"Time to get jackknife covariance: {(end - start)/60:.3f} min")

