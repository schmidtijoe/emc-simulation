import os

# set maximum number of threads for multiprocessing -> numpy is sometimes greedy,
# if code is vectorized neatly it might take more resources than wanted by operator
os.environ["OMP_NUM_THREADS"] = "64"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "64"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "64"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "64"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "64"  # export NUMEXPR_NUM_THREADS=6

import numpy as np
import pandas as pd
import logging
from emc_sim import utils
from emc_fit import options, modes, denoize, plots
from pathlib import Path
import nibabel as nib


def select_fit_function(fitOpts: options.FitOptions,
                        niiData: np.ndarray,
                        pandas_database: pd.DataFrame,
                        numpy_database: np.ndarray
                        ) -> (np.ndarray, np.ndarray):
    args = (niiData, pandas_database, numpy_database)
    fit_mode_opts = {
        "threshold": modes.Fit(*args).get_maps(),
        "pearson": modes.PearsonFit(*args).get_maps(),
        "mle": modes.MleFit(*args).get_maps(),
        "l2": modes.L2Fit(*args).get_maps()
    }
    # make sure data is 2D
    if niiData.shape.__len__() > 2:
        logging.info(f"nii input data; shape {niiData.shape} ")
        niiData = np.reshape(niiData, [-1, niiData.shape[-1]])
        logging.info(f"reshaping data; shape {niiData.shape} ")

    assert fit_mode_opts.get(fitOpts.opts.FitMetric)
    return fit_mode_opts.get(fitOpts.opts.FitMetric)

    # if fitOpts.opts.FitMetric == "threshold":
    #     return np.empty(0), np.empty(0)
    # elif fitOpts.opts.FitMetric == "pearson":
    #     pearson_fit = modes.PearsonFit(
    #         nifti_data=niiData, pandas_database=pandas_database, numpy_database=numpy_database
    #     )
    #     pearson_fit.fit()
    #     return pearson_fit.get_maps()
    # elif fitOpts.opts.FitMetric == "mle":
    #     mle_fit = modes.MleFit(
    #         nifti_data=niiData, pandas_database=pandas_database, numpy_database=numpy_database
    #     )
    #     mle_fit.fit()
    #     return mle_fit.get_maps()
    # elif fitOpts.opts.FitMetric == "l2":
    #     l2_fit = modes.L2Fit(nifti_data=niiData, pandas_database=pandas_database, numpy_database=numpy_database)
    #     l2_fit.fit_mp()
    #     return l2_fit.get_maps()
    # else:
    #     logging.error("Unrecognized Fitting Option!")
    #     raise AttributeError("Unrecognized Fitting Option!")


def main(fitOpts: options.FitOptions):
    logging.info("Load in data")
    dataPath = Path(fitOpts.config.NiiDataPath).absolute()
    outPath = Path(fitOpts.config.OutputPath).absolute()

    logging.info("Loading data")
    niiData, niiImg = utils.niiDataLoader(
        dataPath,
        test_set=fitOpts.opts.TestingFlag,
        normalize=""
    )
    niiData = np.moveaxis(niiData, 1, 2)

    if fitOpts.opts.Visualize:
        # plot ortho
        plots.plot_ortho_view(niiData)

    # denoizing time series
    logging.info("Denoizing time series")
    d_niiData = denoize.denoize_nii_data(data=niiData, num_iterations=4, visualize=fitOpts.opts.Visualize)

    logging.info("Writing denoized to .nii")
    d_outPath = outPath.joinpath(f"d_{dataPath.name}.nii")
    logging.info(f"Writing File: {d_outPath.__str__()}")
    d_saveData = np.moveaxis(d_niiData, 2, 1)
    d_nii = nib.Nifti1Image(d_saveData, niiImg.affine)
    nib.save(d_nii, d_outPath)

    # load database
    db_pd, db_np = utils.load_database(fitOpts.config.DatabasePath, append_zero=True, normalization="l2")

    # Fit
    logging.info(f"Fitting: {fitOpts.opts.FitMetric}")
    t2_map, b1_map = select_fit_function(fitOpts=fitOpts, niiData=d_niiData, pandas_database=db_pd, numpy_database=db_np)

    if fitOpts.opts.TestingFlag:
        return 0

    fitOpts.saveFit(t2_map, niiImg, "t2")
    fitOpts.saveFit(b1_map, niiImg, "b1")


if __name__ == '__main__':
    parser, args = options.createCmdLineParser()
    opts = options.FitOptions.fromCmdLine(args)
    # set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    try:
        main(opts)
    except AttributeError as ae:
        logging.error(ae)
        parser.print_usage()