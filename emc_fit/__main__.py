import os

# set maximum number of threads for multiprocessing -> numpy is sometimes greedy,
# if code is vectorized neatly it might take more resources than wanted by operator
import typing

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
    fit_mode_opts = {
        "threshold": modes.Fit,
        "pearson": modes.PearsonFit,
        "mle": modes.MleFit,
        "l2": modes.L2Fit
    }
    # make sure data is 2D
    if niiData.shape.__len__() > 2:
        logging.info(f"nii input data; shape {niiData.shape} ")
        niiData = np.reshape(niiData, [-1, niiData.shape[-1]])
        logging.info(f"reshaping data; shape {niiData.shape} ")

    assert fit_mode_opts.get(fitOpts.opts.FitMetric)
    return fit_mode_opts.get(fitOpts.opts.FitMetric)(niiData, pandas_database, numpy_database).get_maps()

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


def mode_denoize(
        fitOpts: options.FitOptions,
        niiData: np.ndarray,
        niiImg: nib.nifti1.Nifti1Image):
    # denoizing time series
    logging.info("Denoizing time series")
    d_niiData = denoize.denoize_nii_data(data=niiData, num_iterations=fitOpts.opts.DenoizeNumIterations, visualize=fitOpts.opts.Visualize)

    logging.info("Writing denoized to .nii")
    name = Path(fitOpts.config.NiiDataPath).absolute().stem
    outPath = Path(fitOpts.config.OutputPath).absolute().joinpath(f"d_{name}")
    logging.info(f"Writing File: {outPath}")
    d_nii = nib.Nifti1Image(d_niiData, niiImg.affine)
    nib.save(d_nii, outPath)
    return d_niiData


def mode_fit(
        fitOpts: options.FitOptions,
        niiData: np.ndarray,
        niiImg: nib.nifti1.Nifti1Image):
    logging.info(f"Loading Database - {fitOpts.config.DatabasePath}")
    # load database
    db_pd, db_np = utils.load_database(fitOpts.config.DatabasePath, append_zero=True, normalization="l2")

    # Fit
    logging.info(f"Fitting: {fitOpts.opts.FitMetric}")
    t2_map, b1_map = select_fit_function(fitOpts=fitOpts, niiData=niiData, pandas_database=db_pd,
                                         numpy_database=db_np)

    if fitOpts.opts.TestingFlag:
        return 0

    # cast to ms
    t2_map *= 1e3
    fitOpts.saveFit(t2_map, niiImg, f"{fitOpts.config.NameId}_t2")
    fitOpts.saveFit(b1_map, niiImg, f"{fitOpts.config.NameId}_b1")


def mode_both(
        fitOpts: options.FitOptions,
        niiData: np.ndarray,
        niiImg: nib.nifti1.Nifti1Image):
    d_niiData = mode_denoize(fitOpts, niiData, niiImg)
    mode_fit(fitOpts, d_niiData, niiImg)


def main(fitOpts: options.FitOptions):
    dataPath = Path(fitOpts.config.NiiDataPath).absolute()

    logging.info(f"Loading data - {dataPath}")
    niiData, niiImg = utils.niiDataLoader(
        dataPath,
        test_set=fitOpts.opts.TestingFlag,
        normalize=""
    )
    # set values to range 0 - 1000
    niiData = np.divide(1e3 * niiData, niiData.max())

    if fitOpts.opts.Visualize:
        # plot ortho
        plots.plot_ortho_view(niiData)

    modeOptions = {
        "Denoize": mode_denoize,
        "denoize": mode_denoize,
        "d": mode_denoize,
        "Fit": mode_fit,
        "fit": mode_fit,
        "f": mode_fit,
        "Both": mode_both,
        "both": mode_both,
        "b": mode_both
    }
    assert modeOptions.get(fitOpts.opts.Mode)
    modeOptions.get(fitOpts.opts.Mode)(fitOpts, niiData, niiImg)


if __name__ == '__main__':
    parser, args = options.createCmdLineParser()
    opts = options.FitOptions.fromCmdLine(args)
    # set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    try:
        main(opts)
    except AttributeError or AssertionError as ae:
        logging.error(ae)
        parser.print_usage()
