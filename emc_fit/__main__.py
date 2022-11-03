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
from emc_fit import options, fit, denoize, plots, b1input
from pathlib import Path
import nibabel as nib


def select_fit_function(fitOpts: options.FitOptions,
                        niiData: np.ndarray,
                        pandas_database: pd.DataFrame,
                        b1_weight: b1input.B1Weight
                        ) -> (np.ndarray, np.ndarray):
    fit_mode_opts = {
        "threshold": fit.Fit,
        "pearson": fit.PearsonFit,
        "mle": fit.MleFit,
        "l2": fit.L2Fit
    }

    if fit_mode_opts.get(fitOpts.opts.FitMetric) is None:
        err = f"fit Mode not recognized, choose one of {fit_mode_opts}"
        logging.error(err)
        raise AttributeError(err)
    return fit_mode_opts.get(fitOpts.opts.FitMetric)(niiData, pandas_database, b1_weight).get_maps()


def mode_denoize(
        fitOpts: options.FitOptions,
        niiData: np.ndarray,
        niiImg: nib.nifti1.Nifti1Image):
    # denoizing time series
    logging.info("Denoizing time series")
    if fitOpts.opts.DenoizeSave:
        save_plot_path = Path(fitOpts.config.OutputPath).absolute().joinpath(
            f"{fitOpts.config.NameId}_denoising_efficiency.png"
        ).__str__()
    else:
        save_plot_path = ""

    d_niiData = denoize.denoize_nii_data(
        data=niiData,
        num_iterations=fitOpts.opts.DenoizeNumIterations,
        visualize=fitOpts.opts.Visualize,
        mpHeadroom=fitOpts.opts.HeadroomMultiprocessing,
        save_plot=save_plot_path
    )

    if fitOpts.opts.DenoizeSave:
        logging.info("Writing denoized to .nii")
        if not fitOpts.config.NameId:
            name = Path(fitOpts.config.NiiDataPath).absolute()
            for _ in name.suffixes:
                name = name.with_suffix("")
            name = name.stem
        else:
            name = fitOpts.config.NameId
        outPath = Path(fitOpts.config.OutputPath).absolute().joinpath(f"d_{name}.nii")
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
    db_pd, _ = utils.load_database(fitOpts.config.DatabasePath, append_zero=True, normalization="l2")
    # check b1 weighting
    b1_weight = b1input.set_b1_weighting(
        data_slice_shape=niiData.shape[:2],
        database_pandas=db_pd,
        opts=fitOpts.opts,
        b1_weight_factor=fitOpts.opts.FitB1WeightingLambda,
    )
    # Fit
    logging.info(f"Fitting: {fitOpts.opts.FitMetric}")
    t2_map, b1_map = select_fit_function(fitOpts=fitOpts, niiData=niiData, pandas_database=db_pd,
                                         b1_weight=b1_weight)

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
        "df": mode_both
    }
    if modeOptions.get(fitOpts.opts.Mode) is None:
        err = f"fitting mode not provided choose one of {modeOptions}"
        logging.error(err)
        raise AttributeError(err)
    modeOptions.get(fitOpts.opts.Mode)(fitOpts, niiData, niiImg)


if __name__ == '__main__':
    parser, args = options.createCmdLineParser()
    opts = options.FitOptions.fromCmdLine(args)
    # set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)
    try:
        main(opts)
    except AttributeError or AssertionError or ValueError as err:
        logging.error(err)
        parser.print_usage()
