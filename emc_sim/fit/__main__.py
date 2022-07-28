import os

os.environ["OMP_NUM_THREADS"] = "16"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "16"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "16"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "16"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "16"  # export NUMEXPR_NUM_THREADS=6

import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib

import logging
from emc_sim import utils
from emc_sim.fit import options, modes, resampling


def select_resampling_method(fitOpts: options.FitOptions):
    if fitOpts.opts.ResamplingOption == "re_db":
        # resample database
        logging.info(f"- Resampling Database")
        logging.info(f"DB File: {fitOpts.config.DatabasePath}")
        logging.info(f"Data File: {fitOpts.config.NiiDataPath}")
        niiData, niiImg, pd_db, np_db = resampling.resampleDatabase(fitOpts)
        return niiData, niiImg, pd_db, np_db
    elif fitOpts.opts.ResamplingOption == "re_data":
        # resample database
        logging.info(f"- Resampling Data")
        logging.info(f"DB File: {fitOpts.config.DatabasePath}")
        logging.info(f"Data File: {fitOpts.config.NiiDataPath}")
        niiData, niiImg = resampling.resampleData(fitOpts)
        pd_db, np_db = utils.load_database(fitOpts.config.DatabasePath)
        return niiData, niiImg, pd_db, np_db
    else:
        # keep everything
        logging.info(f"DB File: {fitOpts.config.DatabasePath}")
        logging.info(f"Data File: {fitOpts.config.NiiDataPath}")
        pd_db, np_db = utils.load_database(fitOpts.config.DatabasePath)
        niiData, niiImg = utils.niiDataLoader(fitOpts.config.NiiDataPath)
        return niiData, niiImg, pd_db, np_db


def select_fit_function(fitOpts: options.FitOptions,
                        niiData: np.ndarray,
                        pandas_database: pd.DataFrame,
                        numpy_database: np.ndarray
                        ) -> (np.ndarray, np.ndarray):
    # make sure data is 2D
    if niiData.shape.__len__() > 2:
        logging.info(f"nii input data; shape {niiData.shape} ")
        niiData = np.reshape(niiData, [-1, niiData.shape[-1]])
        logging.info(f"reshaping data; shape {niiData.shape} ")
    if fitOpts.opts.FitMetric == "threshold":
        return np.empty(0), np.empty(0)
    if fitOpts.opts.FitMetric == "pearson":
        pearson_fit = modes.PearsonFit(
            nifti_data=niiData, pandas_database=pandas_database, numpy_database=numpy_database
        )
        pearson_fit.fit()
        return pearson_fit.get_maps()
    if fitOpts.opts.FitMetric == "mle":
        mle_fit = modes.MleFit(
            nifti_data=niiData, pandas_database=pandas_database, numpy_database=numpy_database
        )
        mle_fit.fit()
        return mle_fit.get_maps()
    if fitOpts.opts.FitMetric == "l2":
        l2_fit = modes.L2Fit(nifti_data=niiData, pandas_database=pandas_database, numpy_database=numpy_database)
        l2_fit.fit()
        return l2_fit.get_maps()
    else:
        logging.error("Unrecognized Fitting Option!")
        raise AttributeError("Unrecognized Fitting Option!")


def main(fitOpts: options.FitOptions):
    logging.info(f"Set Resampling scheme: {fitOpts.opts.ResamplingOption}")
    # choose resampling method: resample data or resample database
    niiData, niiImg, pd_db, np_db = select_resampling_method(fitOpts)

    # select fitting option
    t2_map, b1_map = select_fit_function(
        fitOpts=fitOpts, niiData=niiData,
        pandas_database=pd_db, numpy_database=np_db
    )

    if fitOpts.opts.TestingFlag:
        return 0
    t2_map = np.reshape(t2_map, niiImg.shape[:-1])
    b1_map = np.reshape(b1_map, niiImg.shape[:-1])

    # save
    path = Path(fitOpts.config.FitDataOutputPath).absolute()
    niiImg = nib.Nifti1Image(t2_map, niiImg.affine)
    nib.save(niiImg, path.joinpath(f"{fitOpts.opts.FitMetric}_t2_map.nii"))

    niiImg = nib.Nifti1Image(b1_map, niiImg.affine)
    nib.save(niiImg, path.joinpath(f"{fitOpts.opts.FitMetric}_b1_map.nii"))


if __name__ == '__main__':
    parser, args = options.createCmdLineParser()
    opts = options.FitOptions.fromCmdLine(args)
    # set up logging
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%I:%M:%S', level=logging.INFO)
    try:
        main(opts)
    except AttributeError as ae:
        logging.error(ae)
        parser.print_usage()
