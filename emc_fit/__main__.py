import os

os.environ["OMP_NUM_THREADS"] = "64"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "64"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "64"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "64"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "64"  # export NUMEXPR_NUM_THREADS=6

import numpy as np
import pandas as pd
import logging
from emc_sim import utils
from emc_fit import options, modes, resampling


def select_resampling_method(fitOpts: options.FitOptions):

    logging.info(f"DB File: {fitOpts.config.DatabasePath}")
    logging.info(f"Data File: {fitOpts.config.NiiDataPath}")

    if fitOpts.opts.ResamplingOption == "re_db":
        # resample database
        logging.info(f"- Resampling Database")
        niiData, niiImg, pd_db, np_db = resampling.resampleDatabase(fitOpts)
        return niiData, niiImg, pd_db, np_db
    elif fitOpts.opts.ResamplingOption == "re_data":
        # resample database
        logging.info(f"- Resampling Data")
        niiData, niiImg = resampling.resampleData(fitOpts)
        pd_db, np_db = utils.load_database(fitOpts.config.DatabasePath)
        return niiData, niiImg, pd_db, np_db
    else:
        # keep everything
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

    fitOpts.saveFit(t2_map, niiImg, "t2")
    fitOpts.saveFit(b1_map, niiImg, "b1")


if __name__ == '__main__':
    parser, args = options.createCmdLineParser()
    opts = options.FitOptions.fromCmdLine(args)
    # set up logging
    logging.basicConfig(format='%(asctime)s :: %(name)s - %(message)s ',
                        datefmt='%I:%M:%S', level=logging.INFO)
    try:
        main(opts)
    except AttributeError as ae:
        logging.error(ae)
        parser.print_usage()
