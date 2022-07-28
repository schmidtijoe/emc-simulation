"""
Module to fit database with or without sampled noise mean via l2 norm to nii data
"""
import logging
import os
import pickle

import nibabel as nib
import numpy as np
from emc_sim import utils
from emc_sim.noise import handlers
from emc_sim.noise import distributions
from emc_sim.fit import options
import datetime as dt
import multiprocessing as mp
from operator import itemgetter
from itertools import chain
from pathlib import Path
import tqdm


class DataResampler:
    def __init__(self, fitOpts: options.FitOptions):
        logging.info(f"_____data resampler_____")
        logging.info(f"____________________________")
        self.fitOpts = fitOpts

        logging.info("Loading data")
        # data vars
        self.niiData, self.niiImg = utils.niiDataLoader(
            self.fitOpts.config.NiiDataPath,
            test_set=fitOpts.opts.TestingFlag,
            normalize=""
        )
        # dont normalize, noise extraction is dependent on it
        self.ncChi, self.snrMap = handlers.extract_chi_noise_characteristics_from_nii(
            self.niiData,
            corner_fraction=fitOpts.opts.NoiseBackgroundEstimateCornerFraction,
            visualize=fitOpts.opts.NoiseBackgroundEstimateVisualize
        )
        logging.info("Calculating interpolation array")
        # build array between 0 and max value
        self.interpol_array = np.arange(np.max(self.niiData))
        self.interpol_mean = self.ncChi.mean(self.interpol_array)
        # provide 1d
        self.niiData = self.niiData.flatten()
        self.reNiiData = None
        # misc
        self._time_formatter = "%H:%M:%S"
        # multiprocessing, leave headroom but take at least 4
        self.numCpus = np.max([4, mp.cpu_count() - fitOpts.opts.ProcessingHeadroomMultiprocessing])
        self.multiprocessing = fitOpts.opts.Multiprocessing
        logging.info("Finished Init")

    def _wrap_resample(self, args):
        index_list, data_list, interpol_mean = args
        data = np.array(data_list)
        sampled_arr = np.abs(data[np.newaxis] - interpol_mean[:, np.newaxis])
        # we look for the smallest of those differences
        fit_idx = np.argmin(sampled_arr, axis=0)
        # we take the interpolation value corresponding to the index
        interpolated_data = self.interpol_array[fit_idx]
        return interpolated_data, index_list

    def resample(self):
        logging.info("___Start Processing___")
        start = dt.datetime.now()
        logging.info(f"start time: {start.strftime(self._time_formatter)}")

        if self.multiprocessing:
            logging.info("multiprocessing mode")
            logging.info(f"using {self.numCpus} cpus")
            # divide data in chunks
            mp_data_idx = np.array_split(np.arange(len(self.niiData)), self.numCpus)
            mp_data = np.array_split(self.niiData, self.numCpus)
            mp_args = [(mp_data_idx[mp_idx], mp_data[mp_idx], self.interpol_mean) for mp_idx in range(self.numCpus)]
            if self.fitOpts.opts.TestingFlag:
                result = []
                for idx in range(self.numCpus):
                    result.append(self._wrap_resample(mp_args[idx]))
            else:
                with mp.Pool(self.numCpus) as p:
                    result = p.map(self._wrap_resample, mp_args)
            interpolated_data = np.zeros_like(self.niiData)
            for r in result:
                data, indexes = r
                interpolated_data[indexes] = data
        else:
            logging.info("single cpu mode")
            # watch out here, to vectorize we build a massive array, might not work with small compute systems
            # for each point we take absolute value of difference between interpolation points noise mean and data value
            sampled_arr = np.abs(self.niiData[np.newaxis] - self.interpol_mean[:, np.newaxis])
            # we look for the smallest of those differences
            fit_idx = np.argmin(sampled_arr, axis=0)
            # we take the interpolation value corresponding to the index and reshape
            interpolated_data = self.interpol_array[fit_idx]
        # if in testmode we cant reshape here exit before
        if self.fitOpts.opts.TestingFlag:
            return
        # reshaping
        self.reNiiData = np.reshape(interpolated_data, self.niiImg.shape)
        end = dt.datetime.now()
        logging.info(f"Finished: {end.strftime(self._time_formatter)}")
        t_total = (end - start).total_seconds()
        logging.info(f"Compute time: {t_total / 60:.2f} min ({t_total / 3600:.1f} h)")

    def save_resampled(self):
        path = Path(self.fitOpts.config.ResampledDataOutputPath).absolute().joinpath("resampled_input.nii")
        utils.create_folder_ifn_exist(path.parent)
        img = nib.Nifti1Image(self.reNiiData, self.niiImg.affine)
        nib.save(img, path)

    def get_data(self):
        return self.reNiiData, self.niiImg


class DatabaseResampler:
    def __init__(self, fitOpts: options.FitOptions):
        logging.info(f"_____database resampler_____")
        logging.info(f"____________________________")
        self.fitOpts = fitOpts

        logging.info(f"Load data")
        # database vars
        self.pd_db, self.np_db = utils.load_database(self.fitOpts.config.DatabasePath, append_zero=False)
        # data vars
        self.niiData, self.niiImg = utils.niiDataLoader(self.fitOpts.config.NiiDataPath, test_set=fitOpts.opts.TestingFlag, normalize="")
        # dont normalize, noise extraction is dependent on it
        self.niiDataShape = self.niiData.shape
        self.ncChi, self.snrMap = handlers.extract_chi_noise_characteristics_from_nii(
            self.niiData,
            corner_fraction=fitOpts.opts.NoiseBackgroundEstimateCornerFraction,
            visualize=fitOpts.opts.NoiseBackgroundEstimateVisualize
        )
        # get data to 2d
        self.snrMap = np.reshape(self.snrMap, -1)
        self.numCurves = self.snrMap.shape[0]
        self.re_dbs = None

        # misc
        self._time_formatter = "%H:%M:%S"
        mp_headroom = fitOpts.opts.ProcessingHeadroomMultiprocessing
        # leave some headroom but pick at least 4 thread workers
        self.numCpus = np.max([mp.cpu_count() - mp_headroom, 4])

    def resample(self):
        logging.info("___Start Processing___")
        start = dt.datetime.now()
        logging.info(f"start time: {start.strftime(self._time_formatter)}")
        logging.info(f"using {self.numCpus} cpus")
        # split into datablocks
        snrIndexBlocks = np.array_split(np.arange(self.numCurves), self.fitOpts.opts.ProcessingNumBlocks)
        # trange? loop through bocks -> aim is to have smaller chunks to send of when multiprocessing

        # create folder to temporarily save the processed data
        utils.create_folder_ifn_exist('temp/')
        path = Path('temp').absolute()

        for block_idx in tqdm.trange(self.fitOpts.opts.ProcessingNumBlocks):
            # logging.info(f"Processing Block {block_idx}/{self.fitOpts.opts.ResamplingNumBlocks}")
            # have a list of indices to process
            indexes_to_process = snrIndexBlocks[block_idx]
            if not self.fitOpts.opts.TestingFlag:
                with mp.Pool(self.numCpus) as p:
                    resampledDbList = p.map(self._mp_wrap_sampling, indexes_to_process)
            else:
                resampledDbList = [self._mp_wrap_sampling(indexes_to_process[k]) for k in range(len(indexes_to_process))]

            # temp save list
            with open(path.joinpath(f"re_db_block_{block_idx}.pkl"), "wb") as p_file:
                pickle.dump(resampledDbList, p_file)

        resampledDbList = []
        for read_idx in tqdm.trange(self.fitOpts.opts.ProcessingNumBlocks):
            p_path = path.joinpath(f"re_db_block_{read_idx}.pkl")
            with open(p_path, "rb") as p_file:
                resampledDbList.append(pickle.load(p_file))
            os.remove(p_path)
        results = list(chain(*resampledDbList))
        # cleanup
        del resampledDbList
        os.remove(path)

        results.sort(key=itemgetter(1))
        # keep result in array and del rest
        self.re_dbs = np.array(list(chain(*results))[::2])
        del results
        self.re_dbs = utils.normalize_array(self.re_dbs)
        self.re_dbs = np.reshape(self.re_dbs, [*self.niiDataShape[:-1], *self.np_db.shape])
        end = dt.datetime.now()
        logging.info(f"Finished: {end.strftime(self._time_formatter)}")
        t_total = (end - start).total_seconds()
        logging.info(f"Compute time: {t_total / 60:.2f} min ({t_total / 3600:.1f} h)")

    def _mp_wrap_sampling(self, idx):
        snr = self.snrMap[idx]
        self.ncChi.sigma_from_snr(snr)
        re_db = self.ncChi.mean(self.np_db)
        return re_db, idx

    def save_resampled(self):
        path = Path(self.fitOpts.config.ResampledDataOutputPath).absolute().joinpath("resampled_database.pkl")
        utils.create_folder_ifn_exist(path.parent)
        with open(path, "wb") as fp:
            pickle.dump(self.re_dbs, fp)

    def get_data(self):
        return self.niiData, self.niiImg, self.pd_db, self.re_dbs


def resampleDatabase(fitOpts: options.FitOptions) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    dRe = DatabaseResampler(fitOpts=fitOpts)
    dRe.resample()
    if fitOpts.config.ResampledDataOutputPath:
        dRe.save_resampled()
    niiData, niiImg, pd_db, np_db = dRe.get_data()
    return niiData, niiImg, pd_db, np_db


def resampleData(fitOpts: options.FitOptions) -> (np.ndarray, np.ndarray):
    dRe = DataResampler(fitOpts=fitOpts)
    dRe.resample()
    if fitOpts.config.ResampledDataOutputPath:
        dRe.save_resampled()
    niiData, niiImg = dRe.get_data()
    return niiData, niiImg

