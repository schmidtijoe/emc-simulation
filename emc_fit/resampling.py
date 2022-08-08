"""
Module to emc_fit database with or without sampled noise mean via l2 norm to nii data
"""
import logging
import os
import pickle

import nibabel as nib
import numpy as np
import pandas as pd

from emc_sim import utils
from emc_fit.noise import handlers
from emc_fit import options
import datetime as dt
import multiprocessing as mp
from operator import itemgetter
from itertools import chain
from pathlib import Path
import tqdm

logModule = logging.getLogger(__name__)


class DataResampler:
    """
    Class for Streamlining Noise Mean resampling of nii Input data
    """
    def __init__(self, fitOpts: options.FitOptions):
        logModule.info(f"_____data resampler_____")
        logModule.info(f"____________________________")
        self.fitOpts = fitOpts

        logModule.info("Loading data")
        # data vars
        self.niiData, self.niiImg = utils.niiDataLoader(
            self.fitOpts.config.NiiDataPath,
            test_set=fitOpts.opts.TestingFlag,
            normalize=""
        )
        logModule.info("Extract Noise Characteristics")
        # dont normalize, noise extraction is dependent on it
        self.ncChi, self.snrMap = handlers.extract_chi_noise_characteristics_from_nii(
            self.niiData,
            corner_fraction=fitOpts.opts.NoiseBackgroundEstimateCornerFraction,
            visualize=fitOpts.opts.NoiseBackgroundEstimateVisualize
        )
        # misc
        self.eps = 1e-5
        self._time_formatter = "%H:%M:%S"
        self.num_iter = 20

        logModule.info("Calculating interpolation array")
        # build array between 0 and max value
        self.interpol_array = np.arange(np.max(self.niiData))
        self.interpol_mean = self.ncChi.mean(self.interpol_array)
        # provide 1d
        self.niiData = self.niiData.flatten()
        self.reNiiData = np.zeros_like(self.niiData)
        # first init
        self.iterate_approximation(eps=False)
        # multiprocessing, leave headroom but take at least 4
        self.numCpus = np.max([4, mp.cpu_count() - fitOpts.opts.ProcessingHeadroomMultiprocessing])
        self.multiprocessing = fitOpts.opts.Multiprocessing
        logModule.info("Finished Init")

    def _wrap_resample(self, args):
        """
        wrap resampling function for multithreading
        """
        index_list, data_list, interpol_mean = args
        data = np.array(data_list)
        sampled_arr = np.abs(data[np.newaxis] - interpol_mean[:, np.newaxis])
        # we look for the smallest of those differences
        fit_idx = np.argmin(sampled_arr, axis=0)
        # we take the interpolation value corresponding to the index
        interpolated_data = self.interpol_array[fit_idx]
        return interpolated_data, index_list

    def iterate_approximation(self, eps: bool = True):
        if eps:
            selection = self.reNiiData > self.eps
        else:
            selection = self.niiData >= 0.0
        # calculate offset between mean noised amplitude and data
        diff = np.subract(self.niiData[selection], self.ncChi.mean(self.reNiiData[selection]))
        max_diff = np.max(diff)
        result = np.add(self.reNiiData[selection], diff)
        result = np.clip(result, 0.0, np.max(result))
        self.reNiiData[selection] = result
        return np.count_nonzero(selection), max_diff

    def resample(self):
        logModule.info("___Start Processing___")
        start = dt.datetime.now()
        logModule.info(f"start time: {start.strftime(self._time_formatter)}")
        bar = tqdm.trange(self.num_iter)
        for _ in bar:
            num_curves, max_diff = self.iterate_approximation()
            bar.set_postfix_str(f"maximum offset of approx mean for {num_curves} curves: {max_diff:.4f}")

        # reshaping
        self.reNiiData = np.reshape(self.reNiiData, self.niiImg.shape)
        end = dt.datetime.now()
        logModule.info(f"Finished: {end.strftime(self._time_formatter)}")
        t_total = (end - start).total_seconds()
        logModule.info(f"Compute time: {t_total / 60:.2f} min ({t_total / 3600:.1f} h)")

    def resample_old(self):
        """
        Resample data
        """
        logModule.info("___Start Processing___")
        start = dt.datetime.now()
        logModule.info(f"start time: {start.strftime(self._time_formatter)}")

        if self.multiprocessing:
            logModule.info("multiprocessing mode")
            logModule.info(f"using {self.numCpus} cpus")
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
            logModule.info("single cpu mode")
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
        logModule.info(f"Finished: {end.strftime(self._time_formatter)}")
        t_total = (end - start).total_seconds()
        logModule.info(f"Compute time: {t_total / 60:.2f} min ({t_total / 3600:.1f} h)")

    def save_resampled(self):
        """
        Saving resampled data as .nii
        """
        path = Path(self.fitOpts.config.ResampledDataOutputPath).absolute()
        utils.create_folder_ifn_exist(path)
        img = nib.Nifti1Image(self.reNiiData, self.niiImg.affine)
        nib.save(img, path.joinpath("resampled_input.nii"))

    def get_data(self) -> (np.ndarray, nib.Nifti1Image):
        """
        get resampled data
        """
        return self.reNiiData, self.niiImg


class DatabaseResampler:
    def __init__(self, fitOpts: options.FitOptions):
        logModule.info(f"_____database resampler_____")
        logModule.info(f"____________________________")
        self.fitOpts = fitOpts

        logModule.info(f"Load data")
        # database vars
        self.pd_db, self.np_db = utils.load_database(
            self.fitOpts.config.DatabasePath, append_zero=False, normalization="max"
        )
        # data vars
        self.niiData, self.niiImg = utils.niiDataLoader(
            self.fitOpts.config.NiiDataPath, test_set=fitOpts.opts.TestingFlag, normalize=""
        )
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
        logModule.info("___Start Processing___")
        start = dt.datetime.now()
        logModule.info(f"start time: {start.strftime(self._time_formatter)}")
        logModule.info(f"using {self.numCpus} cpus")
        # split into datablocks
        snrIndexBlocks = np.array_split(np.arange(self.numCurves), self.fitOpts.opts.ProcessingNumBlocks)
        # trange? loop through bocks -> aim is to have smaller chunks to send of when multiprocessing

        # create folder to temporarily save the processed data
        utils.create_folder_ifn_exist('temp/')
        path = Path('temp').absolute()

        for block_idx in tqdm.trange(self.fitOpts.opts.ProcessingNumBlocks):
            # logModule.info(f"Processing Block {block_idx}/{self.fitOpts.opts.ResamplingNumBlocks}")
            # have a list of indices to process
            indexes_to_process = snrIndexBlocks[block_idx]
            if not self.fitOpts.opts.TestingFlag:
                with mp.Pool(self.numCpus) as p:
                    resampledDbList = p.map(self._mp_wrap_sampling, indexes_to_process)
            else:
                resampledDbList = [
                    self._mp_wrap_sampling(indexes_to_process[k]) for k in range(len(indexes_to_process))
                ]

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
        logModule.info(f"Finished: {end.strftime(self._time_formatter)}")
        t_total = (end - start).total_seconds()
        logModule.info(f"Compute time: {t_total / 60:.2f} min ({t_total / 3600:.1f} h)")

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


def resampleDatabase(fitOpts: options.FitOptions) -> (np.ndarray, nib.Nifti1Image, pd.DataFrame, np.ndarray):
    dRe = DatabaseResampler(fitOpts=fitOpts)
    dRe.resample()
    if fitOpts.config.ResampledDataOutputPath:
        dRe.save_resampled()
    niiData, niiImg, pd_db, np_db = dRe.get_data()
    return niiData, niiImg, pd_db, np_db


def resampleData(fitOpts: options.FitOptions) -> (np.ndarray, nib.Nifti1Image):
    dRe = DataResampler(fitOpts=fitOpts)
    dRe.resample()
    if fitOpts.config.ResampledDataOutputPath:
        dRe.save_resampled()
    niiData, niiImg = dRe.get_data()
    return niiData, niiImg
