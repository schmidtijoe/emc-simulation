"""
Module to emc_fit database with or without sampled noise mean via l2 norm to nii data
"""
import logging
import os
import pickle
import typing

import nibabel as nib
import numpy as np
import pandas as pd

from emc_fit.noise import chambollepock
from scipy import special
from emc_sim import utils
from emc_fit.noise import handlers
from emc_fit import options
import datetime as dt
import multiprocessing as mp
from operator import itemgetter
from itertools import chain
from pathlib import Path
import tqdm
import matplotlib.pyplot as plt

logModule = logging.getLogger(__name__)


class DataResampler:
    """
    Class for Streamlining Noise Mean resampling of nii Input data
    """

    def __init__(self, fitOpts: options.FitOptions):
        logModule.info(f"_______data resampler_______")
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

        # first init
        self.reNiiData = self.niiData.copy()
        # multiprocessing, leave headroom but take at least 4
        self.numCpus = np.max([4, mp.cpu_count() - fitOpts.opts.ProcessingHeadroomMultiprocessing])
        self.multiprocessing = fitOpts.opts.Multiprocessing
        logModule.info("Finished Init")

    def _simple_iterate_approximation(self, eps: bool = True, return_stats: bool = True):
        """
        Simple iteration -> approximate denoised voxels by taking noised data as nc_chi distribution - mean
        of denoised voxels
        """
        if eps:
            # take only voxels not already mapped to 0
            selection = self.reNiiData > self.eps
        else:
            selection = self.niiData >= 0.0
        # calculate offset between mean noised amplitude and data
        diff = np.subtract(self.niiData[selection], self.ncChi.mean(self.reNiiData[selection]))
        max_diff = np.max(diff)
        result = np.add(self.reNiiData[selection], diff)
        result = np.clip(result, 0.0, np.max(result))
        self.reNiiData[selection] = result
        if return_stats:
            return np.count_nonzero(selection), max_diff

    def resample(self):
        logModule.info("___Start Processing___")
        start = dt.datetime.now()
        logModule.info(f"start time: {start.strftime(self._time_formatter)}")

        if self.fitOpts.opts.ResampleDataSimple:
            # use simple approximation with noise mean
            bar = tqdm.trange(self.fitOpts.opts.ResampleDataNumIterations)
            for _ in bar:
                num_curves, max_diff = self._simple_iterate_approximation(return_stats=True)
                bar.set_postfix_str(f"maximum offset of approx mean for {num_curves} curves: {max_diff:.4f}")

            # reshaping
        else:
            # use maximum likelihood majorize - minimize approach
            self.mmIteration()

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
        # check for shape
        if not self.reNiiData.shape == self.niiImg.shape:
            self.reNiiData = np.reshape(self.reNiiData, self.niiImg.shape)
        return self.reNiiData, self.niiImg

    def _majorante(self, arg_arr: typing.Union[np.ndarray, float, int]) -> typing.Union[np.ndarray, float]:
        is_single_val = False
        if isinstance(arg_arr, (float, int)):
            arg_arr = np.array([arg_arr])
            is_single_val = True
        result = np.zeros_like(arg_arr)

        gam = 7e2
        # for smaller eps result array remains 0
        # for small enough args but bigger than eps we compute the given formula
        sel = np.logical_and(self.eps < arg_arr, gam > arg_arr)
        result[sel] = np.divide(
            special.iv(self.ncChi.num_channels, arg_arr[sel]),
            special.iv(self.ncChi.num_channels - 1, arg_arr[sel])
        )
        # for big args we linearly approach asymptote to 1 @ input arg 30000 (random choice
        len_asymptote = 3e4
        start_val = np.divide(
            special.iv(self.ncChi.num_channels, gam),
            special.iv(self.ncChi.num_channels - 1, gam)
        )
        sel = arg_arr >= gam
        result[sel] = start_val + (1.0 - start_val) / len_asymptote * (arg_arr[sel] - gam)
        if is_single_val:
            result = result[0]
        return result

    def _y_tilde(self, y_obs: typing.Union[np.ndarray, float, int], x_approx: typing.Union[np.ndarray, float, int]
                 ) -> typing.Union[np.ndarray, float]:
        arg = np.multiply(
            y_obs,
            x_approx
        ) / self.ncChi.sigma ** 2
        factor = self._majorante(arg)
        return y_obs * factor

    def test_ytilde(self, y_obs, x_approx):
        return self._y_tilde(y_obs=y_obs, x_approx=x_approx)

    def mmIteration(self):
        # want at least 2d planes for regularization
        if self.reNiiData.shape.__len__() < 2:
            if self.niiImg.shape.__len__() >= 2:
                self.reNiiData = np.reshape(self.reNiiData, self.niiImg.shape)
            else:
                logModule.error(f"Data Shape {self.reNiiData.shape} "
                                f"found not compatible with TV regularization across voxels: "
                                f"at least 2D Data needed")
                raise AttributeError(f"Data Shape {self.reNiiData.shape} "
                                     f"found not compatible with TV regularization across voxels: "
                                     f"at least 2D Data needed")

        if self.niiData.shape.__len__() < 4:
            logModule.info(f"Data shape: {self.niiData.shape} - 3D: assuming no echoes")

        else:
            logModule.info(f"Data shape: {self.niiData.shape} - 4D: assuming {self.niiData.shape[-1]} echoes")

            # bar = tqdm.trange(self.niiData.shape[-1], desc="Processing Echoes", ncols=100)
            # for echo_idx in bar:
            #     # choose 3d volume as input data
            #                 data = self.niiData[:, :, :, echo_idx].copy()
            echo_num = self.niiData.shape[-1]
            mp_list = [[self.niiData[:, :, :, k], k] for k in range(echo_num)]
            num_cpus = np.min([echo_num, self.numCpus])
            if self.multiprocessing:
                # results = parallelbar.progress_imap(self._echo_iteration, mp_list, n_cpu=4, core_progress=True, total=echo_num)
                logModule.info(f"using {num_cpus} CPU for processing {echo_num} echo images")
                with mp.Pool(num_cpus) as pool:
                    results = list(tqdm.tqdm(pool.imap_unordered(self._echo_iteration, mp_list), total=echo_num))
            else:

                results = []
                for _, arg in enumerate(tqdm.tqdm(mp_list)):
                    results.append(self._echo_iteration(arg))

            for res in results:
                self.reNiiData[:, :, :, res[0]] = res[1]

    def _echo_iteration(self, args):
        echoImg, echo_idx = args
        denoised_data_approx = echoImg.copy()

        # initialize approximation with same set
        # iter_bar = tqdm.trange(self.fitOpts.opts.ResampleDataNumIterations, desc="iteration", ncols=50)
        for _ in range(self.fitOpts.opts.ResampleDataNumIterations):
            # set majorante = y tilde from data and previous parameter approximation
            data = self._y_tilde(y_obs=echoImg, x_approx=denoised_data_approx)
            # solve ls with regularization -> argmin_x(denoized data)>0  || K(x) - y_tilde ||_2^2 + TV(x)
            denoised_data_approx = chambollepock.chambolle_pock_tv(
                data=data,
                Lambda=self.fitOpts.opts.ResampleDataRegularizationLambda,
                n_it=30,
                return_all=False
            )
        return echo_idx, denoised_data_approx


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


def plot(data, x, en):
    data_hist, data_bins = np.histogram(data, bins=1000)
    data_bins = data_bins[1:] - np.diff(data_bins)

    x_hist, x_bins = np.histogram(x, bins=1000)
    x_bins = x_bins[1:] - np.diff(x_bins)

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(2, 6, height_ratios=[3, 1])

    ax = fig.add_subplot(gs[0, :2])
    ax.axis(False)
    ax.grid(False)
    img = ax.imshow(data)
    plt.colorbar(img, ax=ax, shrink=0.5)

    ax = fig.add_subplot(gs[0, 2:4])
    ax.axis(False)
    ax.grid(False)
    img = ax.imshow(x)
    plt.colorbar(img, ax=ax, shrink=0.5)

    ax = fig.add_subplot(gs[0, 4:])
    ax.axis(False)
    ax.grid(False)
    img = ax.imshow((data - x) / data)
    plt.colorbar(img, ax=ax, shrink=0.5)

    ax = fig.add_subplot(gs[1, :2])
    ax.plot(np.arange(len(en)), en)

    ax = fig.add_subplot(gs[1, 2:4])
    ax.set_ylim(0, np.max(data_hist[100:]))
    ax.fill_between(data_bins, data_hist)

    ax = fig.add_subplot(gs[1, 4:])
    ax.fill_between(x_bins, x_hist)
    ax.set_ylim(0, np.max(x_hist[20:]))

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # set up logging
    logging.basicConfig(format='%(asctime)s %(levelname)s :: %(name)s --  %(message)s',
                        datefmt='%I:%M:%S', level=logging.INFO)

    configFile = "D:\\Daten\\01_Work\\11_owncloud\\ds_mese_cbs_js\\" \
                 "02_postmortem_scan_data\\01\\t2_semc_0p5_30slice\\fit_config_l2_win.json"
    path = Path(configFile).absolute()
    fitSet = options.FitOptions.load(path)
    dRe = DataResampler(fitSet)
    dRe.resample()
    dRe.save_resampled()

