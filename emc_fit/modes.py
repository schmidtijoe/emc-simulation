"""
Module to emc_fit database with or without sampled noise meanto nii data.

Different fitting options are build in as objects.
Taking in the niiData (in 2D) and pandas/numpy objects of the database
database is assumed to be either 2D [t2/b1 variants, echos] or 3D [corresponding niiData voxel, t2/b1 variants, echoes]

"""
import logging

import pandas as pd

from emc_sim import utils
import numpy as np
import tqdm
from scipy import stats
import multiprocessing as mp
import matplotlib.pyplot as plt

logModule = logging.getLogger(__name__)


class B1Prior:
    def __init__(self, data_slice_shape: tuple, database_pandas: pd.DataFrame,
                 b1_map_input: bool = False, b1_weighting: bool = True,
                 b1_weight_factor: float = 0.1, b1_weight_width: float = 1.1,
                 visualize: bool = True):
        logModule.info("B1Prior -- Inititalize")
        self.visualize = visualize
        # toggle for use of b1 weighting
        self.use_weighting = b1_weighting
        # external b1 map provided
        self.map_input = b1_map_input
        # if not set shape width for b1 prior
        self.width = b1_weight_width
        # set shape
        self.data_shape = data_slice_shape
        if len(self.data_shape) > 2:
            # catch shape mismatch -> only x & y dim
            self.data_shape = self.data_shape[:2]

        # initialize
        self.b1_values = np.unique(database_pandas.b1)
        self.t2_values = np.unique(database_pandas.t2)
        self.etl = len(database_pandas.iloc[0].emcSignal)

        # default values for no weighting at all
        self.weighting_factor = b1_weight_factor
        self.b1_weighting_matrix = np.ones((*data_slice_shape, len(self.b1_values)))

        self.database = database_pandas
        # rebuild database with slice dimensions
        self.slice_database = self.rebuild_database()

    def set_weighting_matrix(self):
        if self.use_weighting:        # b1 weighting factor
            logModule.info(f"B1Prior -- Use weighting!")
            if self.map_input:
                logModule.info(f"B1Prior -- Input B1 Map")
                # ToDo: code for processing input nii here, maybe resampling
                pass
            else:
                logModule.info(f"B1Prior -- Weight: {self.weighting_factor:.3f}, Width: {self.width:.2f}")
                # calculate weighting
                # calls to reset width and weights would take effect here
                self.b1_weighting_matrix = self._set_slice_weighting()
                # visualize
                if self.visualize:
                    self._visualize_b1_weighting()
        else:
            logModule.info(f"B1Prior -- No B1 prior weighting!")
            self.weighting_factor = 0.0

    def get_slice_shape_database(self):
        return self.slice_database

    def get_b1_weighting_matrix(self):
        return self.weighting_factor * self.b1_weighting_matrix

    def set_weight_width(self, weight: float, width: float):
        self.weighting_factor = weight
        self.width = width

    def _set_slice_weighting(self):
        logModule.info("B1Prior -- set slice b1 weighting")
        b1_weighting_profile = self._create_gauss_b1_matrix()
        slice_db_b1_weight = np.zeros([*self.data_shape, len(self.b1_values)])
        for idx_b1 in range(len(self.b1_values)):
            slice_db_b1_weight[:, :, idx_b1] = np.abs(self.b1_values[idx_b1] - b1_weighting_profile)
        return slice_db_b1_weight

    def rebuild_database(self):
        # need to rearrange database to pick curves based on b1
        logModule.info("B1Prior -- Rearranging database")
        slice_database = np.zeros([len(self.t2_values), len(self.b1_values), self.etl])
        # rearrange database
        for idx_b1 in range(len(self.b1_values)):
            for idx_t2 in range(len(self.t2_values)):
                sub_db = self.database[self.database.t2 == self.t2_values[idx_t2]]
                curve = sub_db[sub_db.b1 == self.b1_values[idx_b1]].emcSignal.to_numpy()[0]
                nc = np.linalg.norm(curve)
                slice_database[idx_t2, idx_b1] = np.divide(curve, nc, where=nc > 0, out=np.zeros_like(curve))
        return slice_database

    def _create_gauss_b1_matrix(self, width_factor: float = 1.0):
        x_dim = self.data_shape[0]
        y_dim = self.data_shape[1]
        x_mid = int(x_dim / 2)
        y_mid = int(y_dim / 2)

        X, Y = np.meshgrid(np.arange(x_dim), np.arange(y_dim))
        b1_range = (np.min(self.b1_values), np.max(self.b1_values))

        g2d = b1_range[0] + np.diff(b1_range) * np.exp(
            -4 * np.log(2) * (
                        (X - x_mid) ** 2 / (width_factor * x_mid) ** 2 + (Y - y_mid) ** 2 / (width_factor * y_mid) ** 2)
        )
        return np.swapaxes(g2d, 0, 1)

    def _visualize_b1_weighting(self):
        weight = self.weighting_factor
        width = self.width
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(f"weight: {weight:.2f}, width: {width:.2f}")
        gs_y = int(len(self.b1_values) / 2)
        gs = fig.add_gridspec(2, gs_y+1)
        for k in range(len(self.b1_values)):
            ax = fig.add_subplot(gs[k])
            ax.axis(False)
            ax.set_title(f"B1: {self.b1_values[k]}")
            img = ax.imshow(self.b1_weighting_matrix[:, :, k])
        ax_cb = fig.add_subplot(gs[-1])
        plt.colorbar(img, cax=ax_cb)
        plt.tight_layout()
        plt.show()


class Fit:
    def __init__(self, nifti_data: np.ndarray, pandas_database: pd.DataFrame, b1_prior: B1Prior):
        self.pd_db = pandas_database
        self.nii_data = nifti_data
        logModule.info("______ FIT ______")
        logModule.info("Fit -- emc_fit")
        if np.max(nifti_data) > 1.001:
            logModule.info(f"Fit -- Nifti Input Data range exceeded; max: {np.max(nifti_data)}")
            logModule.info("Fit --Rescaling")
            self.nii_data = utils.normalize_array(self.nii_data)
        if nifti_data.shape.__len__() < 2:
            err = "Nifti Input Data assumed to be at least 2D: [voxels, echoes] but found 1D, exiting..."
            logModule.error(err)
            raise AttributeError(err)
        self.num_curves = np.prod(self.nii_data.shape[:-1])

        # rearrange database
        self.np_db = b1_prior.get_slice_shape_database()
        # set weighting matrix
        b1_prior.set_weighting_matrix()
        # get weight matrix
        self.b1_weight_matrix = b1_prior.get_b1_weighting_matrix()

        # set values
        self.b1_values = np.unique(self.pd_db.b1)
        self.t2_values = np.unique(self.pd_db.t2)

        # l2 normalize
        logModule.info("Fit -- L2 normalize data")
        self.nii_data = utils.normalize_array(self.nii_data, normalization="l2")
        self.np_db = utils.normalize_array(self.np_db, normalization="l2")

        self.t2_map = np.zeros(self.nii_data.shape[:-1])
        self.b1_map = np.zeros(self.nii_data.shape[:-1])

    def fit(self):
        return

    def get_maps(self) -> (np.ndarray, np.ndarray):
        self.fit()
        return self.t2_map, self.b1_map


class L2Fit(Fit):
    def __init__(self, nifti_data: np.ndarray, pandas_database: pd.DataFrame, b1_prior: B1Prior):
        super(L2Fit, self).__init__(nifti_data, pandas_database, b1_prior)
        # self.chunk_size = 2000
        # self.chunk_num = int(np.ceil(self.num_curves / self.chunk_size))

    def fit(self):
        logModule.info(f"__ Fitting L2 Norm Minimization __")
        # we do the fitting in blocks, (whole volume seems to be to expensive)
        # data shape = [num_curves, num_echoes]
        for slice_idx in tqdm.trange(self.nii_data.shape[2]):
            data = self.nii_data[:, :, slice_idx]
            # data dim [x, y, t], db dim [x, y, t2, b1, t]
            differenceCurveDb = self.np_db[np.newaxis, np.newaxis, :, :, :] - data[:, :, np.newaxis, np.newaxis, :]
            l2_db = np.linalg.norm(differenceCurveDb, axis=-1)
            l2_b1 = l2_db + self.b1_weight_matrix[:, :, np.newaxis, :]
            # total variation part
            lam = 0.1
            tv = np.sum(np.abs(np.array(np.gradient(l2_b1, axis=-1))))      # want to minimize total variation across B1 map
            penalty = l2_b1    # + lam * tv
            minimum = np.min(penalty, axis=(-1, -2))
            for x in range(penalty.shape[0]):
                for y in range(penalty.shape[1]):
                    idx_t2, idx_b1 = np.where(penalty[x, y] == minimum[x, y])
                    self.t2_map[x, y, slice_idx] = self.t2_values[idx_t2[0]]
                    self.b1_map[x, y, slice_idx] = self.b1_values[idx_b1[0]]

        logModule.info("Fit L2 -- Finished!")

    # @staticmethod
    # def _wrap_l2(args):
    #     idx_arr, data_arr, db_arr = args
    #     differenceCurveDb = db_arr[np.newaxis, :] - data_arr[:, np.newaxis]
    #     l2 = np.linalg.norm(differenceCurveDb, axis=-1)
    #     fit_idx = np.argmin(l2, axis=-1)
    #     return idx_arr, fit_idx

    # def fit_mp(self):
    #     num_cpus = np.max([mp.cpu_count() - 16, 4])
    #     logModule.info(f"multiprocessing, using {num_cpus} CPU")
    #     chunks = np.array_split(np.arange(self.num_curves), self.chunk_num)
    #     mp_list = [(chunks[i], self.nii_data[chunks[i]], self.np_db) for i in range(len(chunks))]
    #
    #     with mp.Pool(num_cpus) as pool:
    #         results = list(tqdm.tqdm(pool.imap_unordered(self._wrap_l2, mp_list), total=self.chunk_num))
    #
    #     for res in results:
    #         chunk_idx = res[0]
    #         fit_idx = res[1]
    #         self.t2_map[chunk_idx] = self.pd_db.iloc[fit_idx].t2
    #         self.b1_map[chunk_idx] = self.pd_db.iloc[fit_idx].b1


class PearsonFit(Fit):
    def __init__(self, nifti_data, pandas_database, b1_prior):
        super(PearsonFit, self).__init__(nifti_data, pandas_database, b1_prior)
        logModule.info("____________")
        logModule.info("pearson correlation coefficient emc_fit")

    @staticmethod
    def pearsons_multidim(array_a, array_b):
        num = np.inner(array_a, array_b)
        den = np.linalg.norm(array_a, axis=-1) * np.linalg.norm(array_b, axis=-1)

        # Finally get corr coeff, prevent 0 division
        return np.divide(num, den, where=den > 0, out=np.zeros_like(num))

    def pearson_1d(self, curve):
        r = -2
        fit_idx = -1
        for db_idx in range(self.np_db.shape[0]):
            r_, _ = stats.pearsonr(curve, self.np_db[db_idx])
            if not np.isnan(r_):
                if r_ > r:
                    r = r_
                    fit_idx = db_idx
        return fit_idx

    def fit(self):
        logModule.info(f"__ Fitting Pearson Correlation Coefficient Maximization __")
        for bar_idx in tqdm.trange(self.num_curves):
            r_ = self.pearsons_multidim(self.nii_data[bar_idx], self.np_db)
            fit_idx = np.unravel_index(np.argmax(r_), shape=r_.shape)

            self.t2_map[bar_idx] = self.pd_db.iloc[fit_idx].t2
            self.b1_map[bar_idx] = self.pd_db.iloc[fit_idx].b1

        logModule.info("Finished!")


class MleFit(Fit):
    def __init__(self, nifti_data, pandas_database, numpy_database):
        super(MleFit, self).__init__(nifti_data, pandas_database, numpy_database)

    def fit(self):
        logModule.info(f"__ Fitting MLE __")

        logModule.info("Finished!")


if __name__ == '__main__':
    logging.getLogger(__name__)
    dbFile = "/data/pt_np-jschmidt/data/00_phantom_scan_data/pulseq_2022-10-14/emc/database_0p7_fa180_esp10_etl8.pkl"
    inFile = "/data/pt_np-jschmidt/data/00_phantom_scan_data/pulseq_2022-10-14/processed/se_mc_0p7_grappa2_denoised_loraks_recon_mag.nii.gz"
    pd_db, _ = utils.load_database(dbFile)
    niiData, niiImg = utils.niiDataLoader(inFile)

    # set prior
    b1_prior = B1Prior(
        data_slice_shape=niiData.shape[:2],
        database_pandas=pd_db,
        b1_map_input=False,
        b1_weighting=True,
        b1_weight_factor=0.1,
        b1_weight_width=1.1,
        visualize=False
    )

    nw = 5
    b1_width = np.linspace(0.8, 1.4, nw)
    nwe = 5
    b1_weights = np.linspace(0, 0.2, nwe)

    wwe_t2 = np.zeros((nw, nwe, *niiData.shape[:2]))
    wwe_b1 = np.zeros_like(wwe_t2)
    for w_idx in range(nw):
        for we_idx in range(nwe):
            b1_prior.set_weight_width(b1_weights[we_idx], b1_width[w_idx])
            fitModule = L2Fit(nifti_data=niiData, pandas_database=pd_db, b1_prior=b1_prior)
            t2, b1 = fitModule.get_maps()
            wwe_t2[w_idx, we_idx] = t2[:, :, 5]
            wwe_b1[w_idx, we_idx] = b1[:, :, 5]

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(f"T2")
    gs = fig.add_gridspec(5, 5)
    for w_idx in range(nw):
        for we_idx in range(nwe):
            ax = fig.add_subplot(gs[w_idx, we_idx])
            ax.axis(False)
            ax.set_title(f"w:{w_idx},we:{we_idx}")
            img = ax.imshow(wwe_t2[w_idx, we_idx])
    plt.show()

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(f"T2")
    gs = fig.add_gridspec(5, 5)
    for w_idx in range(nw):
        for we_idx in range(nwe):
            ax = fig.add_subplot(gs[w_idx, we_idx])
            ax.axis(False)
            ax.set_title(f"w:{w_idx},we:{we_idx}")
            img = ax.imshow(wwe_b1[w_idx, we_idx])
    plt.show()
