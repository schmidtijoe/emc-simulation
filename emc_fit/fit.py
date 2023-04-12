"""
Module to emc_fit database with or without denoising nii data.

Different fitting options are build in as objects.
Taking in the niiData (in 2D) and pandas/numpy objects of the database
is assumed to be either 2D [t2/b1 variants, echos] or 3D [corresponding niiData voxel, t2/b1 variants, echoes]

"""
import logging
import pandas as pd
from emc_sim import utils
from emc_fit import b1input
import numpy as np
import tqdm
from scipy import stats
import matplotlib.pyplot as plt
import time
logModule = logging.getLogger(__name__)


class Fit:
    def __init__(self, nifti_data: np.ndarray, pandas_database: pd.DataFrame, b1_weight: b1input.B1Weight):
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
        self.np_db = b1_weight.get_t2_b1_etl_shape_database()
        self.b1_weight = b1_weight

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
        raise NotImplementedError

    def get_maps(self) -> (np.ndarray, np.ndarray):
        self.fit()
        return self.t2_map, self.b1_map


class L2Fit(Fit):
    def __init__(self, nifti_data: np.ndarray, pandas_database: pd.DataFrame, b1_weighting: b1input.B1Weight):
        super(L2Fit, self).__init__(nifti_data, pandas_database, b1_weighting)
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
            b1_slice_weight_matrix = self.b1_weight.get_b1_weighting_matrix(slice_id=slice_idx)
            l2_b1 = (1.0 - self.b1_weight.get_weighting_value()) * l2_db + \
                    self.b1_weight.get_weighting_value() * b1_slice_weight_matrix[:, :, np.newaxis, :]
            # total variation part
            # lam = 0.1
            # tv = np.sum(np.abs(np.array(np.gradient(l2_b1, axis=-1))))
            # want to minimize total variation across B1 map
            penalty = l2_b1  # + lam * tv
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


def compute_chunks(chunksize: int, data_flat: np.ndarray):
    finShape = data_flat.shape[:-1]
    resultT2 = np.zeros(finShape)
    resultB1 = np.zeros(finShape)
    num_chunks = int(np.ceil(finShape[0] / chunksize))
    for chunk_idx in tqdm.trange(num_chunks):
        start = chunk_idx * chunksize
        end = np.min([(chunk_idx+1) * chunksize, finShape[0]])
        chunk = data_flat[start:end]
        source = np_db[:, np.newaxis, :] - chunk
        # calculate distance
        l2_matrix = np.sqrt(np.einsum('ijk, ijk -> ij', source, source))
        # find minimum along row
        fit_idx = np.argmin(l2_matrix, axis=0)
        resultT2[start:end] = pd_db.iloc[fit_idx].t2
        resultB1[start:end] = pd_db.iloc[fit_idx].b1

    resultT2 = np.reshape(resultT2, data.shape[:-1])
    resultB1 = np.reshape(resultB1, data.shape[:-1])
    return resultT2, resultB1


if __name__ == '__main__':
    logging.getLogger(__name__)
    dbFile = "/data/pt_np-jschmidt/data/00_phantom_scan_data/pulseq_2022-10-14/emc/database_0p7_fa180_esp10_etl8.pkl"
    inFile = "/data/pt_np-jschmidt/data/00_phantom_scan_data/pulseq_2022-10-14/processed/se_mc_0p7_grappa2_denoised_loraks_recon_mag.nii.gz"
    pd_db, np_db = utils.load_database(dbFile)
    niiData, niiImg = utils.niiDataLoader(inFile, normalize="")

    # normalize
    db_nom = np.sqrt(np.einsum('ij, ij -> i', np_db, np_db))[:, np.newaxis]
    np_db = np.divide(np_db, db_nom, where=db_nom>1e-12, out=np.zeros_like(np_db))
    data = niiData[:200, :200, 4]
    data_nom = np.sqrt(np.einsum('ijk, ijk -> ij', data, data))[:, :, np.newaxis]
    data = np.divide(data, data_nom, where=data_nom > 1e-12, out=np.zeros_like(data))
    data_1d = np.reshape(data, [-1, data.shape[-1]])
    
    chunksizes = np.linspace(20,4000, 100, dtype=int)
    times = []
    for chunk in chunksizes:
        runs = []
        for run in range(3):
            start = time.time()
            _ = compute_chunks(chunk, data_1d)
            runs.append(time.time() - start)
        times.append(np.mean(runs))

    t2_test, _ = compute_chunks(200, data_1d)
    t2_test = np.reshape(t2_test, data.shape[:-1])
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(f"T2")
    ax = fig.add_subplot(211)
    ax.axis(False)
    img = ax.imshow(t2_test[:, :])

    ax = fig.add_subplot(212)
    ax.set_xlabel("chunksizes")
    ax.set_ylabel("compute time average of 3 runs")
    ax.plot(chunksizes, times)

    plt.savefig("chunksize_speed_testing.png")
    plt.show()
    print("lol")
