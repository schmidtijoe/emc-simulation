"""
Module to fit database with or without sampled noise meanto nii data.

Different fitting options are build in as objects.
Taking in the niiData (in 2D) and pandas/numpy objects of the database
database is assumed to be either 2D [t2/b1 variants, echos] or 3D [corresponding niiData voxel, t2/b1 variants, echoes]

"""
import logging

from emc_sim import utils
import numpy as np
import tqdm
from scipy import stats


class L2Fit:
    def __init__(self, nifti_data, pandas_database, numpy_database):
        self.pd_db = pandas_database
        self.np_db = numpy_database
        self.nii_data = nifti_data
        logging.info("____________")
        logging.info("l2 norm minimization fit")
        # data supposed to be 2d and max normed to 1
        if nifti_data.shape.__len__() > 2:
            logging.info(f"Nifti Input Data assumed shape not 2D; shape: {nifti_data.shape}")
            logging.info("Reshaping")
            self.nii_data = np.reshape(nifti_data, [-1, nifti_data.shape[-1]])
        if np.max(nifti_data) > 1.001:
            logging.info(f"Nifti Input Data range exceeded; max: {np.max(nifti_data)}")
            logging.info("Rescaling")
            self.nii_data = utils.normalize_array(self.nii_data)
        if nifti_data.shape.__len__() < 2:
            err = "Nifti Input Data assumed to be at least 2D: [voxels, echoes] but found 1D, exiting..."
            logging.error(err)
            raise AttributeError(err)
        self.num_curves = self.nii_data.shape[0]

        self.t2_map = np.zeros(self.num_curves)
        self.b1_map = np.zeros(self.num_curves)

    def fit(self):
        logging.info(f"__ Fitting L2 Norm Minimization __")
        # we do the fitting in blocks, (whole volume seems to be to expensive)
        # data shape = [num_curves, num_echoes]
        # need some timing checks
        for data_idx in tqdm.trange(self.num_curves):
            data = self.nii_data[data_idx]
            differenceCurveDb = self.np_db - data
            l2 = np.linalg.norm(differenceCurveDb, axis=-1)
            fit_idx = np.argmin(l2, axis=-1)

            self.t2_map[data_idx] = self.pd_db.iloc[fit_idx].t2
            self.b1_map[data_idx] = self.pd_db.iloc[fit_idx].b1

        logging.info("Finished!")

    def get_maps(self) -> (np.ndarray, np.ndarray):
        return self.t2_map, self.b1_map


class PearsonFit:
    def __init__(self, nifti_data, pandas_database, numpy_database):
        logging.info("____________")
        logging.info("pearson correlation coefficient fit")
        self.pd_db = pandas_database
        self.np_db = numpy_database
        self.nii_data = nifti_data
        # data supposed to be 2d and max normed to 1
        if nifti_data.shape.__len__() != 2:
            logging.info(f"Nifti Input Data assumed shape not 2D; shape: {nifti_data.shape}")
            logging.info("Reshaping")
            self.nii_data = np.reshape(nifti_data, [-1, nifti_data.shape[-1]])
        if np.max(nifti_data) > 1.001:
            logging.info(f"Nifti Input Data range exceeded; max: {np.max(nifti_data)}")
            logging.info("Rescaling")
            self.nii_data = utils.normalize_array(self.nii_data)
        self.num_curves = self.nii_data.shape[0]

        self.t2_map = np.zeros(self.num_curves)
        self.b1_map = np.zeros(self.num_curves)

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
        logging.info(f"__ Fitting Pearson Correlation Coefficient Maximization __")
        for bar_idx in tqdm.trange(self.num_curves):
            r_ = self.pearsons_multidim(self.nii_data[bar_idx], self.np_db)
            fit_idx = np.unravel_index(np.argmax(r_), shape=r_.shape)

            self.t2_map[bar_idx] = self.pd_db.iloc[fit_idx].t2
            self.b1_map[bar_idx] = self.pd_db.iloc[fit_idx].b1

        logging.info("Finished!")

    def get_maps(self) -> (np.ndarray, np.ndarray):
        return self.t2_map, self.b1_map


class MleFit:
    def __init__(self, nifti_data, pandas_database, numpy_database):
        self.pd_db = pandas_database
        self.np_db = numpy_database
        self.nii_data = nifti_data
        logging.info("____________")
        logging.info("Maximum likelihood fit")
        # data supposed to be 2d and max normed to 1
        if nifti_data.shape.__len__() != 2:
            logging.info(f"Nifti Input Data assumed shape not 2D; shape: {nifti_data.shape}")
            logging.info("Reshaping")
            self.nii_data = np.reshape(nifti_data, [-1, nifti_data.shape[-1]])
        if np.max(nifti_data) > 1.001:
            logging.info(f"Nifti Input Data range exceeded; max: {np.max(nifti_data)}")
            logging.info("Rescaling")
            self.nii_data = utils.normalize_array(self.nii_data)

        self.t2_map = None
        self.b1_map = None

    def get_maps(self) -> (np.ndarray, np.ndarray):
        return self.t2_map, self.b1_map

    def fit(self):
        logging.info(f"__ Fitting MLE __")

        logging.info("Finished!")
