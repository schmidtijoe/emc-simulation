import matplotlib.pyplot as plt
import numpy as np
import emc_db
from emc_db import DB
import tqdm
import logging
import time

logModule = logging.getLogger(__name__)


class Fitter:
    def __init__(self, nifti_data: np.ndarray, database: emc_db.DB):
        logModule.info(f"emc fit setup")

        # check dims
        if nifti_data.shape.__len__() < 2:
            err = "Nifti Input Data assumed to be at least 2D: [voxels, echoes] but found 1D, exiting..."
            logModule.error(err)
            raise ValueError(err)
        self.nii_shape = nifti_data.shape

        # check normalization
        logModule.info(f"check normalization")
        nii_norm = np.linalg.norm(nifti_data, axis=-1, keepdims=True)
        self.nii_data: np.ndarray = np.divide(nifti_data, nii_norm, where=nii_norm > 1e-12,
                                              out=np.zeros_like(nifti_data))
        # make it always 4D
        if self.nii_data.shape.__len__() == 2:
            self.nii_data = self.nii_data[:, np.newaxis, np.newaxis, :]
        if self.nii_data.shape.__len__() == 3:
            self.nii_data = self.nii_data[:, :, np.newaxis, :]
        if self.nii_data.shape.__len__() > 4:
            err = f"fit not designed for data exceeding 4D"
            logModule.error(err)
            raise ValueError(err)

        self.database: emc_db.DB = database
        self.database.normalize()

        # set vars
        self.num_curves = np.prod(self.nii_data.shape[:-1])
        self.t2_values, self.b1_values = self.database.get_t2_b1_values()

        self.t2_map = np.zeros(self.nii_data.shape[:-1])
        self.b1_map = np.zeros(self.nii_data.shape[:-1])
        self.err_map = np.zeros(self.nii_data.shape[:-1])

        # set B1 input - only manipulate this if needed
        self.b1_weight_input: np.ndarray = np.ones_like(nifti_data)
        self.b1_weight_lambda: float = 0.0

        self.slice_dim = self.nii_data.shape[-2]

    # public
    def reset(self):
        self.t2_map = np.zeros(self.nii_data.shape[:-1])
        self.b1_map = np.zeros(self.nii_data.shape[:-1])
        self.err_map = np.zeros(self.nii_data.shape[:-1])

    def get_maps(self) -> (np.ndarray, np.ndarray):
        if not self.t2_map.any() > 1e-9:
            # if we still have 0 array for t2 map
            self._fit()
        # retrieve arrays -> squeeze possible additional axes
        t2 = np.squeeze(self.t2_map)
        b1 = np.squeeze(self.b1_map)
        return t2, b1

    def set_b1_weight(self, b1_map: np.ndarray, b1_lambda: float = 0.0, plot: bool = True):
        self.b1_weight_input = self._check_b1_input_dims(b1_map)
        b1_lambda = np.clip(b1_lambda, 0, 1)
        self.b1_weight_lambda = b1_lambda
        if plot:
            self._plot_ortho_b1_weight()

    def set_b1_simple_prior(self, b1_lambda: float = 0.2, b1_max: float = 1.5, b1_min: float = 0.5, plot: bool = True,
                            z_middle_shift: float = 0.0, voxel_dims_mm: np.ndarray = np.array([0.7, 0.7, 0.7]),
                            sphere_width: float = 1.0):
        """
        sets a simple B1 prior -> spherical shape with b1_max in middle, falling to ~b1_min at edges
        z_middle sets the midpoint for the maximum along z dimension, half wavelength roughly headsize ~ 25 cm
        """
        b1_prior = np.zeros(self.nii_shape[:-1])
        g_ax = np.linspace(0, 0.3, 10000)
        g_shape = b1_min + (b1_max - b1_min) * np.cos(g_ax * np.pi * 4 / sphere_width)
        g_shape = np.clip(g_shape, b1_min, b1_max)
        half_ax = [1e-3 * voxel_dims_mm[dim_idx] * self.nii_shape[dim_idx] / 2 for dim_idx in range(3)]
        ax_dim = [np.linspace(-half_ax[k], half_ax[k], self.nii_shape[k]) for k in range(3)]
        ax_dim[2] = ax_dim[2] + z_middle_shift
        for x_idx in range(self.nii_shape[0]):
            for y_idx in range(self.nii_shape[1]):
                for z_idx in range(self.nii_shape[2]):
                    vector = np.sqrt(
                        np.square(ax_dim[0][x_idx]) + np.square(ax_dim[1][y_idx]) + np.square(ax_dim[2][z_idx])
                    )
                    b1_prior[x_idx, y_idx, z_idx] = g_shape[g_ax > vector][0]

        self.set_b1_weight(b1_map=b1_prior, b1_lambda=b1_lambda, plot=plot)

    # private
    def _plot_ortho_b1_weight(self):
        self._plot_ortho_view(self.b1_weight_input)

    def _plot_ortho_nii_data(self):
        # take first echo
        self.set_b1_simple_prior(self.nii_data[:, :, :, 0])

    @staticmethod
    def _plot_ortho_view(arr_3d: np.ndarray):
        if arr_3d.shape.__len__() != 3:
            err = f"plot function only for 3d data. data given has {arr_3d.shape.__len__()} dims"
            logModule.error(err)
            raise ValueError
        dim = np.array([*arr_3d.shape]) / 2
        fig = plt.figure(figsize=(10, 3))
        gs = fig.add_gridspec(1, 4, width_ratios=[15, 15, 15, 1])
        img = None
        for k in range(3):
            ax = fig.add_subplot(gs[k])
            ax.grid(False)
            ax.axis(False)
            plot_arr = np.swapaxes(arr_3d, k, 0)
            img = ax.imshow(np.transpose(plot_arr[dim[k]]))
        cb_ax = fig.add_subplot(gs[-1])
        plt.colorbar(img, cax=cb_ax, label='B1')
        plt.show()

    def _check_b1_input_dims(self, b1_map: np.ndarray) -> np.ndarray:
        # fix dims
        if b1_map.shape.__len__() < 3:
            b1_map = b1_map[:, :, np.newaxis]
        if b1_map.shape.__len__() > 3:
            err = f"fit not designed for B1 input data extending 3D"
            logModule.error(err)
            raise ValueError(err)
        if b1_map.shape[2] == 1:
            logModule.info(f"b1 input 2D. Extending slice dimension by repeating slice")
            b1_map = np.repeat(b1_map, self.nii_shape[2], axis=2)
            # check dimensions
        if b1_map.shape[:3] != self.nii_shape[:3]:
            err = f"mismatch in b1 map input shapes to echo data shape {b1_map.shape[:3]} != {self.nii_shape[:3]}." \
                  f"Resample data and try again"
            logModule.error(err)
            raise ValueError(err)
        return b1_map

    def _fit(self):
        logModule.info("Fitting")
        # slice wise
        t_start = time.time()

        dim_x, dim_y = self.nii_shape[:2]
        for slice_idx in tqdm.trange(self.slice_dim, desc="processing"):
            data = np.reshape(self.nii_data[:, :, slice_idx], (-1, self.nii_shape[-1]))
            # L2 norm difference to database
            # data dim [xy, t] ---  db dim [t2b1, t]
            dot_matrix = np.einsum('ik, jk -> ij', data, self.database.np_array)
            # dot matrix [xy, db_t2b1]
            if self.b1_weight_lambda > 1e-9:
                # B1 prior
                # we want to weight the database curves with b1 close to prior more
                # need to calculate penalty for db curves dependent on b1 value
                # data dim [xy, b1] ---  db_b1 dim [idx, b1]
                data_b1 = np.reshape(self.b1_weight_input[:, :, slice_idx], -1)[:, np.newaxis]
                db_b1 = self.database.pd_dataframe.b1.to_numpy()[np.newaxis, :]
                b1_matrix = np.sqrt(np.square(data_b1 - db_b1))
                b1_matrix = 0.5 * (np.max(b1_matrix) - b1_matrix)
            else:
                b1_matrix = np.zeros_like(dot_matrix)
            # square lambda for more decent ranging (otherwise effect of 0.1 already quite drastic)
            max_matrix = (1.0 - self.b1_weight_lambda ** 2) * dot_matrix + self.b1_weight_lambda ** 2 * b1_matrix
            fit_indices = np.argmax(max_matrix, axis=-1)

            t2s = self.database.pd_dataframe.loc[self.database.pd_dataframe.index[fit_indices]].t2.to_numpy()
            b1s = self.database.pd_dataframe.loc[self.database.pd_dataframe.index[fit_indices]].b1.to_numpy()
            self.t2_map[:, :, slice_idx] = np.reshape(t2s, (dim_x, dim_y))
            self.b1_map[:, :, slice_idx] = np.reshape(b1s, (dim_x, dim_y))
        print(f"total processing time, slice wise: {time.time() - t_start:.3f} s")


if __name__ == '__main__':
    nii_data = np.square(np.random.random(size=(100, 100, 10, 16)))
    db = emc_db.DB.load("D:\\Daten\\01_Work\\03_code\\emc-simulation-python\\emc_db\\test\\test_db_database_file.pkl")

    fit_test = Fitter(nifti_data=nii_data, database=db)
    fit_test.set_b1_simple_prior(voxel_dims_mm=np.array([2.1, 2.1, 2.1]), b1_lambda=0.1)
    t2_map_l0p1, b1_map_l0p1 = fit_test.get_maps()

    fit_test.reset()
    fit_test.set_b1_simple_prior(voxel_dims_mm=np.array([2.1, 2.1, 2.1]), b1_lambda=0.8)
    t2_map_l0p8, b1_map_l0p8 = fit_test.get_maps()

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[15, 15, 1])
    ax = fig.add_subplot(gs[0])
    ax.set_title('lambda b1 = 0.1')
    ax.grid(False)
    ax.axis(False)
    ax.imshow(t2_map_l0p1[:, :, 4])
    ax = fig.add_subplot(gs[1])
    ax.set_title('lambda b1 = 0.8')
    ax.grid(False)
    ax.axis(False)
    img = ax.imshow(t2_map_l0p8[:, :, 4])
    cb_ax = fig.add_subplot(gs[2])
    plt.colorbar(img, cax=cb_ax, label="T2")
    ax = fig.add_subplot(gs[3])
    ax.grid(False)
    ax.axis(False)
    ax.imshow(b1_map_l0p1[:, :, 4])
    ax = fig.add_subplot(gs[4])
    ax.grid(False)
    ax.axis(False)
    ax.imshow(b1_map_l0p8[:, :, 4])
    cb_ax = fig.add_subplot(gs[5])
    plt.colorbar(img, cax=cb_ax, label="B1")
    plt.show()
