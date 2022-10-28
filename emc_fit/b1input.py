import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from emc_sim import utils
from emc_fit import options

logModule = logging.getLogger(__name__)


class B1Weight:
    def __init__(self, data_slice_shape: tuple, database_pandas: pd.DataFrame,
                 b1_weighting: bool = True, b1_weight_factor: float = 0.1,
                 visualize: bool = True):
        logModule.info("B1Weight -- Initialize")
        self.visualize = visualize
        # toggle for use of b1 weighting
        self.use_weighting = b1_weighting
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
        # we build the matrix on a per slice bases!
        self.b1_weighting_matrix = np.ones((*data_slice_shape, len(self.b1_values)))

        self.database = database_pandas

    def get_t2_b1_etl_shape_database(self):
        reshaped_database = self.rebuild_database()
        return reshaped_database

    def get_b1_weighting_matrix(self, slice_id: int = 0):
        if not self.use_weighting:
            return np.zeros_like(self.b1_weighting_matrix)
        else:
            logModule.info(f"B1Weight -- Use weighting!")
            logModule.info(f"B1Weight: {self.weighting_factor:.3f}")
            return self.module_set_b1_weighting_matrix(slice_id)

    def module_set_b1_weighting_matrix(self, slice_id: int = 0):
        # index if needed for 3d object
        # this probably needs to be adapted for modules
        return self.weighting_factor * self.b1_weighting_matrix

    def rebuild_database(self):
        # need to rearrange database to pick curves based on b1
        logModule.info("B1Weight -- Rearranging database")
        slice_database = np.zeros([len(self.t2_values), len(self.b1_values), self.etl])
        # rearrange database
        for idx_b1 in range(len(self.b1_values)):
            for idx_t2 in range(len(self.t2_values)):
                sub_db = self.database[self.database.t2 == self.t2_values[idx_t2]]
                curve = sub_db[sub_db.b1 == self.b1_values[idx_b1]].emcSignal.to_numpy()[0]
                nc = np.linalg.norm(curve)
                slice_database[idx_t2, idx_b1] = np.divide(curve, nc, where=nc > 0, out=np.zeros_like(curve))
        return slice_database

    def _visualize_b1_weighting(self):
        fig = plt.figure(figsize=(12, 6))
        fig.suptitle(f"")
        gs_y = int(len(self.b1_values) / 2)
        gs = fig.add_gridspec(2, gs_y + 1)
        for k in range(len(self.b1_values)):
            ax = fig.add_subplot(gs[k])
            ax.axis(False)
            ax.set_title(f"B1: {self.b1_values[k]}")
            img = ax.imshow(self.b1_weighting_matrix[:, :, k])
        ax_cb = fig.add_subplot(gs[-1])
        plt.colorbar(img, cax=ax_cb)
        plt.tight_layout()
        plt.show()


class B1Prior(B1Weight):
    def __init__(self, data_slice_shape: tuple, database_pandas: pd.DataFrame,
                 b1_weighting: bool = True, b1_weight_factor: float = 0.1,
                 visualize: bool = True,
                 # special args:
                 b1_weight_width: float = 1.1):
        super().__init__(
            data_slice_shape=data_slice_shape, database_pandas=database_pandas,
            b1_weighting=b1_weighting, b1_weight_factor=b1_weight_factor,
            visualize=visualize
        )
        logModule.info("B1Prior -- Initialize")
        # if not set shape width for b1 prior
        self.width = b1_weight_width
        # create in-plane pos. dep. weighting
        self.b1_weighting_matrix = self._set_slice_weighting()

    def module_set_b1_weighting_matrix(self, slice_id: int = 0):
        # reset module dependent slice b1 weighting
        # nothing to do here! independent of slice position
        return self.weighting_factor * self.b1_weighting_matrix

    def set_weight_width(self, weight: float, width: float):
        self.weighting_factor = weight
        self.width = width

    def _set_slice_weighting(self):
        # specifier not needed, take the same for every slice: could use different shape dep. on slice pos!
        logModule.info("B1Prior -- set slice b1 weighting")
        b1_weighting_profile = self._create_gauss_b1_matrix()
        slice_db_b1_weight = np.zeros([*self.data_shape, len(self.b1_values)])
        for idx_b1 in range(len(self.b1_values)):
            slice_db_b1_weight[:, :, idx_b1] = np.abs(self.b1_values[idx_b1] - b1_weighting_profile)
        return slice_db_b1_weight

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


class B1Input(B1Weight):
    def __init__(self, data_slice_shape: tuple, database_pandas: pd.DataFrame,
                b1_weighting: bool = True, b1_weight_factor: float = 0.1,
                visualize: bool = True,
                # special args:
                input_path: str = "",
                input_scaling: float = 1e-2):
        super().__init__(
            data_slice_shape=data_slice_shape, database_pandas=database_pandas,
            b1_weighting=b1_weighting, b1_weight_factor=b1_weight_factor,
            visualize=visualize
        )

        path = Path(input_path).absolute()
        if path.is_file():
            self.input_path = path.__str__()
        else:
            err = f"provided input path not a file: {input_path}"
            logModule.error(err)
            raise AttributeError(err)

        b1_nii_data, self.b1_nii_image = utils.niiDataLoader(self.input_path, test_set=False, normalize="")
        self.b1_weighting_matrix = input_scaling * b1_nii_data

    def module_set_b1_weighting_matrix(self, slice_id: int = 0):
        # reset module dependent weighting matrix
        # specifier needed, dep. on slice pos!
        logModule.info("B1Input -- set slice b1 weighting")
        # need to match len b1s in last dim [x, y, len_b1s]
        slice_db_b1_weight = self.b1_weighting_matrix[:, :, slice_id]
        # need to output dimensions [x, y, num_b1_values]
        return self.weighting_factor * np.repeat(slice_db_b1_weight[:, :, np.newaxis], len(self.b1_values), axis=-1)


def set_b1_weighting(opts: options.FitOptions.opts,
                     data_slice_shape: tuple,
                     database_pandas: pd.DataFrame,
                     b1_weight_factor: float = 0.1) -> B1Weight:
    if opts.FitB1WeightingInput:
        b1_weight = B1Input(
            data_slice_shape=data_slice_shape,
            database_pandas=database_pandas,
            b1_weighting=opts.FitB1Weighting,
            b1_weight_factor=b1_weight_factor,
            visualize=opts.Visualize,
            input_path=opts.FitB1WeightingInput,
            input_scaling=1e-2
        )
    else:
        b1_weight = B1Prior(
            data_slice_shape=data_slice_shape,
            database_pandas=database_pandas,
            b1_weighting=opts.opts.FitB1Weighting,
            b1_weight_factor=b1_weight_factor,
            visualize=opts.opts.Visualize,
            b1_weight_width=1.1
        )
    return b1_weight
