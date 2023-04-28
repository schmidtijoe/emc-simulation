"""
Denoizing algorithm for magnitude image data subject to multi-channel receive SoS creation.
According to Dietrich et al. (2008) Influence of multichannel combination, parallel imaging and other reconstruction techniques on MRI noise characteristics
follows a non-central chi distribution.
Denoizing is done following Varadarajan et al. (2015) A Majorize-Minimize Framework for Rician and Non-Central Chi MR Images
"""
import numpy as np
from emc_fit.noise import handlers, chambollepock
from emc_fit import plots
import logging
import typing
from scipy import special
import tqdm
import multiprocessing as mp

logModule = logging.getLogger(__name__)


class MajMinNcChiDenoizer:
    def __init__(self, max_num_runs: int = 4, use_mp: bool = True, mp_headroom: int = 4,
                 visualize: bool = True, single_iteration: bool = True):
        self.single_iteration: bool = single_iteration
        self.num_channels: int = NotImplemented
        self.sigma: float = NotImplemented
        self.gam: float = 7e2      # set from which signal size onwards we approx with gaussian behavior
        self.eps: float = 1e-5     # for comparing 0
        self.num_cp_runs: int = max_num_runs   # set number of runs
        self.use_mp: bool = use_mp
        self.mp_headroom: int = mp_headroom
        self.chambolle_pock_lambda: float = 0.05        # set influence of TV part of algorithm
        self.chambolle_pock_num_iter: int = 25          # set number of iterations of algorithm per run
        self.visualize: bool = visualize
        self.nc_chi_mean_noise: float = -1.0
        self.first_run: bool = True

    # private
    @staticmethod
    def _op_id(x_input):
        """ identity operator"""
        return x_input

    def _majorante(
            self, arg_arr: typing.Union[np.ndarray, float, int]) -> typing.Union[np.ndarray, float]:
        is_single_val = False

        if isinstance(arg_arr, (float, int)):
            arg_arr = np.array([arg_arr])
            is_single_val = True
        result = np.zeros_like(arg_arr)

        # for smaller eps result array remains 0
        # for small enough args but bigger than eps we compute the given formula
        sel = np.logical_and(self.eps < arg_arr, self.gam > arg_arr)
        result[sel] = np.divide(
            special.iv(self.num_channels, arg_arr[sel]),
            special.iv(self.num_channels - 1, arg_arr[sel])
        )
        # for big args we linearly approach asymptote to 1 @ input arg 30000 (random choice
        len_asymptote = 3e4
        start_val = np.divide(
            special.iv(self.num_channels, self.gam),
            special.iv(self.num_channels - 1, self.gam)
        )
        sel = arg_arr >= self.gam
        result[sel] = start_val + (1.0 - start_val) / len_asymptote * (arg_arr[sel] - self.gam)
        if is_single_val:
            result = result[0]
        return result

    def _y_tilde(
            self,
            y_obs: typing.Union[np.ndarray, float, int],
            x_approx: typing.Union[np.ndarray, float, int]) -> typing.Union[np.ndarray, float]:
        arg = np.multiply(
            y_obs,
            x_approx
        ) / self.sigma ** 2
        factor = self._majorante(arg)
        return y_obs * factor

    def _denoize_wrap_mp(self, args):
        data, idx = args
        y = data.copy()
        x = data.copy()
        if self.single_iteration:
            iterate = 1
        else:
            iterate = self.num_cp_runs
        for _ in range(iterate):
            y = self._y_tilde(y_obs=y, x_approx=x)
            x = chambollepock.chambolle_pock_tv(
                y, self.chambolle_pock_lambda, n_it=self.chambolle_pock_num_iter, return_all=False
            )
        return idx, x

    # public
    def set_channels_sigma(self, num_channels: int, sigma: float):
        self.num_channels = num_channels
        self.sigma = sigma

    def get_nc_stats(self, data: np.ndarray):
        logModule.info("extract Noise characteristics")
        nc_chi, _ = handlers.extract_chi_noise_characteristics_from_nii(
            niiData=data,
            visualize=self.visualize,
            corner_fraction=12.0
        )

        if self.visualize and self.first_run:
            # plot curve selection
            plots.plot_curve_selection(data=data, noise_mean=nc_chi.mean(0))
            self.first_run = False
        self.num_channels = nc_chi.num_channels
        self.sigma = nc_chi.sigma
        self.nc_chi_mean_noise = nc_chi.mean(0)

    def denoize_nii_data(self, data: np.ndarray, save_plot: str = ""):
        # majorize nc-chi problem -> becomes least squares problem
        # can solve this with least squares solver eg: chambollepock algorithm,
        # get additionally a total variation (TV) term
        mp_list = []
        for phase_idx in tqdm.trange(data.shape[1], desc="prepare mp"):
            mp_list.append([data[:, phase_idx], phase_idx])

        num_cpus = np.max([4, mp.cpu_count() - self.mp_headroom])  # take at least 4, leave mp Headroom
        logModule.info(f"multiprocessing using {num_cpus} cpus")
        with mp.Pool(num_cpus) as p:
            results = list(tqdm.tqdm(p.imap(self._denoize_wrap_mp, mp_list), total=data.shape[1], desc="mp pes"))

        d_data = np.zeros_like(data)
        for mp_idx in tqdm.trange(data.shape[1], desc="join mp"):
            phase_idx = results[mp_idx][0]
            d_data[:, phase_idx] = results[mp_idx][1]

        if self.visualize:
            plots.plot_denoized(origData=data, denoizedData=d_data, save=save_plot)

        return d_data

    def check_low_noise(self, data_max: float, snr: float = 20.0) -> bool:
        acceptable_noise = data_max / snr
        current_noise_level = self.nc_chi_mean_noise
        return current_noise_level < acceptable_noise
