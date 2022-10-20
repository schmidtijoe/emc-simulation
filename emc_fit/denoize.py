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


def op_id(x_input):
    """ identity operator"""
    return x_input


def _majorante(
        arg_arr: typing.Union[np.ndarray, float, int],
        num_channels: int = 18) -> typing.Union[np.ndarray, float]:
    is_single_val = False
    eps = 1e-5
    if isinstance(arg_arr, (float, int)):
        arg_arr = np.array([arg_arr])
        is_single_val = True
    result = np.zeros_like(arg_arr)

    gam = 7e2
    # for smaller eps result array remains 0
    # for small enough args but bigger than eps we compute the given formula
    sel = np.logical_and(eps < arg_arr, gam > arg_arr)
    result[sel] = np.divide(
        special.iv(num_channels, arg_arr[sel]),
        special.iv(num_channels - 1, arg_arr[sel])
    )
    # for big args we linearly approach asymptote to 1 @ input arg 30000 (random choice
    len_asymptote = 3e4
    start_val = np.divide(
        special.iv(num_channels, gam),
        special.iv(num_channels - 1, gam)
    )
    sel = arg_arr >= gam
    result[sel] = start_val + (1.0 - start_val) / len_asymptote * (arg_arr[sel] - gam)
    if is_single_val:
        result = result[0]
    return result


def _y_tilde(
        y_obs: typing.Union[np.ndarray, float, int],
        x_approx: typing.Union[np.ndarray, float, int],
        sigma: float = 31.0,
        num_channels: int = 16) -> typing.Union[np.ndarray, float]:
    arg = np.multiply(
        y_obs,
        x_approx
    ) / sigma ** 2
    factor = _majorante(arg, num_channels=num_channels)
    return y_obs * factor


def denoize_nii_data(data: np.ndarray, num_iterations: int = 4, mpHeadroom: int = 4,
                     visualize: bool = True, save_plot: str = ""):
    logModule.info("extract Noise characteristics")
    ncChi, snrMap = handlers.extract_chi_noise_characteristics_from_nii(
        niiData=data,
        visualize=visualize,
        corner_fraction=15.0
    )

    if visualize:
        # plot curve selection
        plots.plot_curve_selection(data=data, noise_mean=ncChi.mean(0))

    # majorize nc-chi problem -> becomes least squares problem
    # can solve this with least squares solver eg: chambollepock algorithm,
    # get additionally a total variation (TV) term
    mp_list = []
    for phase_idx in tqdm.trange(data.shape[1], desc="prepare mp"):
        mp_list.append([data[:, phase_idx], phase_idx, num_iterations, ncChi])

    num_cpus = mp.cpu_count() - mpHeadroom
    logModule.info(f"multiprocessing using {num_cpus} cpus")
    with mp.Pool(num_cpus) as p:
        results = list(tqdm.tqdm(p.imap(denoize_wrap_mp, mp_list), total=data.shape[1], desc="mp pes"))

    d_data = np.zeros_like(data)
    for mp_idx in tqdm.trange(data.shape[1], desc="join mp"):
        phase_idx = results[mp_idx][0]
        d_data[:, phase_idx] = results[mp_idx][1]

    if visualize:
        plots.plot_denoized(origData=data, denoizedData=d_data, save=save_plot)

    return d_data


def denoize_wrap_mp(args):
    data, idx, num_iterations, ncChi = args
    y = data.copy()
    x = data.copy()
    for _ in range(num_iterations):
        y = _y_tilde(y_obs=y, x_approx=x, sigma=ncChi.sigma, num_channels=ncChi.num_channels)
        x = chambollepock.chambolle_pock_tv(y, 0.05, n_it=25, return_all=False)
    return idx, x
