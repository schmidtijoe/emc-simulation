"""
Non Central Chi distribution, pdf and sampling function

Using definitions from Dietrich et al. Magn Reson Imag 2008
_____
Jochen Schmidt 09.08.2021
"""
from scipy.stats import chi
from scipy.special import ive, factorial, factorial2, hyp1f1
import numpy as np
import logging
import typing

logModule = logging.getLogger(__name__)


class NcChi:
    """nc-chi distribution"""

    def __init__(self, name: str = 'nc_chi_distribution'):
        self.num_channels: int = -1
        self.sigma: float = 0.0
        self.log_sigma: float = 1.0
        self.name: str = name
        self._mean_factor: float = 1.0
        self._sigma_min: float = 10.0
        self._sigma_max: float = -10.0
        self._amp_sig: typing.Union[int, float, np.ndarray] = 1.0
        self._amp_sig_min: float = 10.0
        self._amp_sig_max: float = -10.0

    def set_channels(self, num_channels: int):
        self.num_channels = num_channels
        # calculate all things we can once when channel changes and keep them as constants
        denominator = np.power(2, self.num_channels - 1) * factorial(self.num_channels - 1)
        numerator = factorial2(2 * self.num_channels - 1) * np.sqrt(np.pi / 2)
        self._mean_factor = numerator / denominator

    def set_sigma(self, sigma):
        if sigma < self._sigma_min:
            self._sigma_min = sigma
        if sigma > self._sigma_max:
            self._sigma_max = sigma
        self.sigma = sigma
        self.log_sigma = - 2 * self._log_func(self.sigma)

    def get_stats(self, evaluate: bool = False) -> (int, float):
        logModule.info(f"___{self.name}___: \t"
                       f"number of uncorrelated channels: {self.num_channels}; \t"
                       f"sigma: {self.sigma:.2f}")
        if evaluate:
            logModule.info(f"sigma values :\n"
                           f"min: {self._sigma_min:.4f}\t max: {self._sigma_max:.4f}\t"
                           f"amp values (z**2 / 2 sigma **2):\t"
                           f"min: {self._amp_sig_min:.4f}\t max: {self._amp_sig_max:.4f}"
                           )
        return self.num_channels, self.sigma

    def fit_noise(self, noise_values):
        # normalize nosie vals
        data = noise_values.copy()
        norm = np.linalg.norm(data)
        data = np.divide(data, norm)
        params = chi.fit(data, 20, floc=0)
        if params[0] < 4:
            params = chi.fit(data, f0=4, floc=0)
        self.set_channels(int(round(params[0] / 2, 0)))
        self.set_sigma(norm*params[2])

    def pdf(self, x: typing.Union[int, float, np.ndarray],
            amplitude: typing.Union[int, float, np.ndarray]) -> np.ndarray:
        small_limit = 1e-6
        result_sl = chi(2 * self.num_channels, loc=amplitude, scale=self.sigma).pdf(x)
        log_res = self._log(x, amplitude)
        result = np.exp(log_res)
        if isinstance(amplitude, (float, int)):
            if amplitude < small_limit:
                result = result_sl
        else:
            result[amplitude < small_limit] = result_sl[amplitude < small_limit]
        return result

    @staticmethod
    def _log_func(val: typing.Union[int, float, np.ndarray]) -> typing.Union[float, np.ndarray]:
        return np.log(val, where=val > 0, out=np.full_like(np.array(val, dtype=float), -np.inf))

    def _square_min_sigma(self) -> float:
        return np.square(np.max([1e-4, self.sigma]))

    def _log(self, x: typing.Union[int, float, np.ndarray],
             amplitude: typing.Union[int, float, np.ndarray]) -> typing.Union[int, float, np.ndarray]:
        a = self._log_func(amplitude)
        # b = log sigma
        c = self.num_channels * self._log_func(x)
        d = - self.num_channels * self._log_func(amplitude)
        e = - (x ** 2 + amplitude ** 2) / (2 * self._square_min_sigma())
        f = ive(self.num_channels - 1, x * amplitude / self._square_min_sigma())
        log_f = self._log_func(f) + x * amplitude / self._square_min_sigma()
        return a + self.log_sigma + c + d + e + log_f

    def sampler(self, amplitude: typing.Union[int, float, np.ndarray],
                size: int = 1, p_sample_size: int = 1000) -> typing.Union[int, float, np.ndarray]:
        """
        This function samples from the non central chi distribution.
        If the amplitude is 0 a quicker sample is drawn from the builtin chi distribution.

        :param amplitude: value for the amplitude, aka unbiased signal point ("ground truth")
        :param size: number of samples drawn for that point
        :param p_sample_size: resolution, i.e. number of points, of pdf to draw from
        :return:
        """
        small_limit: float = 1e-5
        single_val: bool = False
        # check if singular value or array
        if isinstance(amplitude, (float, int)):
            single_val = True
            amplitude = np.array([amplitude])
            shape = (1,)
        else:
            shape = amplitude.shape
        # instantiate result array
        amplitude = np.reshape(amplitude, -1)
        result = np.zeros([len(amplitude), size])
        # check for 0 or small values in the amplitude array, use builtin chi disto for sampling (quicker)
        if len(amplitude[amplitude < small_limit]) > 0:
            result[amplitude < small_limit] = chi(2 * self.num_channels,
                                                  loc=amplitude[amplitude < small_limit],
                                                  scale=self.sigma).rvs(size=(len(amplitude[amplitude < small_limit]),
                                                                              size))

        # use drawing from pdf built per point for nonzero points
        if len(amplitude[amplitude >= small_limit]) > 0:
            result[amplitude >= small_limit] = self._result_for_nonzero(
                amplitude[amplitude >= small_limit],
                size=size,
                p_sample_size=p_sample_size
            )
        # reshaping
        result = np.reshape(result, [*shape, size])
        # some sanity casting, size axis always last axis, independent of shape
        if size == 1:
            result = np.moveaxis(result, -1, 0)[0]
        if single_val:
            result = result[0]
        return result

    def _result_for_nonzero(self, amps: np.ndarray,
                            size: int = 1, p_sample_size: int = 1000) -> np.ndarray:
        # define maximum value to which pdf is built
        max_val = np.max(amps) + 10.0 * self.sigma
        # init x axis
        _x = np.linspace(0, max_val, p_sample_size)
        # init result array
        draw = np.zeros([len(amps), size])
        # for each amplitude point: build the pdf until the max value, normalize, and draw sample from it
        for k_idx in range(len(amps)):
            _pdf = self.pdf(_x, amps[k_idx])
            _pdf = np.divide(_pdf, np.sum(_pdf))
            draw[k_idx] = np.random.choice(_x, size=size, p=_pdf)
        return draw

    def mean(self, amplitude: typing.Union[int, float, np.ndarray]):
        self._amp_sig = amplitude ** 2 / (2 * self.sigma ** 2)
        if np.max(self._amp_sig) > self._amp_sig_max:
            self._amp_sig_max = np.max(self._amp_sig)
        if np.min(self._amp_sig) < self._amp_sig_min:
            self._amp_sig_min = np.min(self._amp_sig)
        return self.sigma * self._mean_factor * hyp1f1(-0.5, self.num_channels, -self._amp_sig)

    def sigma_from_snr(self, f_snr, max_value=1):
        # snr defined as curve_maximum_value / noise_floor_mean
        if f_snr > 0.0:
            noise_floor_mean = max_value / f_snr
            self.set_sigma(noise_floor_mean / self._mean_factor)
        else:
            self.set_sigma(1.0)

    def reset_value_tracker(self):
        self._sigma_min = 10.0
        self._sigma_max = -10.0
        self._amp_sig_min = 10.0
        self._amp_sig_max = -10.0

    def likelihood(self, x: typing.Union[int, float, np.ndarray],
                   amplitude: typing.Union[int, float, np.ndarray]) -> typing.Union[float, np.ndarray]:
        if isinstance(x, float):
            return self._log(x, amplitude)
        else:
            return np.prod([self._log(x_i, amplitude) for x_i in x])


if __name__ == '__main__':
    # Testing
    snr_array = np.geomspace(0.8, 20, 1000)
    amp_array = np.linspace(0, 1, 1000)
    nc_chi = NcChi()
    nc_chi.set_channels(17)
    for snr in snr_array:
        nc_chi.sigma_from_snr(snr)
        nc_chi.mean(amp_array)

    print(nc_chi.get_stats())

    print(nc_chi.mean(1000))

    db = np.reshape(np.arange(160), (10, 16))
    db = np.divide(db, np.max(db))
    re_db = nc_chi.sampler(db, size=1)
    re2_db = nc_chi.sampler(db, size=3)
    re_db3 = nc_chi.sampler(0.8)
