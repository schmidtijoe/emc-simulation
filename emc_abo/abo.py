"""
framweork for optimization
"""
import numpy as np
import pathlib
import typing
from emc_sim import options as eo
from emc_sim import simulations as es
import logging
import multiprocessing as mp
import tqdm
import scipy.optimize as spo

logModule = logging.getLogger(__name__)


class Optimizer:
    def __init__(self,
                 config_path: typing.Union[str, pathlib.Path] = "",
                 multiprocessing: bool = False, mp_num_cpus: int = 4):
        logModule.info("Init Optimizer")
        self.config_path: pathlib.Path = pathlib.Path(config_path).absolute()
        if self.config_path.is_file():
            self.emc_params: eo.SimulationParameters = eo.SimulationParameters.load(config_path)
        else:
            self.emc_params: eo.SimulationParameters = eo.SimulationParameters()
        self.emc_tmp_data: eo.SimulationData = eo.SimulationData()

        # bounds and initial guess
        self.bounds_fa: tuple = (90.0, 180.0)
        self.bounds_phase: tuple = (-180.0, 180.0)
        self.bounds: list = [
            *[self.bounds_fa]*self.emc_params.sequence.ETL,
            *[self.bounds_phase]*self.emc_params.sequence.ETL
        ]

        self.rf_0: np.ndarray = np.zeros(2*self.emc_params.sequence.ETL)
        self.rf_0[:self.emc_params.sequence.ETL] = np.array(self.emc_params.sequence.refocusAngle)

        # for multiprocessing setup
        self.mp : bool = multiprocessing
        max_cpus = np.max([mp.cpu_count() - 8, 4])        # leave at least 8 threads, take at least 4
        self.num_cpus = np.min([mp_num_cpus, max_cpus])

        # for optimization
        self.opt_max_iter: int = 20

        # result
        self.result: np.ndarray = np.zeros_like(self.rf_0)

    def _func_to_optimize(self, x: np.ndarray):
        self.emc_params.refocusAngle = x[:self.emc_params.sequence.ETL].tolist()
        self.emc_params.refocusPhase = x[self.emc_params.sequence.ETL:].tolist()

        # calculate database
        param_list = self.emc_params.settings.get_complete_param_list()
        idx_list = np.arange(len(param_list))

        def wrap_for_mp(idx):
            self.emc_tmp_data.set_run_params(*param_list[idx])
            emcAmplitude, _ = es.simulate_mese(
                simParams=self.emc_params,
                simData=self.emc_tmp_data
            )
            return emcAmplitude.emcSignal
        logModule.debug("Simulate")

        if self.mp:
            logModule.info(f"Multiprocessing using {self.num_cpus} threads")
            with mp.Pool(self.emc_params.config.mpNumCpus) as p:
                results = list(tqdm.tqdm(p.imap(wrap_for_mp, idx_list), total=param_list.__len__(), desc="mp processing sim"))
        else:
            logModule.info("Single thread processing")
            results = []
            for idx in tqdm.trange(idx_list.__len__(), desc="processing sim"):
                results.append(wrap_for_mp(idx))

        signal_curves = np.zeros((len(results), self.emc_params.sequence.ETL))

        for idx in tqdm.trange(len(results), desc="read mp results"):
            signal_curves[idx] = results[idx]

        # calculate correlation
        corr_matrix = np.corrcoef(signal_curves)
        objective = np.sum(corr_matrix)  # diagonals always sum to 1
        return objective

    def optimize(self):
        self.result = spo.differential_evolution(
            self._func_to_optimize, bounds=self.bounds,
            maxiter=20, x0=self.rf_0
        )

    def get_optimal_pulse_specs(self):
        return self.result

    def test_obj_func(self):
        return self._func_to_optimize(self.rf_0)


if __name__ == '__main__':
    abo_optimizer = Optimizer(config_path="config/emc_config.json")
    test_objective = abo_optimizer.test_obj_func()
    print(test_objective)
    abo_optimizer.optimize()
    test_res = abo_optimizer.get_optimal_pulse_specs()
    print(test_res)
